# SPDX-License-Identifier: MIT
"""
make_val_splits.py — generate deterministic fixed validation index files.

A fixed val split lets you compare training runs that vary `--seed`
(or any other training knob) without confounding the comparison with
val-set variance. Run this once per dataset, then pass the resulting
.npy file to the trainer via `--val-indices`.

Usage:
    PYTHONPATH=build/python python examples/python/make_val_splits.py \\
        --cnnp test_data/farseerT76_1b.cnnp \\
        --output-dir test_data/val_splits \\
        --size 500000 \\
        --seed 20260513

Output file is named to encode the dataset size, so a wrong file/dataset
pairing trips the bounds check inside the trainer:

    test_data/val_splits/val_500k_seed20260513_n1000000000.npy

The indices are produced by Floyd's algorithm — O(k) memory and time,
no permutation(N) anywhere — so this script runs in milliseconds even
for 1B-position datasets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _format_size(n: int) -> str:
    if n >= 1_000_000:
        return f'{n // 1000}k' if n % 1_000_000 else f'{n // 1_000_000}M'
    if n >= 1000:
        return f'{n // 1000}k'
    return str(n)


def main():
    p = argparse.ArgumentParser(
        description='Generate a deterministic fixed validation index file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--cnnp', required=True,
                   help='Path to the CNNP V2 file the indices will reference.')
    p.add_argument('--output-dir', default='test_data/val_splits',
                   help='Directory to write the .npy file into.')
    p.add_argument('--size', type=int, default=500_000,
                   help='Number of distinct val indices to sample.')
    p.add_argument('--seed', type=int, default=20260513,
                   help='Sampling seed. Default is the date of first use; '
                        'pin this in commit history if you regenerate.')
    p.add_argument('--cnnp-module-dir', default='build/python',
                   help='Path containing cnnp.<ext> (pybind11 build output).')
    p.add_argument('--force', action='store_true',
                   help='Overwrite if output file already exists.')
    args = p.parse_args()

    # Load cnnp module
    sys.path.insert(0, args.cnnp_module_dir)
    import cnnp

    # Pull sample_unique_sorted from the trainer module (same directory).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_nnue  # noqa: E402

    # Determine dataset size
    print(f'Reading {args.cnnp}...', flush=True)
    r = cnnp.Reader(args.cnnp, validate=False)
    n = int(r.num_positions)
    print(f'  num_positions: {n:,}', flush=True)

    if args.size <= 0:
        raise SystemExit('error: --size must be positive')
    if args.size > n:
        raise SystemExit(f'error: --size ({args.size:,}) exceeds num_positions ({n:,})')

    # Prepare output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f'val_{_format_size(args.size)}_seed{args.seed}_n{n}.npy'
    out_path = out_dir / fname

    if out_path.exists() and not args.force:
        raise SystemExit(
            f'error: {out_path} already exists. Pass --force to overwrite, '
            f'or pick a different --seed.')

    # Generate (Floyd, O(k) memory)
    print(f'Sampling {args.size:,} unique indices (seed={args.seed})...', flush=True)
    idx = train_nnue.sample_unique_sorted(n, args.size, args.seed)

    # Sanity
    assert idx.dtype == np.int64
    assert len(idx) == args.size
    assert len(set(idx.tolist())) == args.size, 'duplicates in Floyd output'
    assert (np.diff(idx) > 0).all(), 'output not strictly sorted'
    assert int(idx.min()) >= 0 and int(idx.max()) < n

    np.save(out_path, idx)

    print('')
    print(f'Generated: {out_path}')
    print(f'  size       : {len(idx):,}')
    print(f'  dtype      : {idx.dtype}')
    print(f'  min index  : {int(idx.min()):,}')
    print(f'  max index  : {int(idx.max()):,}')
    print(f'  bytes      : {out_path.stat().st_size:,}')
    print('')
    print('Use it with:')
    print(f'  --val-indices {out_path}')


if __name__ == '__main__':
    main()
