# SPDX-License-Identifier: MIT
"""
PyTorch DataLoader example for CNNP V2 files.

Wraps `cnnp.Reader.get_batch()` as an IterableDataset so PyTorch's
DataLoader can pull pre-batched, ready-for-`torch.from_numpy` arrays in
a single C++ call per batch — avoiding the per-position Python overhead
that would otherwise starve the GPU.

Usage:
    PYTHONPATH=build/python python examples/python/cnnp_dataset.py \
        --cnnp test_data/farseerT76_50m.cnnp \
        [--batch-size 16384] [--max-batches 200] [--cpu] [--num-workers 0]

The script prints two benchmarks:
  Phase 1 — data-only iteration (no GPU): how fast can we feed batches?
  Phase 2 — full loop with a tiny dummy NNUE: where does wall time go?
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

import cnnp

PADDING_SENTINEL = 65535  # UINT16_MAX — what get_batch() uses for unused slots


# ─── Dataset ──────────────────────────────────────────────────────────────────

class CNNPBatchedDataset(IterableDataset):
    """IterableDataset that yields full batches via Reader.get_batch().

    Use with `DataLoader(..., batch_size=None)` so PyTorch doesn't try to
    re-batch (each `__iter__` step already returns a complete batch).

    Multi-worker safe: the reader is opened LAZILY in `__iter__`, so each
    DataLoader worker process gets its own mmap (the Reader holds an OS
    file handle that can't be pickled across the worker fork/spawn).
    Each worker handles a disjoint stride of batches.
    """

    def __init__(self, path: str, batch_size: int = 16384,
                 shuffle: bool = True, seed: int = 0):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        # Cache the position count so __len__ works without opening the
        # reader in every worker. The temporary reader is GC'd at the end
        # of this expression.
        self._num_positions = cnnp.Reader(path, validate=False).num_positions

    def __len__(self) -> int:
        return (self._num_positions + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        # Open the reader fresh per __iter__ — works for num_workers=0
        # (single-process) and num_workers>0 (one reader per worker).
        reader = cnnp.Reader(self.path, validate=False)

        info = torch.utils.data.get_worker_info()
        worker_id   = info.id          if info is not None else 0
        num_workers = info.num_workers if info is not None else 1

        N = reader.num_positions
        rng = np.random.default_rng(self.seed + worker_id)
        indices = rng.permutation(N) if self.shuffle else np.arange(N, dtype=np.int64)

        # Each worker takes every num_workers-th BATCH (stride by batch_size).
        first = worker_id * self.batch_size
        step  = num_workers * self.batch_size
        for start in range(first, N, step):
            chunk = indices[start:start + self.batch_size]
            if len(chunk) == 0:
                break
            yield reader.get_batch(chunk.astype(np.uint64))


# ─── Tiny NNUE-flavored model (just to exercise GPU compute) ─────────────────

def build_dummy_nnue(max_feature_id: int = 768, hidden: int = 256):
    """Sum-of-embeddings + 2 dense layers. ~250k params; mimics a small NNUE
    architecture's data flow but is NOT trained for chess. Use as a proxy
    for measuring forward+backward cost."""
    import torch
    import torch.nn as nn

    class DummyNNUE(nn.Module):
        def __init__(self):
            super().__init__()
            # Embedding has max_feature_id+1 rows so that PADDING_SENTINEL
            # (after remapping to max_feature_id) gets a zero-row entry that
            # contributes nothing to the sum.
            self.embedding = nn.Embedding(
                max_feature_id + 1, hidden, padding_idx=max_feature_id)
            self.fc1 = nn.Linear(hidden, hidden)
            self.fc2 = nn.Linear(hidden, 1)
            self.act = nn.ReLU()
            self.max_feature_id = max_feature_id

        def forward(self, features, counts):
            # features: (B, 32) int64, with PADDING_SENTINEL in unused slots.
            # Remap sentinel → padding_idx so the embedding emits zeros there.
            features = torch.where(
                features == PADDING_SENTINEL,
                torch.tensor(self.max_feature_id, dtype=features.dtype,
                             device=features.device),
                features)
            embs = self.embedding(features)   # (B, 32, hidden); padding row = 0
            x    = embs.sum(dim=1)            # (B, hidden)
            x    = self.act(self.fc1(x))
            return self.fc2(x).squeeze(-1)    # (B,)

    return DummyNNUE()


# ─── Benchmarks ───────────────────────────────────────────────────────────────

def bench_data_only(path: str, batch_size: int, max_batches: int | None,
                    num_workers: int):
    """Just iterate batches and count positions — no GPU, no model."""
    import torch
    from torch.utils.data import DataLoader

    ds = CNNPBatchedDataset(path, batch_size=batch_size)
    if num_workers > 0:
        loader = DataLoader(ds, batch_size=None, num_workers=num_workers,
                            persistent_workers=True)
    else:
        loader = ds  # iterate directly, no DataLoader wrapping

    total = 0
    n_batches = 0
    t0 = time.perf_counter()
    for batch in loader:
        # `batch` is a dict of NumPy arrays
        total += len(batch["counts"])
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    elapsed = time.perf_counter() - t0
    return dict(positions=total, elapsed_s=elapsed,
                pos_per_s=total / max(elapsed, 1e-9),
                ms_per_batch=1000 * elapsed / max(n_batches, 1),
                n_batches=n_batches)


def bench_full_loop(path: str, batch_size: int, max_batches: int | None,
                    device: str, num_workers: int, max_feature_id: int = 768):
    """Fetch + GPU forward + backward + optimiser step. Reports time split."""
    import torch
    from torch.utils.data import DataLoader

    model    = build_dummy_nnue(max_feature_id=max_feature_id).to(device)
    opt      = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn  = torch.nn.MSELoss()

    ds = CNNPBatchedDataset(path, batch_size=batch_size)
    if num_workers > 0:
        loader = DataLoader(ds, batch_size=None, num_workers=num_workers,
                            persistent_workers=True)
    else:
        loader = ds

    # Warmup (1 batch, ignored). Compiles CUDA kernels, fills caches.
    for batch in loader:
        f = torch.from_numpy(batch["features"]).long().to(device)
        c = torch.from_numpy(batch["counts"]).to(device)
        t = torch.from_numpy(batch["evals_norm"]).to(device)
        pred = model(f, c); loss = loss_fn(pred, t)
        loss.backward(); opt.step(); opt.zero_grad()
        if device == "cuda": torch.cuda.synchronize()
        break

    t_fetch = 0.0
    t_gpu   = 0.0
    total   = 0
    n_batches = 0
    wall_t0 = time.perf_counter()
    fetch_t0 = time.perf_counter()
    for batch in loader:
        t_fetch += time.perf_counter() - fetch_t0

        gpu_t0 = time.perf_counter()
        f = torch.from_numpy(batch["features"]).long().to(device, non_blocking=True)
        c = torch.from_numpy(batch["counts"]).to(device, non_blocking=True)
        t = torch.from_numpy(batch["evals_norm"]).to(device, non_blocking=True)
        pred = model(f, c); loss = loss_fn(pred, t)
        loss.backward(); opt.step(); opt.zero_grad()
        if device == "cuda": torch.cuda.synchronize()
        t_gpu += time.perf_counter() - gpu_t0

        total     += f.shape[0]
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
        fetch_t0 = time.perf_counter()

    wall = time.perf_counter() - wall_t0
    return dict(
        positions=total, elapsed_s=wall, n_batches=n_batches,
        pos_per_s=total / max(wall, 1e-9),
        ms_per_batch_fetch=1000 * t_fetch / max(n_batches, 1),
        ms_per_batch_gpu  =1000 * t_gpu   / max(n_batches, 1),
        fetch_pct=100 * t_fetch / max(wall, 1e-9),
        gpu_pct  =100 * t_gpu   / max(wall, 1e-9))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cnnp", required=True, help="path to .cnnp file")
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--max-batches", type=int, default=None,
                   help="cap batches per pass (default: full file)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (default 0 = single-threaded)")
    p.add_argument("--cpu", action="store_true",
                   help="force CPU even if CUDA is available")
    args = p.parse_args()

    print(f"=== CNNPDataset benchmark ===")
    print(f"  file        : {args.cnnp}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  num_workers : {args.num_workers}")

    print(f"\n[Phase 1] data-only iteration (no GPU)")
    r1 = bench_data_only(args.cnnp, args.batch_size,
                         args.max_batches, args.num_workers)
    print(f"  positions   : {r1['positions']:,}")
    print(f"  batches     : {r1['n_batches']:,}")
    print(f"  elapsed     : {r1['elapsed_s']:.2f} s")
    print(f"  throughput  : {r1['pos_per_s']/1e6:.2f} M pos/s")
    print(f"  per batch   : {r1['ms_per_batch']:.2f} ms")

    import torch
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    label  = (f"CUDA ({torch.cuda.get_device_name(0)})"
              if use_cuda else "CPU")

    print(f"\n[Phase 2] full loop on {label}")
    r2 = bench_full_loop(args.cnnp, args.batch_size, args.max_batches,
                         device, args.num_workers)
    print(f"  positions   : {r2['positions']:,}")
    print(f"  batches     : {r2['n_batches']:,}")
    print(f"  elapsed     : {r2['elapsed_s']:.2f} s")
    print(f"  throughput  : {r2['pos_per_s']/1e6:.2f} M pos/s")
    print(f"  fetch       : {r2['ms_per_batch_fetch']:7.2f} ms/batch"
          f"  ({r2['fetch_pct']:5.1f}% wall)")
    print(f"  compute     : {r2['ms_per_batch_gpu']:7.2f} ms/batch"
          f"  ({r2['gpu_pct']:5.1f}% wall)")

    print()
    if r2["fetch_pct"] > r2["gpu_pct"]:
        print(f"  ⚠ Data fetch dominates ({r2['fetch_pct']:.1f}% vs "
              f"{r2['gpu_pct']:.1f}%). Try --num-workers > 0 or larger batch.")
    else:
        print(f"  ✓ Compute-bound. Data pipeline keeps up "
              f"({r2['fetch_pct']:.1f}% fetch vs {r2['gpu_pct']:.1f}% compute).")


if __name__ == "__main__":
    main()
