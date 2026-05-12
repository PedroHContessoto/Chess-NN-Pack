# SPDX-License-Identifier: MIT
"""
Drop-in adapter to feed CNNP V2 files into the Pelanca v25 NNUE training
notebook (`pelanca_nnue_v25_pure_t76.ipynb`) without touching the model,
loss, batcher, or training loop.

The v25 pipeline currently uses `FastNnueSparseDataset` which loads an
HDF5 sparse_v1 file into RAM as numpy arrays and exposes:

    n            : int            total positions
    evals        : float32[n]     eval (white-POV, normalized = cp / 400)
    has_eval     : uint8[n]       mask (all 1s in v25)
    has_wdl      : uint8[n]       mask (all 1s in v25)
    wdl          : float32[n]     game outcome (white-POV, {-1, 0, +1})
    stm          : uint8[n]       side to move (0=white, 1=black)
    piece_count  : uint8[n]       2..32
    offsets      : int64[n+1]     prefix sum into w_flat (CSR)
    w_flat       : uint16[total_f] feature ids (white-POV HalfP). Stored as
                                  uint16 (CNNP native) instead of int16 to
                                  avoid a 38 GB astype copy on huge datasets.
                                  Downstream code that does
                                  `ds.w_flat[flat].astype(np.int64)` works
                                  identically — values 0..767 fit either dtype.
    mirror       : bool           horizontal-mirror augmentation flag

`CnnpPelancaDataset` exposes the SAME public surface but reads from one
or more CNNP V2 files, so the existing `make_batches()` and
`BatchedNnueIterable` continue to work unchanged.

Key conversions performed at load time:
  - CNNP stores eval in side-to-move POV (binpack convention).
    v25 expects white-POV and flips per-position at batch time.
    We invert here so the v25 flip produces the correct STM target.
  - CNNP's `flags_array()` packs stm + piece_count into one byte;
    we unpack into separate `stm` and `piece_count` arrays.
  - CNNP's `block_anchors + block_prefix` (compact O(1) random-access
    scheme) is flattened to a plain CSR `offsets[n+1]` for v25 compat.
  - CNNP's `w_flat` is uint16 (max id ≤ 1024); we cast to int16 (safe
    for HalfP's 0..767 range).

Memory cost is the same as HDF5: everything is materialized in RAM
because the v25 batcher does numpy fancy-indexing over the arrays.
A future "true streaming" variant could use `cnnp.Reader.get_batch()`
inside `_make_batch` to avoid loading w_flat upfront — that would
unlock the full mmap advantage for huge datasets, but requires
modifying `BatchedNnueIterable._make_batch` too.

Usage in the notebook:

    # Cell 4 (data discovery): replace find_hdf5_files() with paths to .cnnp
    CNNP_FILES = ['/workspace/data/farseerT76_50m.cnnp']

    # Cell 5 (eval perspective sanity): set manually since CNNP doesn't
    # carry the perspective in metadata; our adapter always outputs white-POV
    TARGET_IS_WHITE_POV = True

    # Cell 6 (dataset class): replace FastNnueSparseDataset with this adapter
    import sys
    sys.path.insert(0, '/workspace/Chess-NN-Pack/python')
    from cnnp_pelanca import CnnpPelancaDataset
    NnueSparseDataset = CnnpPelancaDataset   # drop-in alias

    # Rest of the notebook unchanged.
"""

from __future__ import annotations

import gc
import time
from typing import List, Optional, Union

import numpy as np

import cnnp


class CnnpPelancaDataset:
    """Loads one or more CNNP V2 files into RAM and exposes the
    `FastNnueSparseDataset` (v25) interface."""

    def __init__(self,
                 cnnp_paths: Union[str, List[str]],
                 split: str = "train",       # accepted but ignored — CNNP has no split
                 max_pos: Optional[int] = None,
                 mirror: bool = False,
                 verbose: bool = True):
        if isinstance(cnnp_paths, str):
            cnnp_paths = [cnnp_paths]
        self.mirror = mirror

        # `split` is in the v25 signature for API parity with HDF5;
        # CNNP files don't carry train/val internally — convert two
        # binpacks (or split positions externally) if you need a split.
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        t0 = time.time()
        readers = [cnnp.Reader(p, validate=False) for p in cnnp_paths]

        # ─── First pass: figure out totals so we can pre-allocate ─────────
        per_file_n = []
        per_file_w = []   # how many features we'll keep from each file
        kept_so_far = 0
        for r in readers:
            n_avail = r.num_positions
            n_take = n_avail if max_pos is None else min(n_avail, max_pos - kept_so_far)
            if n_take <= 0:
                per_file_n.append(0)
                per_file_w.append(0)
                continue

            # Compute features-up-to-position-n_take using anchors+prefix
            # (the END offset of position n_take-1 = start of position n_take).
            BS = r.header["block_size"]
            anchors = r.block_anchors()   # zero-copy uint64 view
            prefix  = r.block_prefix()    # zero-copy uint16 view
            last_i = n_take - 1
            last_b = last_i // BS
            last_j = last_i % BS
            w_take = (int(anchors[last_b])
                      + int(prefix[last_b * (BS + 1) + last_j + 1]))
            per_file_n.append(n_take)
            per_file_w.append(w_take)
            kept_so_far += n_take

        total_n = sum(per_file_n)
        total_w = sum(per_file_w)
        if total_n == 0:
            raise ValueError(
                "CnnpPelancaDataset: no positions to load "
                "(check max_pos / file paths)")

        # ─── Pre-allocate output arrays ───────────────────────────────────
        if verbose:
            est_mem = (total_n * (4 + 1 + 1 + 4 + 1 + 1 + 8) + total_w * 2) / 1e9
            print(f"[CnnpPelancaDataset] allocating ~{est_mem:.2f} GB "
                  f"for {total_n:,} positions, {total_w:,} features...")

        self.evals       = np.empty(total_n, dtype=np.float32)
        self.has_eval    = np.ones(total_n, dtype=np.uint8)   # always 1 in V2
        self.has_wdl     = np.ones(total_n, dtype=np.uint8)   # always 1 in V2
        self.wdl         = np.empty(total_n, dtype=np.float32)
        self.stm         = np.empty(total_n, dtype=np.uint8)
        self.piece_count = np.empty(total_n, dtype=np.uint8)
        self.offsets     = np.empty(total_n + 1, dtype=np.int64)
        self.offsets[0]  = 0
        # uint16 (matches CNNP native) instead of int16 — avoids a 2x peak
        # on .astype during huge-dataset loads. Downstream code casts to
        # int64 anyway for indexing, so the dtype change is transparent.
        self.w_flat      = np.empty(total_w, dtype=np.uint16)

        # Chunk sizes for the load loop. These cap the size of any
        # numpy temporary during the cast/where/divide operations,
        # keeping peak memory close to the final array size + ~1 GB.
        POS_CHUNK = 50_000_000        # 50M positions per chunk (~200 MB temp)
        W_CHUNK   = 200_000_000       # 200M features per chunk (~400 MB temp)

        # ─── Second pass: copy + derive (chunked, in-place where possible) ─
        n_off = 0
        w_off = 0
        for r, n_take, w_take in zip(readers, per_file_n, per_file_w):
            if n_take == 0:
                continue

            # Cache cnnp views (zero-copy refs into the mmap)
            flags_view  = r.flags_array()
            eval_view   = r.eval_array()
            wdl_view    = r.wdl_array()
            wflat_view  = r.w_flat()
            anchors_view = r.block_anchors()
            prefix_view  = r.block_prefix()
            fixed_scale  = float(r.header["fixed_scale"])
            BS           = int(r.header["block_size"])

            # Process per-position fields in chunks (caps temporaries)
            for cs in range(0, n_take, POS_CHUNK):
                ce = min(cs + POS_CHUNK, n_take)
                gs = slice(n_off + cs, n_off + ce)

                # Flags → stm + piece_count (in-place arithmetic into target)
                flags_chunk = np.array(flags_view[cs:ce], copy=True)  # ~50 MB
                np.bitwise_and(flags_chunk, 0x01, out=self.stm[gs])
                pc_tmp = (flags_chunk >> 1) & 0x1F
                np.add(pc_tmp, 2, out=self.piece_count[gs], casting='unsafe')

                # Eval: int16 → float32 / fixed_scale, with STM-POV → white-POV flip
                eraw = np.array(eval_view[cs:ce], copy=True)         # int16, ~100 MB
                etmp = eraw.astype(np.float32)                       # float32, ~200 MB
                etmp /= fixed_scale                                  # in-place
                # Invert eval where stm == 1 (CNNP STM-POV → white-POV)
                black_mask = (flags_chunk & 0x01).astype(bool)
                etmp[black_mask] *= -1                               # in-place
                self.evals[gs] = etmp

                # WDL: int8 → float32 directly (numpy handles cast on assign)
                self.wdl[gs] = wdl_view[cs:ce]

                del flags_chunk, eraw, etmp, pc_tmp, black_mask

            gc.collect()  # release the per-position temporaries

            # Offsets: derive from anchors + prefix (small enough not to chunk;
            # ~10 MB anchors + 4 GB prefix at 1B positions, but we only touch
            # them as int64 indices). For 1B: works in one vectorised pass.
            anchors_arr = np.array(anchors_view, copy=True).astype(np.int64)
            prefix_arr  = np.array(prefix_view,  copy=True).astype(np.int64)
            i = np.arange(n_take, dtype=np.int64)
            b = i // BS
            j = i % BS
            local_starts = anchors_arr[b] + prefix_arr[b * (BS + 1) + j]
            last_b = (n_take - 1) // BS
            last_j_plus_1 = (n_take - 1) % BS + 1
            local_end = (int(anchors_arr[last_b])
                         + int(prefix_arr[last_b * (BS + 1) + last_j_plus_1]))

            self.offsets[n_off:n_off + n_take] = local_starts + w_off
            self.offsets[n_off + n_take]       = local_end + w_off
            del anchors_arr, prefix_arr, local_starts, i, b, j
            gc.collect()

            # w_flat: chunked copy (uint16 → uint16, no cast required).
            # Direct assignment from mmap view to target slice; numpy
            # internally does memcpy without allocating an intermediate.
            for ws in range(0, w_take, W_CHUNK):
                we = min(ws + W_CHUNK, w_take)
                self.w_flat[w_off + ws:w_off + we] = wflat_view[ws:we]

            n_off += n_take
            w_off += w_take
            del flags_view, eval_view, wdl_view, wflat_view
            del anchors_view, prefix_view
            gc.collect()

        # Sanity: totals match
        assert n_off == total_n
        assert w_off == total_w
        assert self.offsets[total_n] == total_w

        self.n = total_n

        # Free reader handles (mmaps); their data has been copied above.
        del readers
        gc.collect()

        if verbose:
            elapsed = time.time() - t0
            mem_gb = sum(a.nbytes for a in [
                self.evals, self.has_eval, self.has_wdl, self.wdl,
                self.stm, self.piece_count, self.offsets, self.w_flat]) / 1e9
            print(f"[CnnpPelancaDataset] loaded {self.n:,} positions, "
                  f"{len(self.w_flat):,} features in {elapsed:.1f}s "
                  f"({self.n / elapsed / 1e6:.1f} M pos/s); RAM: {mem_gb:.2f} GB")

    def __len__(self):
        return self.n * 2 if self.mirror else self.n


# ─── Convenience: smoke test from the command line ──────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cnnp", nargs="+", help="path(s) to CNNP V2 file(s)")
    p.add_argument("--max-pos", type=int, default=None)
    p.add_argument("--mirror", action="store_true")
    args = p.parse_args()

    ds = CnnpPelancaDataset(args.cnnp, max_pos=args.max_pos, mirror=args.mirror)
    print(f"\nDataset stats:")
    print(f"  n           : {ds.n:,}")
    print(f"  len(ds)     : {len(ds):,} (mirror={ds.mirror})")
    print(f"  total feats : {len(ds.w_flat):,}")
    print(f"  avg feat/pos: {len(ds.w_flat) / ds.n:.2f}")
    print(f"  eval range  : [{ds.evals.min():.3f}, {ds.evals.max():.3f}]")
    print(f"  WDL counts  : white={int((ds.wdl == 1).sum()):,}, "
          f"draw={int((ds.wdl == 0).sum()):,}, "
          f"black={int((ds.wdl == -1).sum()):,}")
    pc_unique, pc_counts = np.unique(ds.piece_count, return_counts=True)
    print(f"  piece_count : min={int(pc_unique.min())}, max={int(pc_unique.max())}")

    # Sanity: pos 0 features via offsets
    s, e = int(ds.offsets[0]), int(ds.offsets[1])
    print(f"  pos[0]      : stm={ds.stm[0]} pc={ds.piece_count[0]} "
          f"eval={ds.evals[0]:.3f} wdl={ds.wdl[0]} feats={ds.w_flat[s:e].tolist()}")
