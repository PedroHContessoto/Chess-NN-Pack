# SPDX-License-Identifier: MIT
"""
train_nnue.py — Reference CNNP NNUE training script.

End-to-end recipe for training a Pelanca v23-style NNUE directly from CNNP V2
files via streaming `cnnp.Reader.get_batch()`. Designed to be the canonical
"how to train from CNNP" artifact — single reproducible entry point, scales
from 50M to 1B+ positions without RAM blow-up.

Architecture (fixed; --hidden-size is the one knob that varies):
    input(768)  -> EmbeddingBag(FT=384)  -> SCReLU
                -> Linear(384*2 -> H)    -> SCReLU
                -> Linear(H -> 8 buckets) -> gather by clamp((pc-2)//4, 0, 7)

Streaming design (the reason this scales):
    - Train indices are NEVER materialized as a single array. The sampler
      walks [0, n_positions) in shuffled chunks of `--chunk-size` positions,
      so peak RAM for indexing is O(chunk_size + |val_indices|) — typically
      ~10 MB regardless of dataset size.
    - Val indices ARE materialized (always small by construction: capped by
      --val-max-pos, default 500k positions = 4 MB).
    - When splitting train/val from the same file, val_indices is sampled
      via Floyd's algorithm (see sample_unique_sorted) — strictly O(k) time
      and memory, no permutation(N) anywhere. Excluded from train per-chunk
      via np.searchsorted.

POV conventions (CNNP-native, see notes/CNNP-V2-SPEC.md §5):
    eval_i16 : side-to-move POV (binpack convention; stored as-is)
    wdl_i8   : white-POV (spec §5.3)
    features : HalfP white-POV indices (color*384 + piece*64 + square)

This script keeps eval in STM-POV throughout, inverts WDL to STM-POV at
batch time, and presents two feature streams (STM perspective + non-STM
perspective) via a precomputed LUT.

Usage:
    PYTHONPATH=build/python python examples/python/train_nnue.py \\
        --train test_data/farseerT76_50m.cnnp \\
        --epochs 12 --batch-size 16384 \\
        --output runs/pelanca_v26

For datasets ≥ 100M positions, pass `--val` explicitly (separate file) to
avoid the per-chunk exclusion cost.

Outputs under --output:
    config.json    full hyperparameter dump (with effective val_path)
    train.log      timestamped training log
    metrics.csv    per-epoch metrics
    best.pt        best raw model by composite quality (resumable)
    best_ema.pt    EMA snapshot of best (inference-ready)
    last.pt        last epoch checkpoint (for --resume)
    final.nnue     PLNN v4 binary exported from best EMA
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import struct
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


cnnp = None  # populated by main() after sys.path is patched


# ─── Architecture constants (engine-fixed) ────────────────────────────────────

INPUT_SIZE       = 768            # HalfP vocab: color*384 + piece*64 + square
FT_SIZE          = 384
NUM_OUT_BUCKETS  = 8              # bucket = clamp((piece_count - 2) // 4, 0, 7)

FT_QUANT     = 64
HIDDEN_QUANT = 64
OUTPUT_QUANT = 64
FT_WCLIP     = 32767.0 / FT_QUANT   # FT params clipped to keep i16 quant safe
HIDDEN_WCLIP = 127.0   / HIDDEN_QUANT
OUT_WCLIP    = 127.0   / OUTPUT_QUANT

PIECE_VALUES_CP          = [100, 320, 330, 500, 900, 0]  # P, N, B, R, Q, K
MATERIAL_INIT_NUDGE_FRAC = 0.05

SANITY_FENS = [
    ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'Initial',   0),
    ('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1', '1.e4',     0),
    ('8/8/8/8/8/8/4K3/R3k3 w - - 0 1',                            'KR vs K',  300),
    ('4k3/8/8/8/8/8/8/3QK3 w - - 0 1',                            'KQ vs K',  800),
    ('4k3/3q4/8/8/8/8/8/4K3 w - - 0 1',                           'K vs KQ', -800),
]


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # I/O
    train_cnnp: str = ''
    val_cnnp: Optional[str] = None
    val_frac: float = 0.02
    val_max_pos: int = 500_000
    output_dir: str = 'runs/default'
    resume: Optional[str] = None
    cnnp_module_dir: str = 'build/python'

    # Architecture
    hidden_size: int = 8
    num_out_buckets: int = NUM_OUT_BUCKETS

    # Schedule
    epochs: int = 12
    batch_size: int = 16384
    warmup_epochs: int = 2
    lr_ft: float = 0.6e-3
    lr_min: float = 2e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999

    # Loss (calibrated for CNNP's [-10, 10] normalized eval range)
    eval_scale: float = 400.0
    filter_threshold: float = 4.5
    filter_weight: float = 0.5
    skip_threshold: float = 100.0     # disabled; CNNP clips eval to ±10
    mate_threshold: float = 6.5
    # Default OFF — for a "reference" trainer, eval-only baseline first;
    # turn on WDL with --wdl-weight 0.5 after baseline is measured.
    wdl_weight_max: float = 0.0
    wdl_ramp_end_frac: float = 0.4
    wdl_temperature: float = 350.0
    use_real_wdl: bool = True

    bucket_weights: Tuple[float, ...] = (1.0, 1.0, 1.2, 1.5, 2.0, 2.0, 1.5, 1.0)

    # Augmentation / init
    mirror_aug: bool = True
    material_init: bool = True

    # Validation / selection
    score_phase_weights: dict = field(default_factory=lambda:
        {'opening': 0.05, 'middle': 0.90, 'endgame': 0.05})
    quality_scale_penalty_w: float = 0.01
    select_wdl_weight: float = 0.0

    # Streaming sampler
    chunk_size: int = 1_048_576       # ~1M positions per shuffle chunk (8 MB int64)
    samples_per_epoch: Optional[int] = None   # None = full epoch (n_positions × mirror_factor)

    # DataLoader / perf
    num_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    use_amp: bool = True

    # Misc
    seed: int = 0
    validate_before_train: bool = False
    sanity_test: bool = True


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── HalfP white-POV ↔ black-POV LUT ──────────────────────────────────────────

def build_lut() -> np.ndarray:
    """white-POV feature id -> black-POV feature id (involution).

    HalfP: f = color*384 + piece_type*64 + square.
    Flipping POV: new_color = 1 - color, new_square = square ^ 56 (vflip).
    """
    lut = np.empty(INPUT_SIZE, dtype=np.int64)
    for i in range(INPUT_SIZE):
        color = i // 384
        pt_sq = i % 384
        pt = pt_sq // 64
        sq = pt_sq % 64
        lut[i] = (1 - color) * 384 + pt * 64 + (sq ^ 56)
    return lut


LUT_W_TO_B = build_lut()
assert (LUT_W_TO_B[LUT_W_TO_B] == np.arange(INPUT_SIZE)).all(), 'LUT is not an involution'


# ─── Unique sampling without permutation(N) ───────────────────────────────────

def sample_unique_sorted(n: int, k: int, seed: int) -> np.ndarray:
    """Floyd's algorithm — sample k distinct ints from [0, n) in O(k).

    The classic NumPy alternative `rng.choice(n, k, replace=False)` does not
    contractually guarantee O(k); historically it could allocate `perm(n)`
    internally when shuffle=True (the default). For n=1B and k=500k that's
    8 GB peak. Floyd's algorithm is O(k) by construction and never touches
    memory proportional to n. See Bentley & Floyd (1987).

    Args:
        n:    upper bound of the universe (exclusive)
        k:    number of unique samples
        seed: PRNG seed (Python's random.Random for reproducibility)

    Returns:
        Sorted np.int64 array of length k.
    """
    if k > n:
        raise ValueError(f'sample_unique_sorted: k={k} > n={n}')
    rng = random.Random(seed)
    selected: set = set()
    for j in range(n - k, n):
        t = rng.randrange(0, j + 1)
        if t in selected:
            selected.add(j)
        else:
            selected.add(t)
    return np.fromiter(sorted(selected), dtype=np.int64, count=k)


# ─── Streaming CNNP dataset ───────────────────────────────────────────────────

def _excludes_in_range(exclude_sorted: np.ndarray, start: int, end: int) -> np.ndarray:
    """Return positions (relative to `start`) within [start, end) that are in
    the sorted exclude array. Uses searchsorted — O(log |exclude|)."""
    lo = np.searchsorted(exclude_sorted, start, side='left')
    hi = np.searchsorted(exclude_sorted, end,   side='left')
    return exclude_sorted[lo:hi] - start


class CnnpStreamingDataset(IterableDataset):
    """IterableDataset over a .cnnp via chunked shuffle + Reader.get_batch().

    RAM is O(chunk_size + |exclude_indices|), NOT O(n_positions).

    Per worker, per __iter__ call:
      1. Enumerate chunks [c*chunk_size, (c+1)*chunk_size) over [0, n_positions).
      2. Stride chunks across workers (worker_id::num_workers).
      3. Shuffle chunk order using seed = base_seed + epoch*1_000_003 + wid*13.
      4. For each chunk: arange, exclude val_indices (if any), shuffle locally,
         optionally double for mirror augmentation, emit batches.

    `set_epoch(e)` lets the main loop bump the shuffle stream between epochs.
    Persistent workers don't propagate set_epoch; the main loop instead
    rebuilds the DataLoader each epoch.
    """

    def __init__(self, cnnp_path: str, n_positions: int, batch_size: int,
                 *, shuffle: bool = True, mirror: bool = False,
                 base_seed: int = 0, drop_last: bool = True,
                 samples_per_epoch: Optional[int] = None,
                 exclude_indices: Optional[np.ndarray] = None,
                 chunk_size: int = 1_048_576):
        super().__init__()
        self.cnnp_path        = cnnp_path
        self.n_positions      = int(n_positions)
        self.batch_size       = int(batch_size)
        self.shuffle          = bool(shuffle)
        self.mirror           = bool(mirror)
        self.base_seed        = int(base_seed)
        self.drop_last        = bool(drop_last)
        self.samples_per_epoch = samples_per_epoch
        self.exclude_indices  = (None if exclude_indices is None
                                 else np.ascontiguousarray(exclude_indices, dtype=np.int64))
        self.chunk_size       = max(int(chunk_size), int(batch_size))
        self._reader = None

    def __len__(self):
        if self.samples_per_epoch is not None:
            samples = int(self.samples_per_epoch)
        else:
            samples = self.n_positions
            if self.exclude_indices is not None:
                samples -= len(self.exclude_indices)
            if self.mirror:
                samples *= 2
        bs = self.batch_size
        return samples // bs if self.drop_last else (samples + bs - 1) // bs

    def _ensure_reader(self):
        if self._reader is None:
            self._reader = cnnp.Reader(self.cnnp_path, validate=False)

    def __iter__(self):
        self._ensure_reader()

        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        seed = self.base_seed + wid * 13
        rng = np.random.default_rng(seed)

        n = self.n_positions
        cs = self.chunk_size
        n_chunks = (n + cs - 1) // cs
        chunk_order = np.arange(n_chunks, dtype=np.int64)
        if self.shuffle:
            rng.shuffle(chunk_order)
        chunk_order = chunk_order[wid::nw]   # stride across workers

        budget = None
        if self.samples_per_epoch is not None:
            budget = int(self.samples_per_epoch) // nw
        samples_emitted = 0

        tail_idx = np.empty(0, dtype=np.int64)
        tail_mir = np.empty(0, dtype=bool)

        for ci in chunk_order:
            if budget is not None and samples_emitted >= budget:
                return

            start = int(ci) * cs
            end   = min(start + cs, n)

            # Build chunk indices, applying exclusion (if any)
            if self.exclude_indices is not None:
                ex = _excludes_in_range(self.exclude_indices, start, end)
                if len(ex) > 0:
                    mask = np.ones(end - start, dtype=bool)
                    mask[ex] = False
                    chunk = np.arange(start, end, dtype=np.int64)[mask]
                else:
                    chunk = np.arange(start, end, dtype=np.int64)
            else:
                chunk = np.arange(start, end, dtype=np.int64)

            if len(chunk) == 0:
                continue

            if self.shuffle:
                rng.shuffle(chunk)

            # Mirror augmentation: duplicate chunk with mirror flag set
            if self.mirror:
                chunk_mir = np.concatenate([
                    np.zeros(len(chunk), dtype=bool),
                    np.ones(len(chunk),  dtype=bool),
                ])
                chunk = np.concatenate([chunk, chunk])
                if self.shuffle:
                    perm = rng.permutation(len(chunk))
                    chunk = chunk[perm]
                    chunk_mir = chunk_mir[perm]
            else:
                chunk_mir = np.zeros(len(chunk), dtype=bool)

            # Combine with any leftover tail from the previous chunk
            if len(tail_idx) > 0:
                chunk = np.concatenate([tail_idx, chunk])
                chunk_mir = np.concatenate([tail_mir, chunk_mir])
                tail_idx = np.empty(0, dtype=np.int64)
                tail_mir = np.empty(0, dtype=bool)

            # Emit full batches from this combined buffer
            bs = self.batch_size
            n_full = len(chunk) // bs
            for b in range(n_full):
                s = b * bs
                e = s + bs
                yield self._make_batch(chunk[s:e], chunk_mir[s:e])
                samples_emitted += bs
                if budget is not None and samples_emitted >= budget:
                    return

            # Anything that doesn't fill a batch becomes the tail
            tail_start = n_full * bs
            if tail_start < len(chunk):
                tail_idx = chunk[tail_start:]
                tail_mir = chunk_mir[tail_start:]

        # End-of-epoch: flush partial batch if drop_last is False
        if not self.drop_last and len(tail_idx) > 0:
            yield self._make_batch(tail_idx, tail_mir)

    def _make_batch(self, idx: np.ndarray, mirror_flags: np.ndarray):
        b = self._reader.get_batch(idx.astype(np.uint64))
        features = b['features']    # (B, 32) uint16, padded UINT16_MAX
        counts   = b['counts']      # (B,) uint8 — piece_count == nnz
        evals    = b['evals_norm']  # (B,) float32 — STM-POV
        wdls     = b['wdls']        # (B,) int8 — WHITE-POV
        stm      = b['stm']         # (B,) uint8 — 0 white, 1 black

        B = features.shape[0]
        counts_i64 = counts.astype(np.int64)

        # CSR offsets for EmbeddingBag
        offsets = np.zeros(B, dtype=np.int64)
        np.cumsum(counts_i64[:-1], out=offsets[1:])

        # Drop padding
        valid = np.arange(32, dtype=np.int64)[None, :] < counts_i64[:, None]
        flat  = features[valid].astype(np.int64)

        stm_per_feat    = np.repeat(stm,          counts_i64)
        mirror_per_feat = np.repeat(mirror_flags, counts_i64)

        # Mirror: XOR 7 flips file (HalfP's `square` low bits)
        if mirror_per_feat.any():
            flat[mirror_per_feat] ^= 7

        white_pov = flat
        black_pov = LUT_W_TO_B[flat]
        is_black  = stm_per_feat == 1
        stm_feats  = np.where(is_black, black_pov, white_pov)
        nstm_feats = np.where(is_black, white_pov, black_pov)

        # WDL white-POV → STM-POV (eval is already STM-POV)
        wdls_stm = wdls.astype(np.float32)
        wdls_stm[stm == 1] *= -1.0

        has_eval = np.ones(B, dtype=np.uint8)
        has_wdl  = np.ones(B, dtype=np.uint8)

        return (
            torch.from_numpy(stm_feats),
            torch.from_numpy(offsets),
            torch.from_numpy(nstm_feats),
            torch.from_numpy(offsets.copy()),
            torch.from_numpy(evals.astype(np.float32, copy=False)),
            torch.from_numpy(has_eval),
            torch.from_numpy(counts.astype(np.uint8, copy=False)),
            torch.from_numpy(wdls_stm),
            torch.from_numpy(has_wdl),
        )


# ─── Model ────────────────────────────────────────────────────────────────────

class SquaredCReLU(nn.Module):
    """SCReLU(x) = clamp(x, 0, 1)^2."""
    def forward(self, x):
        c = torch.clamp(x, 0.0, 1.0)
        return c * c


class PelancaNNUE(nn.Module):
    """Pelanca v23 (SCReLU + 8 output buckets via single Linear + gather)."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.ft        = nn.EmbeddingBag(INPUT_SIZE, FT_SIZE, mode='sum', sparse=False)
        self.ft_bias   = nn.Parameter(torch.zeros(FT_SIZE))
        self.hidden    = nn.Linear(FT_SIZE * 2, cfg.hidden_size)
        self.out       = nn.Linear(cfg.hidden_size, cfg.num_out_buckets)
        self.scrl      = SquaredCReLU()
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.ft.weight,     nonlinearity='relu')
        nn.init.zeros_(self.ft_bias)
        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        if self.cfg.material_init:
            self._material_init()

    def _material_init(self):
        spread_dims = self.cfg.hidden_size
        with torch.no_grad():
            for color in range(2):
                for pt in range(6):
                    sign = 1.0 if color == 0 else -1.0
                    nudge = (sign * PIECE_VALUES_CP[pt] / self.cfg.eval_scale *
                             MATERIAL_INIT_NUDGE_FRAC / spread_dims)
                    for sq in range(64):
                        f = color * 384 + pt * 64 + sq
                        self.ft.weight.data[f, :spread_dims] += nudge

    @staticmethod
    def bucket_idx(pc: torch.Tensor) -> torch.Tensor:
        return torch.clamp((pc.long() - 2) // 4, 0, NUM_OUT_BUCKETS - 1)

    def forward(self, stm_idx, stm_off, nstm_idx, nstm_off, pc):
        # FT in fp32 even under AMP — i16-quant inference path uses fp32
        # accumulation and we want training to match.
        with torch.amp.autocast('cuda', enabled=False):
            stm_acc  = self.scrl(self.ft(stm_idx,  stm_off)  + self.ft_bias)
            nstm_acc = self.scrl(self.ft(nstm_idx, nstm_off) + self.ft_bias)
        h = self.scrl(self.hidden(torch.cat([stm_acc, nstm_acc], dim=1)))
        all_out = self.out(h)                           # (B, K)
        idx = self.bucket_idx(pc).unsqueeze(-1)         # (B, 1)
        return all_out.gather(1, idx)                   # (B, 1)

    def clip_weights(self):
        """Constrain all weight matrices to stay quantizable.

        FT clipping: int16 with FT_QUANT=64 admits |w| ≤ 32767/64 ≈ 512.
        Real FT magnitudes are <1.0 in practice (kaiming init), so this clip
        is defensive but trivially cheap.
        """
        with torch.no_grad():
            self.ft.weight.clamp_(-FT_WCLIP, FT_WCLIP)
            self.ft_bias.clamp_(-FT_WCLIP, FT_WCLIP)
            self.hidden.weight.clamp_(-HIDDEN_WCLIP, HIDDEN_WCLIP)
            self.out.weight.clamp_(-OUT_WCLIP, OUT_WCLIP)


class EMA:
    def __init__(self, model: nn.Module, decay: float, cfg: TrainConfig):
        self.decay = decay
        self.ema = PelancaNNUE(cfg)
        self.ema.load_state_dict(model.state_dict())
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def to(self, device):
        self.ema.to(device)
        return self

    def update(self, model: nn.Module):
        with torch.no_grad():
            for pe, p in zip(self.ema.parameters(), model.parameters()):
                pe.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
            for be, b in zip(self.ema.buffers(), model.buffers()):
                be.copy_(b)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)


# ─── Loss ─────────────────────────────────────────────────────────────────────

def recompute_wdl_target(eval_stm: torch.Tensor, eval_scale: float,
                         wdl_temp: float, mate_threshold: float) -> torch.Tensor:
    is_mate = eval_stm.abs() > mate_threshold
    eval_cp = torch.where(is_mate,
                          torch.sign(eval_stm) * 32000.0,
                          eval_stm * eval_scale)
    return 2.0 * torch.sigmoid(eval_cp / wdl_temp) - 1.0


def blended_loss(pred, target_eval, has_eval, wdl_w,
                 target_wdl_real=None, has_wdl=None,
                 piece_count=None, bucket_weights=None,
                 cfg: TrainConfig = None):
    pred_sig        = torch.tanh(pred.float())
    target_eval_f   = target_eval.float()
    target_eval_sig = torch.tanh(target_eval_f)

    abs_e     = target_eval_f.abs()
    keep_mask = ~(abs_e > cfg.skip_threshold)
    weights   = torch.where(abs_e > cfg.filter_threshold,
                            torch.full_like(target_eval_f, cfg.filter_weight),
                            torch.ones_like(target_eval_f)) * keep_mask.float()

    if piece_count is not None and bucket_weights is not None:
        bidx = torch.clamp((piece_count.long() - 2) // 4, 0, NUM_OUT_BUCKETS - 1)
        # bucket_weights is expected to already live on the right device;
        # main() and eval_model() create it once with device=device.
        bw = bucket_weights[bidx].unsqueeze(-1)
        weights = weights * bw

    mask_e = has_eval.bool().squeeze() & keep_mask.squeeze()
    if not mask_e.any():
        return torch.zeros((), device=pred.device)

    pred_e   = pred_sig[mask_e].squeeze(-1)
    target_e = target_eval_sig[mask_e].squeeze(-1)
    w_e      = weights[mask_e].squeeze(-1)
    eval_loss = (w_e * (pred_e - target_e).pow(2)).sum() / w_e.sum().clamp_min(1e-8)

    if wdl_w <= 0.0:
        return eval_loss

    if cfg.use_real_wdl and target_wdl_real is not None and has_wdl is not None:
        mask_w = has_eval.bool().squeeze() & has_wdl.bool().squeeze() & keep_mask.squeeze()
        if not mask_w.any():
            return eval_loss
        pred_w   = pred_sig[mask_w].squeeze(-1)
        target_w = target_wdl_real.float()[mask_w].squeeze(-1)
        w_w      = weights[mask_w].squeeze(-1)
        wdl_loss = (w_w * (pred_w - target_w).pow(2)).sum() / w_w.sum().clamp_min(1e-8)
    else:
        synth = recompute_wdl_target(target_eval_f, cfg.eval_scale,
                                     cfg.wdl_temperature, cfg.mate_threshold)
        target_w_e = synth[mask_e].squeeze(-1)
        wdl_loss = (w_e * (pred_e - target_w_e).pow(2)).sum() / w_e.sum().clamp_min(1e-8)

    return (1.0 - wdl_w) * eval_loss + wdl_w * wdl_loss


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_model(model, val_loader, device, cfg: TrainConfig,
               bucket_weights: torch.Tensor, with_calibration: bool = False):
    """Validation loop returning sample-weighted average losses.

    Sample-weighting (rather than simple per-batch mean) matters when the
    val loader emits partial batches at the end and/or when multiple
    workers each produce their own tail — averaging the per-batch means
    would bias toward smaller batches.
    """
    model.eval()
    # Per-bucket: accumulate (loss * n_samples, n_samples) then divide once.
    loss_sum = {'all': 0.0, 'opening': 0.0, 'middle': 0.0, 'endgame': 0.0}
    loss_n   = {'all': 0,   'opening': 0,   'middle': 0,   'endgame': 0}

    cal = dict(pred_sum=0.0, pred_sq=0.0, ev_sum=0.0, ev_sq=0.0,
               sign_ok=0, sign_total=0, abs_diff=0.0, n=0)

    for stm_i, stm_o, nstm_i, nstm_o, ev, hev, pc, wdl, hwdl in val_loader:
        stm_i  = stm_i.to(device, non_blocking=True)
        stm_o  = stm_o.to(device, non_blocking=True)
        nstm_i = nstm_i.to(device, non_blocking=True)
        nstm_o = nstm_o.to(device, non_blocking=True)
        ev     = ev.to(device, non_blocking=True).unsqueeze(1)
        hev    = hev.to(device, non_blocking=True).unsqueeze(1)
        pc     = pc.to(device, non_blocking=True)
        wdl    = wdl.to(device, non_blocking=True).unsqueeze(1)
        hwdl   = hwdl.to(device, non_blocking=True).unsqueeze(1)

        pred = model(stm_i, stm_o, nstm_i, nstm_o, pc)

        B = pred.shape[0]
        L_all = blended_loss(
            pred, ev, hev, cfg.select_wdl_weight,
            target_wdl_real=wdl, has_wdl=hwdl,
            piece_count=pc, bucket_weights=bucket_weights, cfg=cfg).item()
        loss_sum['all'] += L_all * B
        loss_n['all']   += B

        for phase, mask in [('opening', pc > 20),
                            ('middle',  (pc > 10) & (pc <= 20)),
                            ('endgame', pc <= 10)]:
            m_count = int(mask.sum().item())
            if m_count > 0:
                L_p = blended_loss(
                    pred[mask], ev[mask], hev[mask], cfg.select_wdl_weight,
                    target_wdl_real=wdl[mask], has_wdl=hwdl[mask],
                    piece_count=pc[mask],
                    bucket_weights=bucket_weights, cfg=cfg).item()
                loss_sum[phase] += L_p * m_count
                loss_n[phase]   += m_count

        if with_calibration:
            m = hev.bool().squeeze()
            if m.any():
                pred_t = torch.tanh(pred[m].squeeze().float())
                ev_t   = torch.tanh(ev[m].squeeze().float())
                pred_cp = pred_t * cfg.eval_scale
                ev_cp   = ev_t   * cfg.eval_scale
                cal['pred_sum'] += pred_cp.sum().item()
                cal['pred_sq']  += (pred_cp ** 2).sum().item()
                cal['ev_sum']   += ev_cp.sum().item()
                cal['ev_sq']    += (ev_cp ** 2).sum().item()
                cal['abs_diff'] += (pred_cp - ev_cp).abs().sum().item()
                cal['n']        += int(m.sum().item())
                dec = ev_t.abs() > 0.05
                if dec.any():
                    cal['sign_ok']    += int(((pred_t[dec] > 0) == (ev_t[dec] > 0)).sum().item())
                    cal['sign_total'] += int(dec.sum().item())

    out = {k: (loss_sum[k] / loss_n[k] if loss_n[k] else float('nan'))
           for k in loss_sum}
    if with_calibration and cal['n'] > 0:
        pm = cal['pred_sum'] / cal['n']
        em = cal['ev_sum']   / cal['n']
        pv = max(0.0, cal['pred_sq'] / cal['n'] - pm ** 2)
        ev_v = max(0.0, cal['ev_sq'] / cal['n'] - em ** 2)
        out['scale_ratio'] = (pv ** 0.5) / max(ev_v ** 0.5, 1e-6)
        out['mae_cp']      = cal['abs_diff'] / cal['n']
        out['sign_acc']    = 100.0 * cal['sign_ok'] / max(cal['sign_total'], 1)
    return out


def composite_score(d, cfg: TrainConfig) -> float:
    w = cfg.score_phase_weights
    return w['opening'] * d['opening'] + w['middle'] * d['middle'] + w['endgame'] * d['endgame']


def quality_score(d, cfg: TrainConfig):
    sr = d.get('scale_ratio', 1.0)
    score = composite_score(d, cfg)
    penalty = cfg.quality_scale_penalty_w * max(0.0, 1.0 - sr)
    return score + penalty, score, penalty


# ─── Export (PLNN v4 binary) ──────────────────────────────────────────────────

def export_nnue(model: PelancaNNUE, path: str, cfg: TrainConfig, verbose: bool = True):
    """Write PLNN v4 binary expected by the Pelanca inference engine.

    Header (28 B):
        magic 'PLNN' | version=4 | input_size | ft_size | hidden_size
                     | num_buckets | activation_id (1=SCReLU) | eval_scale
    """
    model.eval()
    H = cfg.hidden_size
    K = cfg.num_out_buckets

    ft_w = (model.ft.weight.data.cpu().T * FT_QUANT).round().clamp(-32767, 32767).to(torch.int16)
    ft_b = (model.ft_bias.data.cpu()       * FT_QUANT).round().clamp(-32767, 32767).to(torch.int16)
    h_w  = (model.hidden.weight.data.cpu() * HIDDEN_QUANT).round().clamp(-127, 127).to(torch.int8)
    h_b_scale = FT_QUANT * FT_QUANT * HIDDEN_QUANT
    h_b  = (model.hidden.bias.data.cpu() * h_b_scale).round().clamp(-2**30, 2**30).to(torch.int32)
    out_w_scale = OUTPUT_QUANT
    out_b_scale = FT_QUANT * FT_QUANT * HIDDEN_QUANT * HIDDEN_QUANT * OUTPUT_QUANT
    out_w = (model.out.weight.data.cpu() * out_w_scale).round().clamp(-127, 127).to(torch.int8)
    out_b = (model.out.bias.data.cpu()   * out_b_scale).round().clamp(-2**30, 2**30).to(torch.int32)

    with open(path, 'wb') as f:
        f.write(b'PLNN')
        f.write(struct.pack('<I', 4))
        f.write(struct.pack('<I', INPUT_SIZE))
        f.write(struct.pack('<I', FT_SIZE))
        f.write(struct.pack('<I', H))
        f.write(struct.pack('<I', K))
        f.write(struct.pack('<I', 1))                     # SCReLU
        f.write(struct.pack('<I', int(cfg.eval_scale)))
        f.write(ft_w.contiguous().numpy().tobytes())
        f.write(ft_b.numpy().tobytes())
        f.write(h_w.numpy().tobytes())
        f.write(h_b.numpy().tobytes())
        for k in range(K):
            f.write(out_w[k:k+1].numpy().tobytes())
            f.write(out_b[k:k+1].numpy().tobytes())

    if verbose:
        sz = os.path.getsize(path)
        h_sat = ((h_w.abs() == 127).float().mean().item()) * 100.0
        o_sat = ((out_w.abs() == 127).float().mean().item()) * 100.0
        print(f'  exported {path} ({sz:,} B / {sz/1024:.1f} KB) — '
              f'PLNN v4 H={H} K={K} SCReLU scale={int(cfg.eval_scale)} | '
              f'h_sat={h_sat:.2f}% o_sat={o_sat:.2f}%')


# ─── Sanity test on known FENs ────────────────────────────────────────────────

PIECE_MAP = {
    'P': (0, 0), 'N': (0, 1), 'B': (0, 2), 'R': (0, 3), 'Q': (0, 4), 'K': (0, 5),
    'p': (1, 0), 'n': (1, 1), 'b': (1, 2), 'r': (1, 3), 'q': (1, 4), 'k': (1, 5),
}


def fen_to_features(fen: str):
    parts = fen.split(' ')
    stm = 0 if parts[1] == 'w' else 1
    wi, bi = [], []
    sq = 56
    for ch in parts[0]:
        if ch == '/':
            sq -= 16
        elif ch.isdigit():
            sq += int(ch)
        else:
            color, pt = PIECE_MAP[ch]
            wi.append(color * 384 + pt * 64 + sq)
            bi.append((1 - color) * 384 + pt * 64 + (sq ^ 56))
            sq += 1
    return stm, wi, bi, len(wi)


def sanity_test(model, device, cfg: TrainConfig, log):
    model.eval()
    log('')
    log('=== Sanity test ===')
    log(f'{"Position":<14} {"raw":>10} {"tanh":>8} {"~cp":>8} {"expected":>10}')
    log('-' * 56)
    with torch.no_grad():
        for fen, desc, expected in SANITY_FENS:
            stm, wi, bi, pc = fen_to_features(fen)
            if stm == 1:
                wi, bi = bi, wi
            stm_i  = torch.tensor(wi, dtype=torch.long, device=device)
            stm_o  = torch.tensor([0], dtype=torch.long, device=device)
            nstm_i = torch.tensor(bi, dtype=torch.long, device=device)
            nstm_o = torch.tensor([0], dtype=torch.long, device=device)
            pc_t   = torch.tensor([pc], dtype=torch.long, device=device)
            r = model(stm_i, stm_o, nstm_i, nstm_o, pc_t).item()
            t = math.tanh(r)
            t_clip = max(-0.9999, min(0.9999, t))
            cp = 0.5 * math.log((1 + t_clip) / (1 - t_clip)) * cfg.eval_scale
            if stm == 1:
                cp = -cp   # model outputs STM-POV; report from white's POV
            flag = '' if abs(cp - expected) < 500 else ' [SUSPECT]'
            log(f'{desc:<14} {r:>+10.3f} {t:>+8.3f} {cp:>+8.0f} {expected:>+10d}{flag}')


# ─── Schedules ────────────────────────────────────────────────────────────────

def lr_factor(epoch: int, cfg: TrainConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return (epoch + 1) / cfg.warmup_epochs
    span = max(1, cfg.epochs - cfg.warmup_epochs - 1)
    t = min(1.0, (epoch - cfg.warmup_epochs) / span)
    floor = cfg.lr_min / cfg.lr_ft
    return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * t))


def wdl_weight(epoch: int, cfg: TrainConfig) -> float:
    if cfg.wdl_weight_max <= 0.0 or epoch < cfg.warmup_epochs:
        return 0.0
    ramp_end = max(cfg.warmup_epochs + 1, int(cfg.epochs * cfg.wdl_ramp_end_frac))
    if epoch >= ramp_end:
        return cfg.wdl_weight_max
    return cfg.wdl_weight_max * (epoch - cfg.warmup_epochs) / (ramp_end - cfg.warmup_epochs)


# ─── Train/val split (without materializing perm(N)) ──────────────────────────

def prepare_split(cfg: TrainConfig) -> Tuple[int, np.ndarray, str]:
    """Return (n_train_positions, val_indices_sorted, val_path_used).

    val_indices is ALWAYS materialized (small by construction — capped by
    val_max_pos). n_train_positions is just a count; train indices are NEVER
    materialized as a single array (the streaming sampler walks them by
    chunk). If val came from the same file, the trainer excludes val_indices
    per-chunk via searchsorted.
    """
    train_r = cnnp.Reader(cfg.train_cnnp, validate=False)
    n_train = int(train_r.num_positions)

    if cfg.val_cnnp and cfg.val_cnnp != cfg.train_cnnp:
        val_r = cnnp.Reader(cfg.val_cnnp, validate=False)
        n_val_avail = int(val_r.num_positions)
        n_val = min(cfg.val_max_pos, n_val_avail) if cfg.val_max_pos > 0 else n_val_avail
        return n_train, np.arange(n_val, dtype=np.int64), cfg.val_cnnp

    n_val = min(cfg.val_max_pos, int(n_train * cfg.val_frac))
    # Contractually O(n_val), not O(n_train) — see sample_unique_sorted().
    val_indices = sample_unique_sorted(n_train, n_val, cfg.seed)
    return n_train, val_indices, cfg.train_cnnp


# ─── DataLoader factory (called per-epoch for proper shuffle) ─────────────────

def make_train_loader(cfg: TrainConfig, n_train: int,
                      exclude: Optional[np.ndarray], epoch: int) -> DataLoader:
    ds = CnnpStreamingDataset(
        cnnp_path=cfg.train_cnnp,
        n_positions=n_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        mirror=cfg.mirror_aug,
        # Different shuffle stream per epoch — prime mixed with seed
        base_seed=cfg.seed + epoch * 1_000_003,
        drop_last=True,
        samples_per_epoch=cfg.samples_per_epoch,
        exclude_indices=exclude,
        chunk_size=cfg.chunk_size,
    )
    return DataLoader(
        ds, batch_size=None, num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=False,           # we rebuild every epoch
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )


def make_val_loader(cfg: TrainConfig, val_path: str, val_idx: np.ndarray) -> DataLoader:
    # Val dataset uses materialized indices directly via a tiny wrapper —
    # we still drive it through CnnpStreamingDataset with n_positions = max(idx)+1
    # and a no-shuffle config, but easier is a dedicated path: build CSR-style
    # walk by handing the sorted val_idx as the only "chunk".
    # Simpler: reuse the same dataset class with exclude_indices = complement.
    # Simplest of all: define a small sequential val IterableDataset.
    return DataLoader(
        _ValIterableDataset(val_path, val_idx, cfg.batch_size),
        batch_size=None, num_workers=min(2, cfg.num_workers),
        pin_memory=cfg.pin_memory,
        persistent_workers=False,
        prefetch_factor=cfg.prefetch_factor if min(2, cfg.num_workers) > 0 else None,
    )


class _ValIterableDataset(IterableDataset):
    """Sequential walk over a fixed val_indices array (no shuffle, no mirror,
    drop_last=False). Workers split the index array contiguously."""

    def __init__(self, cnnp_path: str, val_indices: np.ndarray, batch_size: int):
        super().__init__()
        self.cnnp_path = cnnp_path
        self.val_indices = np.ascontiguousarray(val_indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self._reader = None

    def __len__(self):
        bs = self.batch_size
        return (len(self.val_indices) + bs - 1) // bs

    def _ensure_reader(self):
        if self._reader is None:
            self._reader = cnnp.Reader(self.cnnp_path, validate=False)

    def __iter__(self):
        self._ensure_reader()
        info = get_worker_info()
        if info is None:
            indices = self.val_indices
        else:
            wid, nw = info.id, info.num_workers
            per = len(self.val_indices) // nw
            start = wid * per
            end = (start + per) if wid < nw - 1 else len(self.val_indices)
            indices = self.val_indices[start:end]

        bs = self.batch_size
        n = len(indices)
        for s in range(0, n, bs):       # drop_last=False — last partial batch emitted
            e = min(s + bs, n)
            yield self._make_batch(indices[s:e])

    def _make_batch(self, idx: np.ndarray):
        b = self._reader.get_batch(idx.astype(np.uint64))
        features = b['features']
        counts   = b['counts']
        evals    = b['evals_norm']
        wdls     = b['wdls']
        stm      = b['stm']

        B = features.shape[0]
        counts_i64 = counts.astype(np.int64)
        offsets = np.zeros(B, dtype=np.int64)
        np.cumsum(counts_i64[:-1], out=offsets[1:])
        valid = np.arange(32, dtype=np.int64)[None, :] < counts_i64[:, None]
        flat  = features[valid].astype(np.int64)

        stm_per_feat = np.repeat(stm, counts_i64)
        white_pov = flat
        black_pov = LUT_W_TO_B[flat]
        is_black  = stm_per_feat == 1
        stm_feats  = np.where(is_black, black_pov, white_pov)
        nstm_feats = np.where(is_black, white_pov, black_pov)

        wdls_stm = wdls.astype(np.float32)
        wdls_stm[stm == 1] *= -1.0

        has_eval = np.ones(B, dtype=np.uint8)
        has_wdl  = np.ones(B, dtype=np.uint8)

        return (
            torch.from_numpy(stm_feats),
            torch.from_numpy(offsets),
            torch.from_numpy(nstm_feats),
            torch.from_numpy(offsets.copy()),
            torch.from_numpy(evals.astype(np.float32, copy=False)),
            torch.from_numpy(has_eval),
            torch.from_numpy(counts.astype(np.uint8, copy=False)),
            torch.from_numpy(wdls_stm),
            torch.from_numpy(has_wdl),
        )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Reference CNNP NNUE trainer (streaming, Pelanca v23 arch).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--train', required=True)
    p.add_argument('--val', default=None,
                   help='Path to validation .cnnp (default: random split from --train; '
                        'pass a separate file for datasets ≥ 100M positions)')
    p.add_argument('--val-frac',    type=float, default=0.02)
    p.add_argument('--val-max-pos', type=int,   default=500_000)
    p.add_argument('--output',      default='runs/default')
    p.add_argument('--resume',      default=None)

    p.add_argument('--epochs',        type=int,   default=12)
    p.add_argument('--batch-size',    type=int,   default=16384)
    p.add_argument('--warmup-epochs', type=int,   default=2)
    p.add_argument('--lr',            type=float, default=0.6e-3,
                   help='Peak LR (calibrated for batch=16384; scale linearly with batch)')
    p.add_argument('--lr-min',        type=float, default=2e-5)
    p.add_argument('--weight-decay',  type=float, default=1e-4)
    p.add_argument('--ema-decay',     type=float, default=0.999)

    p.add_argument('--eval-scale',    type=float, default=400.0)
    p.add_argument('--wdl-weight',    type=float, default=0.0,
                   help='WDL blending weight (default 0 = pure eval baseline; '
                        '0.5 is the Stockfish/Cosmo recommended value after baseline)')
    p.add_argument('--no-wdl',        action='store_true', help='Force WDL off')

    p.add_argument('--hidden-size',       type=int, default=8)
    p.add_argument('--no-mirror',         action='store_true')
    p.add_argument('--no-material-init',  action='store_true')
    p.add_argument('--no-amp',            action='store_true')

    # Streaming sampler
    p.add_argument('--chunk-size',        type=int, default=1_048_576,
                   help='Positions per shuffle chunk (RAM = ~8 MB at default)')
    p.add_argument('--samples-per-epoch', type=int, default=None,
                   help='Cap on samples emitted per epoch, COUNTED POST-AUGMENTATION. '
                        'A full epoch over N positions is 2N samples when mirror is on '
                        '(default). Use this to make a fixed schedule comparable across '
                        '50M / 200M / 1B datasets (e.g. --samples-per-epoch 100_000_000).')

    p.add_argument('--num-workers',     type=int, default=4)
    p.add_argument('--prefetch-factor', type=int, default=4)

    p.add_argument('--validate-before-train', action='store_true',
                   help='Run full O(N) cnnp validation before training')
    p.add_argument('--no-sanity', action='store_true')
    p.add_argument('--seed',      type=int, default=0)
    p.add_argument('--cnnp-module-dir', default='build/python')
    return p.parse_args()


def args_to_config(args) -> TrainConfig:
    return TrainConfig(
        train_cnnp=args.train,
        val_cnnp=args.val,
        val_frac=args.val_frac,
        val_max_pos=args.val_max_pos,
        output_dir=args.output,
        resume=args.resume,
        cnnp_module_dir=args.cnnp_module_dir,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        lr_ft=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        eval_scale=args.eval_scale,
        wdl_weight_max=0.0 if args.no_wdl else args.wdl_weight,
        mirror_aug=not args.no_mirror,
        material_init=not args.no_material_init,
        use_amp=not args.no_amp,
        chunk_size=args.chunk_size,
        samples_per_epoch=args.samples_per_epoch,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        validate_before_train=args.validate_before_train,
        sanity_test=not args.no_sanity,
        seed=args.seed,
    )


def make_logger(log_path: Path):
    fh = open(log_path, 'a', buffering=1, encoding='utf-8')

    def log(msg: str = ''):
        ts = time.strftime('%H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line, flush=True)
        fh.write(line + '\n')

    return log, fh


def main():
    args = parse_args()
    cfg = args_to_config(args)
    set_seed(cfg.seed)

    global cnnp
    sys.path.insert(0, cfg.cnnp_module_dir)
    import cnnp as cnnp_mod
    cnnp = cnnp_mod

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log, log_fh = make_logger(out / 'train.log')

    log('=== Reference CNNP NNUE trainer ===')
    log(f'cwd:      {Path.cwd()}')
    log(f'output:   {out.resolve()}')
    log(f'device:   {"cuda" if torch.cuda.is_available() else "cpu"}'
        + (f' ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ''))
    log(f'cnnp:     {cnnp.__file__} (VERSION={cnnp.VERSION})')

    if cfg.validate_before_train:
        log(f'validating {cfg.train_cnnp} (full scan)...')
        cnnp.Reader(cfg.train_cnnp).header
        if cfg.val_cnnp and cfg.val_cnnp != cfg.train_cnnp:
            log(f'validating {cfg.val_cnnp}...')
            cnnp.Reader(cfg.val_cnnp).header
        log('  validation OK')

    n_train, val_idx, val_path = prepare_split(cfg)
    split_note = ' (split from train, exclusion per chunk)' if val_path == cfg.train_cnnp else ''
    # When val is split from the same file, train excludes those indices
    # — account for that in the reported epoch size.
    effective_train_positions = (n_train - len(val_idx)
                                 if val_path == cfg.train_cnnp else n_train)
    full_epoch_size = effective_train_positions * (2 if cfg.mirror_aug else 1)
    if cfg.samples_per_epoch:
        cap_note = (f' (cap {cfg.samples_per_epoch:,} samples/epoch post-aug; '
                    f'full epoch would be {full_epoch_size:,})')
    else:
        cap_note = f' (full epoch = {full_epoch_size:,} samples post-aug)'
    log(f'train:    {n_train:,} positions from {cfg.train_cnnp}{cap_note}')
    log(f'val:      {len(val_idx):,} positions from {val_path}{split_note}')
    log(f'chunked:  chunk_size={cfg.chunk_size:,}, mirror={cfg.mirror_aug}, '
        f'workers={cfg.num_workers}')

    # Effective config dump
    cfg_dump = asdict(cfg)
    cfg_dump['_val_path_used']     = val_path
    cfg_dump['_n_train_positions'] = int(n_train)
    cfg_dump['_n_val_positions']   = int(len(val_idx))
    with open(out / 'config.json', 'w') as f:
        json.dump(cfg_dump, f, indent=2, default=str)

    # Val loader is built once (val_indices are materialized; no per-epoch reshuffle)
    val_loader = make_val_loader(cfg, val_path, val_idx)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = PelancaNNUE(cfg).to(device)
    ema    = EMA(model, decay=cfg.ema_decay, cfg=cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(f'arch:     input={INPUT_SIZE} ft={FT_SIZE} hidden={cfg.hidden_size} '
        f'buckets={cfg.num_out_buckets} params={n_params:,}')

    wd_params, no_wd_params = [], []
    for name, p in model.named_parameters():
        if 'bias' in name or name == 'ft_bias':
            no_wd_params.append(p)
        else:
            wd_params.append(p)
    optimizer = optim.AdamW([
        {'params': wd_params,    'weight_decay': cfg.weight_decay, 'name': 'wd'},
        {'params': no_wd_params, 'weight_decay': 0.0,              'name': 'no_wd'},
    ], lr=cfg.lr_ft)

    use_amp = cfg.use_amp and torch.cuda.is_available()
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None
    # Single device-resident copy reused across train + eval (saves per-batch
    # .to(device) overhead in the inner loop).
    bucket_weights = torch.tensor(cfg.bucket_weights, dtype=torch.float32, device=device)

    start_epoch  = 0
    best_quality = float('inf')
    if cfg.resume:
        log(f'resuming from {cfg.resume}')
        ck = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ck['model'])
        ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optim'])
        if scaler and ck.get('scaler') is not None:
            scaler.load_state_dict(ck['scaler'])
        start_epoch  = ck['epoch'] + 1
        best_quality = ck.get('best_quality', float('inf'))
        log(f'  resumed at epoch {start_epoch} (best_quality={best_quality:.6f})')

    metrics_path = out / 'metrics.csv'
    if not metrics_path.exists() or start_epoch == 0:
        with open(metrics_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'train_loss', 'val_all', 'val_op', 'val_md', 'val_eg',
                'score', 'quality', 'scale_ratio', 'sign_acc', 'mae_cp',
                'lr', 'wdl_w', 'data_time_s', 'compute_time_s', 'samples_per_s',
                'gpu_peak_gb', 'epoch_wall_s',
            ])

    log('')
    log(f'=== training: {cfg.epochs} epochs, batch={cfg.batch_size}, '
        f'lr={cfg.lr_ft:.1e}→{cfg.lr_min:.0e} cosine, '
        f'EMA={cfg.ema_decay}, AMP={use_amp}, '
        f'wdl_max={cfg.wdl_weight_max} ===')

    train_exclude = val_idx if val_path == cfg.train_cnnp else None
    t_start_global = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        # Schedule
        f = lr_factor(epoch, cfg)
        for g in optimizer.param_groups:
            g['lr'] = cfg.lr_ft * f
        cur_lr = cfg.lr_ft * f
        wdl_w  = wdl_weight(epoch, cfg)

        # New DataLoader (fresh shuffle stream tied to (seed, epoch))
        train_loader = make_train_loader(cfg, n_train, train_exclude, epoch)

        # Train
        model.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        tl, tn, n_samples = 0.0, 0, 0
        t_data, t_compute = 0.0, 0.0
        t_epoch = time.time()
        t_batch = time.time()

        for stm_i, stm_o, nstm_i, nstm_o, ev, hev, pc, wdl, hwdl in train_loader:
            t_data += time.time() - t_batch

            stm_i  = stm_i.to(device, non_blocking=True)
            stm_o  = stm_o.to(device, non_blocking=True)
            nstm_i = nstm_i.to(device, non_blocking=True)
            nstm_o = nstm_o.to(device, non_blocking=True)
            ev     = ev.to(device, non_blocking=True).unsqueeze(1)
            hev    = hev.to(device, non_blocking=True).unsqueeze(1)
            pc     = pc.to(device, non_blocking=True)
            wdl    = wdl.to(device, non_blocking=True).unsqueeze(1)
            hwdl   = hwdl.to(device, non_blocking=True).unsqueeze(1)

            t_cs = time.time()
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    pred = model(stm_i, stm_o, nstm_i, nstm_o, pc)
                    loss = blended_loss(pred, ev, hev, wdl_w,
                                        target_wdl_real=wdl, has_wdl=hwdl,
                                        piece_count=pc, bucket_weights=bucket_weights,
                                        cfg=cfg)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(stm_i, stm_o, nstm_i, nstm_o, pc)
                loss = blended_loss(pred, ev, hev, wdl_w,
                                    target_wdl_real=wdl, has_wdl=hwdl,
                                    piece_count=pc, bucket_weights=bucket_weights,
                                    cfg=cfg)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            model.clip_weights()
            ema.update(model)
            t_compute += time.time() - t_cs

            tl += loss.item(); tn += 1; n_samples += stm_o.shape[0]
            t_batch = time.time()

        train_loss = tl / max(tn, 1)
        epoch_wall = time.time() - t_epoch
        sps        = n_samples / max(epoch_wall, 1e-9)
        gpu_peak   = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0

        # Validate (EMA)
        val = eval_model(ema.ema, val_loader, device, cfg, bucket_weights,
                         with_calibration=True)
        q, score, _ = quality_score(val, cfg)
        sr = val.get('scale_ratio', 0.0)
        sa = val.get('sign_acc',    0.0)
        mc = val.get('mae_cp',      0.0)

        # Update best_quality BEFORE building ck so last.pt records the
        # true latest value (otherwise a resume from last.pt would see the
        # stale best_quality from the previous epoch).
        marker = ''
        if q < best_quality:
            best_quality = q
            marker = ' ** BEST'

        ck = {
            'epoch':  epoch,
            'model':  model.state_dict(),
            'ema':    ema.state_dict(),
            'optim':  optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'val':    val,
            'quality': q,
            'score':  score,
            'cfg':    asdict(cfg),
            'best_quality': best_quality,
        }
        torch.save(ck, out / 'last.pt')
        if marker:
            torch.save(ck, out / 'best.pt')
            torch.save({'ema': ema.state_dict(),
                        'cfg': asdict(cfg),
                        'epoch': epoch}, out / 'best_ema.pt')

        log(f'E{epoch:3d} | train={train_loss:.5f} val={val["all"]:.5f} '
            f'score={score:.5f} q={q:.5f} | scale={sr:.3f} sign={sa:.1f}% mae={mc:.0f}cp '
            f'| op={val["opening"]:.4f} md={val["middle"]:.4f} eg={val["endgame"]:.4f} '
            f'| lr={cur_lr:.1e} wdl={wdl_w:.2f} | {epoch_wall:.0f}s '
            f'({sps/1e6:.2f}Mpos/s data={t_data:.0f}s compute={t_compute:.0f}s '
            f'gpu={gpu_peak:.1f}GB){marker}')

        with open(metrics_path, 'a', newline='') as fcsv:
            csv.writer(fcsv).writerow([
                epoch, train_loss, val['all'], val['opening'], val['middle'], val['endgame'],
                score, q, sr, sa, mc, cur_lr, wdl_w,
                round(t_data, 3), round(t_compute, 3), round(sps, 1),
                round(gpu_peak, 3), round(epoch_wall, 1),
            ])

    log('')
    log(f'=== done ({(time.time() - t_start_global)/60:.1f} min total) ===')
    log(f'best quality: {best_quality:.6f}')

    best_ema_path = out / 'best_ema.pt'
    if best_ema_path.exists():
        ck = torch.load(best_ema_path, map_location='cpu')
        final_model = PelancaNNUE(cfg)
        final_model.load_state_dict(ck['ema'])
        export_nnue(final_model, str(out / 'final.nnue'), cfg)
        if cfg.sanity_test:
            sanity_test(final_model.to(device), device, cfg, log)
    else:
        log('No best_ema.pt found — skipping final export (training may have crashed).')

    log_fh.close()


if __name__ == '__main__':
    main()
