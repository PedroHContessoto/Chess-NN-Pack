# Tooling Roadmap

> Date: 2026-05-13
> Scope: Auxiliary tools and CLI extensions identified after the
> 8-run ablation battery. Items are prioritized by value/cost ratio
> and by whether they prevent silent bugs.

## Priority queue

| # | Tool | Priority | Effort | Value |
|---|---|---|---|---|
| 1 | Fixed val index splits | **P0** | 1-2 h | Removes seed-induced confounding from val set |
| 2 | NNUE parity test (Python) | **P0** | 4-6 h | Detects 8/9 families of silent export bugs |
| 3 | `compare-runs` CLI | P1 | 3-5 h | Automates the cross-run analysis done manually |
| 4 | `cnnp inspect --full-stats` | P2 | 2-3 h | Histograms over the dataset (descriptive only) |
| 5 | `cnnp check-targets` | P2 | 3-4 h | Dataset/encoding QA. Defer until 2nd dataset arrives |
| 6 | `benchmark-dataloader` formalized | P3 | 2-3 h | Currently informal. Already proven `data_time < 5%` |

Total P0: ~5-8 h. P0 + P1: ~10-13 h.

---

## P0 #1 — Fixed val index splits

**Problem.** Today the val split is computed via Floyd sampling seeded by
`--seed`. When two runs use different seeds (e.g. seed=0 vs seed=42 for
variance probing), they also see different val positions. The reported
quality difference therefore mixes architectural variance with val-set
variance. We saw this in the variance battery — for genuine ablations
we want the val set held constant across runs.

**Solution.** Pre-generate and version a small set of `.npy` files
containing sorted int64 index arrays:

```
test_data/val_splits/
    val_500k_seed20260513_n1000000000.npy
    val_100k_seed20260513_n1000000000.npy   (smaller, faster eval)
```

Add `--val-indices PATH` to `train_nnue.py`. When supplied:
- Load the file as the val_indices array
- Train sampler excludes those indices per chunk (already supported)
- Skip the Floyd sampling step entirely

Filename encodes the dataset size (`n1000000000`) so a wrong-file pair
trips on the bounds check.

**Status:** implemented (this commit). Generator at
[`examples/python/make_val_splits.py`](../examples/python/make_val_splits.py).
Trainer flag in [`examples/python/train_nnue.py`](../examples/python/train_nnue.py).

---

## P0 #2 — NNUE parity test (Python loader)

**Problem.** The `.nnue` export path (PyTorch fp32 → int16/int8 quant +
binary serialization) is the single biggest source of silent bugs:

| Bug class | Detection |
|---|---|
| SCReLU vs CReLU activation mismatch | parity output diverges |
| Quantization scale wrong (FT_QUANT, etc.) | parity diff scales with input magnitude |
| Hidden bias scale wrong | constant offset in output |
| Output bucket indexing wrong | matches in one phase, fails in another |
| STM/NSTM stream swap | wrong sign for black-to-move positions |
| Mirror flip wrong | matches normal FEN, fails on file-flipped |
| Endianness or byte order | total garbage |
| Header field misalignment | crash or wrong layer dims |

Validation loss alone catches none of these reliably.

**Solution.** Implement a Python loader that reads PLNN v4 binary, runs
a pure-Python (numpy) forward, and compares against the PyTorch model
on a held-out FEN list. Max diff in centipawns is the metric.

Pure-Python loader replicates the engine's path:
1. Parse 28-byte header
2. Read FT weights (i16) + bias (i16), reshape (FT_SIZE, INPUT_SIZE)
3. Read hidden weights (i8) + bias (i32)
4. Read NUM_BUCKETS × (out weights i8 + out bias i32)
5. Forward: sum embeddings → dequant → SCReLU → hidden → SCReLU → bucket select

Test surface:
- 20-50 hand-picked FENs (varied phases, material, in/out of book)
- Pass FEN through both PyTorch model AND Python binary loader
- Assert `max(|cp_pytorch - cp_loader|) < 5 cp` (quant rounding bound)

**Bug coverage:** 8 of 9 classes listed above. Misses runtime engine
bugs (SIMD lanes, incremental update) which need cooperative testing
in the Pelanca engine repo.

**Status:** designed, not implemented. Target output:
`tools/nnue_parity.py` (or `examples/python/nnue_parity.py`).

---

## P1 — `compare-runs` CLI

Walks `runs/*/{metrics.csv,config.json}`, prints a comparison table:

```
$ python tools/compare_runs.py runs/
                                 best_q   sign    mae   scale  epochs   wall
baseline_seed0                  0.0819   91.0%   59   0.899     12   11.3m
var_seed42                      0.0804   91.1%   59   0.902     12   11.3m
var_seed1337                    0.0812   91.1%   59   0.900     12   11.3m
h16_samples500M                 0.0784   91.3%   57   0.906     12   28.0m
abl_wdl010_e6                   0.0888   90.5%   63   0.810      6    5.7m
...

config diff (vs baseline_seed0):
  var_seed42:        seed: 0 -> 42
  h16_samples500M:   hidden_size: 8 -> 16
                     samples_per_epoch: 200000000 -> 500000000
```

Bonus features:
- `--group-by hidden_size,samples_per_epoch` to aggregate by knob
- `--metric sign` to sort by something else
- Variance-aware: when ≥ 2 runs share the same config, report `mean ± std`

**Status:** designed, not implemented.

---

## P2 — `cnnp inspect --full-stats`

Extend the existing `cnnp inspect` with histogram output:

```
$ ./cnnp inspect --full-stats train.cnnp

[... existing header dump ...]

Piece-count distribution:
   2- 5: ........... 12.3 %  (123,400,000)
   6- 9: ......... 18.7 %  (187,000,000)
  ...
  30+ : .... 8.2 %  (82,000,000)

Eval distribution (normalized, target_clip=10):
  -10.0 .. -6.5: ▏ 3.4 %  (mate-like, negative)
   -6.5 .. -4.5: ▎ 6.8 %
   ...
    0.0 ..  0.0: ████  too narrow to count
   ...
   +6.5 .. +10.0: ▏ 3.6 %

WDL distribution:
  white wins:  33.1 %
  draws:       38.7 %
  black wins:  28.2 %

STM balance:
  white to move: 50.05 %
  black to move: 49.95 %

Avg features/position: 19.34
```

Pure C++, no new deps. Single O(N) pass over the flags / eval / wdl
arrays.

**Status:** designed.

---

## P2 — `cnnp check-targets` (deferred)

Defer until a second dataset arrives. The current farseerT76 was
manually validated against engine sanity FENs and metric coherence,
which suffices for the first dataset.

When implemented, surface anomalies like:
- Mean eval by STM (POV inversion check)
- WDL vs eval sign agreement (encoding sanity)
- Draws with `|eval| > 4.5` (potentially noisy positions)
- Wins/losses with `|eval| < 0.5` (engine disagreement)

**Status:** deferred.

---

## P3 — `benchmark-dataloader`

Already proven `data_time = 1.04 %` of compute across all 9 runs.
Formalize as a CI-style check only when a regression risk appears
(e.g. after architectural changes to the sampler).

**Status:** deferred.

---

## What this roadmap does NOT include

- **Streaming Writer.** Already in the V2 spec roadmap, separate effort.
- **Match-Elo tooling.** Lives in the downstream Pelanca engine repo;
  not in scope for CNNP.
- **Multi-dataset interleaving.** Not needed until a 2nd dataset exists.
- **HalfKAv2_hm / new feature sets.** Spec-level change, out of scope
  for the current V2 era.
