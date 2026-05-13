# End-to-End Pipeline Validation & Baseline Discovery

> Date: 2026-05-12 / 2026-05-13
> Scope: Full `binpack → cnnp → trained .nnue` pipeline, validated on
> the 1B-position farseerT76 dataset, followed by an 8-run ablation
> battery to characterize the training surface.
> Status: **Pipeline structurally validated. Baseline #1 frozen.
> Production config identified.**

---

## Executive summary

- **Pipeline works end-to-end** on a real 1B-position dataset, 11.3 min
  for 12 epochs on a single RTX 5070 Ti, no HDF5, no RAM blow-up.
- **Streaming claim validated:** data fetch is 1.04% of wall time across
  every run. The 5070 Ti is GPU-compute-bound, not data-bound.
- **Baseline #1 frozen:** `pelanca_v26_farseer_1b_evalonly_h8_b65536_lr2.4e3`
  — H=8, eval-only, batch 65k, lr 2.4e-3 cosine, 12 epochs, 200 M
  samples/epoch. Final: sign 91.0 %, mae 59 cp, scale 0.90.
- **8-run ablation battery (~2 h GPU)** characterized the training
  surface. Of every knob tested, only `--samples-per-epoch` produced
  a quality gain that beat seed-to-seed variance.
- **Three default knobs proven to be placebo** — material init, bucket
  weights, WDL blending. Should be removed from the canonical defaults.

---

## Environment

| | Value |
|---|---|
| Provider | Vast.ai |
| GPU | NVIDIA RTX 5070 Ti |
| CPU / RAM | 125 GiB total, 122 GiB free at run start |
| Disk | 125 GB overlay, 118 GB free |
| OS | Linux (Docker), GCC 13, CMake ≥ 3.20 |
| Python | 3.12 in `/venv/main`, PyTorch with CUDA |
| Source data | `https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/farseerT76.binpack` |

## Pipeline stages — measured once

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCNNP_BUILD_PYTHON=ON \
      -DCNNP_BUILD_BINPACK_CONVERTER=ON \
      -DPYTHON_EXECUTABLE=$(which python3)
cmake --build build --config Release -j $(nproc)
```

| Stage | Time | Output |
|---|---|---|
| Build core + CLI + Python module | ~2-4 min | `cnnp_cli`, `build/python/cnnp.cpython-312-x86_64-linux-gnu.so` |
| Build `binpack_to_cnnp` (incl. FetchContent) | ~60 s | `build/binpack_to_cnnp` (GPL-3.0, isolated) |
| 129 C++ tests + 61 Python checks | passed | gtest binaries |
| Download `farseerT76.binpack` | < 1 min | 5.8 GB |
| Convert binpack → CNNP (1B positions cap) | 5 min 6 s | `farseerT76_1b.cnnp` (42 GB) |
| `cnnp validate` (full O(N) scan) | ~30 s | num_positions=1,000,000,000, num_features=19,334,552,948 |

**Conversion details:**
```
Read   : 1,307,755,287
Kept   : 1,000,000,000  (76.47 %)
Skipped: 236,129,680 captures + 71,625,607 in-check
Throughput: 4,275 K pos/s (read)
```

Peak RAM during conversion: ~38 GB. Without `--max-positions`, the full
2.17 B-position binpack would have buffered ~130 GB and triggered OOM —
documented separately in the writer's rule-of-thumb table at the end.

---

## Baseline #1 — `pelanca_v26_farseer_1b_evalonly_h8_b65536_lr2.4e3`

```sh
PYTHONPATH=build/python python3 examples/python/train_nnue.py \
    --train test_data/farseerT76_1b.cnnp \
    --epochs 12 --batch-size 65536 --lr 2.4e-3 \
    --samples-per-epoch 200000000 --num-workers 4 \
    --output runs/pelanca_v26_farseer_1b
```

Run completed 23:50 → 00:01 UTC. **All 12 epochs were monotonic `BEST`.**

Per-epoch metrics:

| E | train | val | sign | scale | mae | lr |
|---|------:|----:|----:|------:|---:|---:|
| 0  | 0.0895 | 0.0860 | 89.2 % | 0.770 | 76 | 1.2e-3 |
| 1  | 0.0722 | 0.0720 | 90.2 % | 0.870 | 65 | 2.4e-3 |
| 2  | 0.0683 | 0.0690 | 90.5 % | 0.884 | 63 | 2.4e-3 |
| 3  | 0.0666 | 0.0676 | 90.6 % | 0.887 | 62 | 2.3e-3 |
| 4  | 0.0655 | 0.0669 | 90.7 % | 0.889 | 61 | 2.1e-3 |
| 5  | 0.0648 | 0.0662 | 90.8 % | 0.892 | 61 | 1.8e-3 |
| 6  | 0.0644 | 0.0656 | 90.9 % | 0.894 | 60 | 1.4e-3 |
| 7  | 0.0638 | 0.0650 | 90.9 % | 0.896 | 60 | 1.0e-3 |
| 8  | 0.0634 | 0.0645 | 91.0 % | 0.898 | 60 | 6.2e-4 |
| 9  | 0.0634 | 0.0641 | 91.0 % | 0.899 | 59 | 3.0e-4 |
| 10 | 0.0634 | 0.0639 | 91.0 % | 0.900 | 59 | 9.2e-5 |
| 11 | 0.0636 | 0.0638 | 91.0 % | 0.899 | 59 | 2.0e-5 |

Headline numbers:

| | |
|---|---|
| Total wall time | **11.3 min** |
| Steady-state throughput | 3.6 M pos/s |
| `data_time` / epoch | 1 s (1.04 % of compute) |
| GPU memory peak | 1.0 / 16 GB (6 %) |
| Final `final.nnue` | 583 KB (PLNN v4, H=8, K=8, SCReLU, scale=400) |
| Best composite quality | 0.081864 |
| h_sat / o_sat | 0.00 % / 3.12 % |

Sanity test on EMA:

| FEN | Predicted | Expected (script) | Notes |
|---|---:|---:|---|
| Initial | +69 cp | 0 | minor white bias — typical |
| 1.e4 (B to move) | +55 cp | 0 | matches engine baseline |
| KR vs K (W) | +917 cp | +300 | sign correct; script threshold conservative — Stockfish at d20 also evaluates K+R vs K in this range |
| KQ vs K (W) | +1110 cp | +800 | sign correct |
| K vs KQ (W) | −1139 cp | −800 | symmetric, perfect |

---

## Ablation battery — 8 experimental runs

Total wall time: ~2 h. Each ablation isolated one variable from the
baseline. Master table (all `q` values at E11, except where noted):

| Run | Config delta from baseline | best q | sign | mae | Verdict |
|---|---|---:|---:|---:|---|
| #1 baseline | (reference) | 0.08186 | 91.04 % | 59 | **reference** |
| #2 H=16 | `--hidden-size 16` | 0.08073 | 91.11 % | 58 | within variance |
| #3 WDL=0.10 | `--wdl-weight 0.10` (epochs=6) | 0.08876 (E2 best) | 90.5 % | 63 | **refuted — quality regression** |
| #4 batch 131k | `--batch-size 131072 --lr 4.8e-3` (epochs=6) | 0.08590 (E4 best) | 90.7 % | 61 | refuted — no throughput gain |
| A | `--seed 42` | 0.08040 | 91.10 % | 59 | variance probe |
| B | `--seed 1337` | 0.08123 | 91.10 % | 59 | variance probe |
| C | `--no-material-init` (epochs=6) | 0.08529 (E6) | 90.8 % | 61 | **placebo** |
| D | `--bucket-weights uniform` (epochs=6) | 0.08254 (E6) | 90.8 % | 60 | **placebo (numeric artifact)** |
| E | `--hidden-size 16 --samples-per-epoch 500000000` | **0.07839** | **91.30 %** | **57** | 🟢 **real signal (beats variance)** |

### Detailed notes per run

**#2 — H=16 (12 ep, 200M samples):** marginal. FT layer (97.8 % of params)
is unchanged; doubling hidden only adds 6 k params (+2.1 % total). `o_sat`
halves (3.12 → 1.56 %) so quantization headroom improves, but val gain
(−1.4 %) is **within variance** measured later by runs A/B.

**#3 — WDL=0.10 (6 ep):** **degraded every metric** the moment WDL ramped
in at E3. Quality −3.8 %, mae +5 cp, scale_ratio collapsed from 0.884 →
0.810. Root cause: the CNNP target is already Stockfish-normalized eval
(`cp → tanh(cp / EVAL_SCALE)`), which implicitly encodes WDL via the
sigmoid mapping. Adding a separate MSE on WDL fights the eval gradient
without providing independent information. **Hypothesis refuted.**

**#4 — batch 131k, lr 4.8e-3 (6 ep):** **wall time identical to baseline
(~55 s/epoch).** Doubling batch halved steps but doubled work per step;
the 5070 Ti was already compute-bound at batch 65 k for this tiny model.
E0 also showed instability from the linear-scaled lr=4.8e-3 (val 0.154 vs
baseline 0.086). Linear LR scaling is not safe above 65 k on this setup.
**Escalation to 262 k / 524 k stopped.**

**A — `--seed 42` (12 ep, baseline config):** Δq vs baseline = −1.8 %.
Variance probe.

**B — `--seed 1337` (12 ep, baseline config):** Δq vs baseline = −0.8 %.
Variance probe.

**C — `--no-material-init` (6 ep):** E0-E2 essentially identical to
baseline (differences < 0.05 % in quality). The piece-value nudge during
FT init is **dominated by kaiming initialization within 1-2 epochs** at
this dataset scale. **Placebo.**

**D — `--bucket-weights uniform` (6 ep):** apparent quality improvement
(0.08254 vs baseline E6=~0.0863) is a **numeric artifact**: the default
weights `[1,1,1.2,1.5,2,2,1.5,1]` have mean ≈ 1.4, so all loss values
scale by ~1.4 vs the uniform `[1]*8` profile. **Variance-independent**
metrics (sign 90.8 vs 90.8, mae 60 vs 60.5, scale 0.898 vs 0.892) are
identical to baseline. **Placebo.**

**E — H=16 + 500M samples (12 ep):** **the only experiment that beats
the noise floor.** Final q = 0.07839, which is 0.002 below the **lowest**
seed in the variance probes (A: 0.08040). All phases improved
significantly (op −3.7 %, md −4.2 %, eg −6.3 %). Sign hit 91.3 %, the
first reading above the 91.0-91.1 % plateau. **Real signal.**

---

## Variance baseline — the noise floor

Three runs with identical config but different seeds (0 / 42 / 1337):

| Seed | best q | sign | mae | scale |
|---|---:|---:|---:|---:|
| 0 (baseline) | 0.08186 | 91.04 % | 59 | 0.899 |
| 42 (Run A) | 0.08040 | 91.10 % | 59 | 0.902 |
| 1337 (Run B) | 0.08123 | 91.10 % | 59 | 0.900 |
| **min / max** | 0.08040 / 0.08186 | | | |
| **range** | **0.00146 (1.8 %)** | < 0.1 pp | 0 | 0.003 |
| **mean ± std** | 0.0812 ± 0.0007 | | | |

**Decision rule:** any experiment showing Δq < 1.8 % vs baseline is
statistically indistinguishable from a seed flip. Re-interpreting all
the ablations against this:

| Δq vs baseline | Within variance? |
|---|---|
| H=16 (#2): −1.4 % | Yes — could be noise |
| WDL (#3 E5 vs baseline E5): +5.0 % | No — clearly worse |
| batch 131k (#4 E5 vs baseline E5): +1.2 % | Borderline; throughput nil → no value |
| no material init: ~0 % | Yes — placebo |
| bucket uniform: ~0 % (variance-indep metrics) | Yes — placebo |
| H=16 + 500M (E): −4.2 % | **No — real signal (2.7 σ)** |

---

## Consolidated findings

### F1. Data pipeline is empirically irrelevant (1.04 % of compute)

Steady-state throughput: 3.6 M pos/s, `data_time = 0-1 s` per epoch
across every run. The CNNP streaming sampler design is validated on a
real 1B-position dataset; I/O is decisively not the bottleneck.

### F2. Natural seed-to-seed variance is 1.8 % of quality

Sets the noise floor for all comparisons. Any single-run Δ smaller
than this is statistically meaningless without multi-seed replication.

### F3. The only real lever found is `samples_per_epoch`

Out of 8 ablations, only Run E (which raised samples_per_epoch from 200M
to 500M) produced a quality gain that beat the variance baseline. All
other knobs (H=16, WDL, batch, init, bucket weights) gave Δ within noise.

The H=16/500M combination confounds two variables — a follow-up run with
**H=8 + 500M** is needed to confirm whether the gain transfers fully to
H=8 or whether H=16 contributes some.

### F4. Pelanca v23 architecture is FT-dominated

```
FT (768 → 384):  294,912 params  (97.8 %)
hidden:           6-12 k params   (~2 %)
out:              64-128 params   (negligible)
```

`--hidden-size` is a weak lever — H=8 → H=16 only adds 2.1 % to total
parameters and yields proportional gains. To make a real capacity jump,
the lever would be `ft_size` — but that requires a new PLNN binary
format, so it's deferred.

### F5. GPU is compute-bound at batch=65 k for this network

```
batch  65 k → 3.6 M pos/s → 55 s/epoch
batch 131 k → 3.6 M pos/s → 55 s/epoch  (same wall, no speedup)
```

The 5070 Ti is saturated by a 300 k-param network at batch 65 k. AMP is
on, cudnn benchmark is on; there's no per-step overhead left to amortize.
Linear LR scaling above 65 k is also unsafe (Run #4 E0 showed warmup
instability). **Sweet spot is batch=65 k / lr=2.4e-3 — do not escalate.**

### F6. Three default knobs are placebo

- **`material_init`** — piece-value nudge in FT init is washed out by
  kaiming within 1-2 epochs at this dataset scale.
- **`bucket_weights` (MD profile)** — apparent loss reduction is a
  numeric artifact; variance-independent metrics (sign/mae/scale) are
  identical to uniform weighting.
- **`wdl_weight` > 0** — redundant with Stockfish-normalized targets;
  introduces conflicting gradients without independent information.

**Recommended default cleanup** (see Production config below).

### F7. sign_acc plateau near 91 % is real for H=8

Saturates at 91.0-91.1 % across seeds, epochs, and schedule variants.
The only run that exceeded was Run E (H=16 + 500M, 91.3 %), and even
that is only +0.2 pp. **Probably a capacity ceiling**, not a training-
schedule issue. The real way past would be either a bigger FT or a
richer feature set (HalfKAv2_hm), both engine-level changes.

### F8. Cosine tail (last ~2 epochs) is mostly cosmetic

Epoch-over-epoch quality improvement at lr < 1e-4 is < 0.05 % per
epoch — within rounding for 6-epoch runs. To save time on ablations,
`--epochs 6` captures ≥ 83 % of the signal. Reserve `--epochs 12` for
final candidate runs.

---

## Production config recommendation (H=8)

Based on the eight runs:

```sh
PYTHONPATH=build/python python3 examples/python/train_nnue.py \
    --train test_data/farseerT76_1b.cnnp \
    --epochs 12 --batch-size 65536 --lr 2.4e-3 \
    --hidden-size 8 \
    --samples-per-epoch 500000000 \
    --no-material-init \
    --bucket-weights uniform \
    --num-workers 4 \
    --output runs/production_h8_v1
```

Rationale by knob:

| Knob | Value | Why |
|---|---|---|
| `--hidden-size` | **8** | H=16 gain (1.4 %) is within variance; H=8 keeps NPS in the engine |
| `--batch-size` | **65,536** | GPU sweet spot; higher batch does not accelerate (F5) |
| `--lr` | **2.4e-3** | Pairs with batch=65 k; higher linear scaling is unstable |
| `--epochs` | **12** | Final convergence; ablations can use 6 |
| `--samples-per-epoch` | **500,000,000** | F3 — only real lever found; unconfirmed for H=8 specifically |
| `--no-material-init` | **on** | F6 — placebo |
| `--bucket-weights uniform` | **on** | F6 — placebo |
| `--wdl-weight` | **0.0** (default) | F6 — refuted |
| `--no-mirror` | **off** (mirror stays on) | mirror is a real symmetry, not a placebo |

Expected outcome (extrapolated from Run E): q ≈ 0.078-0.080, sign ≈
91.2-91.3 %, mae ≈ 57-58 cp. To be confirmed by running this exact
command (the H=8 + 500M cell is the one unfilled gap).

---

## Open questions

1. **Does samples_per_epoch=500M help H=8?** — Run G unblocks the production
   config. Should be run as the next experiment if GPU time is available.
2. **Does H=16 actually win Elo in match play?** — `o_sat` halved means
   quantization rounding bites less. May translate to small Elo even
   with similar val. Needs `pelanca bench` + fastchess in the engine repo.
3. **Where does sign plateau if FT is widened?** — Would require a new
   PLNN binary format and engine update. Not blocking.
4. **Is the cap of 1B positions still the right scale?** — Run E suggests
   more positions help; eventually limited by writer RAM (~130 GB for
   2 B positions without StreamingWriter).

## Next milestones

- **Downstream (Pelanca engine repo):** integrate `final.nnue` files,
  run `pelanca bench`, fastchess A/B match (≥ 400 games at 10+0.1).
- **Upstream (Chess-NN-Pack):**
  - Run G (H=8 + 500M) to close the production gap
  - Apply default cleanup (material_init off, bucket_weights uniform)
    in `train_nnue.py`
  - Implement `StreamingWriter` to remove the 1B-position ceiling

---

## Appendix — converter memory rule of thumb

The in-memory `Writer` requires ~`(8 + 4 × avg_features)` bytes per
position at peak (`finalize()` allocates a second buffer of equal size
to the in-memory arrays). For HalfP with avg ~16 features:

| Total RAM | Safe `--max-positions` cap |
|---|---|
| 32 GB | 100 M |
| 64 GB | 300 M |
| 128 GB | 1 B (this run) |
| 256 GB+ | 2 B+ (or wait for `StreamingWriter`) |

The `--max-mem-gb` guardrail in the CLI only triggers at `finalize()`;
it does not prevent OOM during incremental buffering. Always pair big
conversions with `--max-positions N`.

---

## References

- Wire format spec: [`notes/CNNP-V2-SPEC.md`](CNNP-V2-SPEC.md)
- Trainer source: [`examples/python/train_nnue.py`](../examples/python/train_nnue.py)
- Converter source: [`tools/binpack_to_cnnp/binpack_to_cnnp.cpp`](../tools/binpack_to_cnnp/binpack_to_cnnp.cpp)
- Source dataset: [official-stockfish/master-binpacks on Hugging Face](https://huggingface.co/datasets/official-stockfish/master-binpacks)
