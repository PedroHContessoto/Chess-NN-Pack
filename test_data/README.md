# test_data/

Local-only directory for large test datasets. **Nothing here is
committed to git** (everything except this README and `.gitkeep` is
covered by the project's `.gitignore`).

## Suggested layout

```
test_data/
├── *.binpack       Stockfish binpack training data (input)
├── *.cnnp          CNNP V2 files (output of binpack→cnnp converter)
├── *.h5            Legacy HDF5 sparse_v1 files (for A/B benchmarks)
└── golden/         Optional: hand-crafted small fixtures for regression tests
```

## Where to get binpacks

- **Linrock's filtered collection** (recommended for first tests):
  https://robotmoon.com/nnue-training-data/
  Smallest official: `test77-jan2022-2tb7p.high-simple-eval-1k.min-v2.binpack` (~60 MB)
- **Stockfish testing infrastructure** (fishtest): per-iteration small binpacks
- **nnue-pytorch test fixtures**: tiny (<1 MB) but only useful as smoke tests

## Why ignored

Binpacks routinely run from 60 MB (filtered subsets) to 6+ GB (full
training corpora). HDF5 conversions and CNNP files inherit those
sizes. Committing them would balloon the repo and require Git LFS;
keeping them local matches how every other NNUE pipeline handles
training data.
