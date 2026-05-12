# Chess-NN-Pack (CNNP)

A purpose-built binary format for **NNUE training data** — a flat,
mmap-first, zero-dependency-core layout designed to replace HDF5
sparse_v1 in chess-specific training pipelines.

> **Status:** V2 spec frozen + reference implementation complete.
> 190 tests passing (129 C++ + 61 Python). Built and verified on
> Windows MSVC 19.44 / C++20 / CMake 3.20+. Linux/macOS builds should
> work but are not yet validated in CI.

---

## What is CNNP?

```text
binpack (Stockfish) ──converter──▶ .cnnp ──mmap──▶ training (PyTorch / C++ / Rust / ...)
                                       │
                                       ├─ flat 256-byte header
                                       ├─ flags + eval + wdl + features (per-position)
                                       ├─ block-anchored CSR-style prefix sums
                                       └─ optional UTF-8 JSON metadata trailer
```

Three things make it useful:

1. **O(1) random access by position index** — no B-tree, no chunks; an
   anchored block-prefix scheme gets you to any position in two
   pointer dereferences over the mmap.
2. **Zero runtime dependencies in the reader** — POSIX `mmap` /
   Windows `MapViewOfFile`. No `libhdf5`, no `libarrow`, no `libzstd`.
3. **Self-describing** — minimal JSON metadata trailer (forward-compatible)
   plus a fixed binary header that every field is documented byte-by-byte
   in [`notes/CNNP-V2-SPEC.md`](notes/CNNP-V2-SPEC.md).

## Why a new format?

| | Streaming binpack | HDF5 sparse_v1 | **CNNP V2** |
|---|---|---|---|
| Reader deps | `sfbinpack` | `libhdf5` (~10 MB) | **none** |
| Random access | sequential only | O(log n) B-tree | **O(1) flat** |
| mmap | no | partial (chunked) | **yes (whole file)** |
| Single file | yes | yes | **yes** |
| Size vs HDF5 | n/a | baseline | **~25% smaller** |
| Windows pain | low | high | **low** |

CNNP keeps the HDF5 benefits (random shuffle, single-pass conversion)
while dropping the dependency surface and getting genuinely O(1)
indexing.

---

## Quick start

### Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Build options (all default ON except Python):

```
-DCNNP_BUILD_TESTS=ON     # gtest fetched via FetchContent
-DCNNP_BUILD_CLI=ON       # cnnp CLI tool
-DCNNP_BUILD_EXAMPLES=ON  # make_sample helper
-DCNNP_BUILD_PYTHON=OFF   # pybind11 bindings (requires Python dev headers)
```

For Python bindings you must point CMake at a real Python install
(not the Windows Store stub):

```sh
cmake -S . -B build -DCNNP_BUILD_PYTHON=ON \
      "-DPYTHON_EXECUTABLE=C:/path/to/python.exe"
cmake --build build --config Release --target cnnp_python
```

The module lands in `build/python/cnnp.cp3XX-<arch>.pyd` (or `.so`).

For the binpack→cnnp converter (GPL-3.0, isolated):

```sh
cmake -S . -B build -DCNNP_BUILD_BINPACK_CONVERTER=ON
cmake --build build --config Release --target binpack_to_cnnp
```

This downloads nnue-pytorch's binpack reader headers (~9000 LoC,
GPL-3.0) into `build/_deps/`. The resulting `binpack_to_cnnp` binary
is GPL-3.0; the rest of the build stays MIT. See
[`tools/binpack_to_cnnp/README.md`](tools/binpack_to_cnnp/README.md)
for the licensing rationale.

### CLI

```sh
# Generate a tiny sample file (5 hand-crafted positions)
build/Release/make_sample sample.cnnp

# Spec validation (binary layer + JSON trailer lexical check)
build/Release/cnnp validate sample.cnnp
# → OK  sample.cnnp  num_positions=5  num_blocks=1  num_features=56  metadata=50B

# Header summary + WDL / feature stats + JSON trailer dump
build/Release/cnnp inspect sample.cnnp

# Dump positions in human-readable form
build/Release/cnnp dump sample.cnnp --range 0:3
# [0] stm=0 pc=16 wdl=1  eval=0.3   features=[1,200,320,...]
# [1] stm=1 pc=16 wdl=0  eval=-0.11 features=[5,10,20,...]
# [2] stm=0 pc=8  wdl=-1 eval=1.25  features=[100,200,...]
```

Exit codes: `0` success, `1` validation failure, `2` usage error,
`3` file/IO error.

### Read from C++

```cpp
#include "cnnp/cnnp.hpp"  // umbrella header

cnnp::Reader r = cnnp::Reader::open("train.cnnp");  // mmap + validate
std::cout << r.num_positions() << " positions\n";

// Random access — O(1)
cnnp::PositionView v = r.at(42);
std::cout << "stm=" << int(v.stm)
          << " pc=" << int(v.piece_count)
          << " wdl=" << int(v.wdl)
          << " eval=" << v.eval_normalized << "\n";
for (auto fid : v.features) std::cout << fid << " ";

// Bulk zero-copy spans (point into the mmap)
std::span<const std::int16_t> evals = r.eval_array();
std::span<const std::int8_t>  wdls  = r.wdl_array();
std::span<const std::byte>    md    = r.metadata();  // raw JSON bytes
```

### Read from Python (NumPy zero-copy)

```python
import sys; sys.path.insert(0, "build/python")
import cnnp
import numpy as np

r = cnnp.Reader("train.cnnp")        # mmap + validate
print(len(r), "positions")

# Per-position view (features copied; small)
pos = r.at(42)
print(pos.stm, pos.piece_count, pos.wdl, pos.eval_normalized)
print(pos.features)                  # numpy uint16

# Bulk arrays — zero-copy NumPy views over the mmap
evals  = r.eval_array()              # int16[N]   (view)
wdls   = r.wdl_array()               # int8[N]    (view)
wflat  = r.w_flat()                  # uint16[total_features] (view)
print(evals.base is not None)        # True — view holds Reader alive

# Batch gather (single C++ loop) — for DataLoader workloads
indices = np.random.permutation(len(r))[:65536].astype(np.uint64)
batch = r.get_batch(indices)
batch["features"]    # (65536, 32) uint16, padded with UINT16_MAX
batch["counts"]      # (65536,) uint8 — actual feature count per position
batch["evals_norm"]  # (65536,) float32 — decoded eval
batch["wdls"]        # (65536,) int8 — game outcome (white-POV)
batch["stm"]         # (65536,) uint8 — side to move

# Metadata as bytes; parse with stdlib json
import json
meta = json.loads(r.metadata.decode("utf-8"))
print(meta["format"])                # "cnnp_sparse_v2"
```

### Convert a binpack to CNNP

```sh
binpack_to_cnnp train.binpack train.cnnp
# [read=1000000 kept=845022 rate=2450980 pos/s]
# [read=2000000 kept=1689521 rate=2577319 pos/s]
# ...
# Finalising 15443679 positions → train.cnnp ...
#
# --- Summary ---
#   Read       : 18271788
#   Kept       : 15443679 (84.522%)
#   Skipped:
#     captures : 1055883
#     in-check : 1772226
#     rejected : 0
#   Wall time  : 13.0 s
#   Throughput : 1.4M pos/s
```

Default filter matches Stockfish-style training (skip captures + in-check).
Disable with `--no-filter`. Cap output with `--max-positions N` for tests.

Real numbers from a 60 MB Linrock T77 binpack (`high-simple-eval-1k`):

| Metric | Value |
|---|---|
| Input binpack | 60 MB (compressed) |
| Output CNNP | 314 MB (uncompressed, mmap-friendly) |
| Positions read | 18.27M |
| Positions kept | 15.44M (84.5%) |
| Conversion time | 13 s on a single CPU core |
| Read throughput | ~1.4M positions/s |
| `cnnp validate` | 446 ms (full O(N) scan over all invariants) |
| Per-position random access (Python) | 4–5 µs (mmap O(1) + Python overhead) |
| **Batched random access (`get_batch`)** | **~225 ns per position** (C++ loop) |

For DataLoader-scale workloads the per-position Python overhead dominates;
`Reader.get_batch(indices)` runs the gather entirely in C++ and is **30–80×
faster than calling `at(i)` in a Python loop**:

| Batch size | `at(i)` loop | `get_batch()` | Speedup |
|---:|---:|---:|---:|
| 256 | 10.2 ms | 0.3 ms | 32× |
| 1024 | 23.1 ms | 0.3 ms | 74× |
| 8192 | 219.7 ms | 3.8 ms | 58× |
| 16384 | 303.2 ms | 3.8 ms | 79× |
| 65536 | 757.4 ms | 14.8 ms | 51× |

For batch=65536 this drops data-fetch from 757 ms (GPU starves) to 14.8 ms
(GPU saturated by typical 50–200 ms forward+backward step).

### PyTorch DataLoader integration

[`examples/python/cnnp_dataset.py`](examples/python/cnnp_dataset.py)
shows a complete `IterableDataset` + DataLoader integration plus a
self-contained benchmark. End-to-end measurements on the 50M-position
T76 file (2.1 GB cnnp):

| Configuration | Phase | Throughput | Per-batch breakdown |
|---|---|---:|---|
| batch=16384, 1000 batches | data-only | **2.01 M pos/s** | 8.16 ms fetch (steady-state) |
| batch=16384, 1000 batches | full loop on GTX 1050 | 0.23 M pos/s | fetch 9.79 ms (14%) / compute 61.75 ms (86%) — **GPU-bound** ✓ |
| batch=65536, 250 batches | data-only | 1.66 M pos/s | 39.54 ms (slightly slower per pos due to 4 MB output buffer overflowing L2) |
| batch=65536, 250 batches | full loop on CPU | 0.03 M pos/s | fetch 45.66 ms (2%) / compute 1907 ms (98%) — CPU obviously compute-bound |

For the typical NNUE training range (batch=8k–32k on a modern GPU),
data fetch sustains **~2 M pos/s** and consumes roughly 5–10 ms per
batch — well under the typical 50–200 ms forward+backward step on
A100/H100. The pipeline does NOT need `num_workers > 0` for most
real workloads.

Run it yourself:

```sh
PYTHONPATH=build/python python examples/python/cnnp_dataset.py \
    --cnnp test_data/farseerT76_50m.cnnp \
    --batch-size 16384 --max-batches 1000
```

### Write a file

```cpp
#include "cnnp/cnnp.hpp"

cnnp::Writer w;
std::vector<std::uint16_t> features = {1, 2, 3, 4};
w.add(/*score_cp=*/100, /*wdl=*/1, /*stm=*/0,
      /*piece_count=*/4, std::span<const std::uint16_t>(features));
// ... add more positions ...
w.finalize("out.cnnp");  // computes layout, validates, atomic write
```

```python
import cnnp, numpy as np

w = cnnp.Writer(max_in_memory_bytes=16 * 1024**3)  # optional 16 GB cap
feats = np.array([1, 2, 3, 4], dtype=np.uint16)
w.add(score_cp=100, wdl=1, stm=0, piece_count=4, features=feats)
w.finalize("out.cnnp")
```

---

## What's in the box

```
include/cnnp/
  byte_reader.hpp       endian-safe LE primitive read/write helpers
  header.hpp            Header struct + constants + parse/serialize
  encoding.hpp          flags / eval / wdl encoders & decoders
  validator.hpp         spec §7 invariants (binary + cross-array checks)
  mmap_region.hpp       RAII mmap (POSIX + Windows)
  reader.hpp            random-access Reader + PositionView
  writer.hpp            incremental Writer + write_cnnp_file()
  cnnp.hpp              umbrella header (one include for everything)

src/
  *.cpp                 implementations
  cli/cnnp_main.cpp     `cnnp` CLI (validate / inspect / dump)

examples/
  make_sample.cpp       generates a 5-position sample.cnnp for manual testing

python/
  cnnp_py.cpp           pybind11 module (Reader, Writer, PositionView, ...)
  test_cnnp.py          smoke test suite (run after building cnnp_python)

tools/binpack_to_cnnp/
  binpack_to_cnnp.cpp   GPL-3.0 — converter using vendored nnue-pytorch
  README.md             explains the GPL/MIT isolation

examples/python/
  cnnp_dataset.py       PyTorch DataLoader integration + benchmark

tests/
  test_header.cpp       23 tests — wire layout, parse/serialize, rejection paths
  test_encoding.cpp     28 tests — flags, eval, wdl, mate range, throw cases
  test_validator.cpp    40 tests — every spec §7 invariant, corruption tests
  test_reader.cpp       12 tests — mmap, at(), bulk views, defensive bounds
  test_writer.cpp       22 tests — round-trips, validation, metadata, guardrails
  test_golden.cpp        4 tests — hand-derived 4-position file, byte-exact
  test_fixtures.hpp     shared ValidFile builder

notes/
  CNNP-V2-SPEC.md       authoritative wire-format spec (FROZEN)
```

## Spec

The format is fully specified in
[**`notes/CNNP-V2-SPEC.md`**](notes/CNNP-V2-SPEC.md) (~900 lines).
Highlights:

- 256-byte fixed header with explicit byte offsets per field
- Block size 1024, max 32 features per position, count_base 2
- HalfP feature encoding (768 distinct ids by default)
- `int16` normalized fixed-point eval at scale 3000
- WDL stored as `int8` ∈ {-1, 0, +1}
- Anchored block-prefix sums for O(1) random access
- UTF-8 JSON metadata trailer required (`format` + `layout` fields)
- Little-endian only, 64-bit only

Any spec change requires a new format version (`magic = "CNN3"`).

## Testing

```sh
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
# ───
# Test #1: test_header     — 23 ✓
# Test #2: test_encoding   — 28 ✓
# Test #3: test_validator  — 40 ✓
# Test #4: test_reader     — 12 ✓
# Test #5: test_writer     — 22 ✓
# Test #6: test_golden     —  4 ✓
# 100% tests passed (~0.7 s)
```

Python smoke (after building `cnnp_python`):

```sh
PYTHONPATH=build/python python python/test_cnnp.py
# ── 9 sub-suites, 61 individual checks ──
```

## Limits and caveats

The **Writer is in-memory** in this V2 implementation. Approximate
peak RAM (HalfP, ~16 features/position):

| Dataset | Peak RAM | Verdict |
|---|---|---|
| 100M positions | ~8 GB | viable on 16+ GB workstations |
| 500M positions | ~40 GB | needs a high-RAM machine |
| 2B positions | ~160 GB | **not viable** — wait for StreamingWriter |

Use `WriterConfig::max_in_memory_bytes` (or `max_in_memory_bytes=` in
Python) as an explicit guardrail to fail loudly instead of running
the OS out of memory.

A future `StreamingWriter` will write the `w_flat` array directly to
disk, capping peak memory under ~10 GB regardless of dataset size.

## Roadmap

Done:
- [x] V2 wire-format spec, frozen
- [x] Header parser/serializer (field-by-field, no `reinterpret_cast`)
- [x] Encoding (`flags`, `eval`, `wdl`, mate-range fixed-point)
- [x] Validator (~30 invariants, including cross-array `piece_count == nnz`)
- [x] mmap-based Reader + zero-copy bulk accessors
- [x] In-memory Writer + atomic finalize
- [x] CLI (`cnnp validate | inspect | dump`)
- [x] Python bindings (pybind11, NumPy zero-copy views)
- [x] Golden file regression test (hand-derived bytes)
- [x] 190 unit tests across 7 binaries

Next:
- [x] **`binpack_to_cnnp` converter** (closes the binpack → cnnp loop)
- [ ] **HDF5 vs CNNP benchmark** (size, read speed, batch latency)
- [ ] **`StreamingWriter`** for datasets > 500M positions
- [ ] **Linux + macOS CI matrix** (currently Windows MSVC verified only)
- [ ] **PyPI wheel** via scikit-build-core (today: manual `cmake` + `PYTHONPATH`)
- [ ] **PyTorch DataLoader reference example**

## Related work

- [Stockfish binpack](https://github.com/official-stockfish/Stockfish) — source format and filter semantics
- [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) — reference NNUE training pipeline
- [HDF5 sparse_v1](https://www.hdfgroup.org/solutions/hdf5/) — schema layout inspiration (offsets + values)
- [AnnData (.h5ad)](https://anndata.readthedocs.io/) — CSR-on-disk pattern
- [Apache Arrow](https://arrow.apache.org/) — mmap-first design philosophy

## License

[MIT](LICENSE).
