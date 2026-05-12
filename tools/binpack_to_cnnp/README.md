# binpack_to_cnnp

A small command-line tool that converts a Stockfish `.binpack` training
file into the CNNP V2 format. Lives in `tools/` because of its
licensing posture (see below) — the rest of the project is unaffected.

## Status

Stage 1: **compile-check stub.** The vendoring + build wiring is
verified; the actual conversion logic is stage 3.

## Build

Disabled by default. Enable with:

```sh
cmake -S . -B build -DCNNP_BUILD_BINPACK_CONVERTER=ON
cmake --build build --config Release --target binpack_to_cnnp
```

On first configure, CMake fetches nnue-pytorch's headers via
`FetchContent` into `build/_deps/nnue_pytorch-src/`. The repo itself
never commits those files.

## Licensing

This tool **vendors** nnue-pytorch's data loader headers
(`nnue_training_data_formats.h`, `nnue_training_data_stream.h`,
`rng.h`, `thread_safe_types.h`), which are
[GPL-3.0-or-later](https://github.com/official-stockfish/nnue-pytorch/blob/master/LICENSE).

As a consequence:

- The resulting **`binpack_to_cnnp` binary is GPL-3.0-or-later.**
- The `cnnp` static/shared library it links against remains **MIT**.
- The main `cnnp` CLI (`validate`, `inspect`, `dump`), the Python
  bindings (`cnnp.pyd` / `cnnp.so`), and every header under
  `include/cnnp/` remain **MIT**.
- MIT code can be linked into a GPL binary — that absorption is
  one-way and explicitly allowed by both licenses.

If you redistribute the `binpack_to_cnnp` binary, you must follow
GPL-3.0-or-later requirements (provide the source, keep the license
notice, etc.). If you only redistribute `cnnp.lib` / `cnnp.exe` /
`cnnp.pyd`, MIT applies and you have no GPL obligations.

## Why this isolation matters

The whole point of CNNP V2 is "zero external dependencies in the
reader" (spec §12). Vendoring GPL code into the core library would:

1. Violate the zero-deps promise (binpack reader is huge).
2. Force every downstream consumer of `cnnp.lib` to comply with GPL.

Keeping the GPL territory in `tools/` preserves the MIT API for
training pipelines that just want to read CNNP files — which is the
common case.

## Why nnue-pytorch instead of writing our own parser

The Stockfish binpack format is non-trivial: PackedSfen with Huffman
board encoding plus a delta-encoded streaming layer. Writing a
correct parser from scratch is ~2000 LoC of error-prone bit-level
work. Vendoring nnue-pytorch's reader (~4000 LoC) gets us a
production-tested implementation maintained by the Stockfish team
itself, at the cost of the GPL constraint above.
