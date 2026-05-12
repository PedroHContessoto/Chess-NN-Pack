// SPDX-License-Identifier: MIT
//
// cnnp/cnnp.hpp — umbrella header.
//
// Single include exposing the entire public CNNP API. Use this header
// for friction-free integration; the project also ships individual
// component headers (`cnnp/header.hpp`, `cnnp/reader.hpp`, etc.) so
// downstream code can include only what it needs.
//
// Modules:
//   * byte_reader  — endian-safe LE primitive read/write helpers
//   * header       — Header struct, constants, parse/serialize
//   * encoding     — flags / eval / wdl encoders & decoders
//   * validator    — header consistency + per-array invariants
//   * mmap_region  — RAII memory-mapped file (POSIX + Windows)
//   * reader       — random-access Reader + PositionView
//   * writer       — incremental Writer + write_cnnp_file() one-shot

#pragma once

#include "cnnp/byte_reader.hpp"
#include "cnnp/encoding.hpp"
#include "cnnp/header.hpp"
#include "cnnp/mmap_region.hpp"
#include "cnnp/reader.hpp"
#include "cnnp/validator.hpp"
#include "cnnp/writer.hpp"
