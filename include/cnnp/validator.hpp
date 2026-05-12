// SPDX-License-Identifier: MIT
//
// cnnp/validator.hpp — full spec §7 validator.
//
// Layered validation:
//
//   parse_header()                  → wire-level field validity (§4)
//                                     (already enforced; cheap, mandatory)
//
//   validate_header_consistency()   → cross-field invariants (§7 Header
//                                     + Consistency + Alignment + canonical
//                                     array sizes). No data scan needed.
//
//   validate_*_array()              → per-array invariants (§7 Per-position
//                                     and Per-block prefix). Caller provides
//                                     typed spans.
//
//   validate_full()                 → orchestrates everything from a Header
//                                     and a raw byte view of the file.
//
// Every check that fails throws `ValidationError` with a human-readable
// message indicating which invariant was violated.

#pragma once

#include "cnnp/header.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace cnnp {

// ─── Errors ───────────────────────────────────────────────────────────────────

class ValidationError : public std::runtime_error {
public:
    explicit ValidationError(const std::string& what) : std::runtime_error(what) {}
};

// ─── Checked u64 arithmetic (utility) ─────────────────────────────────────────
//
// Spec §7 mandates that all size computations use checked u64 arithmetic
// to detect overflow. These helpers return `true` on overflow and write
// the (possibly garbage) result via reference.

[[nodiscard]] bool checked_add_u64(std::uint64_t a, std::uint64_t b,
                                   std::uint64_t& out) noexcept;

[[nodiscard]] bool checked_mul_u64(std::uint64_t a, std::uint64_t b,
                                   std::uint64_t& out) noexcept;

// ─── Header-only validation ───────────────────────────────────────────────────
//
// Validates everything that can be checked from the Header alone given
// the total file size. Catches the vast majority of malformed-header
// scenarios without scanning data arrays.
//
// Checks (§7):
//   * num_positions > 0, num_blocks > 0
//   * num_blocks == ceil(num_positions / block_size)
//   * num_blocks fits in u32 (UINT32_MAX bound)
//   * storage_target_clip * fixed_scale ≤ 32767
//   * block_size * max_count ≤ 65535 (already enforced by parser, re-checked)
//   * 64-bit address space (num_features_total ≤ SIZE_MAX)
//   * all array offsets within [header_size, file_size)
//   * metadata_offset + metadata_length ≤ file_size
//   * canonical layout array-size invariants (each array fits before next)
//   * alignment of typed array offsets (eval/anchors/prefix/w_flat)
//
// Throws ValidationError on the first failure with an informative message.
void validate_header_consistency(const Header& h, std::uint64_t file_size);

// ─── Per-array validators ────────────────────────────────────────────────────
//
// Each takes a typed span (caller is responsible for alignment when
// converting from raw bytes; use `validate_full` to get this for free).

void validate_flags_array(std::span<const std::uint8_t> flags);

void validate_eval_array(std::span<const std::int16_t> evals,
                         float fixed_scale,
                         float storage_target_clip);

void validate_wdl_array(std::span<const std::int8_t> wdls);

/// Validates anchor/prefix consistency, monotonicity, padding rule for
/// the last block, and that the running total matches `num_features_total`.
void validate_block_prefix(std::span<const std::uint64_t> anchors,
                           std::span<const std::uint16_t> prefix,
                           const Header& h);

void validate_w_flat(std::span<const std::uint16_t> w_flat,
                     std::uint32_t max_feature_id);

// ─── Full validation ──────────────────────────────────────────────────────────
//
// Convenience: runs `validate_header_consistency` then derives typed
// spans from the header offsets and runs all per-array validators.
// `file_bytes.size()` must equal the actual file size (this is what
// validate_header_consistency checks against).
//
// The byte view's data pointer must be at least 8-byte aligned (mmap
// returns page-aligned addresses, so this is naturally satisfied).
void validate_full(const Header& h, std::span<const std::byte> file_bytes);

}  // namespace cnnp
