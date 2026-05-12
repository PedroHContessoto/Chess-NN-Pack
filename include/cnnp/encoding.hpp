// SPDX-License-Identifier: MIT
//
// cnnp/encoding.hpp — encode/decode of per-position fields.
//
// Covers the three per-position scalar arrays from spec §5:
//   * `flags_u8`      (stm + count_minus_2)            — §5.1
//   * `eval_i16`      (normalized fixed-point cp)       — §5.2
//   * `wdl_i8`        (game outcome, white-POV)         — §5.3
//
// The HalfP feature index encoding (§5.4) lives elsewhere because it
// depends on chess types from the binpack input, not on cnnp itself.

#pragma once

#include "cnnp/header.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace cnnp {

// ─── Errors ───────────────────────────────────────────────────────────────────

class EncodeError : public std::runtime_error {
public:
    explicit EncodeError(const std::string& what) : std::runtime_error(what) {}
};

// ─── flags_u8 ─────────────────────────────────────────────────────────────────
//
// Bit layout:
//   bit 0       stm                          0=white, 1=black
//   bits 1-5    count_minus_2                0..30 (sentinel 31 reserved)
//   bits 6-7    reserved                     must be 0

struct DecodedFlags {
    std::uint8_t stm;          // 0 or 1
    std::uint8_t piece_count;  // 2..32
};

/// Pack `stm` + `piece_count` into a single byte.
/// Throws `EncodeError` if inputs are out of range:
///   stm > 1
///   piece_count < 2 or piece_count > 32
[[nodiscard]] std::uint8_t encode_flags(std::uint8_t stm, std::uint8_t piece_count);

/// Unpack `flags` back into `{stm, piece_count}`.
/// Throws `EncodeError` if the encoded value is malformed:
///   reserved bits 6-7 are non-zero
///   stored count == 31 (sentinel reserved)
[[nodiscard]] DecodedFlags decode_flags(std::uint8_t flags);

// ─── eval_i16 ─────────────────────────────────────────────────────────────────
//
// Pipeline:  cp ──normalize_target──▶ target [-clip, +clip]
//                ──encode_eval──▶  i16 (target * fixed_scale, clamped)
//
// Decoder:  i16 / fixed_scale = target_normalized

/// Convert a centipawn score into the normalized target space the
/// model trains on. Mate-range scores (|cp| > 25_000) are encoded
/// outside the normal_eval_clip range so they remain distinguishable
/// from saturated normal evals.
///
/// Output range: approximately [-9.9, -7.0] ∪ [-6.5, +6.5] ∪ [+7.0, +9.9].
[[nodiscard]] float normalize_target(std::int32_t score_cp) noexcept;

/// Encode a normalized target into i16 storage. The value is clamped
/// to [-storage_target_clip, +storage_target_clip] before scaling and
/// rounded to nearest.
///
/// Throws `EncodeError` on:
///   - non-finite `target_normalized` (NaN, ±inf)
///   - non-finite or non-positive `fixed_scale`
///   - non-finite or non-positive `storage_target_clip`
///   - `storage_target_clip * fixed_scale > 32767` (would overflow i16)
///
/// The previous "safety clamp" behavior was removed: silent saturation
/// on inconsistent parameters is unsafe for training data — preferring
/// loud failure makes upstream bugs visible.
[[nodiscard]] std::int16_t encode_eval(
    float target_normalized,
    float fixed_scale         = CNNP_FIXED_SCALE,
    float storage_target_clip = CNNP_STORAGE_TARGET_CLIP
);

/// Decode an i16 back into the normalized target space.
/// Throws `EncodeError` if `fixed_scale` is non-finite or non-positive.
[[nodiscard]] float decode_eval(
    std::int16_t eval_i16,
    float fixed_scale = CNNP_FIXED_SCALE
);

/// Convenience: cp → i16 in one call (normalize + encode).
/// Propagates any `EncodeError` from `encode_eval`.
[[nodiscard]] std::int16_t encode_eval_from_cp(
    std::int32_t score_cp,
    float fixed_scale         = CNNP_FIXED_SCALE,
    float storage_target_clip = CNNP_STORAGE_TARGET_CLIP
);

// ─── wdl_i8 ───────────────────────────────────────────────────────────────────
//
// Game outcome from white's perspective:
//   -1  black won
//    0  draw
//   +1  white won

/// Validate and pack a WDL outcome into i8.
/// Throws `EncodeError` if `wdl` is not in {-1, 0, +1}.
[[nodiscard]] std::int8_t encode_wdl(int wdl);

/// Validate-on-read: ensures the stored value is one of the three legal
/// outcomes. Throws `EncodeError` otherwise.
/// Returns the value unchanged on success.
[[nodiscard]] std::int8_t decode_wdl(std::int8_t wdl_i8);

}  // namespace cnnp
