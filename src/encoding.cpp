// SPDX-License-Identifier: MIT
//
// cnnp/src/encoding.cpp — implementation of per-position field codecs.

#include "cnnp/encoding.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>

namespace cnnp {

// ─── flags_u8 ─────────────────────────────────────────────────────────────────

namespace {

constexpr std::uint8_t FLAGS_STM_MASK    = 0x01;     // bit 0
constexpr std::uint8_t FLAGS_COUNT_MASK  = 0x1F;     // bits 1-5 (after >> 1)
constexpr std::uint8_t FLAGS_RESERVED    = 0xC0;     // bits 6-7
constexpr std::uint8_t FLAGS_COUNT_SENTINEL = 31;    // stored value reserved

}  // anonymous namespace

std::uint8_t encode_flags(std::uint8_t stm, std::uint8_t piece_count) {
    if (stm > 1) {
        std::ostringstream os;
        os << "encode_flags: invalid stm " << static_cast<unsigned>(stm)
           << " (must be 0 or 1)";
        throw EncodeError(os.str());
    }
    if (piece_count < 2 || piece_count > 32) {
        std::ostringstream os;
        os << "encode_flags: invalid piece_count "
           << static_cast<unsigned>(piece_count) << " (must be in [2, 32])";
        throw EncodeError(os.str());
    }

    const std::uint8_t stored_count = static_cast<std::uint8_t>(piece_count - 2);
    // stored_count is in [0, 30] here — the sentinel 31 is reachable only via
    // corrupt input, never via a valid encode call.
    return static_cast<std::uint8_t>((stored_count << 1) | (stm & FLAGS_STM_MASK));
}

DecodedFlags decode_flags(std::uint8_t flags) {
    if ((flags & FLAGS_RESERVED) != 0) {
        std::ostringstream os;
        os << "decode_flags: reserved bits 6-7 must be zero (got 0x"
           << std::hex << static_cast<unsigned>(flags) << ")";
        throw EncodeError(os.str());
    }
    const std::uint8_t stored_count = static_cast<std::uint8_t>((flags >> 1) & FLAGS_COUNT_MASK);
    if (stored_count == FLAGS_COUNT_SENTINEL) {
        throw EncodeError("decode_flags: stored count == 31 (sentinel reserved)");
    }
    return DecodedFlags{
        static_cast<std::uint8_t>(flags & FLAGS_STM_MASK),
        static_cast<std::uint8_t>(stored_count + 2)
    };
}

// ─── eval_i16 ─────────────────────────────────────────────────────────────────

namespace {

constexpr std::int64_t MATE_THRESHOLD_CP = 25'000;
constexpr float        MATE_BASE         = 10.0f;
constexpr float        MATE_DECAY        = 0.1f;
constexpr float        NORMAL_EVAL_CLIP  = 6.5f;
constexpr float        NORMAL_EVAL_SCALE = 400.0f;  // cp / 400 → normalized
constexpr float        PLY_DECAY_NUM     = 32'000.0f;
constexpr float        PLY_DECAY_DEN     = 100.0f;
constexpr float        PLY_MIN           = 1.0f;
constexpr float        PLY_MAX           = 30.0f;

}  // anonymous namespace

float normalize_target(std::int32_t score_cp) noexcept {
    // Promote to int64 to avoid UB on std::abs(INT32_MIN) (per spec §5.2).
    const std::int64_t cp     = static_cast<std::int64_t>(score_cp);
    const std::int64_t abs_cp = (cp < 0) ? -cp : cp;
    const float        sign   = (cp >= 0) ? 1.0f : -1.0f;

    if (abs_cp > MATE_THRESHOLD_CP) {
        // Mate encoding: above the normal_eval_clip range, distinguishable.
        // Range produced ≈ [7.0, 9.9].
        const float ply_estimate = std::clamp(
            (PLY_DECAY_NUM - static_cast<float>(abs_cp)) / PLY_DECAY_DEN,
            PLY_MIN, PLY_MAX);
        return sign * (MATE_BASE - ply_estimate * MATE_DECAY);
    }

    return sign * std::min(static_cast<float>(abs_cp) / NORMAL_EVAL_SCALE,
                           NORMAL_EVAL_CLIP);
}

std::int16_t encode_eval(float target_normalized,
                         float fixed_scale,
                         float storage_target_clip) noexcept {
    // Clamp to spec-provided bounds first, then scale.
    const float clamped = std::clamp(target_normalized,
                                     -storage_target_clip, storage_target_clip);
    const long  rounded = std::lroundf(clamped * fixed_scale);

    // Final safety clamp to i16 range. With well-formed parameters
    // (storage_target_clip * fixed_scale ≤ 32767), this is a no-op,
    // but defends against pathological caller input.
    constexpr long I16_MIN = -32768L;
    constexpr long I16_MAX =  32767L;
    const long final_v = std::clamp(rounded, I16_MIN, I16_MAX);
    return static_cast<std::int16_t>(final_v);
}

float decode_eval(std::int16_t eval_i16, float fixed_scale) noexcept {
    return static_cast<float>(eval_i16) / fixed_scale;
}

std::int16_t encode_eval_from_cp(std::int32_t score_cp,
                                 float fixed_scale,
                                 float storage_target_clip) noexcept {
    return encode_eval(normalize_target(score_cp), fixed_scale, storage_target_clip);
}

// ─── wdl_i8 ───────────────────────────────────────────────────────────────────

std::int8_t encode_wdl(int wdl) {
    if (wdl != -1 && wdl != 0 && wdl != 1) {
        std::ostringstream os;
        os << "encode_wdl: value " << wdl << " not in {-1, 0, +1}";
        throw EncodeError(os.str());
    }
    return static_cast<std::int8_t>(wdl);
}

std::int8_t decode_wdl(std::int8_t wdl_i8) {
    if (wdl_i8 != -1 && wdl_i8 != 0 && wdl_i8 != 1) {
        std::ostringstream os;
        os << "decode_wdl: value " << static_cast<int>(wdl_i8) << " not in {-1, 0, +1}";
        throw EncodeError(os.str());
    }
    return wdl_i8;
}

}  // namespace cnnp
