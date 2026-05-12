// SPDX-License-Identifier: MIT
//
// Unit tests for cnnp/encoding.{hpp,cpp}.
//
// Coverage:
//   * encode_flags / decode_flags  — validation, roundtrip, sentinel rejection
//   * normalize_target             — zero, normal, mate, boundary, INT32_MIN
//   * encode_eval / decode_eval    — roundtrip, clamp, i16 range safety
//   * encode_wdl / decode_wdl      — validation, roundtrip

#include "cnnp/encoding.hpp"
#include "cnnp/header.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using namespace cnnp;

// ─── flags_u8 ─────────────────────────────────────────────────────────────────

TEST(EncodeFlags, RoundtripAllValid) {
    for (std::uint8_t stm : {std::uint8_t{0}, std::uint8_t{1}}) {
        for (std::uint8_t pc = 2; pc <= 32; ++pc) {
            const std::uint8_t encoded = encode_flags(stm, pc);
            const DecodedFlags d = decode_flags(encoded);
            EXPECT_EQ(d.stm, stm)
                << "stm=" << +stm << " pc=" << +pc;
            EXPECT_EQ(d.piece_count, pc)
                << "stm=" << +stm << " pc=" << +pc;
        }
    }
}

TEST(EncodeFlags, RejectStmTooLarge) {
    EXPECT_THROW((void)encode_flags(2, 16), EncodeError);    // stm > 1
    EXPECT_THROW((void)encode_flags(255, 16), EncodeError);
}

TEST(EncodeFlags, RejectPieceCountOutOfRange) {
    EXPECT_THROW((void)encode_flags(0, 0), EncodeError);     // pc < 2
    EXPECT_THROW((void)encode_flags(0, 1), EncodeError);     // pc < 2
    EXPECT_THROW((void)encode_flags(0, 33), EncodeError);    // pc > 32
    EXPECT_THROW((void)encode_flags(0, 200), EncodeError);   // pc > 32
}

TEST(DecodeFlags, RejectReservedBitsSet) {
    // Build a "valid" encoding then twiddle the reserved bits.
    const std::uint8_t valid = encode_flags(1, 16);
    EXPECT_NO_THROW((void)decode_flags(valid));

    EXPECT_THROW((void)decode_flags(static_cast<std::uint8_t>(valid | 0x40)), EncodeError);  // bit 6
    EXPECT_THROW((void)decode_flags(static_cast<std::uint8_t>(valid | 0x80)), EncodeError);  // bit 7
    EXPECT_THROW((void)decode_flags(static_cast<std::uint8_t>(valid | 0xC0)), EncodeError);  // both
}

TEST(DecodeFlags, RejectSentinelStoredCount) {
    // stored_count = 31 is the sentinel; raw byte = (31 << 1) | stm
    const std::uint8_t sentinel_white = static_cast<std::uint8_t>(31u << 1);
    const std::uint8_t sentinel_black = static_cast<std::uint8_t>((31u << 1) | 1u);
    EXPECT_THROW((void)decode_flags(sentinel_white), EncodeError);
    EXPECT_THROW((void)decode_flags(sentinel_black), EncodeError);
}

// ─── normalize_target ─────────────────────────────────────────────────────────

TEST(NormalizeTarget, Zero) {
    EXPECT_FLOAT_EQ(normalize_target(0), 0.0f);
}

TEST(NormalizeTarget, NormalRange) {
    // 200 cp → 200 / 400 = 0.5
    EXPECT_FLOAT_EQ(normalize_target(200),  0.5f);
    EXPECT_FLOAT_EQ(normalize_target(-200), -0.5f);

    // 100 cp → 0.25
    EXPECT_FLOAT_EQ(normalize_target(100),  0.25f);
    EXPECT_FLOAT_EQ(normalize_target(-100), -0.25f);
}

TEST(NormalizeTarget, ClipsAtNormalBoundary) {
    // ≥ 2600 cp clipped to 6.5 (the normal_eval_clip)
    EXPECT_FLOAT_EQ(normalize_target(2600),  6.5f);
    EXPECT_FLOAT_EQ(normalize_target(2700),  6.5f);
    EXPECT_FLOAT_EQ(normalize_target(25000), 6.5f);
    EXPECT_FLOAT_EQ(normalize_target(-2600), -6.5f);
    EXPECT_FLOAT_EQ(normalize_target(-25000), -6.5f);
}

TEST(NormalizeTarget, MateRangeUpperBound) {
    // 31_900 → ply_estimate = (32000-31900)/100 = 1.0 → 10.0 - 0.1 = 9.9
    EXPECT_NEAR(normalize_target(31'900),  9.9f, 1e-4f);
    EXPECT_NEAR(normalize_target(-31'900), -9.9f, 1e-4f);
}

TEST(NormalizeTarget, MateRangeLowerBound) {
    // 29_000 → ply_estimate clamped to 30 → 10.0 - 3.0 = 7.0
    EXPECT_NEAR(normalize_target(29'000),  7.0f, 1e-4f);
    EXPECT_NEAR(normalize_target(-29'000), -7.0f, 1e-4f);
}

TEST(NormalizeTarget, JustAboveThresholdEntersMateBranch) {
    // 25_001 enters mate branch; produces ≈ 7.0 (clamped at PLY_MAX=30)
    const float v = normalize_target(25'001);
    EXPECT_GT(v, 6.5f);   // strictly above normal_eval_clip
    EXPECT_LE(v, 9.9f + 1e-4f);
}

TEST(NormalizeTarget, IntMinDoesNotUB) {
    // INT32_MIN passed to abs() would be UB on int32; spec mandates int64 path.
    // Just verify it doesn't crash and returns a sensible (saturated mate) value.
    const float v = normalize_target(std::numeric_limits<std::int32_t>::min());
    EXPECT_LT(v, -6.5f);
    EXPECT_GE(v, -10.0f);
}

// ─── encode_eval / decode_eval ────────────────────────────────────────────────

TEST(EncodeEval, ZeroRoundtrip) {
    EXPECT_EQ(encode_eval(0.0f), 0);
    EXPECT_FLOAT_EQ(decode_eval(0), 0.0f);
}

TEST(EncodeEval, NormalValueRoundtrip) {
    // 0.5 * 3000 = 1500
    const std::int16_t i = encode_eval(0.5f);
    EXPECT_EQ(i, 1500);
    EXPECT_NEAR(decode_eval(i), 0.5f, 1e-3f);
}

TEST(EncodeEval, MateValueFitsAndRoundtrips) {
    // 9.9 * 3000 = 29700, fits in i16
    const std::int16_t i = encode_eval(9.9f);
    EXPECT_EQ(i, 29700);
    EXPECT_NEAR(decode_eval(i), 9.9f, 1e-3f);

    // Maximum allowed: 10.0 * 3000 = 30000
    const std::int16_t imax = encode_eval(10.0f);
    EXPECT_EQ(imax, 30000);
}

TEST(EncodeEval, ClampsBeyondStorageTargetClip) {
    // Passing 12.0 with default clip 10.0 should clamp to 10.0 → 30000
    EXPECT_EQ(encode_eval( 12.0f), 30000);
    EXPECT_EQ(encode_eval(-12.0f), -30000);
}

TEST(EncodeEval, RejectsPathologicalParams) {
    // Replaces the previous "silent saturation" behavior. Per spec
    // §7, a header with `storage_target_clip * fixed_scale > 32767`
    // is invalid; the encoder mirrors that contract by throwing instead
    // of silently saturating training data.
    EXPECT_THROW((void)encode_eval(1.0f, /*fixed_scale=*/100'000.0f,
                                   /*storage_target_clip=*/100.0f),
                 cnnp::EncodeError);
}

TEST(EncodeEval, RejectsNanTarget) {
    EXPECT_THROW((void)encode_eval(std::numeric_limits<float>::quiet_NaN()),
                 cnnp::EncodeError);
}

TEST(EncodeEval, RejectsInfTarget) {
    EXPECT_THROW((void)encode_eval(std::numeric_limits<float>::infinity()),
                 cnnp::EncodeError);
}

TEST(EncodeEval, RejectsZeroFixedScale) {
    EXPECT_THROW((void)encode_eval(1.0f, /*fixed_scale=*/0.0f),
                 cnnp::EncodeError);
}

TEST(DecodeEval, RejectsZeroFixedScale) {
    EXPECT_THROW((void)decode_eval(100, /*fixed_scale=*/0.0f),
                 cnnp::EncodeError);
}

TEST(DecodeEval, RejectsNanFixedScale) {
    EXPECT_THROW((void)decode_eval(100,
                                   std::numeric_limits<float>::quiet_NaN()),
                 cnnp::EncodeError);
}

TEST(EncodeEvalFromCp, FullPipelineWorks) {
    // 200 cp → 0.5 → 1500
    EXPECT_EQ(encode_eval_from_cp(200), 1500);
    // Mate: 31900 cp → 9.9 → 29700
    EXPECT_EQ(encode_eval_from_cp(31'900), 29700);
    // Saturated normal: 5000 cp → 6.5 → 19500
    EXPECT_EQ(encode_eval_from_cp(5000), 19500);
}

TEST(EncodeEvalFromCp, NegativeRoundtrip) {
    EXPECT_EQ(encode_eval_from_cp(-200), -1500);
    EXPECT_EQ(encode_eval_from_cp(-31'900), -29700);
}

TEST(EncodeEval, ResolutionApproximatesOneOverFixedScale) {
    // Two values 0.001 apart should map to consecutive i16 values
    const std::int16_t a = encode_eval(0.0f);
    const std::int16_t b = encode_eval(1.0f / 3000.0f);
    EXPECT_EQ(b - a, 1);
}

// ─── wdl_i8 ───────────────────────────────────────────────────────────────────

TEST(EncodeWdl, RoundtripAllValid) {
    for (int wdl : {-1, 0, 1}) {
        const std::int8_t e = encode_wdl(wdl);
        EXPECT_EQ(static_cast<int>(e), wdl);
        EXPECT_EQ(static_cast<int>(decode_wdl(e)), wdl);
    }
}

TEST(EncodeWdl, RejectInvalidValues) {
    EXPECT_THROW((void)encode_wdl(2),    EncodeError);
    EXPECT_THROW((void)encode_wdl(-2),   EncodeError);
    EXPECT_THROW((void)encode_wdl(100),  EncodeError);
    EXPECT_THROW((void)encode_wdl(-100), EncodeError);
}

TEST(DecodeWdl, RejectInvalidStoredValues) {
    EXPECT_THROW((void)decode_wdl(static_cast<std::int8_t>(2)),   EncodeError);
    EXPECT_THROW((void)decode_wdl(static_cast<std::int8_t>(-2)),  EncodeError);
    EXPECT_THROW((void)decode_wdl(static_cast<std::int8_t>(127)), EncodeError);
}
