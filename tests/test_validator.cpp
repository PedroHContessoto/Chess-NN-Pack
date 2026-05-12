// SPDX-License-Identifier: MIT
//
// Unit tests for cnnp/validator.{hpp,cpp}.
//
// Strategy:
//   * Build a small but valid Header + matching byte buffer (via the
//     shared fixture in test_fixtures.hpp).
//   * Verify validate_full() passes.
//   * Tweak one invariant at a time and verify ValidationError is thrown.

#include "cnnp/validator.hpp"
#include "cnnp/header.hpp"
#include "cnnp/encoding.hpp"
#include "cnnp/byte_reader.hpp"

#include "test_fixtures.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace cnnp;
using cnnp::fixtures::make_valid_file;
using cnnp::fixtures::ValidFile;

// ─── Sanity: the fixture itself is valid ─────────────────────────────────────

TEST(Validator, FixtureIsValid) {
    auto v = make_valid_file();
    EXPECT_NO_THROW(validate_full(v.header, std::span<const std::byte>(v.bytes)));
}

TEST(Validator, FixtureLargerIsValid) {
    // Multi-block fixture (2 blocks, second is partially full)
    auto v = make_valid_file(/*n_positions=*/1500, /*ppp=*/12);
    EXPECT_NO_THROW(validate_full(v.header, std::span<const std::byte>(v.bytes)));
}

// ─── Header consistency: cross-field invariants ──────────────────────────────

TEST(Validator, RejectZeroPositions) {
    auto v = make_valid_file();
    v.header.num_positions = 0;
    EXPECT_THROW(validate_header_consistency(v.header, v.bytes.size()), ValidationError);
}

TEST(Validator, RejectZeroBlocks) {
    auto v = make_valid_file();
    v.header.num_blocks = 0;
    EXPECT_THROW(validate_header_consistency(v.header, v.bytes.size()), ValidationError);
}

TEST(Validator, RejectMismatchedNumBlocks) {
    auto v = make_valid_file();
    v.header.num_blocks += 1;  // claim one more block than computed
    EXPECT_THROW(validate_header_consistency(v.header, v.bytes.size()), ValidationError);
}

TEST(Validator, RejectFixedScaleClipOverflow) {
    auto v = make_valid_file();
    v.header.fixed_scale = 4096.0f;  // 10.0 * 4096 = 40960 > 32767
    EXPECT_THROW(validate_header_consistency(v.header, v.bytes.size()), ValidationError);
}

TEST(Validator, RejectMisalignedEvalOffset) {
    auto v = make_valid_file();
    v.header.eval_offset += 1;  // odd address — i16 misaligned
    EXPECT_THROW(validate_header_consistency(v.header, v.bytes.size()), ValidationError);
}

TEST(Validator, RejectMisalignedAnchorsOffset) {
    auto v = make_valid_file();
    v.header.block_anchors_offset += 1;  // not 8-byte aligned
    EXPECT_THROW(validate_header_consistency(v.header, v.bytes.size()), ValidationError);
}

TEST(Validator, RejectOffsetWithinHeader) {
    auto v = make_valid_file();
    v.header.flags_offset = 100;  // inside header region
    EXPECT_THROW(validate_header_consistency(v.header, v.bytes.size()), ValidationError);
}

TEST(Validator, RejectOffsetBeyondFile) {
    auto v = make_valid_file();
    v.header.metadata_offset = v.bytes.size() + 1000;
    EXPECT_THROW(validate_header_consistency(v.header, v.bytes.size()), ValidationError);
}

TEST(Validator, RejectMetadataExtendingBeyondFile) {
    auto v = make_valid_file();
    v.header.metadata_length = 10'000;  // > available room after metadata_offset
    EXPECT_THROW(validate_header_consistency(v.header, v.bytes.size()), ValidationError);
}

TEST(Validator, RejectArrayOverlap) {
    auto v = make_valid_file();
    // Make eval_offset point INSIDE flags region by setting it equal to flags_offset
    v.header.eval_offset = v.header.flags_offset;  // overlapping arrays
    EXPECT_THROW(validate_header_consistency(v.header, v.bytes.size()), ValidationError);
}

// ─── Per-array validators ────────────────────────────────────────────────────

TEST(Validator, FlagsArrayRejectsReservedBitsSet) {
    std::vector<std::uint8_t> flags = {encode_flags(0, 16),
                                       encode_flags(1, 18),
                                       static_cast<std::uint8_t>(encode_flags(0, 8) | 0x40)};
    EXPECT_THROW(validate_flags_array(std::span<const std::uint8_t>(flags)),
                 ValidationError);
}

TEST(Validator, FlagsArrayRejectsSentinelStoredCount) {
    std::vector<std::uint8_t> flags = {encode_flags(0, 16),
                                       static_cast<std::uint8_t>(31u << 1)};
    EXPECT_THROW(validate_flags_array(std::span<const std::uint8_t>(flags)),
                 ValidationError);
}

TEST(Validator, EvalArrayRejectsOutOfRange) {
    // bound = 10.0 * 3000 = 30000
    std::vector<std::int16_t> evals = {0, 1500, -1500, 30001 /* outside */};
    EXPECT_THROW(validate_eval_array(std::span<const std::int16_t>(evals),
                                     CNNP_FIXED_SCALE, CNNP_STORAGE_TARGET_CLIP),
                 ValidationError);
}

TEST(Validator, EvalArrayAcceptsBoundaryValues) {
    std::vector<std::int16_t> evals = {0, 30000, -30000};
    EXPECT_NO_THROW(validate_eval_array(std::span<const std::int16_t>(evals),
                                        CNNP_FIXED_SCALE, CNNP_STORAGE_TARGET_CLIP));
}

TEST(Validator, WdlArrayRejectsInvalidValues) {
    std::vector<std::int8_t> wdls = {-1, 0, 1, 2 /* invalid */};
    EXPECT_THROW(validate_wdl_array(std::span<const std::int8_t>(wdls)),
                 ValidationError);
}

TEST(Validator, WdlArrayAcceptsAllValid) {
    std::vector<std::int8_t> wdls = {-1, 0, 1, 0, -1, 1};
    EXPECT_NO_THROW(validate_wdl_array(std::span<const std::int8_t>(wdls)));
}

TEST(Validator, WFlatRejectsOutOfRangeId) {
    std::vector<std::uint16_t> w = {0, 100, 768 /* equals max_feature_id; rejected */};
    EXPECT_THROW(validate_w_flat(std::span<const std::uint16_t>(w), 768),
                 ValidationError);
}

// ─── block_prefix validation ─────────────────────────────────────────────────

TEST(Validator, BlockPrefixCorruptionDetected) {
    auto v = make_valid_file(/*n=*/16, /*ppp=*/4);
    // Corrupt prefix value somewhere
    write_u16_le(v.bytes, v.header.block_prefix_offset + 4 * 2, 999);  // huge spike
    EXPECT_THROW(validate_full(v.header, std::span<const std::byte>(v.bytes)),
                 ValidationError);
}

TEST(Validator, AnchorsCorruptionDetected) {
    auto v = make_valid_file(/*n=*/2000, /*ppp=*/8);  // 2 blocks
    // Corrupt anchor[1] (should equal sum of block 0)
    write_u64_le(v.bytes, v.header.block_anchors_offset + 8, 99'999);
    EXPECT_THROW(validate_full(v.header, std::span<const std::byte>(v.bytes)),
                 ValidationError);
}

TEST(Validator, LastBlockPaddingViolationDetected) {
    auto v = make_valid_file(/*n=*/1500, /*ppp=*/4);  // 2 blocks; last has 476 valid pos
    // Tamper with a padding slot in the last block.
    const std::uint64_t last_block_base =
        v.header.block_prefix_offset + static_cast<std::uint64_t>(1) * (v.header.block_size + 1) * 2;
    // valid_in_last = 1500 - 1024 = 476; slot 600 is padding.
    write_u16_le(v.bytes, last_block_base + 600 * 2, 12345);
    EXPECT_THROW(validate_full(v.header, std::span<const std::byte>(v.bytes)),
                 ValidationError);
}

TEST(Validator, GlobalFeatureCountMismatch) {
    auto v = make_valid_file();
    v.header.num_features_total += 1;  // claim more than block prefixes sum to
    EXPECT_THROW(validate_full(v.header, std::span<const std::byte>(v.bytes)),
                 ValidationError);
}

// ─── Checked arithmetic helpers ──────────────────────────────────────────────

TEST(CheckedArithmetic, AddNoOverflow) {
    std::uint64_t r = 0;
    EXPECT_FALSE(checked_add_u64(100, 200, r));
    EXPECT_EQ(r, 300u);
}

TEST(CheckedArithmetic, AddOverflow) {
    std::uint64_t r = 0;
    EXPECT_TRUE(checked_add_u64(std::numeric_limits<std::uint64_t>::max(), 1, r));
}

TEST(CheckedArithmetic, MulNoOverflow) {
    std::uint64_t r = 0;
    EXPECT_FALSE(checked_mul_u64(2'000'000'000ull, 20ull, r));
    EXPECT_EQ(r, 40'000'000'000ull);
}

TEST(CheckedArithmetic, MulOverflow) {
    std::uint64_t r = 0;
    EXPECT_TRUE(checked_mul_u64(std::numeric_limits<std::uint64_t>::max(), 2, r));
}

TEST(CheckedArithmetic, MulZeroIsSafe) {
    std::uint64_t r = 99;
    EXPECT_FALSE(checked_mul_u64(0, std::numeric_limits<std::uint64_t>::max(), r));
    EXPECT_EQ(r, 0u);
}
