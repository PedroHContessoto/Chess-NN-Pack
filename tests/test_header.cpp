// SPDX-License-Identifier: MIT
//
// Unit tests for cnnp::parse_header / cnnp::serialize_header.
//
// Strategy:
//   1. Roundtrip: build a Header, serialize, parse back, compare fields.
//   2. Validate the on-disk byte layout against the spec at known offsets.
//   3. Exercise rejection paths: bad magic, wrong version, etc.

#include "cnnp/header.hpp"
#include "cnnp/byte_reader.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <vector>

using namespace cnnp;

namespace {

/// Build a Header populated with realistic non-default values to exercise
/// every field. Field values chosen so they're recognizable in hex dumps.
Header make_canonical_header() {
    Header h;  // start from defaults
    h.feature_set         = FeatureSet::HalfP;
    h.num_positions       = 1'234'567'890ULL;
    h.num_features_total  = 24'691'357'800ULL;          // ~num_positions * 20
    h.max_feature_id      = 768;
    h.num_blocks          = static_cast<std::uint32_t>(
        (h.num_positions + h.block_size - 1) / h.block_size);

    // Canonical layout offsets (matches §7 "Array sizes match declared offsets")
    h.flags_offset         = CNNP_HEADER_SIZE;
    h.eval_offset          = h.flags_offset         + h.num_positions * 1;
    h.wdl_offset           = h.eval_offset          + h.num_positions * 2;
    h.block_anchors_offset = h.wdl_offset           + h.num_positions * 1;
    h.block_prefix_offset  = h.block_anchors_offset + h.num_blocks    * 8;
    h.w_flat_offset        = h.block_prefix_offset  + h.num_blocks    * (h.block_size + 1) * 2;
    h.metadata_offset      = h.w_flat_offset        + h.num_features_total * 2;
    h.metadata_length      = 4096;

    return h;
}

/// Compare two Headers field-by-field, with explicit failures on mismatch.
void expect_headers_equal(const Header& a, const Header& b) {
    EXPECT_EQ(a.version, b.version);
    EXPECT_EQ(a.header_size, b.header_size);
    EXPECT_EQ(a.endian, b.endian);
    EXPECT_EQ(a.layout_kind, b.layout_kind);
    EXPECT_EQ(a.feature_set, b.feature_set);
    EXPECT_EQ(a.count_semantics, b.count_semantics);
    EXPECT_EQ(a.eval_encoding, b.eval_encoding);
    EXPECT_EQ(a.mate_encoding, b.mate_encoding);
    EXPECT_EQ(a.flags_encoding, b.flags_encoding);
    EXPECT_EQ(a.shard_id, b.shard_id);
    EXPECT_EQ(a.num_shards, b.num_shards);
    EXPECT_EQ(a.global_position_start, b.global_position_start);
    EXPECT_EQ(a.num_positions, b.num_positions);
    EXPECT_EQ(a.num_features_total, b.num_features_total);
    EXPECT_EQ(a.max_feature_id, b.max_feature_id);
    EXPECT_EQ(a.num_blocks, b.num_blocks);
    EXPECT_EQ(a.block_size, b.block_size);
    EXPECT_EQ(a.max_count, b.max_count);
    EXPECT_EQ(a.count_base, b.count_base);
    EXPECT_FLOAT_EQ(a.fixed_scale, b.fixed_scale);
    EXPECT_FLOAT_EQ(a.normal_eval_clip, b.normal_eval_clip);
    EXPECT_FLOAT_EQ(a.storage_target_clip, b.storage_target_clip);
    EXPECT_EQ(a.has_eval_global, b.has_eval_global);
    EXPECT_EQ(a.has_wdl_global, b.has_wdl_global);
    EXPECT_EQ(a.header_flags, b.header_flags);
    EXPECT_EQ(a.flags_offset, b.flags_offset);
    EXPECT_EQ(a.eval_offset, b.eval_offset);
    EXPECT_EQ(a.wdl_offset, b.wdl_offset);
    EXPECT_EQ(a.block_anchors_offset, b.block_anchors_offset);
    EXPECT_EQ(a.block_prefix_offset, b.block_prefix_offset);
    EXPECT_EQ(a.w_flat_offset, b.w_flat_offset);
    EXPECT_EQ(a.metadata_offset, b.metadata_offset);
    EXPECT_EQ(a.metadata_length, b.metadata_length);
}

}  // anonymous namespace

// ─── Roundtrip tests ──────────────────────────────────────────────────────────

TEST(Header, RoundtripCanonical) {
    const Header original = make_canonical_header();
    std::array<std::byte, CNNP_HEADER_SIZE> buf{};
    serialize_header(original, buf);
    const Header parsed = parse_header(buf);
    expect_headers_equal(original, parsed);
}

TEST(Header, RoundtripDefaults) {
    Header h;
    h.num_positions = 1;  // num_positions > 0 invariant
    h.num_blocks    = 1;
    std::array<std::byte, CNNP_HEADER_SIZE> buf{};
    serialize_header(h, buf);
    const Header parsed = parse_header(buf);
    expect_headers_equal(h, parsed);
}

TEST(Header, RoundtripAllFeatureSets) {
    for (FeatureSet fs : {FeatureSet::HalfP, FeatureSet::HalfKAv2_hm, FeatureSet::HalfKA}) {
        Header h = make_canonical_header();
        h.feature_set = fs;
        std::array<std::byte, CNNP_HEADER_SIZE> buf{};
        serialize_header(h, buf);
        EXPECT_EQ(parse_header(buf).feature_set, fs);
    }
}

// ─── Wire-format checks (spec §4) ─────────────────────────────────────────────

TEST(Header, WireMagicAtOffsetZero) {
    const Header h = make_canonical_header();
    std::array<std::byte, CNNP_HEADER_SIZE> buf{};
    serialize_header(h, buf);

    EXPECT_EQ(buf[0], std::byte{'C'});
    EXPECT_EQ(buf[1], std::byte{'N'});
    EXPECT_EQ(buf[2], std::byte{'N'});
    EXPECT_EQ(buf[3], std::byte{'2'});
}

TEST(Header, WireVersionAndHeaderSize) {
    const Header h = make_canonical_header();
    std::array<std::byte, CNNP_HEADER_SIZE> buf{};
    serialize_header(h, buf);

    EXPECT_EQ(read_u16_le(buf, 4), CNNP_VERSION);
    EXPECT_EQ(read_u16_le(buf, 6), CNNP_HEADER_SIZE);
}

TEST(Header, WireKnownConstantsAtExpectedOffsets) {
    Header h = make_canonical_header();
    std::array<std::byte, CNNP_HEADER_SIZE> buf{};
    serialize_header(h, buf);

    // block_size at offset 56 (u16)
    EXPECT_EQ(read_u16_le(buf, 56), CNNP_BLOCK_SIZE);
    // max_count at offset 58 (u8)
    EXPECT_EQ(read_u8(buf, 58), CNNP_MAX_COUNT);
    // count_base at offset 59 (u8)
    EXPECT_EQ(read_u8(buf, 59), CNNP_COUNT_BASE);
    // fixed_scale at offset 60 (f32)
    EXPECT_FLOAT_EQ(read_f32_le(buf, 60), CNNP_FIXED_SCALE);
}

TEST(Header, WireReservedBytesAreZero) {
    const Header h = make_canonical_header();
    std::array<std::byte, CNNP_HEADER_SIZE> buf{};
    serialize_header(h, buf);

    // Reserved byte at offset 15
    EXPECT_EQ(buf[15], std::byte{0});
    // Reserved 4 bytes at offset 76..79
    for (std::size_t i = 76; i < 80; ++i) EXPECT_EQ(buf[i], std::byte{0});
    // Reserved padding at offset 144..255
    for (std::size_t i = 144; i < 256; ++i) EXPECT_EQ(buf[i], std::byte{0});
}

// ─── Rejection tests ──────────────────────────────────────────────────────────

TEST(Header, RejectShortBuffer) {
    std::array<std::byte, 100> buf{};
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectBadMagic) {
    Header h = make_canonical_header();
    std::vector<std::byte> buf(CNNP_HEADER_SIZE);
    serialize_header(h, buf);
    buf[0] = std::byte{'X'};
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectWrongVersion) {
    Header h = make_canonical_header();
    std::vector<std::byte> buf(CNNP_HEADER_SIZE);
    serialize_header(h, buf);
    write_u16_le(buf, 4, 99);
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectWrongHeaderSize) {
    Header h = make_canonical_header();
    std::vector<std::byte> buf(CNNP_HEADER_SIZE);
    serialize_header(h, buf);
    write_u16_le(buf, 6, 128);
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectBigEndian) {
    Header h = make_canonical_header();
    std::vector<std::byte> buf(CNNP_HEADER_SIZE);
    serialize_header(h, buf);
    write_u8(buf, 8, 1);  // BE marker — V2 LE only
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectInvalidLayoutKind) {
    Header h = make_canonical_header();
    std::vector<std::byte> buf(CNNP_HEADER_SIZE);
    serialize_header(h, buf);
    write_u8(buf, 9, 99);
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectInvalidFeatureSet) {
    Header h = make_canonical_header();
    std::vector<std::byte> buf(CNNP_HEADER_SIZE);
    serialize_header(h, buf);
    write_u8(buf, 10, 0);  // 0 not in {1, 2, 3}
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectShardIdNonZeroV2) {
    Header h = make_canonical_header();
    std::vector<std::byte> buf(CNNP_HEADER_SIZE);
    serialize_header(h, buf);
    write_u32_le(buf, 16, 5);  // shard_id != 0
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectMultipleShardsV2) {
    Header h = make_canonical_header();
    std::vector<std::byte> buf(CNNP_HEADER_SIZE);
    serialize_header(h, buf);
    write_u32_le(buf, 20, 2);  // num_shards != 1
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectHasEvalGlobalZero) {
    Header h = make_canonical_header();
    std::vector<std::byte> buf(CNNP_HEADER_SIZE);
    serialize_header(h, buf);
    write_u8(buf, 72, 0);  // V2 requires == 1
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectHasWdlGlobalZero) {
    Header h = make_canonical_header();
    std::vector<std::byte> buf(CNNP_HEADER_SIZE);
    serialize_header(h, buf);
    write_u8(buf, 73, 0);  // V2 requires == 1
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectNonZeroHeaderFlags) {
    Header h = make_canonical_header();
    std::vector<std::byte> buf(CNNP_HEADER_SIZE);
    serialize_header(h, buf);
    write_u16_le(buf, 74, 0x0001);  // V2 reserves header_flags entirely
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectNonZeroReservedByte15) {
    Header h = make_canonical_header();
    std::vector<std::byte> buf(CNNP_HEADER_SIZE);
    serialize_header(h, buf);
    buf[15] = std::byte{0xFF};
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectNonZeroReservedPadding) {
    Header h = make_canonical_header();
    std::vector<std::byte> buf(CNNP_HEADER_SIZE);
    serialize_header(h, buf);
    buf[200] = std::byte{0x42};  // somewhere in the 144..255 reserved padding
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectWrongBlockSize) {
    Header h = make_canonical_header();
    std::vector<std::byte> buf(CNNP_HEADER_SIZE);
    serialize_header(h, buf);
    write_u16_le(buf, 56, 512);
    EXPECT_THROW((void)parse_header(buf), ParseError);
}

TEST(Header, RejectShortOutputBufferOnSerialize) {
    const Header h = make_canonical_header();
    std::array<std::byte, 100> buf{};
    EXPECT_THROW(serialize_header(h, buf), ParseError);
}
