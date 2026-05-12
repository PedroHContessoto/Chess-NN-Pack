// SPDX-License-Identifier: MIT
//
// Phase-4 "golden file" integration test (spec roadmap §13 steps 9-10).
//
// Hand-build the EXPECTED byte representation of a tiny 4-position
// CNNP V2 file, then verify three things end-to-end:
//
//   (A) Writer produces those exact bytes at the canonical offsets.
//   (B) parse_header() recovers the canonical layout, and
//       validate_full() accepts the file.
//   (C) Reader round-trips every per-position field back to the
//       original input semantics.
//
// This is the safety net that catches drift between spec, encoding
// helpers, layout math, validator, and Reader/Writer that single-
// component unit tests can miss.
//
// ─── Hand-derived dataset ─────────────────────────────────────────────────────
//
// 4 positions × 4 features each, default block_size = 1024 → 1 block.
//
//                       cp     wdl   stm   features
//   pos 0:              100      1     0   { 1,  2,  3,  4}
//   pos 1:              -50      0     1   { 5,  6,  7,  8}
//   pos 2:              300     -1     0   { 9, 10, 11, 12}
//   pos 3:                0      1     1   {13, 14, 15, 16}
//
// Canonical layout (round_up offsets):
//   header              :    0..256
//   flags  (4 bytes)    :  256..260
//   eval   (8 bytes)    :  260..268    (round_up 260,2 = 260)
//   wdl    (4 bytes)    :  268..272
//   anchors (8 bytes)   :  272..280    (round_up 272,8 = 272)
//   prefix (2050 bytes) :  280..2330   (1*1025*2)
//   w_flat (32 bytes)   : 2330..2362   (16 features * 2)
//   metadata (empty)    : 2362..2362
//
// flags  (stm | (pc-2)<<1):
//   0x04, 0x05, 0x04, 0x05
//
// eval (i16 LE; cp/400 * 3000):
//   cp=100 → 750  = 0x02EE → LE 0xEE 0x02
//   cp=-50 → -375 = 0xFE89 → LE 0x89 0xFE
//   cp=300 → 2250 = 0x08CA → LE 0xCA 0x08
//   cp=0   → 0    = 0x0000 → LE 0x00 0x00
//
// wdl (i8): 0x01, 0x00, 0xFF, 0x01
//
// anchors[0]  = 0   (single block)
// prefix[0..4] = {0, 4, 8, 12, 16}; prefix[5..1024] = 16 (padding).
// w_flat = {1, 2, 3, 4, ..., 16}.

#include "cnnp/encoding.hpp"
#include "cnnp/header.hpp"
#include "cnnp/reader.hpp"
#include "cnnp/validator.hpp"
#include "cnnp/writer.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

using cnnp::CNNP_FIXED_SCALE;
using cnnp::CNNP_HEADER_SIZE;
using cnnp::CNNP_VERSION;
using cnnp::decode_eval;
using cnnp::encode_eval_from_cp;
using cnnp::Header;
using cnnp::parse_header;
using cnnp::Reader;
using cnnp::validate_full;
using cnnp::Writer;

namespace {

class TempPath {
public:
    explicit TempPath(const char* tag) {
        static std::atomic<std::uint64_t> counter{0};
        const auto id = counter.fetch_add(1, std::memory_order_relaxed);
        m_path = std::filesystem::temp_directory_path() /
                 (std::string("cnnp_golden_") + tag + "_" +
                  std::to_string(id) + ".cnnp");
    }
    ~TempPath() {
        std::error_code ec;
        std::filesystem::remove(m_path, ec);
    }
    TempPath(const TempPath&)            = delete;
    TempPath& operator=(const TempPath&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return m_path; }

private:
    std::filesystem::path m_path;
};

std::vector<std::byte> read_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    EXPECT_TRUE(f.good());
    const auto sz = f.tellg();
    f.seekg(0);
    std::vector<std::byte> buf(static_cast<std::size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(buf.size()));
    return buf;
}

unsigned char u8at(const std::vector<std::byte>& bytes, std::uint64_t off) {
    return static_cast<unsigned char>(bytes[static_cast<std::size_t>(off)]);
}

// Canonical offsets for the golden dataset (see comment block above).
constexpr std::uint64_t G_FLAGS_OFF   = 256;
constexpr std::uint64_t G_EVAL_OFF    = 260;
constexpr std::uint64_t G_WDL_OFF     = 268;
constexpr std::uint64_t G_ANCHORS_OFF = 272;
constexpr std::uint64_t G_PREFIX_OFF  = 280;
constexpr std::uint64_t G_WFLAT_OFF   = 2330;
constexpr std::uint64_t G_FILE_SIZE   = 2362;

// Hand-built dataset (shared by all golden tests).
struct GoldenInput {
    std::vector<std::vector<std::uint16_t>> features;
    std::vector<std::int32_t> cps;
    std::vector<std::int32_t> wdls;
    std::vector<std::uint8_t> stms;
};

GoldenInput make_input() {
    return GoldenInput{
        /* features */ {{1, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12},
                        {13, 14, 15, 16}},
        /* cps      */ {100, -50, 300,  0},
        /* wdls     */ {  1,   0,  -1,  1},
        /* stms     */ {  0,   1,   0,  1},
    };
}

void write_golden_file(const std::filesystem::path& path) {
    const auto in = make_input();
    // Disable the JSON trailer so the file matches the hand-derived
    // 2362-byte layout exactly. Tests that exercise the trailer live
    // in test_writer.cpp.
    cnnp::WriterConfig cfg;
    cfg.metadata_json.clear();
    Writer w(cfg);
    for (std::size_t i = 0; i < in.features.size(); ++i) {
        w.add(in.cps[i], in.wdls[i], in.stms[i],
              static_cast<std::uint8_t>(in.features[i].size()),
              std::span<const std::uint16_t>(in.features[i]));
    }
    w.finalize(path);
}

}  // anonymous namespace

// ─── (A) Writer produces the EXPECTED bytes ──────────────────────────────────

TEST(Golden, WriterProducesExpectedBytes) {
    TempPath tp("bytes");
    write_golden_file(tp.path());

    auto bytes = read_file(tp.path());
    ASSERT_EQ(bytes.size(), G_FILE_SIZE);

    // ── Magic + version
    EXPECT_EQ(u8at(bytes, 0), 'C');
    EXPECT_EQ(u8at(bytes, 1), 'N');
    EXPECT_EQ(u8at(bytes, 2), 'N');
    EXPECT_EQ(u8at(bytes, 3), '2');
    EXPECT_EQ(u8at(bytes, 4), 0x02u);  // version low
    EXPECT_EQ(u8at(bytes, 5), 0x00u);  // version high

    // ── flags (stm | (pc-2)<<1)
    EXPECT_EQ(u8at(bytes, G_FLAGS_OFF + 0), 0x04u);
    EXPECT_EQ(u8at(bytes, G_FLAGS_OFF + 1), 0x05u);
    EXPECT_EQ(u8at(bytes, G_FLAGS_OFF + 2), 0x04u);
    EXPECT_EQ(u8at(bytes, G_FLAGS_OFF + 3), 0x05u);

    // ── eval (i16 LE) — see derivation in file header comment
    EXPECT_EQ(u8at(bytes, G_EVAL_OFF + 0), 0xEEu);
    EXPECT_EQ(u8at(bytes, G_EVAL_OFF + 1), 0x02u);
    EXPECT_EQ(u8at(bytes, G_EVAL_OFF + 2), 0x89u);
    EXPECT_EQ(u8at(bytes, G_EVAL_OFF + 3), 0xFEu);
    EXPECT_EQ(u8at(bytes, G_EVAL_OFF + 4), 0xCAu);
    EXPECT_EQ(u8at(bytes, G_EVAL_OFF + 5), 0x08u);
    EXPECT_EQ(u8at(bytes, G_EVAL_OFF + 6), 0x00u);
    EXPECT_EQ(u8at(bytes, G_EVAL_OFF + 7), 0x00u);

    // ── wdl (i8 reinterpreted as byte)
    EXPECT_EQ(u8at(bytes, G_WDL_OFF + 0), 0x01u);
    EXPECT_EQ(u8at(bytes, G_WDL_OFF + 1), 0x00u);
    EXPECT_EQ(u8at(bytes, G_WDL_OFF + 2), 0xFFu);  // -1
    EXPECT_EQ(u8at(bytes, G_WDL_OFF + 3), 0x01u);

    // ── block_anchors (u64 LE) — single block at offset 0
    for (std::uint64_t k = 0; k < 8; ++k) {
        EXPECT_EQ(u8at(bytes, G_ANCHORS_OFF + k), 0x00u)
            << "anchor byte " << k;
    }

    // ── block_prefix (u16 LE)
    // prefix[0..4] = {0, 4, 8, 12, 16}; prefix[5..1024] = 16 (padding).
    const std::uint16_t expected_prefix[5] = {0, 4, 8, 12, 16};
    for (std::size_t e = 0; e < 5; ++e) {
        EXPECT_EQ(u8at(bytes, G_PREFIX_OFF + e * 2 + 0),
                  static_cast<unsigned char>(expected_prefix[e] & 0xFF))
            << "prefix[" << e << "] lo";
        EXPECT_EQ(u8at(bytes, G_PREFIX_OFF + e * 2 + 1),
                  static_cast<unsigned char>(expected_prefix[e] >> 8))
            << "prefix[" << e << "] hi";
    }
    // padding entries 5..1024 all = 16 (0x10 0x00)
    for (std::uint64_t e = 5; e <= 1024; ++e) {
        EXPECT_EQ(u8at(bytes, G_PREFIX_OFF + e * 2 + 0), 0x10u)
            << "prefix[" << e << "] lo (padding)";
        EXPECT_EQ(u8at(bytes, G_PREFIX_OFF + e * 2 + 1), 0x00u)
            << "prefix[" << e << "] hi (padding)";
    }

    // ── w_flat (u16 LE) — features 1..16
    for (std::uint16_t k = 1; k <= 16; ++k) {
        EXPECT_EQ(u8at(bytes, G_WFLAT_OFF + (k - 1) * 2 + 0),
                  static_cast<unsigned char>(k & 0xFF))
            << "w_flat[" << (k - 1) << "] lo";
        EXPECT_EQ(u8at(bytes, G_WFLAT_OFF + (k - 1) * 2 + 1), 0x00u)
            << "w_flat[" << (k - 1) << "] hi";
    }
}

// ─── (B) parse_header + validate_full accept the canonical file ──────────────

TEST(Golden, ParseHeaderRecoversCanonicalLayout) {
    TempPath tp("parse");
    write_golden_file(tp.path());

    auto bytes = read_file(tp.path());
    Header h = parse_header(std::span<const std::byte>(bytes));

    EXPECT_EQ(h.version,             CNNP_VERSION);
    EXPECT_EQ(h.header_size,         CNNP_HEADER_SIZE);
    EXPECT_EQ(h.num_positions,       4u);
    EXPECT_EQ(h.num_blocks,          1u);
    EXPECT_EQ(h.num_features_total,  16u);

    EXPECT_EQ(h.flags_offset,         G_FLAGS_OFF);
    EXPECT_EQ(h.eval_offset,          G_EVAL_OFF);
    EXPECT_EQ(h.wdl_offset,           G_WDL_OFF);
    EXPECT_EQ(h.block_anchors_offset, G_ANCHORS_OFF);
    EXPECT_EQ(h.block_prefix_offset,  G_PREFIX_OFF);
    EXPECT_EQ(h.w_flat_offset,        G_WFLAT_OFF);
    EXPECT_EQ(h.metadata_offset,      G_FILE_SIZE);
    EXPECT_EQ(h.metadata_length,      0u);

    EXPECT_NO_THROW(validate_full(h, std::span<const std::byte>(bytes)));
}

// ─── (C) End-to-end struct → write → read → validate → compare ───────────────

TEST(Golden, EndToEndStructWriteReadCompare) {
    // Spec roadmap §13 Phase 4 step 10:
    //   "roundtrip test: struct → write .cnnp → read → validate → compare"
    TempPath tp("e2e");
    write_golden_file(tp.path());

    // Reader::open runs validate_full internally, so successful open
    // covers both "read" and "validate".
    auto r = Reader::open(tp.path());
    ASSERT_EQ(r.num_positions(), 4u);
    EXPECT_EQ(r.num_blocks(),    1u);

    const auto in = make_input();
    for (std::size_t i = 0; i < 4; ++i) {
        auto v = r.at(i);
        EXPECT_EQ(v.stm,         in.stms[i])                            << "i=" << i;
        EXPECT_EQ(v.piece_count, 4u)                                     << "i=" << i;
        EXPECT_EQ(v.wdl,         static_cast<std::int8_t>(in.wdls[i]))   << "i=" << i;

        // Eval roundtrip: cp → encode → decode == decode(encode(cp))
        const float expected_eval = decode_eval(
            encode_eval_from_cp(in.cps[i]), CNNP_FIXED_SCALE);
        EXPECT_FLOAT_EQ(v.eval_normalized, expected_eval)                << "i=" << i;

        ASSERT_EQ(v.features.size(), in.features[i].size())              << "i=" << i;
        for (std::size_t k = 0; k < in.features[i].size(); ++k) {
            EXPECT_EQ(v.features[k], in.features[i][k])
                << "i=" << i << " k=" << k;
        }
    }
}

// ─── Bonus: total file size matches hand-derived value ───────────────────────

TEST(Golden, FileSizeMatchesHandDerivedTotal) {
    TempPath tp("size");
    write_golden_file(tp.path());
    EXPECT_EQ(std::filesystem::file_size(tp.path()), G_FILE_SIZE);
}
