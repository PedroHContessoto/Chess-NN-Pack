// SPDX-License-Identifier: MIT
//
// Unit tests for cnnp/reader.{hpp,cpp}.
//
// Strategy:
//   * Build an in-memory valid file via the shared fixture.
//   * Write to a unique temp path.
//   * Open with Reader and verify header, per-position views, bulk
//     accessors, and bounds checking.

#include "cnnp/reader.hpp"

#include "cnnp/byte_reader.hpp"
#include "cnnp/encoding.hpp"
#include "cnnp/mmap_region.hpp"
#include "cnnp/validator.hpp"

#include "test_fixtures.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using cnnp::DecodedFlags;
using cnnp::decode_flags;
using cnnp::Header;
using cnnp::MmapError;
using cnnp::PositionView;
using cnnp::Reader;
using cnnp::ValidationError;
using cnnp::write_u16_le;
using cnnp::fixtures::make_valid_file;
using cnnp::fixtures::ValidFile;

namespace {

/// Unique temp file holder. Deletes the file on destruction.
class TempFile {
public:
    TempFile(const ValidFile& vf, const char* tag) {
        static std::atomic<std::uint64_t> counter{0};
        const std::uint64_t id = counter.fetch_add(1, std::memory_order_relaxed);
        m_path = std::filesystem::temp_directory_path() /
                 (std::string("cnnp_test_") + tag + "_" +
                  std::to_string(id) + ".cnnp");
        std::ofstream out(m_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to open temp file for writing: " +
                                     m_path.string());
        }
        out.write(reinterpret_cast<const char*>(vf.bytes.data()),
                  static_cast<std::streamsize>(vf.bytes.size()));
        out.close();
    }

    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(m_path, ec);
    }

    TempFile(const TempFile&)            = delete;
    TempFile& operator=(const TempFile&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return m_path; }

private:
    std::filesystem::path m_path;
};

}  // anonymous namespace

// ─── Open + header roundtrip ─────────────────────────────────────────────────

TEST(Reader, OpenSmallFile) {
    auto vf = make_valid_file();
    TempFile tmp(vf, "small");

    auto reader = Reader::open(tmp.path());
    EXPECT_EQ(reader.num_positions(), vf.header.num_positions);
    EXPECT_EQ(reader.num_blocks(),    vf.header.num_blocks);
    EXPECT_EQ(reader.header().w_flat_offset, vf.header.w_flat_offset);
    EXPECT_EQ(reader.header().num_features_total, vf.header.num_features_total);
}

TEST(Reader, OpenRejectsMissingFile) {
    EXPECT_THROW((void)Reader::open("__nonexistent_cnnp_file_xyz123__.cnnp"),
                 MmapError);
}

// ─── Per-position access ─────────────────────────────────────────────────────

TEST(Reader, AtReturnsExpectedDefaultsView) {
    auto vf = make_valid_file(/*n=*/4, /*ppp=*/16);
    TempFile tmp(vf, "at");
    auto reader = Reader::open(tmp.path());

    for (std::uint64_t i = 0; i < vf.header.num_positions; ++i) {
        auto v = reader.at(i);
        EXPECT_EQ(v.stm, 0u);
        EXPECT_EQ(v.piece_count, 16u);
        EXPECT_EQ(v.wdl, 0);
        EXPECT_FLOAT_EQ(v.eval_normalized, 0.0f);
        EXPECT_EQ(v.features.size(), 16u);
    }
}

TEST(Reader, FeaturesValuesMatchEncoded) {
    auto vf = make_valid_file(/*n=*/4, /*ppp=*/8);
    TempFile tmp(vf, "feat");
    auto reader = Reader::open(tmp.path());

    // Position 0 should have features 0..7
    auto v0 = reader.at(0);
    ASSERT_EQ(v0.features.size(), 8u);
    for (std::uint64_t k = 0; k < 8; ++k) {
        EXPECT_EQ(v0.features[k], k);
    }

    // Position 1 should have features 8..15
    auto v1 = reader.at(1);
    ASSERT_EQ(v1.features.size(), 8u);
    for (std::uint64_t k = 0; k < 8; ++k) {
        EXPECT_EQ(v1.features[k], 8u + k);
    }
}

TEST(Reader, AtThrowsOutOfRange) {
    auto vf = make_valid_file();
    TempFile tmp(vf, "oob");
    auto reader = Reader::open(tmp.path());

    EXPECT_THROW((void)reader.at(reader.num_positions()),       std::out_of_range);
    EXPECT_THROW((void)reader.at(reader.num_positions() + 1000), std::out_of_range);
}

// ─── Bulk accessors ──────────────────────────────────────────────────────────

TEST(Reader, BulkAccessorSizes) {
    auto vf = make_valid_file(/*n=*/4, /*ppp=*/16);
    TempFile tmp(vf, "bulk");
    auto reader = Reader::open(tmp.path());

    EXPECT_EQ(reader.flags_array().size(),   vf.header.num_positions);
    EXPECT_EQ(reader.eval_array().size(),    vf.header.num_positions);
    EXPECT_EQ(reader.wdl_array().size(),     vf.header.num_positions);
    EXPECT_EQ(reader.block_anchors().size(), vf.header.num_blocks);
    EXPECT_EQ(reader.block_prefix().size(),
              static_cast<std::size_t>(vf.header.num_blocks) *
              static_cast<std::size_t>(vf.header.block_size + 1u));
    EXPECT_EQ(reader.w_flat().size(),        vf.header.num_features_total);
    EXPECT_EQ(reader.metadata().size(),      vf.header.metadata_length);
}

TEST(Reader, BulkAccessorContentsMatchFixture) {
    auto vf = make_valid_file(/*n=*/4, /*ppp=*/8);
    TempFile tmp(vf, "bulk2");
    auto reader = Reader::open(tmp.path());

    auto flags = reader.flags_array();
    for (auto f : flags) {
        DecodedFlags df = decode_flags(f);
        EXPECT_EQ(df.stm, 0u);
        EXPECT_EQ(df.piece_count, 8u);
    }

    auto evals = reader.eval_array();
    for (auto e : evals) EXPECT_EQ(e, 0);

    auto wdls = reader.wdl_array();
    for (auto w : wdls) EXPECT_EQ(w, 0);

    auto wflat = reader.w_flat();
    ASSERT_EQ(wflat.size(), 32u);  // 4 positions * 8 features
    for (std::size_t k = 0; k < wflat.size(); ++k) {
        EXPECT_EQ(wflat[k], k);
    }
}

// ─── Cross-block random access ───────────────────────────────────────────────

TEST(Reader, MultipleBlocksRandomAccess) {
    auto vf = make_valid_file(/*n=*/2500, /*ppp=*/12);  // 3 blocks
    TempFile tmp(vf, "multi");
    auto reader = Reader::open(tmp.path());

    EXPECT_EQ(reader.num_blocks(), 3u);

    // Spot-check across block boundaries.
    for (std::uint64_t i : {std::uint64_t{0},
                            std::uint64_t{1},
                            std::uint64_t{1023},
                            std::uint64_t{1024},
                            std::uint64_t{1025},
                            std::uint64_t{2047},
                            std::uint64_t{2048},
                            std::uint64_t{2499}}) {
        auto v = reader.at(i);
        EXPECT_EQ(v.piece_count, 12u);
        ASSERT_EQ(v.features.size(), 12u);
        // Fixture writes feature ids as (k % 768) for k = global feature
        // index. The first feature of position i is at global index i*12.
        EXPECT_EQ(v.features[0],
                  static_cast<std::uint16_t>((i * 12u) % 768u))
            << "at i=" << i;
        // Last feature: global index i*12 + 11
        EXPECT_EQ(v.features[11],
                  static_cast<std::uint16_t>((i * 12u + 11u) % 768u))
            << "at i=" << i;
    }
}

// ─── open vs open_unchecked ──────────────────────────────────────────────────

TEST(Reader, OpenUncheckedSkipsArrayValidation) {
    // Build a valid file then corrupt one w_flat entry to be out of range.
    auto vf = make_valid_file(/*n=*/4, /*ppp=*/8);
    write_u16_le(vf.bytes, vf.header.w_flat_offset + 0, 9999);  // > max_feature_id
    TempFile tmp(vf, "unchk");

    EXPECT_THROW((void)Reader::open(tmp.path()), ValidationError);
    EXPECT_NO_THROW((void)Reader::open_unchecked(tmp.path()));
}

TEST(Reader, OpenUncheckedStillValidatesHeaderAlignment) {
    auto vf = make_valid_file();
    // Misalign eval_offset → header consistency must reject even in unchecked.
    vf.header.eval_offset += 1;
    // Re-serialize header so the file reflects the bad offset.
    cnnp::serialize_header(vf.header, std::span<std::byte>(vf.bytes));
    TempFile tmp(vf, "unchk_align");

    EXPECT_THROW((void)Reader::open_unchecked(tmp.path()), ValidationError);
}
