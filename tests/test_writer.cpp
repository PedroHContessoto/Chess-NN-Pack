// SPDX-License-Identifier: MIT
//
// Unit tests for cnnp/writer.{hpp,cpp}.
//
// Strategy:
//   * Build positions in memory.
//   * Write to a unique temp path with Writer (or write_cnnp_file).
//   * Re-open with Reader and verify per-position fields round-trip.

#include "cnnp/writer.hpp"

#include "cnnp/encoding.hpp"
#include "cnnp/header.hpp"
#include "cnnp/reader.hpp"
#include "cnnp/validator.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>      // std::memcmp (GCC requires explicit include; MSVC pulls transitively)
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using cnnp::DecodedFlags;
using cnnp::decode_eval;
using cnnp::decode_flags;
using cnnp::encode_eval_from_cp;
using cnnp::Header;
using cnnp::InputPosition;
using cnnp::Reader;
using cnnp::ValidationError;
using cnnp::WriteError;
using cnnp::Writer;
using cnnp::WriterConfig;
using cnnp::write_cnnp_file;

namespace {

class TempPath {
public:
    explicit TempPath(const char* tag) {
        static std::atomic<std::uint64_t> counter{0};
        const auto id = counter.fetch_add(1, std::memory_order_relaxed);
        m_path = std::filesystem::temp_directory_path() /
                 (std::string("cnnp_writer_test_") + tag + "_" +
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

}  // anonymous namespace

// ─── Validation of inputs ────────────────────────────────────────────────────

TEST(Writer, FinalizeWithoutPositionsThrows) {
    Writer w;
    TempPath tp("empty");
    EXPECT_THROW(w.finalize(tp.path()), WriteError);
}

TEST(Writer, RejectsBadStm) {
    Writer w;
    std::vector<std::uint16_t> feats = {0, 1};
    EXPECT_THROW(w.add(0, 0, /*stm=*/2, /*pc=*/2,
                       std::span<const std::uint16_t>(feats)),
                 WriteError);
}

TEST(Writer, RejectsBadPieceCount) {
    Writer w;
    std::vector<std::uint16_t> too_few = {0};
    EXPECT_THROW(w.add(0, 0, 0, /*pc=*/1, std::span<const std::uint16_t>(too_few)),
                 WriteError);

    std::vector<std::uint16_t> too_many(33, 0);
    EXPECT_THROW(w.add(0, 0, 0, /*pc=*/33, std::span<const std::uint16_t>(too_many)),
                 WriteError);
}

TEST(Writer, RejectsMismatchedFeatureCount) {
    Writer w;
    std::vector<std::uint16_t> feats = {0, 1};
    EXPECT_THROW(w.add(0, 0, 0, /*pc=*/3, std::span<const std::uint16_t>(feats)),
                 WriteError);
}

TEST(Writer, RejectsBadFeatureId) {
    WriterConfig cfg;
    cfg.max_feature_id = 100;
    Writer w(cfg);
    std::vector<std::uint16_t> feats = {0, 100};  // 100 >= max_feature_id
    EXPECT_THROW(w.add(0, 0, 0, 2, std::span<const std::uint16_t>(feats)),
                 WriteError);
}

TEST(Writer, RejectsBadWdl) {
    Writer w;
    std::vector<std::uint16_t> feats = {0, 1};
    // encode_wdl throws EncodeError, which we propagate (not WriteError, by design,
    // because it surfaces from the encoding layer).
    EXPECT_ANY_THROW(w.add(0, /*wdl=*/2, 0, 2, std::span<const std::uint16_t>(feats)));
}

// ─── Roundtrip: small file ───────────────────────────────────────────────────

TEST(Writer, SingleBlockRoundtrip) {
    TempPath tp("rt1");
    Writer w;

    std::vector<std::vector<std::uint16_t>> all_features = {
        {1, 2, 3, 4, 5, 6},
        {10, 20, 30},
        {100, 200},
        {7, 8, 9, 11, 13, 17, 19},
    };
    const std::vector<std::int32_t> cps  = {15, -42, 320, -800};
    const std::vector<std::int32_t> wdls = {1, 0, -1, 0};
    const std::vector<std::uint8_t> stms = {0, 1, 0, 1};

    for (std::size_t i = 0; i < all_features.size(); ++i) {
        w.add(cps[i], wdls[i], stms[i],
              static_cast<std::uint8_t>(all_features[i].size()),
              std::span<const std::uint16_t>(all_features[i]));
    }

    EXPECT_EQ(w.num_positions(), 4u);
    w.finalize(tp.path());

    auto r = Reader::open(tp.path());
    EXPECT_EQ(r.num_positions(), 4u);
    EXPECT_EQ(r.num_blocks(),    1u);

    for (std::size_t i = 0; i < all_features.size(); ++i) {
        auto v = r.at(i);
        EXPECT_EQ(v.stm,         stms[i])                        << "i=" << i;
        EXPECT_EQ(v.piece_count, all_features[i].size())          << "i=" << i;
        EXPECT_EQ(v.wdl,         static_cast<std::int8_t>(wdls[i])) << "i=" << i;

        const float expected_eval = decode_eval(
            encode_eval_from_cp(cps[i]), cnnp::CNNP_FIXED_SCALE);
        EXPECT_FLOAT_EQ(v.eval_normalized, expected_eval) << "i=" << i;

        ASSERT_EQ(v.features.size(), all_features[i].size()) << "i=" << i;
        for (std::size_t k = 0; k < all_features[i].size(); ++k) {
            EXPECT_EQ(v.features[k], all_features[i][k])
                << "i=" << i << " k=" << k;
        }
    }
}

// ─── Roundtrip: multiple blocks ──────────────────────────────────────────────

TEST(Writer, MultiBlockRoundtrip) {
    TempPath tp("rt_multi");
    Writer w;

    constexpr std::size_t  N   = 1500;  // 2 blocks (1024 + 476)
    constexpr std::uint8_t PPP = 8;
    std::vector<std::uint16_t> feats(PPP);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t k = 0; k < PPP; ++k) {
            feats[k] = static_cast<std::uint16_t>((i * PPP + k) % 768);
        }
        const std::int32_t wdl = static_cast<std::int32_t>(i % 3) - 1;
        const std::uint8_t stm = static_cast<std::uint8_t>(i % 2);
        w.add(static_cast<std::int32_t>(i % 100), wdl, stm,
              PPP, std::span<const std::uint16_t>(feats));
    }
    w.finalize(tp.path());

    auto r = Reader::open(tp.path());
    EXPECT_EQ(r.num_positions(), N);
    EXPECT_EQ(r.num_blocks(),    2u);

    for (std::size_t i : {std::size_t{0}, std::size_t{1023},
                          std::size_t{1024}, std::size_t{1499}}) {
        auto v = r.at(i);
        EXPECT_EQ(v.piece_count, PPP)                                  << "i=" << i;
        EXPECT_EQ(v.stm,         static_cast<std::uint8_t>(i % 2))     << "i=" << i;
        EXPECT_EQ(v.wdl,
                  static_cast<std::int8_t>(static_cast<int>(i % 3) - 1))
            << "i=" << i;
        ASSERT_EQ(v.features.size(), PPP)                              << "i=" << i;
        EXPECT_EQ(v.features[0],
                  static_cast<std::uint16_t>((i * PPP) % 768))         << "i=" << i;
    }
}

// ─── Free function vs incremental writer ─────────────────────────────────────

TEST(Writer, FreeFunctionMatchesIncremental) {
    TempPath tp1("free");
    TempPath tp2("incr");

    std::vector<std::uint16_t> feats = {0, 1, 2, 3};
    std::vector<InputPosition> positions;
    for (int i = 0; i < 5; ++i) {
        positions.push_back(InputPosition{
            /*score_cp*/   i * 10,
            /*wdl*/        (i % 3) - 1,
            /*stm*/        static_cast<std::uint8_t>(i % 2),
            /*piece_count*/4,
            /*features*/   std::span<const std::uint16_t>(feats),
        });
    }

    write_cnnp_file(tp1.path(), std::span<const InputPosition>(positions));

    Writer w;
    for (const auto& p : positions) w.add(p);
    w.finalize(tp2.path());

    // Files should be byte-identical.
    const auto sz1 = std::filesystem::file_size(tp1.path());
    const auto sz2 = std::filesystem::file_size(tp2.path());
    ASSERT_EQ(sz1, sz2);

    std::ifstream f1(tp1.path(), std::ios::binary);
    std::ifstream f2(tp2.path(), std::ios::binary);
    ASSERT_TRUE(f1 && f2);

    std::vector<char> b1(static_cast<std::size_t>(sz1));
    std::vector<char> b2(static_cast<std::size_t>(sz2));
    f1.read(b1.data(), static_cast<std::streamsize>(b1.size()));
    f2.read(b2.data(), static_cast<std::streamsize>(b2.size()));
    EXPECT_EQ(b1, b2);
}

// ─── Output validation ───────────────────────────────────────────────────────

TEST(Writer, OutputPassesReaderOpen) {
    TempPath tp("validate");
    std::vector<std::uint16_t> feats = {0, 1, 2};
    Writer w;
    w.add( 50, 1, 0, 3, std::span<const std::uint16_t>(feats));
    w.add(-10, 0, 1, 3, std::span<const std::uint16_t>(feats));
    w.finalize(tp.path());

    EXPECT_NO_THROW((void)Reader::open(tp.path()));
}

TEST(Writer, ValidateFlagDisablesInternalCheck) {
    TempPath tp("novalidate");
    WriterConfig cfg;
    cfg.validate = false;

    std::vector<std::uint16_t> feats = {0, 1};
    Writer w(cfg);
    w.add(0, 0, 0, 2, std::span<const std::uint16_t>(feats));
    EXPECT_NO_THROW(w.finalize(tp.path()));

    // The file is still valid (cfg only controls the redundant internal check).
    EXPECT_NO_THROW((void)Reader::open(tp.path()));
}

// ─── Statefulness ────────────────────────────────────────────────────────────

TEST(Writer, ResetsAfterFinalize) {
    TempPath tp("reset");
    std::vector<std::uint16_t> feats = {0, 1};
    Writer w;
    w.add(0, 0, 0, 2, std::span<const std::uint16_t>(feats));
    w.finalize(tp.path());

    EXPECT_EQ(w.num_positions(), 0u);

    // Throws on second finalize because no positions were re-added.
    TempPath tp2("reset2");
    EXPECT_THROW(w.finalize(tp2.path()), WriteError);

    // After re-adding, finalize succeeds again.
    w.add(123, 1, 0, 2, std::span<const std::uint16_t>(feats));
    EXPECT_NO_THROW(w.finalize(tp2.path()));
    EXPECT_EQ(w.num_positions(), 0u);
}

// ─── Eval encoding edge cases ────────────────────────────────────────────────

TEST(Writer, EvalRoundtripCoversMateRange) {
    TempPath tp("mate");
    std::vector<std::uint16_t> feats = {0, 1};
    Writer w;
    // Mate scores: |cp| > 25_000 routes through the mate branch in
    // normalize_target. Verify decoder yields the same encoded value.
    const std::vector<std::int32_t> mate_cps = {30000, -30000, 28000, -28000};
    for (auto cp : mate_cps) {
        w.add(cp, 0, 0, 2, std::span<const std::uint16_t>(feats));
    }
    w.finalize(tp.path());

    auto r = Reader::open(tp.path());
    for (std::size_t i = 0; i < mate_cps.size(); ++i) {
        const float expected = decode_eval(
            encode_eval_from_cp(mate_cps[i]), cnnp::CNNP_FIXED_SCALE);
        EXPECT_FLOAT_EQ(r.at(i).eval_normalized, expected) << "i=" << i;
    }
}

// ─── Variable-length feature blocks ──────────────────────────────────────────

// ─── Metadata trailer (spec §6) ──────────────────────────────────────────────

TEST(Writer, MetadataDefaultIsMinimalValidJson) {
    TempPath tp("md_default");
    std::vector<std::uint16_t> feats = {0, 1};
    Writer w;  // default config → minimal JSON trailer
    w.add(0, 0, 0, 2, std::span<const std::uint16_t>(feats));
    w.finalize(tp.path());

    auto r = Reader::open(tp.path());
    auto md = r.metadata();
    const std::string expected =
        R"({"format":"cnnp_sparse_v2","layout":"single_file"})";
    ASSERT_EQ(md.size(), expected.size());
    EXPECT_EQ(0, std::memcmp(md.data(), expected.data(), expected.size()));
    EXPECT_EQ(r.header().metadata_length, expected.size());
    // metadata_offset must equal w_flat_offset + 2 * num_features_total
    EXPECT_EQ(r.header().metadata_offset,
              r.header().w_flat_offset + r.header().num_features_total * 2);
}

TEST(Writer, MetadataCustomJsonRoundtrip) {
    TempPath tp("md_custom");
    WriterConfig cfg;
    cfg.metadata_json = R"({"format":"cnnp_sparse_v2","layout":"single_file","extra":42})";

    std::vector<std::uint16_t> feats = {0, 1, 2};
    Writer w(cfg);
    w.add(50, 1, 0, 3, std::span<const std::uint16_t>(feats));
    w.finalize(tp.path());

    auto r = Reader::open(tp.path());
    auto md = r.metadata();
    ASSERT_EQ(md.size(), cfg.metadata_json.size());
    EXPECT_EQ(0, std::memcmp(md.data(), cfg.metadata_json.data(), md.size()));
}

TEST(Writer, MetadataEmptyTriggersDefault) {
    // Spec §6 + §12 require a non-empty UTF-8 JSON trailer. The Writer
    // auto-fills the spec minimum if the user supplies an empty string,
    // so no path produces a non-conforming file.
    TempPath tp("md_empty_default");
    WriterConfig cfg;
    cfg.metadata_json.clear();  // user explicitly opts for "empty"

    std::vector<std::uint16_t> feats = {0, 1};
    Writer w(cfg);
    w.add(0, 0, 0, 2, std::span<const std::uint16_t>(feats));
    w.finalize(tp.path());

    auto r = Reader::open(tp.path());
    const std::string expected =
        R"({"format":"cnnp_sparse_v2","layout":"single_file"})";
    auto md = r.metadata();
    ASSERT_EQ(md.size(), expected.size());
    EXPECT_EQ(0, std::memcmp(md.data(), expected.data(), expected.size()));
}

TEST(Writer, RejectsNanFixedScaleInConfig) {
    WriterConfig cfg;
    cfg.fixed_scale = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(Writer w(cfg), WriteError);
}

TEST(Writer, RejectsExceedingMaxInMemoryBytes) {
    // Cap at 1024 bytes — far less than even one block's prefix (2050B).
    WriterConfig cfg;
    cfg.max_in_memory_bytes = 1024;

    Writer w(cfg);
    std::vector<std::uint16_t> feats = {0, 1};
    w.add(0, 0, 0, 2, std::span<const std::uint16_t>(feats));

    TempPath tp("max_mem");
    EXPECT_THROW(w.finalize(tp.path()), WriteError);
}

// ─── V2 fixed-field enforcement (fail-fast in ctor) ──────────────────────────

TEST(Writer, RejectsNonV2BlockSizeInConfig) {
    WriterConfig cfg;
    cfg.block_size = 2048;  // V2 mandates 1024
    EXPECT_THROW(Writer w(cfg), WriteError);
}

TEST(Writer, RejectsNonV2MaxCountInConfig) {
    WriterConfig cfg;
    cfg.max_count = 16;  // V2 mandates 32
    EXPECT_THROW(Writer w(cfg), WriteError);
}

TEST(Writer, RejectsNonV2CountBaseInConfig) {
    WriterConfig cfg;
    cfg.count_base = 0;  // V2 mandates 2
    EXPECT_THROW(Writer w(cfg), WriteError);
}

// ─── Metadata user-input validation ─────────────────────────────────────────

TEST(Writer, RejectsMetadataNotJsonObject) {
    WriterConfig cfg;
    cfg.metadata_json = "not really json";
    Writer w(cfg);
    std::vector<std::uint16_t> feats = {0, 1};
    w.add(0, 0, 0, 2, std::span<const std::uint16_t>(feats));

    TempPath tp("md_bad_shape");
    EXPECT_THROW(w.finalize(tp.path()), WriteError);
}

TEST(Writer, RejectsMetadataMissingFormat) {
    WriterConfig cfg;
    cfg.metadata_json = R"({"layout":"single_file"})";  // no format field
    Writer w(cfg);
    std::vector<std::uint16_t> feats = {0, 1};
    w.add(0, 0, 0, 2, std::span<const std::uint16_t>(feats));

    TempPath tp("md_no_format");
    EXPECT_THROW(w.finalize(tp.path()), WriteError);
}

TEST(Writer, RejectsMetadataMissingLayout) {
    WriterConfig cfg;
    cfg.metadata_json = R"({"format":"cnnp_sparse_v2"})";  // no layout field
    Writer w(cfg);
    std::vector<std::uint16_t> feats = {0, 1};
    w.add(0, 0, 0, 2, std::span<const std::uint16_t>(feats));

    TempPath tp("md_no_layout");
    EXPECT_THROW(w.finalize(tp.path()), WriteError);
}

TEST(Writer, MetadataValidationDisabledWhenValidateFalse) {
    // validate=false should let any metadata through (escape hatch).
    WriterConfig cfg;
    cfg.metadata_json = "totally not json";
    cfg.validate      = false;
    Writer w(cfg);
    std::vector<std::uint16_t> feats = {0, 1};
    w.add(0, 0, 0, 2, std::span<const std::uint16_t>(feats));

    TempPath tp("md_no_validate");
    EXPECT_NO_THROW(w.finalize(tp.path()));
    // The file is now non-conforming; cnnp validate (CLI) would reject it.
}

TEST(Writer, MaxInMemoryBytesZeroDisablesCheck) {
    // Default (0) means unlimited — finalize succeeds for any reasonable size.
    WriterConfig cfg;
    cfg.max_in_memory_bytes = 0;

    Writer w(cfg);
    std::vector<std::uint16_t> feats = {0, 1};
    w.add(0, 0, 0, 2, std::span<const std::uint16_t>(feats));

    TempPath tp("max_mem_unlimited");
    EXPECT_NO_THROW(w.finalize(tp.path()));
}

TEST(Writer, RejectsZeroStorageTargetClipInConfig) {
    WriterConfig cfg;
    cfg.storage_target_clip = 0.0f;
    EXPECT_THROW(Writer w(cfg), WriteError);
}

TEST(Writer, MetadataFileSizeMatchesHeaderPlusTrailer) {
    TempPath tp("md_size");
    WriterConfig cfg;
    cfg.metadata_json = R"({"format":"cnnp_sparse_v2","layout":"single_file"})";

    std::vector<std::uint16_t> feats = {0, 1, 2, 3};
    Writer w(cfg);
    for (int i = 0; i < 3; ++i) {
        w.add(i * 10, 0, 0, 4, std::span<const std::uint16_t>(feats));
    }
    w.finalize(tp.path());

    const auto file_size = std::filesystem::file_size(tp.path());
    auto r = Reader::open(tp.path());
    EXPECT_EQ(file_size,
              r.header().metadata_offset + r.header().metadata_length);
}

TEST(Writer, VariablePieceCountsRoundtrip) {
    TempPath tp("varpc");
    Writer w;
    // Pieces decreasing as game progresses (endgame).
    std::vector<std::vector<std::uint16_t>> seqs;
    for (std::uint8_t pc = 32; pc >= 2; pc -= 2) {
        std::vector<std::uint16_t> v;
        for (std::uint8_t k = 0; k < pc; ++k) v.push_back(k);
        seqs.push_back(std::move(v));
    }
    for (const auto& s : seqs) {
        w.add(0, 0, 0, static_cast<std::uint8_t>(s.size()),
              std::span<const std::uint16_t>(s));
    }
    w.finalize(tp.path());

    auto r = Reader::open(tp.path());
    ASSERT_EQ(r.num_positions(), seqs.size());
    for (std::size_t i = 0; i < seqs.size(); ++i) {
        auto v = r.at(i);
        ASSERT_EQ(v.features.size(), seqs[i].size());
        EXPECT_EQ(v.piece_count, seqs[i].size());
        for (std::size_t k = 0; k < seqs[i].size(); ++k) {
            EXPECT_EQ(v.features[k], seqs[i][k]) << "i=" << i << " k=" << k;
        }
    }
}
