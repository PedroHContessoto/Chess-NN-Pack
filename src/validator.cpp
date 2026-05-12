// SPDX-License-Identifier: MIT
//
// cnnp/src/validator.cpp — implementation of spec §7 invariants.

#include "cnnp/validator.hpp"
#include "cnnp/encoding.hpp"  // for FLAGS_RESERVED / sentinel constants

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>

namespace cnnp {

// ─── Checked arithmetic ──────────────────────────────────────────────────────

bool checked_add_u64(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept {
    if (a > std::numeric_limits<std::uint64_t>::max() - b) {
        out = 0;
        return true;  // overflow
    }
    out = a + b;
    return false;
}

bool checked_mul_u64(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept {
    if (a == 0 || b == 0) { out = 0; return false; }
    if (a > std::numeric_limits<std::uint64_t>::max() / b) {
        out = 0;
        return true;
    }
    out = a * b;
    return false;
}

// ─── Internal helpers ────────────────────────────────────────────────────────

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw ValidationError(msg);
}

void must_add(std::uint64_t a, std::uint64_t b, std::uint64_t& out, const char* ctx) {
    if (checked_add_u64(a, b, out)) {
        std::ostringstream os;
        os << "u64 overflow in addition: " << ctx;
        fail(os.str());
    }
}

void must_mul(std::uint64_t a, std::uint64_t b, std::uint64_t& out, const char* ctx) {
    if (checked_mul_u64(a, b, out)) {
        std::ostringstream os;
        os << "u64 overflow in multiplication: " << ctx;
        fail(os.str());
    }
}

}  // anonymous namespace

// ─── Header-only validation ──────────────────────────────────────────────────

void validate_header_consistency(const Header& h, std::uint64_t file_size) {
    // ─── V2 fixed-field invariants (spec §7 Header) ──────────────────────────
    // These re-check what parse_header validates. They matter for in-memory
    // Header structs (e.g., produced by Writer) that bypass the parser.
    if (h.version != CNNP_VERSION) {
        std::ostringstream os;
        os << "version " << h.version << " != " << CNNP_VERSION << " (V2)";
        fail(os.str());
    }
    if (h.header_size != CNNP_HEADER_SIZE) {
        std::ostringstream os;
        os << "header_size " << h.header_size << " != " << CNNP_HEADER_SIZE;
        fail(os.str());
    }
    if (h.endian != Endian::Little) {
        fail("endian != Little (V2 is little-endian only)");
    }
    if (h.layout_kind != LayoutKind::SingleFile) {
        fail("layout_kind != SingleFile (V2)");
    }
    if (h.feature_set != FeatureSet::HalfP &&
        h.feature_set != FeatureSet::HalfKAv2_hm &&
        h.feature_set != FeatureSet::HalfKA) {
        std::ostringstream os;
        os << "feature_set value " << static_cast<int>(h.feature_set)
           << " not in V2 set {HalfP=1, HalfKAv2_hm=2, HalfKA=3}";
        fail(os.str());
    }
    if (h.count_semantics != CountSemantics::PieceCountEqualsNnz) {
        fail("count_semantics != PieceCountEqualsNnz (V2)");
    }
    if (h.eval_encoding != EvalEncoding::Int16FixedNormalized) {
        fail("eval_encoding != Int16FixedNormalized (V2)");
    }
    if (h.mate_encoding != MateEncoding::InBand &&
        h.mate_encoding != MateEncoding::Saturate) {
        std::ostringstream os;
        os << "mate_encoding value " << static_cast<int>(h.mate_encoding)
           << " not in V2 set {InBand=0, Saturate=1}";
        fail(os.str());
    }
    if (h.flags_encoding != FlagsEncoding::StmPlusCountMinus2) {
        fail("flags_encoding != StmPlusCountMinus2 (V2)");
    }
    if (h.has_eval_global != 1) fail("has_eval_global != 1 (V2: arrays always present)");
    if (h.has_wdl_global  != 1) fail("has_wdl_global  != 1 (V2: arrays always present)");
    if (h.shard_id              != 0) fail("shard_id != 0 (V2: single-file only)");
    if (h.num_shards            != 1) fail("num_shards != 1 (V2: single-file only)");
    if (h.global_position_start != 0) fail("global_position_start != 0 (V2)");
    if (h.header_flags          != 0) {
        std::ostringstream os;
        os << "header_flags " << h.header_flags << " != 0 (V2: reserved, must be zero)";
        fail(os.str());
    }
    if (h.block_size != CNNP_BLOCK_SIZE) {
        std::ostringstream os;
        os << "block_size " << h.block_size << " != " << CNNP_BLOCK_SIZE << " (V2)";
        fail(os.str());
    }
    if (h.max_count != CNNP_MAX_COUNT) {
        std::ostringstream os;
        os << "max_count " << static_cast<int>(h.max_count)
           << " != " << static_cast<int>(CNNP_MAX_COUNT) << " (V2)";
        fail(os.str());
    }
    if (h.count_base != CNNP_COUNT_BASE) {
        std::ostringstream os;
        os << "count_base " << static_cast<int>(h.count_base)
           << " != " << static_cast<int>(CNNP_COUNT_BASE) << " (V2)";
        fail(os.str());
    }

    // ─── Float fields must be finite and positive ────────────────────────────
    // Done BEFORE the bound check below: NaN comparisons against 32767 would
    // silently return false, letting garbage through.
    if (!std::isfinite(h.fixed_scale) || h.fixed_scale <= 0.0f) {
        std::ostringstream os;
        os << "fixed_scale must be finite and > 0; got " << h.fixed_scale;
        fail(os.str());
    }
    if (!std::isfinite(h.normal_eval_clip) || h.normal_eval_clip <= 0.0f) {
        std::ostringstream os;
        os << "normal_eval_clip must be finite and > 0; got " << h.normal_eval_clip;
        fail(os.str());
    }
    if (!std::isfinite(h.storage_target_clip) || h.storage_target_clip <= 0.0f) {
        std::ostringstream os;
        os << "storage_target_clip must be finite and > 0; got " << h.storage_target_clip;
        fail(os.str());
    }

    // ─── Sanity ──────────────────────────────────────────────────────────────
    if (h.num_positions == 0) fail("num_positions == 0 (V2 forbids empty datasets)");
    if (h.num_blocks    == 0) fail("num_blocks == 0 (V2 forbids empty datasets)");

    // num_blocks == ceil(num_positions / block_size).
    // Overflow-safe: avoids `num_positions + block_size - 1` wrapping for
    // pathological inputs where num_positions is near UINT64_MAX.
    if (h.block_size == 0) fail("block_size == 0");
    {
        const std::uint64_t expected_blocks =
            h.num_positions / h.block_size +
            ((h.num_positions % h.block_size) != 0 ? 1ull : 0ull);
        if (expected_blocks > std::numeric_limits<std::uint32_t>::max()) {
            std::ostringstream os;
            os << "num_blocks " << expected_blocks << " exceeds UINT32_MAX";
            fail(os.str());
        }
        if (static_cast<std::uint64_t>(h.num_blocks) != expected_blocks) {
            std::ostringstream os;
            os << "num_blocks " << h.num_blocks
               << " != ceil(num_positions/block_size) " << expected_blocks;
            fail(os.str());
        }
    }

    // storage_target_clip * fixed_scale ≤ 32767
    {
        const float bound = h.storage_target_clip * h.fixed_scale;
        if (bound > 32767.0f) {
            std::ostringstream os;
            os << "storage_target_clip * fixed_scale = " << bound
               << " > 32767 (eval_i16 would overflow)";
            fail(os.str());
        }
    }

    // block_size * max_count ≤ 65535 (prefix u16 invariant)
    {
        const std::uint32_t bound =
            static_cast<std::uint32_t>(h.block_size) * static_cast<std::uint32_t>(h.max_count);
        if (bound > std::numeric_limits<std::uint16_t>::max()) {
            std::ostringstream os;
            os << "block_size * max_count = " << bound
               << " > 65535 (prefix u16 would overflow)";
            fail(os.str());
        }
    }

    // 64-bit process: num_features_total ≤ SIZE_MAX (per spec §12)
    if (h.num_features_total > std::numeric_limits<std::size_t>::max()) {
        fail("num_features_total exceeds SIZE_MAX (file requires a 64-bit process)");
    }

    // ─── Alignment of typed array offsets ─────────────────────────────────────
    auto require_aligned = [&](std::uint64_t off, std::size_t align, const char* name) {
        if (off % align != 0) {
            std::ostringstream os;
            os << name << " (" << off << ") is not " << align << "-byte aligned";
            fail(os.str());
        }
    };
    require_aligned(h.eval_offset,          2, "eval_offset");
    require_aligned(h.block_anchors_offset, 8, "block_anchors_offset");
    require_aligned(h.block_prefix_offset,  2, "block_prefix_offset");
    require_aligned(h.w_flat_offset,        2, "w_flat_offset");

    // ─── All offsets within [header_size, file_size) ──────────────────────────
    auto require_in_file = [&](std::uint64_t off, const char* name) {
        if (off < h.header_size) {
            std::ostringstream os;
            os << name << " " << off << " < header_size " << h.header_size;
            fail(os.str());
        }
        if (off >= file_size) {
            std::ostringstream os;
            os << name << " " << off << " >= file_size " << file_size;
            fail(os.str());
        }
    };
    require_in_file(h.flags_offset,         "flags_offset");
    require_in_file(h.eval_offset,          "eval_offset");
    require_in_file(h.wdl_offset,           "wdl_offset");
    require_in_file(h.block_anchors_offset, "block_anchors_offset");
    require_in_file(h.block_prefix_offset,  "block_prefix_offset");
    require_in_file(h.w_flat_offset,        "w_flat_offset");

    // metadata is special: its offset can equal file_size when metadata_length
    // is zero (empty trailer at end of file). Only the combined end matters.
    if (h.metadata_offset < h.header_size) {
        std::ostringstream os;
        os << "metadata_offset " << h.metadata_offset
           << " < header_size " << h.header_size;
        fail(os.str());
    }
    {
        std::uint64_t end = 0;
        must_add(h.metadata_offset, h.metadata_length, end, "metadata end");
        if (end > file_size) {
            std::ostringstream os;
            os << "metadata_offset + metadata_length (" << end
               << ") > file_size " << file_size;
            fail(os.str());
        }
    }

    // ─── Canonical layout: each array fits before the next ────────────────────
    auto check_fits_before = [&](std::uint64_t start, std::uint64_t count,
                                  std::uint64_t elem_size, std::uint64_t next_start,
                                  const char* name) {
        std::uint64_t bytes = 0;
        must_mul(count, elem_size, bytes, name);
        std::uint64_t end = 0;
        must_add(start, bytes, end, name);
        if (end > next_start) {
            std::ostringstream os;
            os << name << " end " << end << " exceeds next array start " << next_start;
            fail(os.str());
        }
    };
    check_fits_before(h.flags_offset,         h.num_positions, 1,
                      h.eval_offset,          "flags array");
    check_fits_before(h.eval_offset,          h.num_positions, 2,
                      h.wdl_offset,           "eval array");
    check_fits_before(h.wdl_offset,           h.num_positions, 1,
                      h.block_anchors_offset, "wdl array");
    check_fits_before(h.block_anchors_offset, h.num_blocks,    8,
                      h.block_prefix_offset,  "block_anchors array");
    {
        // block_prefix has num_blocks * (block_size + 1) elements of u16
        std::uint64_t total_prefix = 0;
        must_mul(h.num_blocks, static_cast<std::uint64_t>(h.block_size) + 1,
                 total_prefix, "block_prefix elements");
        check_fits_before(h.block_prefix_offset, total_prefix, 2,
                          h.w_flat_offset, "block_prefix array");
    }
    check_fits_before(h.w_flat_offset, h.num_features_total, 2,
                      h.metadata_offset, "w_flat array");
}

// ─── Per-array validators ────────────────────────────────────────────────────

void validate_flags_array(std::span<const std::uint8_t> flags) {
    constexpr std::uint8_t RESERVED_BITS  = 0xC0;  // bits 6-7
    constexpr std::uint8_t COUNT_MASK     = 0x1F;  // bits 1-5 (after >> 1)
    constexpr std::uint8_t COUNT_SENTINEL = 31;

    for (std::size_t i = 0; i < flags.size(); ++i) {
        const std::uint8_t f = flags[i];
        if ((f & RESERVED_BITS) != 0) {
            std::ostringstream os;
            os << "flags[" << i << "] = 0x" << std::hex << +f
               << ": reserved bits 6-7 must be zero";
            fail(os.str());
        }
        const std::uint8_t stored_count = static_cast<std::uint8_t>((f >> 1) & COUNT_MASK);
        if (stored_count == COUNT_SENTINEL) {
            std::ostringstream os;
            os << "flags[" << i << "]: stored count == 31 (sentinel reserved)";
            fail(os.str());
        }
    }
}

void validate_eval_array(std::span<const std::int16_t> evals,
                         float fixed_scale,
                         float storage_target_clip) {
    const float bound_f = storage_target_clip * fixed_scale;
    if (bound_f > 32767.0f) {
        // Defensive: should have been rejected by header validator already.
        fail("validate_eval_array: storage_target_clip * fixed_scale > 32767");
    }
    const std::int32_t bound = static_cast<std::int32_t>(bound_f);
    for (std::size_t i = 0; i < evals.size(); ++i) {
        const std::int32_t v = evals[i];
        if (v > bound || v < -bound) {
            std::ostringstream os;
            os << "evals[" << i << "] = " << v << " outside ±" << bound;
            fail(os.str());
        }
    }
}

void validate_wdl_array(std::span<const std::int8_t> wdls) {
    for (std::size_t i = 0; i < wdls.size(); ++i) {
        const std::int8_t v = wdls[i];
        if (v != -1 && v != 0 && v != 1) {
            std::ostringstream os;
            os << "wdls[" << i << "] = " << +v << " not in {-1, 0, +1}";
            fail(os.str());
        }
    }
}

void validate_block_prefix(std::span<const std::uint64_t> anchors,
                           std::span<const std::uint16_t> prefix,
                           const Header& h) {
    if (anchors.size() != h.num_blocks) {
        std::ostringstream os;
        os << "anchors.size() " << anchors.size()
           << " != num_blocks " << h.num_blocks;
        fail(os.str());
    }
    const std::uint64_t expected_prefix_count =
        static_cast<std::uint64_t>(h.num_blocks) * (h.block_size + 1);
    if (prefix.size() != expected_prefix_count) {
        std::ostringstream os;
        os << "prefix.size() " << prefix.size()
           << " != num_blocks * (block_size + 1) " << expected_prefix_count;
        fail(os.str());
    }

    // anchors[0] == 0
    if (anchors[0] != 0) {
        std::ostringstream os;
        os << "anchors[0] = " << anchors[0] << " != 0";
        fail(os.str());
    }

    // valid_positions_in_last_block per spec
    const std::uint64_t valid_in_last =
        (h.num_positions % h.block_size == 0)
            ? h.block_size
            : (h.num_positions % h.block_size);

    const std::uint32_t max_block_features =
        static_cast<std::uint32_t>(h.block_size) * static_cast<std::uint32_t>(h.max_count);

    std::uint64_t computed_anchor = 0;
    for (std::uint32_t b = 0; b < h.num_blocks; ++b) {
        const std::size_t base = static_cast<std::size_t>(b) *
                                 (static_cast<std::size_t>(h.block_size) + 1);

        // anchor matches running sum
        if (anchors[b] != computed_anchor) {
            std::ostringstream os;
            os << "anchors[" << b << "] = " << anchors[b]
               << " != computed " << computed_anchor;
            fail(os.str());
        }

        // prefix[base + 0] == 0
        if (prefix[base] != 0) {
            std::ostringstream os;
            os << "prefix[block " << b << "][0] = " << prefix[base] << " != 0";
            fail(os.str());
        }

        // monotonicity across the full (block_size + 1) entries
        for (std::uint32_t j = 0; j < h.block_size; ++j) {
            if (prefix[base + j + 1] < prefix[base + j]) {
                std::ostringstream os;
                os << "prefix[block " << b << "] not monotonic at j=" << j
                   << " (" << prefix[base + j] << " → " << prefix[base + j + 1] << ")";
                fail(os.str());
            }
        }

        // upper bound
        if (prefix[base + h.block_size] > max_block_features) {
            std::ostringstream os;
            os << "prefix[block " << b << "][block_size] = "
               << prefix[base + h.block_size]
               << " > block_size * max_count " << max_block_features;
            fail(os.str());
        }

        // last block: padding rule (slots beyond valid_in_last must equal prefix[valid_in_last])
        if (b + 1 == h.num_blocks) {
            const std::uint16_t last_valid = prefix[base + valid_in_last];
            for (std::uint64_t j = valid_in_last; j <= h.block_size; ++j) {
                if (prefix[base + j] != last_valid) {
                    std::ostringstream os;
                    os << "last-block padding violation: prefix[" << j
                       << "] = " << prefix[base + j] << " != " << last_valid;
                    fail(os.str());
                }
            }
        }

        // accumulate for next anchor
        std::uint64_t added;
        must_add(computed_anchor, prefix[base + h.block_size], added, "anchor accumulation");
        computed_anchor = added;
    }

    // Global feature count consistency
    if (computed_anchor != h.num_features_total) {
        std::ostringstream os;
        os << "sum of block prefix totals = " << computed_anchor
           << " != num_features_total " << h.num_features_total;
        fail(os.str());
    }
}

void validate_w_flat(std::span<const std::uint16_t> w_flat,
                     std::uint32_t max_feature_id) {
    for (std::size_t i = 0; i < w_flat.size(); ++i) {
        if (w_flat[i] >= max_feature_id) {
            std::ostringstream os;
            os << "w_flat[" << i << "] = " << w_flat[i]
               << " >= max_feature_id " << max_feature_id;
            fail(os.str());
        }
    }
}

void validate_counts_match_prefix(std::span<const std::uint8_t> flags,
                                  std::span<const std::uint16_t> prefix,
                                  const Header& h) {
    // Caller guarantees: flags.size() == h.num_positions,
    //                    prefix.size() == h.num_blocks * (block_size + 1),
    //                    validate_flags_array has run (sentinel rejected).
    constexpr std::uint8_t COUNT_MASK = 0x1F;
    for (std::uint64_t i = 0; i < h.num_positions; ++i) {
        const std::uint64_t b    = i / h.block_size;
        const std::uint64_t j    = i % h.block_size;
        const std::uint64_t base = b * (static_cast<std::uint64_t>(h.block_size) + 1ull);

        const std::uint8_t  stored_count =
            static_cast<std::uint8_t>((flags[i] >> 1) & COUNT_MASK);
        const std::uint16_t piece_count  =
            static_cast<std::uint16_t>(stored_count + h.count_base);
        const std::uint16_t nnz =
            static_cast<std::uint16_t>(prefix[base + j + 1] - prefix[base + j]);

        if (nnz != piece_count) {
            std::ostringstream os;
            os << "position " << i << ": prefix nnz " << nnz
               << " != flags piece_count " << piece_count
               << " (count_semantics = piece_count_equals_nnz)";
            fail(os.str());
        }
    }
}

// ─── Full validation ─────────────────────────────────────────────────────────

namespace {

template <typename T>
std::span<const T> make_span(std::span<const std::byte> bytes,
                             std::uint64_t off, std::uint64_t count,
                             const char* name) {
    // Header validator already verified alignment + bounds. These checks
    // are belt-and-suspenders against accidental misuse.
    if (off % alignof(T) != 0) {
        std::ostringstream os;
        os << name << " offset " << off << " misaligned for sizeof(T)=" << sizeof(T);
        fail(os.str());
    }
    std::uint64_t bytes_needed;
    if (checked_mul_u64(count, sizeof(T), bytes_needed)) {
        std::ostringstream os;
        os << name << ": u64 overflow computing array bytes";
        fail(os.str());
    }
    std::uint64_t end;
    if (checked_add_u64(off, bytes_needed, end)) {
        std::ostringstream os;
        os << name << ": u64 overflow computing array end";
        fail(os.str());
    }
    if (end > bytes.size()) {
        std::ostringstream os;
        os << name << ": array end " << end << " > file_size " << bytes.size();
        fail(os.str());
    }
    return std::span<const T>(
        reinterpret_cast<const T*>(bytes.data() + off),
        static_cast<std::size_t>(count));
}

}  // anonymous namespace

void validate_full(const Header& h, std::span<const std::byte> file_bytes) {
    validate_header_consistency(h, file_bytes.size());

    auto flags = make_span<std::uint8_t>(file_bytes, h.flags_offset,
                                          h.num_positions, "flags");
    auto evals = make_span<std::int16_t>(file_bytes, h.eval_offset,
                                          h.num_positions, "eval");
    auto wdls  = make_span<std::int8_t>(file_bytes, h.wdl_offset,
                                         h.num_positions, "wdl");
    auto anchors = make_span<std::uint64_t>(file_bytes, h.block_anchors_offset,
                                             h.num_blocks, "block_anchors");
    const std::uint64_t prefix_count =
        static_cast<std::uint64_t>(h.num_blocks) * (h.block_size + 1);
    auto prefix = make_span<std::uint16_t>(file_bytes, h.block_prefix_offset,
                                            prefix_count, "block_prefix");
    auto w_flat = make_span<std::uint16_t>(file_bytes, h.w_flat_offset,
                                            h.num_features_total, "w_flat");

    validate_flags_array(flags);
    validate_eval_array(evals, h.fixed_scale, h.storage_target_clip);
    validate_wdl_array(wdls);
    validate_block_prefix(anchors, prefix, h);
    validate_w_flat(w_flat, h.max_feature_id);
    // Cross-check piece_count (flags) vs nnz (prefix delta) — runs after
    // validate_flags_array so the sentinel is already rejected.
    validate_counts_match_prefix(flags, prefix, h);
}

}  // namespace cnnp
