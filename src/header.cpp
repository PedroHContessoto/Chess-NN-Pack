// SPDX-License-Identifier: MIT
//
// cnnp/src/header.cpp — CNNP V2 header parser and serializer.
//
// Field-by-field parsing through endian-safe helpers. NO reinterpret_cast
// or bit_cast of the whole header (per spec §12). This module is the
// single source of truth for the on-disk byte layout described in
// `notes/CNNP-V2-SPEC.md` §4.

#include "cnnp/header.hpp"
#include "cnnp/byte_reader.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace cnnp {

// ─── Wire-format byte offsets (V2) ────────────────────────────────────────────
// These offsets are the canonical layout described in spec §4. Changing any
// of them is a wire-format change and requires bumping `version`.

namespace off {
    constexpr std::size_t magic                  = 0;
    constexpr std::size_t version                = 4;
    constexpr std::size_t header_size            = 6;
    constexpr std::size_t endian                 = 8;
    constexpr std::size_t layout_kind            = 9;
    constexpr std::size_t feature_set            = 10;
    constexpr std::size_t count_semantics        = 11;
    constexpr std::size_t eval_encoding          = 12;
    constexpr std::size_t mate_encoding          = 13;
    constexpr std::size_t flags_encoding         = 14;
    constexpr std::size_t reserved_15            = 15;
    constexpr std::size_t shard_id               = 16;
    constexpr std::size_t num_shards             = 20;
    constexpr std::size_t global_position_start  = 24;
    constexpr std::size_t num_positions          = 32;
    constexpr std::size_t num_features_total     = 40;
    constexpr std::size_t max_feature_id         = 48;
    constexpr std::size_t num_blocks             = 52;
    constexpr std::size_t block_size             = 56;
    constexpr std::size_t max_count              = 58;
    constexpr std::size_t count_base             = 59;
    constexpr std::size_t fixed_scale            = 60;
    constexpr std::size_t normal_eval_clip       = 64;
    constexpr std::size_t storage_target_clip    = 68;
    constexpr std::size_t has_eval_global        = 72;
    constexpr std::size_t has_wdl_global         = 73;
    constexpr std::size_t header_flags           = 74;
    constexpr std::size_t reserved_76            = 76;
    constexpr std::size_t flags_offset           = 80;
    constexpr std::size_t eval_offset            = 88;
    constexpr std::size_t wdl_offset             = 96;
    constexpr std::size_t block_anchors_offset   = 104;
    constexpr std::size_t block_prefix_offset    = 112;
    constexpr std::size_t w_flat_offset          = 120;
    constexpr std::size_t metadata_offset        = 128;
    constexpr std::size_t metadata_length        = 136;
    constexpr std::size_t reserved_padding_start = 144;
    constexpr std::size_t reserved_padding_end   = 256;
}  // namespace off

static_assert(off::reserved_padding_end == CNNP_HEADER_SIZE,
              "header offsets must total CNNP_HEADER_SIZE (256)");

// ─── Internal helpers ─────────────────────────────────────────────────────────

namespace {

[[noreturn]] void throw_parse(const std::string& msg) {
    throw ParseError("cnnp::parse_header: " + msg);
}

// Validate an enum-like u8 falls into a permitted set.
template <typename E>
E parse_enum_u8(std::uint8_t raw, std::initializer_list<E> allowed,
                const char* field_name) {
    for (E v : allowed) {
        if (raw == static_cast<std::uint8_t>(v)) return v;
    }
    std::ostringstream os;
    os << "invalid value " << static_cast<unsigned>(raw)
       << " for field '" << field_name << "'";
    throw_parse(os.str());
}

}  // anonymous namespace

// ─── parse_header ─────────────────────────────────────────────────────────────

Header parse_header(std::span<const std::byte> bytes) {
    if (bytes.size() < CNNP_HEADER_SIZE) {
        std::ostringstream os;
        os << "input too short: " << bytes.size()
           << " bytes (need " << CNNP_HEADER_SIZE << ")";
        throw_parse(os.str());
    }

    // ── Magic ────────────────────────────────────────────────────────────────
    for (std::size_t i = 0; i < CNNP_MAGIC.size(); ++i) {
        if (bytes[off::magic + i] != CNNP_MAGIC[i]) {
            throw_parse("magic mismatch (expected \"CNN2\")");
        }
    }

    Header h;

    // ── Identity ─────────────────────────────────────────────────────────────
    h.version     = read_u16_le(bytes, off::version);
    h.header_size = read_u16_le(bytes, off::header_size);

    if (h.version != CNNP_VERSION) {
        std::ostringstream os;
        os << "unsupported version " << h.version
           << " (this build supports v" << CNNP_VERSION << ")";
        throw_parse(os.str());
    }
    if (h.header_size != CNNP_HEADER_SIZE) {
        std::ostringstream os;
        os << "header_size " << h.header_size
           << " != " << CNNP_HEADER_SIZE;
        throw_parse(os.str());
    }

    h.endian      = parse_enum_u8<Endian>(
        read_u8(bytes, off::endian), {Endian::Little}, "endian");
    h.layout_kind = parse_enum_u8<LayoutKind>(
        read_u8(bytes, off::layout_kind), {LayoutKind::SingleFile}, "layout_kind");

    // ── Semantics ────────────────────────────────────────────────────────────
    h.feature_set     = parse_enum_u8<FeatureSet>(
        read_u8(bytes, off::feature_set),
        {FeatureSet::HalfP, FeatureSet::HalfKAv2_hm, FeatureSet::HalfKA},
        "feature_set");
    h.count_semantics = parse_enum_u8<CountSemantics>(
        read_u8(bytes, off::count_semantics),
        {CountSemantics::PieceCountEqualsNnz}, "count_semantics");
    h.eval_encoding   = parse_enum_u8<EvalEncoding>(
        read_u8(bytes, off::eval_encoding),
        {EvalEncoding::Int16FixedNormalized}, "eval_encoding");
    h.mate_encoding   = parse_enum_u8<MateEncoding>(
        read_u8(bytes, off::mate_encoding),
        {MateEncoding::InBand, MateEncoding::Saturate}, "mate_encoding");
    h.flags_encoding  = parse_enum_u8<FlagsEncoding>(
        read_u8(bytes, off::flags_encoding),
        {FlagsEncoding::StmPlusCountMinus2}, "flags_encoding");

    if (read_u8(bytes, off::reserved_15) != 0) {
        throw_parse("reserved byte at offset 15 is non-zero");
    }

    // ── Shard info ───────────────────────────────────────────────────────────
    h.shard_id              = read_u32_le(bytes, off::shard_id);
    h.num_shards            = read_u32_le(bytes, off::num_shards);
    h.global_position_start = read_u64_le(bytes, off::global_position_start);

    // V2: enforce single-file invariants in the parser (cheap and catches
    // misencoded files early).
    if (h.shard_id != 0 || h.num_shards != 1 || h.global_position_start != 0) {
        throw_parse("V2 requires shard_id=0, num_shards=1, global_position_start=0");
    }

    // ── Sizes ────────────────────────────────────────────────────────────────
    h.num_positions      = read_u64_le(bytes, off::num_positions);
    h.num_features_total = read_u64_le(bytes, off::num_features_total);
    h.max_feature_id     = read_u32_le(bytes, off::max_feature_id);
    h.num_blocks         = read_u32_le(bytes, off::num_blocks);

    // ── Block parameters ─────────────────────────────────────────────────────
    h.block_size = read_u16_le(bytes, off::block_size);
    h.max_count  = read_u8 (bytes, off::max_count);
    h.count_base = read_u8 (bytes, off::count_base);

    if (h.block_size != CNNP_BLOCK_SIZE) {
        std::ostringstream os;
        os << "block_size " << h.block_size << " != " << CNNP_BLOCK_SIZE;
        throw_parse(os.str());
    }
    if (h.max_count != CNNP_MAX_COUNT) {
        std::ostringstream os;
        os << "max_count " << static_cast<unsigned>(h.max_count)
           << " != " << static_cast<unsigned>(CNNP_MAX_COUNT);
        throw_parse(os.str());
    }
    if (h.count_base != CNNP_COUNT_BASE) {
        std::ostringstream os;
        os << "count_base " << static_cast<unsigned>(h.count_base)
           << " != " << static_cast<unsigned>(CNNP_COUNT_BASE);
        throw_parse(os.str());
    }

    // ── Eval encoding ────────────────────────────────────────────────────────
    h.fixed_scale         = read_f32_le(bytes, off::fixed_scale);
    h.normal_eval_clip    = read_f32_le(bytes, off::normal_eval_clip);
    h.storage_target_clip = read_f32_le(bytes, off::storage_target_clip);

    // ── Optional arrays presence ─────────────────────────────────────────────
    h.has_eval_global = read_u8 (bytes, off::has_eval_global);
    h.has_wdl_global  = read_u8 (bytes, off::has_wdl_global);
    h.header_flags    = read_u16_le(bytes, off::header_flags);

    if (h.has_eval_global != 1) {
        throw_parse("V2 requires has_eval_global == 1");
    }
    if (h.has_wdl_global != 1) {
        throw_parse("V2 requires has_wdl_global == 1");
    }
    if (read_u32_le(bytes, off::reserved_76) != 0) {
        throw_parse("reserved 4 bytes at offset 76 are non-zero");
    }

    // ── Array offsets ────────────────────────────────────────────────────────
    h.flags_offset         = read_u64_le(bytes, off::flags_offset);
    h.eval_offset          = read_u64_le(bytes, off::eval_offset);
    h.wdl_offset           = read_u64_le(bytes, off::wdl_offset);
    h.block_anchors_offset = read_u64_le(bytes, off::block_anchors_offset);
    h.block_prefix_offset  = read_u64_le(bytes, off::block_prefix_offset);
    h.w_flat_offset        = read_u64_le(bytes, off::w_flat_offset);
    h.metadata_offset      = read_u64_le(bytes, off::metadata_offset);
    h.metadata_length      = read_u64_le(bytes, off::metadata_length);

    // ── Reserved padding ─────────────────────────────────────────────────────
    for (std::size_t i = off::reserved_padding_start;
         i < off::reserved_padding_end; ++i) {
        if (bytes[i] != std::byte{0}) {
            std::ostringstream os;
            os << "reserved padding byte at offset " << i << " is non-zero";
            throw_parse(os.str());
        }
    }

    return h;
}

// ─── serialize_header ─────────────────────────────────────────────────────────

void serialize_header(const Header& h, std::span<std::byte> bytes) {
    if (bytes.size() < CNNP_HEADER_SIZE) {
        std::ostringstream os;
        os << "cnnp::serialize_header: output buffer too small ("
           << bytes.size() << " < " << CNNP_HEADER_SIZE << ")";
        throw ParseError(os.str());
    }

    // Zero the whole header region first so reserved bytes are clean.
    std::fill_n(bytes.begin(), CNNP_HEADER_SIZE, std::byte{0});

    // Magic
    for (std::size_t i = 0; i < CNNP_MAGIC.size(); ++i) {
        bytes[off::magic + i] = CNNP_MAGIC[i];
    }

    // Identity
    write_u16_le(bytes, off::version,     h.version);
    write_u16_le(bytes, off::header_size, h.header_size);
    write_u8    (bytes, off::endian,      static_cast<std::uint8_t>(h.endian));
    write_u8    (bytes, off::layout_kind, static_cast<std::uint8_t>(h.layout_kind));

    // Semantics
    write_u8(bytes, off::feature_set,     static_cast<std::uint8_t>(h.feature_set));
    write_u8(bytes, off::count_semantics, static_cast<std::uint8_t>(h.count_semantics));
    write_u8(bytes, off::eval_encoding,   static_cast<std::uint8_t>(h.eval_encoding));
    write_u8(bytes, off::mate_encoding,   static_cast<std::uint8_t>(h.mate_encoding));
    write_u8(bytes, off::flags_encoding,  static_cast<std::uint8_t>(h.flags_encoding));
    // reserved_15 already zero

    // Shard info
    write_u32_le(bytes, off::shard_id,              h.shard_id);
    write_u32_le(bytes, off::num_shards,            h.num_shards);
    write_u64_le(bytes, off::global_position_start, h.global_position_start);

    // Sizes
    write_u64_le(bytes, off::num_positions,      h.num_positions);
    write_u64_le(bytes, off::num_features_total, h.num_features_total);
    write_u32_le(bytes, off::max_feature_id,     h.max_feature_id);
    write_u32_le(bytes, off::num_blocks,         h.num_blocks);

    // Block parameters
    write_u16_le(bytes, off::block_size, h.block_size);
    write_u8    (bytes, off::max_count,  h.max_count);
    write_u8    (bytes, off::count_base, h.count_base);

    // Eval encoding
    write_f32_le(bytes, off::fixed_scale,         h.fixed_scale);
    write_f32_le(bytes, off::normal_eval_clip,    h.normal_eval_clip);
    write_f32_le(bytes, off::storage_target_clip, h.storage_target_clip);

    // Optional arrays presence
    write_u8    (bytes, off::has_eval_global, h.has_eval_global);
    write_u8    (bytes, off::has_wdl_global,  h.has_wdl_global);
    write_u16_le(bytes, off::header_flags,    h.header_flags);
    // reserved_76 (4 bytes) already zero

    // Array offsets
    write_u64_le(bytes, off::flags_offset,         h.flags_offset);
    write_u64_le(bytes, off::eval_offset,          h.eval_offset);
    write_u64_le(bytes, off::wdl_offset,           h.wdl_offset);
    write_u64_le(bytes, off::block_anchors_offset, h.block_anchors_offset);
    write_u64_le(bytes, off::block_prefix_offset,  h.block_prefix_offset);
    write_u64_le(bytes, off::w_flat_offset,        h.w_flat_offset);
    write_u64_le(bytes, off::metadata_offset,      h.metadata_offset);
    write_u64_le(bytes, off::metadata_length,      h.metadata_length);

    // Reserved padding bytes 144..256 already zero from the initial fill.
}

}  // namespace cnnp
