// SPDX-License-Identifier: MIT
//
// cnnp/header.hpp — CNNP V2 header types and parser/serializer API.
//
// This header defines:
//   - Compile-time constants for the V2 wire format (magic bytes,
//     header size, default values).
//   - Strongly-typed enums for discriminant fields.
//   - The logical `Header` struct (NOT the wire layout).
//   - `parse_header()` and `serialize_header()` declarations.
//   - `ParseError` exception for malformed/incomplete headers.
//
// See `notes/CNNP-V2-SPEC.md` §4 for the byte-by-byte layout this
// parser/serializer round-trips through.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace cnnp {

// ─── Wire-format constants (V2) ───────────────────────────────────────────────

inline constexpr std::array<std::byte, 4> CNNP_MAGIC = {
    std::byte{'C'}, std::byte{'N'}, std::byte{'N'}, std::byte{'2'}
};
inline constexpr std::uint16_t CNNP_VERSION       = 2;
inline constexpr std::uint16_t CNNP_HEADER_SIZE   = 256;
inline constexpr std::uint16_t CNNP_BLOCK_SIZE    = 1024;
inline constexpr std::uint8_t  CNNP_MAX_COUNT     = 32;
inline constexpr std::uint8_t  CNNP_COUNT_BASE    = 2;
inline constexpr float         CNNP_FIXED_SCALE        = 3000.0f;
inline constexpr float         CNNP_NORMAL_EVAL_CLIP   = 6.5f;
inline constexpr float         CNNP_STORAGE_TARGET_CLIP = 10.0f;

// ─── Discriminant enums ───────────────────────────────────────────────────────

enum class Endian : std::uint8_t {
    Little = 0,
    // Big-endian (1) is reserved but unsupported in V2.
};

enum class LayoutKind : std::uint8_t {
    SingleFile = 0,
    // Sharded (1) is reserved for V3.
};

enum class FeatureSet : std::uint8_t {
    HalfP        = 1,
    HalfKAv2_hm  = 2,
    HalfKA       = 3,
};

enum class CountSemantics : std::uint8_t {
    PieceCountEqualsNnz = 0,
    // V3 may add separate piece_count + nnz arrays (1).
};

enum class EvalEncoding : std::uint8_t {
    Int16FixedNormalized = 1,
};

enum class MateEncoding : std::uint8_t {
    InBand   = 0,  // Stored within ±storage_target_clip; no saturation
    Saturate = 1,  // Allowed but not produced by V2 encoder
    OutBand  = 2,  // Reserved for V3 (dedicated bit + range)
};

enum class FlagsEncoding : std::uint8_t {
    StmPlusCountMinus2 = 0,
};

// ─── Header struct (logical representation) ───────────────────────────────────
//
// This is NOT the wire layout. The on-disk header is a flat 256-byte
// region described in spec §4. This struct is what the parser produces
// and the serializer consumes; field ordering here is for human
// readability, not byte alignment.

struct Header {
    // Identity
    std::uint16_t  version       = CNNP_VERSION;
    std::uint16_t  header_size   = CNNP_HEADER_SIZE;
    Endian         endian        = Endian::Little;
    LayoutKind     layout_kind   = LayoutKind::SingleFile;

    // Semantics
    FeatureSet     feature_set     = FeatureSet::HalfP;
    CountSemantics count_semantics = CountSemantics::PieceCountEqualsNnz;
    EvalEncoding   eval_encoding   = EvalEncoding::Int16FixedNormalized;
    MateEncoding   mate_encoding   = MateEncoding::InBand;
    FlagsEncoding  flags_encoding  = FlagsEncoding::StmPlusCountMinus2;

    // Shard info (V2 single-file: shard_id=0, num_shards=1, start=0)
    std::uint32_t  shard_id              = 0;
    std::uint32_t  num_shards            = 1;
    std::uint64_t  global_position_start = 0;

    // Sizes
    std::uint64_t  num_positions      = 0;
    std::uint64_t  num_features_total = 0;
    std::uint32_t  max_feature_id     = 0;  // exclusive upper bound
    std::uint32_t  num_blocks         = 0;  // ceil(num_positions / block_size)

    // Block parameters
    std::uint16_t  block_size  = CNNP_BLOCK_SIZE;
    std::uint8_t   max_count   = CNNP_MAX_COUNT;
    std::uint8_t   count_base  = CNNP_COUNT_BASE;

    // Eval encoding
    float          fixed_scale         = CNNP_FIXED_SCALE;
    float          normal_eval_clip    = CNNP_NORMAL_EVAL_CLIP;
    float          storage_target_clip = CNNP_STORAGE_TARGET_CLIP;

    // Optional arrays presence (V2: both must be 1)
    std::uint8_t   has_eval_global = 1;
    std::uint8_t   has_wdl_global  = 1;
    std::uint16_t  header_flags    = 0;  // reserved bitfield

    // Array byte offsets into the file (V2 canonical order)
    std::uint64_t  flags_offset         = 0;
    std::uint64_t  eval_offset          = 0;
    std::uint64_t  wdl_offset           = 0;
    std::uint64_t  block_anchors_offset = 0;
    std::uint64_t  block_prefix_offset  = 0;
    std::uint64_t  w_flat_offset        = 0;
    std::uint64_t  metadata_offset      = 0;
    std::uint64_t  metadata_length      = 0;
};

// ─── Errors ───────────────────────────────────────────────────────────────────

class ParseError : public std::runtime_error {
public:
    explicit ParseError(const std::string& what) : std::runtime_error(what) {}
};

// ─── API ──────────────────────────────────────────────────────────────────────

/// Parse a 256-byte CNNP V2 header from a raw byte span.
///
/// Validates structural fields (magic, version, header_size, endian,
/// reserved bytes). Does NOT validate cross-field consistency such as
/// `num_blocks == ceil(num_positions / block_size)` — that is the
/// validator's responsibility (see `cnnp/validator.hpp`, future).
///
/// Throws `ParseError` on:
///   - input span shorter than CNNP_HEADER_SIZE
///   - magic bytes != "CNN2"
///   - version != 2
///   - header_size != 256
///   - endian != 0 (V2 is little-endian only)
///   - invalid enum values
///   - any reserved byte != 0
[[nodiscard]] Header parse_header(std::span<const std::byte> bytes);

/// Serialize a Header into a 256-byte buffer.
///
/// The output span MUST be at least CNNP_HEADER_SIZE bytes.
/// Reserved bytes are zero-filled. The caller is responsible for
/// ensuring the field values are valid (e.g., `header.version == 2`);
/// this function does not re-validate (use the validator beforehand).
///
/// Throws `ParseError` if the output buffer is too small.
void serialize_header(const Header& header, std::span<std::byte> bytes);

}  // namespace cnnp
