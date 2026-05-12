// SPDX-License-Identifier: MIT
//
// tests/test_fixtures.hpp — shared CNNP V2 file fixture builder.
//
// Used by both test_validator and test_reader. Constructs a
// self-consistent canonical-layout V2 file in a `std::vector<std::byte>`
// for easy in-memory validation or write-to-disk + mmap roundtrips.

#pragma once

#include "cnnp/byte_reader.hpp"
#include "cnnp/encoding.hpp"
#include "cnnp/header.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace cnnp::fixtures {

struct ValidFile {
    Header                 header;
    std::vector<std::byte> bytes;
};

inline constexpr std::uint64_t round_up(std::uint64_t v, std::uint64_t a) {
    return (v + (a - 1)) & ~(a - 1);
}

/// Build a self-consistent valid V2 file with `n_positions` positions
/// and `pieces_per_position` features per position (uniform).
///
/// Layout (canonical):
///   header             : 0..256
///   flags              : 256..256+N                       (N bytes)
///   eval               : aligned-2 ..  +N*2                (N*2 bytes)
///   wdl                :          ..  +N                   (N bytes)
///   block_anchors      : aligned-8 ..  +num_blocks*8       (8*num_blocks)
///   block_prefix       : aligned-2 ..  +num_blocks*1025*2  (2050*num_blocks)
///   w_flat             : aligned-2 ..  +total_features*2
///   metadata (empty)   :          ..  end
inline ValidFile make_valid_file(std::uint64_t n_positions = 4,
                                 std::uint8_t  pieces_per_position = 16,
                                 FeatureSet    feature_set = FeatureSet::HalfP,
                                 std::uint32_t max_feature_id = 768) {
    Header h;
    h.feature_set        = feature_set;
    h.num_positions      = n_positions;
    h.num_blocks         = static_cast<std::uint32_t>(
        (n_positions + h.block_size - 1) / h.block_size);
    h.num_features_total = n_positions * pieces_per_position;
    h.max_feature_id     = max_feature_id;

    // Compute aligned offsets.
    h.flags_offset         = CNNP_HEADER_SIZE;
    h.eval_offset          = round_up(h.flags_offset         + n_positions * 1, 2);
    h.wdl_offset           = round_up(h.eval_offset          + n_positions * 2, 1);
    h.block_anchors_offset = round_up(h.wdl_offset           + n_positions * 1, 8);
    h.block_prefix_offset  = round_up(h.block_anchors_offset + h.num_blocks * 8, 2);
    const std::uint64_t prefix_bytes =
        static_cast<std::uint64_t>(h.num_blocks) * (h.block_size + 1) * 2;
    h.w_flat_offset        = round_up(h.block_prefix_offset  + prefix_bytes, 2);
    // Spec §6 + §12 require a non-empty UTF-8 JSON trailer. The fixture
    // emits the minimal spec-compliant document (50 bytes) so that
    // tests share the same invariants as files written by `Writer`.
    constexpr std::string_view DEFAULT_METADATA =
        R"({"format":"cnnp_sparse_v2","layout":"single_file"})";
    h.metadata_offset      = h.w_flat_offset + h.num_features_total * 2;
    h.metadata_length      = DEFAULT_METADATA.size();

    const std::uint64_t file_size = h.metadata_offset + h.metadata_length;
    std::vector<std::byte> bytes(file_size, std::byte{0});

    serialize_header(h, std::span<std::byte>(bytes));

    // flags
    const std::uint8_t flags_byte = encode_flags(0, pieces_per_position);
    for (std::uint64_t i = 0; i < n_positions; ++i) {
        bytes[h.flags_offset + i] = std::byte{flags_byte};
    }

    // eval (zeroes)
    for (std::uint64_t i = 0; i < n_positions; ++i) {
        write_i16_le(bytes, h.eval_offset + i * 2, 0);
    }

    // wdl (zeroes)
    for (std::uint64_t i = 0; i < n_positions; ++i) {
        bytes[h.wdl_offset + i] = std::byte{0};
    }

    // block_anchors + block_prefix
    std::uint64_t running = 0;
    for (std::uint32_t b = 0; b < h.num_blocks; ++b) {
        write_u64_le(bytes, h.block_anchors_offset + b * 8, running);

        const std::uint64_t pos_in_block = std::min<std::uint64_t>(
            h.block_size, n_positions - static_cast<std::uint64_t>(b) * h.block_size);

        std::uint16_t local = 0;
        const std::uint64_t prefix_base = h.block_prefix_offset
                                        + static_cast<std::uint64_t>(b)
                                          * (h.block_size + 1) * 2;
        write_u16_le(bytes, prefix_base, 0);
        for (std::uint64_t j = 0; j < pos_in_block; ++j) {
            local = static_cast<std::uint16_t>(local + pieces_per_position);
            write_u16_le(bytes, prefix_base + (j + 1) * 2, local);
        }
        // padding repeats last valid prefix
        for (std::uint64_t j = pos_in_block; j < h.block_size; ++j) {
            write_u16_le(bytes, prefix_base + (j + 1) * 2, local);
        }

        running += local;
    }

    // w_flat: feature ids cycling 0..max_feature_id-1
    for (std::uint64_t k = 0; k < h.num_features_total; ++k) {
        write_u16_le(bytes, h.w_flat_offset + k * 2,
                     static_cast<std::uint16_t>(k % max_feature_id));
    }

    // metadata trailer (UTF-8 JSON; spec §6)
    std::memcpy(bytes.data() + h.metadata_offset,
                DEFAULT_METADATA.data(), DEFAULT_METADATA.size());

    return ValidFile{h, std::move(bytes)};
}

}  // namespace cnnp::fixtures
