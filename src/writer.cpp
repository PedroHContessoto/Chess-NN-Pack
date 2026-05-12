// SPDX-License-Identifier: MIT
//
// cnnp/writer.cpp — implementation of the in-memory Writer.

#include "cnnp/writer.hpp"

#include "cnnp/byte_reader.hpp"
#include "cnnp/encoding.hpp"
#include "cnnp/header.hpp"
#include "cnnp/validator.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace cnnp {

namespace {

constexpr std::uint64_t round_up(std::uint64_t v, std::uint64_t a) {
    return (v + (a - 1)) & ~(a - 1);
}

}  // anonymous namespace

// ─── Construction ────────────────────────────────────────────────────────────

Writer::Writer() : Writer(WriterConfig{}) {}

Writer::Writer(WriterConfig cfg) : m_cfg(std::move(cfg)) {
    if (m_cfg.feature_set != FeatureSet::HalfP) {
        // V2 supports HalfP only (count_semantics = piece_count == nnz).
        throw WriteError("Writer: only FeatureSet::HalfP is supported in V2");
    }
    if (m_cfg.block_size == 0) {
        throw WriteError("Writer: block_size must be > 0");
    }
    if (m_cfg.max_count < 2 || m_cfg.max_count > 32) {
        throw WriteError("Writer: max_count must be in [2, 32]");
    }
    if (m_cfg.max_feature_id == 0) {
        throw WriteError("Writer: max_feature_id must be > 0");
    }
    if (m_cfg.fixed_scale <= 0.0f) {
        throw WriteError("Writer: fixed_scale must be positive");
    }
    if (m_cfg.storage_target_clip <= 0.0f) {
        throw WriteError("Writer: storage_target_clip must be positive");
    }
}

// ─── add() ───────────────────────────────────────────────────────────────────

void Writer::add(const InputPosition& pos) {
    add(pos.score_cp, pos.wdl, pos.stm, pos.piece_count, pos.features);
}

void Writer::add(std::int32_t score_cp, std::int32_t wdl,
                 std::uint8_t stm, std::uint8_t piece_count,
                 std::span<const std::uint16_t> features) {
    if (stm > 1) {
        throw WriteError("Writer::add: stm must be 0 or 1, got " +
                         std::to_string(stm));
    }
    if (piece_count < 2 || piece_count > m_cfg.max_count) {
        throw WriteError("Writer::add: piece_count " +
                         std::to_string(piece_count) +
                         " out of range [2, " +
                         std::to_string(m_cfg.max_count) + "]");
    }
    if (features.size() != piece_count) {
        throw WriteError("Writer::add: features.size() (" +
                         std::to_string(features.size()) +
                         ") != piece_count (" +
                         std::to_string(piece_count) + ") for HalfP");
    }
    for (auto fid : features) {
        if (fid >= m_cfg.max_feature_id) {
            throw WriteError("Writer::add: feature id " +
                             std::to_string(fid) +
                             " >= max_feature_id " +
                             std::to_string(m_cfg.max_feature_id));
        }
    }

    // Encode into wire form. encode_flags() / encode_wdl() throw on bad
    // inputs but we've already validated above.
    const std::uint8_t flags_byte = encode_flags(stm, piece_count);
    const std::int16_t eval_i16   = encode_eval_from_cp(
        score_cp, m_cfg.fixed_scale, m_cfg.storage_target_clip);
    const std::int8_t  wdl_i8     = encode_wdl(wdl);

    m_flags.push_back(flags_byte);
    m_eval.push_back(eval_i16);
    m_wdl.push_back(wdl_i8);
    m_per_pos_nnz.push_back(static_cast<std::uint16_t>(features.size()));
    m_w_flat.insert(m_w_flat.end(), features.begin(), features.end());
}

// ─── finalize() ──────────────────────────────────────────────────────────────

void Writer::finalize(const std::filesystem::path& path) {
    if (m_flags.empty()) {
        throw WriteError(
            "Writer::finalize: no positions added (num_positions=0 not allowed)");
    }

    // Build the header from accumulated state + config.
    Header h;
    h.feature_set         = m_cfg.feature_set;
    h.num_positions       = m_flags.size();
    h.num_features_total  = m_w_flat.size();
    h.max_feature_id      = m_cfg.max_feature_id;
    h.block_size          = m_cfg.block_size;
    h.max_count           = m_cfg.max_count;
    h.count_base          = m_cfg.count_base;
    h.fixed_scale         = m_cfg.fixed_scale;
    h.normal_eval_clip    = m_cfg.normal_eval_clip;
    h.storage_target_clip = m_cfg.storage_target_clip;
    h.num_blocks          = static_cast<std::uint32_t>(
        (h.num_positions + h.block_size - 1) / h.block_size);

    // Per-block invariant: sum of nnz in any block must fit in u16.
    // With block_size=1024 and max_count=32, max sum = 32768 — but we
    // verify defensively in case the config was relaxed.
    {
        std::uint64_t pos_idx = 0;
        for (std::uint32_t b = 0; b < h.num_blocks; ++b) {
            std::uint32_t local_sum = 0;
            const std::uint64_t block_end = std::min<std::uint64_t>(
                pos_idx + h.block_size, h.num_positions);
            for (std::uint64_t i = pos_idx; i < block_end; ++i) {
                local_sum += m_per_pos_nnz[i];
            }
            if (local_sum > 65535u) {
                throw WriteError("Writer::finalize: block " +
                                 std::to_string(b) +
                                 " feature sum " + std::to_string(local_sum) +
                                 " exceeds u16 prefix limit (65535)");
            }
            pos_idx = block_end;
        }
    }

    // Compute canonical offsets (matches the layout enforced by the validator).
    h.flags_offset         = CNNP_HEADER_SIZE;
    h.eval_offset          = round_up(h.flags_offset + h.num_positions * 1, 2);
    h.wdl_offset           = round_up(h.eval_offset  + h.num_positions * 2, 1);
    h.block_anchors_offset = round_up(h.wdl_offset   + h.num_positions * 1, 8);
    h.block_prefix_offset  = round_up(h.block_anchors_offset + h.num_blocks * 8, 2);
    const std::uint64_t prefix_bytes =
        static_cast<std::uint64_t>(h.num_blocks) * (h.block_size + 1) * 2;
    h.w_flat_offset        = round_up(h.block_prefix_offset + prefix_bytes, 2);
    h.metadata_offset      = h.w_flat_offset + h.num_features_total * 2;
    h.metadata_length      = m_cfg.metadata_json.size();

    const std::uint64_t file_size = h.metadata_offset + h.metadata_length;

    // Allocate the entire file buffer (zero-filled so any padding bytes
    // between sections are deterministic).
    std::vector<std::byte> bytes(static_cast<std::size_t>(file_size),
                                 std::byte{0});

    // Header
    serialize_header(h, std::span<std::byte>(bytes));

    // flags  (1 byte/pos, no endianness concerns)
    std::memcpy(bytes.data() + h.flags_offset,
                m_flags.data(),
                m_flags.size());

    // eval  (2 bytes/pos LE)
    for (std::uint64_t i = 0; i < h.num_positions; ++i) {
        write_i16_le(bytes, h.eval_offset + i * 2, m_eval[i]);
    }

    // wdl  (1 byte/pos, signed)
    for (std::uint64_t i = 0; i < h.num_positions; ++i) {
        bytes[h.wdl_offset + i] = std::byte{static_cast<std::uint8_t>(m_wdl[i])};
    }

    // anchors + prefix (computed from m_per_pos_nnz)
    std::uint64_t running_features = 0;
    std::uint64_t pos_idx = 0;
    for (std::uint32_t b = 0; b < h.num_blocks; ++b) {
        write_u64_le(bytes, h.block_anchors_offset + b * 8, running_features);

        const std::uint64_t prefix_base = h.block_prefix_offset
                                        + static_cast<std::uint64_t>(b)
                                          * (h.block_size + 1) * 2;
        write_u16_le(bytes, prefix_base, 0);

        const std::uint64_t pos_in_block = std::min<std::uint64_t>(
            h.block_size, h.num_positions - pos_idx);

        std::uint16_t local = 0;
        for (std::uint64_t j = 0; j < pos_in_block; ++j) {
            local = static_cast<std::uint16_t>(local + m_per_pos_nnz[pos_idx + j]);
            write_u16_le(bytes, prefix_base + (j + 1) * 2, local);
        }
        // padding: repeat the last valid prefix value
        for (std::uint64_t j = pos_in_block; j < h.block_size; ++j) {
            write_u16_le(bytes, prefix_base + (j + 1) * 2, local);
        }

        running_features += local;
        pos_idx += pos_in_block;
    }

    // w_flat  (2 bytes/feature LE)
    for (std::uint64_t k = 0; k < h.num_features_total; ++k) {
        write_u16_le(bytes, h.w_flat_offset + k * 2, m_w_flat[k]);
    }

    // metadata trailer (UTF-8 JSON; spec §6)
    if (!m_cfg.metadata_json.empty()) {
        std::memcpy(bytes.data() + h.metadata_offset,
                    m_cfg.metadata_json.data(),
                    m_cfg.metadata_json.size());
    }

    // Optional re-validation: catches any disagreement between writer
    // and validator (i.e., a writer bug).
    if (m_cfg.validate) {
        validate_full(h, std::span<const std::byte>(bytes));
    }

    // Atomic write: ofstream + write + close, all-or-nothing for
    // typical platforms (caller can wrap in tmp+rename for true atomicity).
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw WriteError("Writer::finalize: could not open '" +
                             path.string() + "' for writing");
        }
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        out.close();
        if (!out) {
            throw WriteError("Writer::finalize: write failed for '" +
                             path.string() + "'");
        }
    }

    // Reset state — Writer is reusable for another file.
    m_flags.clear();
    m_eval.clear();
    m_wdl.clear();
    m_per_pos_nnz.clear();
    m_w_flat.clear();
}

// ─── One-shot helper ─────────────────────────────────────────────────────────

void write_cnnp_file(const std::filesystem::path& path,
                     std::span<const InputPosition> positions,
                     const WriterConfig& cfg) {
    Writer w(cfg);
    for (const auto& p : positions) w.add(p);
    w.finalize(path);
}

}  // namespace cnnp
