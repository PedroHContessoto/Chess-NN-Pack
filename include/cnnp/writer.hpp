// SPDX-License-Identifier: MIT
//
// cnnp/writer.hpp — produce CNNP V2 files.
//
// Two surfaces:
//   * `Writer` — incremental: add positions one at a time, then finalize().
//   * `write_cnnp_file()` — one-shot wrapper for caller-owned spans.
//
// The writer:
//   * validates each input (stm, piece_count, feature ids) and throws
//     `WriteError` on the first malformed position.
//   * normalizes raw centipawn scores via encode_eval_from_cp().
//   * computes canonical block_anchors / block_prefix / canonical offsets.
//   * serializes a 256-byte header + all arrays in a single atomic write.
//   * by default re-validates the produced file via validate_full().
//
// **In-memory** (intended for tests, golden files, and small-to-medium
// datasets up to a few hundred million positions). The entire dataset
// is buffered in RAM until finalize() and a second buffer of equal size
// is allocated for the output file image. Approximate peak RAM:
//
//     in-RAM arrays:    ~(8 + 2 * avg_features) bytes per position
//     output buffer:    ~(8 + 2 * avg_features) bytes per position
//     total peak:       ~16 + 4 * avg_features bytes per position
//
// For HalfP with avg_features ≈ 16:
//
//     100M positions  →  ~8 GB peak  (viable on 16+ GB workstations)
//     500M positions  →  ~40 GB peak (need a high-RAM machine)
//     2B  positions   →  ~160 GB peak (NOT viable; needs StreamingWriter)
//
// A future `StreamingWriter` will write the w_flat array directly to
// disk while only buffering small per-position arrays + prefix/anchors
// in RAM, capping peak memory below ~10 GB regardless of dataset size.
// Use this `Writer` until then; if your dataset exceeds ~500M positions,
// stop and ask for the streaming variant.

#pragma once

#include "cnnp/header.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cnnp {

// ─── Errors ───────────────────────────────────────────────────────────────────

class WriteError : public std::runtime_error {
public:
    explicit WriteError(const std::string& what) : std::runtime_error(what) {}
};

// ─── Per-position input ───────────────────────────────────────────────────────

/// One position's worth of input. `features` contains u16 feature ids
/// in [0, max_feature_id). For HalfP (count_semantics == nnz),
/// `features.size()` must equal `piece_count`.
struct InputPosition {
    std::int32_t score_cp;     // raw engine score in centipawns (white-POV)
    std::int32_t wdl;          // -1, 0, +1 (white-POV game outcome)
    std::uint8_t stm;          // 0 = white to move, 1 = black to move
    std::uint8_t piece_count;  // 2..32
    std::span<const std::uint16_t> features;
};

// ─── Writer config ───────────────────────────────────────────────────────────

struct WriterConfig {
    FeatureSet     feature_set         = FeatureSet::HalfP;
    std::uint32_t  max_feature_id      = 768;  // exclusive upper bound
    std::uint16_t  block_size          = CNNP_BLOCK_SIZE;
    std::uint8_t   max_count           = CNNP_MAX_COUNT;
    std::uint8_t   count_base          = CNNP_COUNT_BASE;
    float          fixed_scale         = CNNP_FIXED_SCALE;
    float          normal_eval_clip    = CNNP_NORMAL_EVAL_CLIP;
    float          storage_target_clip = CNNP_STORAGE_TARGET_CLIP;
    /// If true, run validate_full() on the in-memory buffer before
    /// flushing to disk. Adds roughly 10 ms per million positions.
    bool           validate            = true;
    /// JSON metadata trailer written at metadata_offset (spec §6).
    /// Default is the minimal spec-compliant document (`format` +
    /// `layout` keys). Set to empty string to omit the trailer entirely;
    /// the resulting file is non-conformant per spec §12 but still
    /// readable by the core Reader (the core does not parse JSON).
    std::string    metadata_json =
        R"({"format":"cnnp_sparse_v2","layout":"single_file"})";
    /// Hard cap on the in-memory output buffer (bytes). If `finalize()`
    /// computes a `file_size` greater than this, it throws `WriteError`
    /// instead of trying to allocate. Use this as an explicit guardrail
    /// when running on memory-constrained machines or when you suspect
    /// your dataset has outgrown the in-memory writer.
    /// `0` disables the check (default behavior). Recommended values:
    ///   *  16 * (1ull << 30)  // 16 GB  — safe on 32 GB workstations
    ///   *  64 * (1ull << 30)  // 64 GB  — high-RAM training rigs
    /// Once the StreamingWriter exists, use that instead of raising the cap.
    std::uint64_t  max_in_memory_bytes = 0;
};

// ─── Writer ──────────────────────────────────────────────────────────────────

class Writer {
public:
    Writer();
    explicit Writer(WriterConfig cfg);

    Writer(const Writer&)            = delete;
    Writer& operator=(const Writer&) = delete;
    Writer(Writer&&) noexcept            = default;
    Writer& operator=(Writer&&) noexcept = default;

    /// Append one position. Throws WriteError on invalid input.
    void add(const InputPosition& pos);

    /// Equivalent overload for callers without an InputPosition struct.
    void add(std::int32_t score_cp, std::int32_t wdl,
             std::uint8_t stm, std::uint8_t piece_count,
             std::span<const std::uint16_t> features);

    [[nodiscard]] std::uint64_t num_positions() const noexcept {
        return m_flags.size();
    }

    [[nodiscard]] const WriterConfig& config() const noexcept { return m_cfg; }

    /// Compute the canonical layout, serialize the header + arrays,
    /// optionally re-validate, then write the file to `path`.
    /// After finalize() returns, the writer is empty (num_positions == 0)
    /// and may be reused for another file.
    /// Throws WriteError on any failure (disk error, layout overflow,
    /// validation failure).
    void finalize(const std::filesystem::path& path);

private:
    WriterConfig m_cfg;

    // Per-position arrays (one entry per add() call).
    std::vector<std::uint8_t>  m_flags;
    std::vector<std::int16_t>  m_eval;
    std::vector<std::int8_t>   m_wdl;
    std::vector<std::uint16_t> m_per_pos_nnz;  // features.size() per position

    // Concatenated feature ids in add() order.
    std::vector<std::uint16_t> m_w_flat;
};

// ─── One-shot helper ─────────────────────────────────────────────────────────

/// Convenience: construct a Writer, add all positions, finalize.
void write_cnnp_file(const std::filesystem::path& path,
                     std::span<const InputPosition> positions,
                     const WriterConfig& cfg = {});

}  // namespace cnnp
