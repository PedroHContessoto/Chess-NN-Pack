// SPDX-License-Identifier: MIT
//
// cnnp/reader.hpp — read-only random-access view over a CNNP V2 file.
//
// `Reader` memory-maps the file (zero-copy), parses & validates the
// header, caches typed pointers into each array, and provides O(1)
// per-position access via `at(i)`.
//
// Two factories:
//   * `Reader::open(path)`           — full validation (per-array scans).
//                                       Use for untrusted input.
//   * `Reader::open_unchecked(path)` — header-only validation. Skips
//                                       per-array scans; alignment and
//                                       offset bounds are still verified
//                                       so pointer casts remain safe.
//
// Reader is move-only and owns the underlying MMapRegion.

#pragma once

#include "cnnp/header.hpp"
#include "cnnp/mmap_region.hpp"

#include <cstdint>
#include <filesystem>
#include <span>

namespace cnnp {

/// Non-owning view of a single position's decoded data.
struct PositionView {
    float        eval_normalized;  // i16 / fixed_scale
    std::uint8_t stm;              // 0 = white, 1 = black
    std::uint8_t piece_count;      // 2..32
    std::int8_t  wdl;              // -1, 0, +1
    /// Feature ids active at this position. Length equals piece_count
    /// for HalfP (count_semantics = piece_count == nnz).
    std::span<const std::uint16_t> features;
};

class Reader {
public:
    /// Open and fully validate a CNNP V2 file.
    /// Throws `MmapError`, `ParseError`, or `ValidationError` on failure.
    [[nodiscard]] static Reader open(const std::filesystem::path& path);

    /// Open with header-only validation. Per-array scans are skipped;
    /// alignment, bounds, and the canonical layout are still verified
    /// so subsequent pointer access is well-defined.
    [[nodiscard]] static Reader open_unchecked(const std::filesystem::path& path);

    Reader(Reader&&) noexcept            = default;
    Reader& operator=(Reader&&) noexcept = default;

    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    // ─── Header / counts ──────────────────────────────────────────────────────
    [[nodiscard]] const Header& header() const noexcept { return m_header; }
    [[nodiscard]] std::uint64_t num_positions() const noexcept {
        return m_header.num_positions;
    }
    [[nodiscard]] std::uint32_t num_blocks() const noexcept {
        return m_header.num_blocks;
    }

    /// O(1) view of position `i`. Throws `std::out_of_range` if
    /// `i >= num_positions()`.
    [[nodiscard]] PositionView at(std::uint64_t i) const;

    // ─── Bulk array accessors (zero-copy spans into the mmap) ─────────────────
    [[nodiscard]] std::span<const std::uint8_t>  flags_array()   const noexcept;
    [[nodiscard]] std::span<const std::int16_t>  eval_array()    const noexcept;
    [[nodiscard]] std::span<const std::int8_t>   wdl_array()     const noexcept;
    [[nodiscard]] std::span<const std::uint64_t> block_anchors() const noexcept;
    [[nodiscard]] std::span<const std::uint16_t> block_prefix()  const noexcept;
    [[nodiscard]] std::span<const std::uint16_t> w_flat()        const noexcept;
    [[nodiscard]] std::span<const std::byte>     metadata()      const noexcept;

private:
    Reader(MMapRegion region, Header h);

    MMapRegion m_region;
    Header     m_header;

    // Cached typed pointers — populated by the constructor after the
    // header has been validated for alignment and bounds.
    const std::uint8_t*  m_flags    = nullptr;
    const std::int16_t*  m_eval     = nullptr;
    const std::int8_t*   m_wdl      = nullptr;
    const std::uint64_t* m_anchors  = nullptr;
    const std::uint16_t* m_prefix   = nullptr;
    const std::uint16_t* m_w_flat   = nullptr;
    const std::byte*     m_metadata = nullptr;
};

}  // namespace cnnp
