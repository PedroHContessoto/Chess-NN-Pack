// SPDX-License-Identifier: MIT
//
// cnnp/reader.cpp — implementation of Reader.

#include "cnnp/reader.hpp"

#include "cnnp/encoding.hpp"
#include "cnnp/validator.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace cnnp {

namespace {

template <typename T>
const T* typed_ptr_at(const std::byte* base, std::uint64_t offset) noexcept {
    // Alignment is enforced by validate_header_consistency before we get
    // here; mmap base is page-aligned (≥4 KiB) on every supported OS, so
    // the resulting pointer is well-aligned for T.
    return reinterpret_cast<const T*>(base + offset);
}

}  // anonymous namespace

Reader::Reader(MMapRegion region, Header h)
    : m_region(std::move(region))
    , m_header(std::move(h))
{
    const std::byte* base = m_region.data().data();
    m_flags    = typed_ptr_at<std::uint8_t> (base, m_header.flags_offset);
    m_eval     = typed_ptr_at<std::int16_t> (base, m_header.eval_offset);
    m_wdl      = typed_ptr_at<std::int8_t>  (base, m_header.wdl_offset);
    m_anchors  = typed_ptr_at<std::uint64_t>(base, m_header.block_anchors_offset);
    m_prefix   = typed_ptr_at<std::uint16_t>(base, m_header.block_prefix_offset);
    m_w_flat   = typed_ptr_at<std::uint16_t>(base, m_header.w_flat_offset);
    m_metadata = base + m_header.metadata_offset;
}

Reader Reader::open(const std::filesystem::path& path) {
    MMapRegion region(path);
    auto bytes = region.data();
    Header h = parse_header(bytes);
    validate_full(h, bytes);
    return Reader(std::move(region), std::move(h));
}

Reader Reader::open_unchecked(const std::filesystem::path& path) {
    MMapRegion region(path);
    auto bytes = region.data();
    Header h = parse_header(bytes);
    // Header-only validation guarantees alignment, bounds, and the
    // canonical layout — the minimum needed for safe pointer casts.
    validate_header_consistency(h, static_cast<std::uint64_t>(bytes.size()));
    return Reader(std::move(region), std::move(h));
}

PositionView Reader::at(std::uint64_t i) const {
    if (i >= m_header.num_positions) {
        throw std::out_of_range("Reader::at: index " + std::to_string(i) +
                                " >= num_positions " +
                                std::to_string(m_header.num_positions));
    }

    const std::uint64_t bsz          = m_header.block_size;
    const std::uint64_t block_idx    = i / bsz;
    const std::uint64_t pos_in_block = i % bsz;
    const std::uint64_t prefix_base  = block_idx * (bsz + 1ull);

    const std::uint64_t anchor = m_anchors[block_idx];
    const std::uint64_t start  = anchor + m_prefix[prefix_base + pos_in_block];
    const std::uint64_t end    = anchor + m_prefix[prefix_base + pos_in_block + 1];

    const DecodedFlags df = decode_flags(m_flags[i]);

    PositionView view{};
    view.eval_normalized = decode_eval(m_eval[i], m_header.fixed_scale);
    view.stm             = df.stm;
    view.piece_count     = df.piece_count;
    view.wdl             = m_wdl[i];
    view.features        = std::span<const std::uint16_t>(
        m_w_flat + start, static_cast<std::size_t>(end - start));
    return view;
}

// ─── Bulk accessors ──────────────────────────────────────────────────────────

std::span<const std::uint8_t> Reader::flags_array() const noexcept {
    return {m_flags, static_cast<std::size_t>(m_header.num_positions)};
}

std::span<const std::int16_t> Reader::eval_array() const noexcept {
    return {m_eval, static_cast<std::size_t>(m_header.num_positions)};
}

std::span<const std::int8_t> Reader::wdl_array() const noexcept {
    return {m_wdl, static_cast<std::size_t>(m_header.num_positions)};
}

std::span<const std::uint64_t> Reader::block_anchors() const noexcept {
    return {m_anchors, static_cast<std::size_t>(m_header.num_blocks)};
}

std::span<const std::uint16_t> Reader::block_prefix() const noexcept {
    const std::size_t total = static_cast<std::size_t>(m_header.num_blocks) *
                              static_cast<std::size_t>(m_header.block_size + 1u);
    return {m_prefix, total};
}

std::span<const std::uint16_t> Reader::w_flat() const noexcept {
    return {m_w_flat, static_cast<std::size_t>(m_header.num_features_total)};
}

std::span<const std::byte> Reader::metadata() const noexcept {
    return {m_metadata, static_cast<std::size_t>(m_header.metadata_length)};
}

}  // namespace cnnp
