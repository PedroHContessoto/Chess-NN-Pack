// SPDX-License-Identifier: MIT
//
// cnnp/byte_reader.hpp — endian-safe little-endian primitives.
//
// All CNNP wire data is little-endian (per spec §3). These helpers parse
// fundamental types from a `std::span<const std::byte>` view (typically a
// memory-mapped region) without UB: no reinterpret_cast, no unaligned
// loads, no dependence on host endianness.
//
// `bit_cast` is used ONLY to type-pun u32 ↔ float (IEEE 754 binary32),
// which is always well-defined. It is NEVER used to type-pun a whole
// struct from raw bytes (forbidden by spec §12).
//
// Bounds checking: callers are expected to validate `off + sizeof(T) <=
// data.size()` before calling. The helpers themselves are noexcept and
// trust the caller for hot-path performance; the parser layer above
// enforces bounds.

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

namespace cnnp {

// ─── Readers ──────────────────────────────────────────────────────────────────

[[nodiscard]] constexpr std::uint8_t
read_u8(std::span<const std::byte> data, std::size_t off) noexcept {
    return static_cast<std::uint8_t>(data[off]);
}

[[nodiscard]] constexpr std::uint16_t
read_u16_le(std::span<const std::byte> data, std::size_t off) noexcept {
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(static_cast<std::uint8_t>(data[off + 0])) |
        static_cast<std::uint16_t>(static_cast<std::uint8_t>(data[off + 1])) << 8
    );
}

[[nodiscard]] constexpr std::uint32_t
read_u32_le(std::span<const std::byte> data, std::size_t off) noexcept {
    return static_cast<std::uint32_t>(static_cast<std::uint8_t>(data[off + 0])) |
           static_cast<std::uint32_t>(static_cast<std::uint8_t>(data[off + 1])) << 8 |
           static_cast<std::uint32_t>(static_cast<std::uint8_t>(data[off + 2])) << 16 |
           static_cast<std::uint32_t>(static_cast<std::uint8_t>(data[off + 3])) << 24;
}

[[nodiscard]] constexpr std::uint64_t
read_u64_le(std::span<const std::byte> data, std::size_t off) noexcept {
    return static_cast<std::uint64_t>(static_cast<std::uint8_t>(data[off + 0])) |
           static_cast<std::uint64_t>(static_cast<std::uint8_t>(data[off + 1])) << 8 |
           static_cast<std::uint64_t>(static_cast<std::uint8_t>(data[off + 2])) << 16 |
           static_cast<std::uint64_t>(static_cast<std::uint8_t>(data[off + 3])) << 24 |
           static_cast<std::uint64_t>(static_cast<std::uint8_t>(data[off + 4])) << 32 |
           static_cast<std::uint64_t>(static_cast<std::uint8_t>(data[off + 5])) << 40 |
           static_cast<std::uint64_t>(static_cast<std::uint8_t>(data[off + 6])) << 48 |
           static_cast<std::uint64_t>(static_cast<std::uint8_t>(data[off + 7])) << 56;
}

[[nodiscard]] inline std::int16_t
read_i16_le(std::span<const std::byte> data, std::size_t off) noexcept {
    return static_cast<std::int16_t>(read_u16_le(data, off));
}

[[nodiscard]] inline std::int32_t
read_i32_le(std::span<const std::byte> data, std::size_t off) noexcept {
    return static_cast<std::int32_t>(read_u32_le(data, off));
}

[[nodiscard]] inline std::int64_t
read_i64_le(std::span<const std::byte> data, std::size_t off) noexcept {
    return static_cast<std::int64_t>(read_u64_le(data, off));
}

[[nodiscard]] inline float
read_f32_le(std::span<const std::byte> data, std::size_t off) noexcept {
    return std::bit_cast<float>(read_u32_le(data, off));
}

// ─── Writers ──────────────────────────────────────────────────────────────────

constexpr void
write_u8(std::span<std::byte> data, std::size_t off, std::uint8_t v) noexcept {
    data[off] = std::byte{v};
}

constexpr void
write_u16_le(std::span<std::byte> data, std::size_t off, std::uint16_t v) noexcept {
    data[off + 0] = std::byte{static_cast<std::uint8_t>(v & 0xFFu)};
    data[off + 1] = std::byte{static_cast<std::uint8_t>((v >> 8) & 0xFFu)};
}

constexpr void
write_u32_le(std::span<std::byte> data, std::size_t off, std::uint32_t v) noexcept {
    data[off + 0] = std::byte{static_cast<std::uint8_t>(v & 0xFFu)};
    data[off + 1] = std::byte{static_cast<std::uint8_t>((v >> 8) & 0xFFu)};
    data[off + 2] = std::byte{static_cast<std::uint8_t>((v >> 16) & 0xFFu)};
    data[off + 3] = std::byte{static_cast<std::uint8_t>((v >> 24) & 0xFFu)};
}

constexpr void
write_u64_le(std::span<std::byte> data, std::size_t off, std::uint64_t v) noexcept {
    data[off + 0] = std::byte{static_cast<std::uint8_t>(v & 0xFFu)};
    data[off + 1] = std::byte{static_cast<std::uint8_t>((v >> 8) & 0xFFu)};
    data[off + 2] = std::byte{static_cast<std::uint8_t>((v >> 16) & 0xFFu)};
    data[off + 3] = std::byte{static_cast<std::uint8_t>((v >> 24) & 0xFFu)};
    data[off + 4] = std::byte{static_cast<std::uint8_t>((v >> 32) & 0xFFu)};
    data[off + 5] = std::byte{static_cast<std::uint8_t>((v >> 40) & 0xFFu)};
    data[off + 6] = std::byte{static_cast<std::uint8_t>((v >> 48) & 0xFFu)};
    data[off + 7] = std::byte{static_cast<std::uint8_t>((v >> 56) & 0xFFu)};
}

inline void
write_i16_le(std::span<std::byte> data, std::size_t off, std::int16_t v) noexcept {
    write_u16_le(data, off, static_cast<std::uint16_t>(v));
}

inline void
write_i32_le(std::span<std::byte> data, std::size_t off, std::int32_t v) noexcept {
    write_u32_le(data, off, static_cast<std::uint32_t>(v));
}

inline void
write_i64_le(std::span<std::byte> data, std::size_t off, std::int64_t v) noexcept {
    write_u64_le(data, off, static_cast<std::uint64_t>(v));
}

inline void
write_f32_le(std::span<std::byte> data, std::size_t off, float v) noexcept {
    write_u32_le(data, off, std::bit_cast<std::uint32_t>(v));
}

}  // namespace cnnp
