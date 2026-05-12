// SPDX-License-Identifier: MIT
//
// cnnp/mmap_region.hpp — RAII wrapper for memory-mapped read-only files.
//
// Provides a portable view of a file's contents as `std::span<const std::byte>`
// using POSIX mmap(2) on Unix and MapViewOfFile on Windows. Move-only.
//
// On any failure (open, stat/size, map) throws `MmapError`.
//
// Lifetime: the mapped region remains valid until the MMapRegion is
// destroyed or moved-from. The underlying file descriptor / handles are
// owned by the region.

#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>

namespace cnnp {

// ─── Errors ───────────────────────────────────────────────────────────────────

class MmapError : public std::runtime_error {
public:
    explicit MmapError(const std::string& what) : std::runtime_error(what) {}
};

// ─── Mapped region ────────────────────────────────────────────────────────────

class MMapRegion {
public:
    /// Open `path` and map its entire contents read-only.
    /// Throws `MmapError` on failure to open, stat, or map.
    /// An empty file is allowed: data() returns an empty span.
    explicit MMapRegion(const std::filesystem::path& path);

    ~MMapRegion();

    MMapRegion(MMapRegion&& other) noexcept;
    MMapRegion& operator=(MMapRegion&& other) noexcept;

    MMapRegion(const MMapRegion&)            = delete;
    MMapRegion& operator=(const MMapRegion&) = delete;

    [[nodiscard]] std::span<const std::byte> data() const noexcept {
        return std::span<const std::byte>(m_data, m_size);
    }

    [[nodiscard]] std::size_t size()  const noexcept { return m_size; }
    [[nodiscard]] bool        empty() const noexcept { return m_size == 0; }

private:
    void close() noexcept;

    const std::byte* m_data = nullptr;
    std::size_t      m_size = 0;

#ifdef _WIN32
    // HANDLE is `void*` on Windows; storing as void* avoids leaking
    // <windows.h> into this public header.
    void* m_file_handle    = nullptr;
    void* m_mapping_handle = nullptr;
#else
    int   m_fd = -1;
#endif
};

}  // namespace cnnp
