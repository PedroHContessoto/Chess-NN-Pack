// SPDX-License-Identifier: MIT
//
// cnnp/mmap_region.cpp — POSIX mmap / Windows MapViewOfFile implementation.

#include "cnnp/mmap_region.hpp"

#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#else
  #include <cerrno>
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

namespace cnnp {

// ─── Windows ─────────────────────────────────────────────────────────────────
#ifdef _WIN32

namespace {

std::string format_winerr(DWORD err) {
    char buf[512] = {0};
    DWORD n = ::FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, err, 0, buf, sizeof(buf), nullptr);
    if (n == 0) return "Windows error " + std::to_string(err);
    while (n > 0 && (buf[n - 1] == '\r' || buf[n - 1] == '\n' || buf[n - 1] == ' ')) {
        buf[--n] = '\0';
    }
    return std::string(buf) + " (code " + std::to_string(err) + ")";
}

}  // anonymous namespace

MMapRegion::MMapRegion(const std::filesystem::path& path) {
    HANDLE file = ::CreateFileW(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        throw MmapError("CreateFileW failed for '" + path.string() + "': " +
                        format_winerr(::GetLastError()));
    }
    m_file_handle = file;

    LARGE_INTEGER sz;
    if (!::GetFileSizeEx(file, &sz)) {
        const DWORD err = ::GetLastError();
        close();
        throw MmapError("GetFileSizeEx failed: " + format_winerr(err));
    }

    // QuadPart is signed LONGLONG; defensively reject pathological values
    // for parity with the POSIX path. Unreachable on a healthy NTFS but
    // documents the invariant.
    if (sz.QuadPart < 0) {
        close();
        throw MmapError("file size is negative (corrupt filesystem?)");
    }
    if (static_cast<std::uint64_t>(sz.QuadPart) >
        std::numeric_limits<std::size_t>::max()) {
        close();
        throw MmapError("file too large for this process address space");
    }

    if (sz.QuadPart == 0) {
        // Empty file: nothing to map. Release the file handle now.
        ::CloseHandle(file);
        m_file_handle = nullptr;
        return;
    }

    m_size = static_cast<std::size_t>(sz.QuadPart);

    HANDLE mapping = ::CreateFileMappingW(
        file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
        const DWORD err = ::GetLastError();
        close();
        throw MmapError("CreateFileMappingW failed: " + format_winerr(err));
    }
    m_mapping_handle = mapping;

    void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        const DWORD err = ::GetLastError();
        close();
        throw MmapError("MapViewOfFile failed: " + format_winerr(err));
    }

    m_data = static_cast<const std::byte*>(view);
}

void MMapRegion::close() noexcept {
    if (m_data) {
        ::UnmapViewOfFile(m_data);
        m_data = nullptr;
    }
    if (m_mapping_handle) {
        ::CloseHandle(static_cast<HANDLE>(m_mapping_handle));
        m_mapping_handle = nullptr;
    }
    if (m_file_handle) {
        ::CloseHandle(static_cast<HANDLE>(m_file_handle));
        m_file_handle = nullptr;
    }
    m_size = 0;
}

// ─── POSIX ───────────────────────────────────────────────────────────────────
#else

MMapRegion::MMapRegion(const std::filesystem::path& path) {
    m_fd = ::open(path.c_str(), O_RDONLY);
    if (m_fd < 0) {
        const int e = errno;
        throw MmapError("open failed for '" + path.string() + "': " +
                        std::strerror(e));
    }

    struct stat st{};
    if (::fstat(m_fd, &st) < 0) {
        const int e = errno;
        close();
        throw MmapError(std::string("fstat failed: ") + std::strerror(e));
    }

    // off_t is signed; defensively reject negative or oversized sizes that
    // would wrap on the cast. Both are unreachable on real x86_64 but
    // documenting the invariant prevents silent UB on weirder targets.
    if (st.st_size < 0) {
        close();
        throw MmapError("file size is negative (corrupt filesystem?)");
    }
    using StSizeUnsigned = std::make_unsigned_t<decltype(st.st_size)>;
    if (static_cast<StSizeUnsigned>(st.st_size) >
        std::numeric_limits<std::size_t>::max()) {
        close();
        throw MmapError("file too large for this process address space");
    }

    if (st.st_size == 0) {
        ::close(m_fd);
        m_fd = -1;
        return;
    }

    m_size = static_cast<std::size_t>(st.st_size);

    void* p = ::mmap(nullptr, m_size, PROT_READ, MAP_PRIVATE, m_fd, 0);
    if (p == MAP_FAILED) {
        const int e = errno;
        close();
        throw MmapError(std::string("mmap failed: ") + std::strerror(e));
    }

    m_data = static_cast<const std::byte*>(p);
}

void MMapRegion::close() noexcept {
    if (m_data) {
        ::munmap(const_cast<std::byte*>(m_data), m_size);
        m_data = nullptr;
    }
    if (m_fd >= 0) {
        ::close(m_fd);
        m_fd = -1;
    }
    m_size = 0;
}

#endif

// ─── Common (dtor + move) ────────────────────────────────────────────────────

MMapRegion::~MMapRegion() {
    close();
}

MMapRegion::MMapRegion(MMapRegion&& other) noexcept
    : m_data(other.m_data)
    , m_size(other.m_size)
#ifdef _WIN32
    , m_file_handle(other.m_file_handle)
    , m_mapping_handle(other.m_mapping_handle)
#else
    , m_fd(other.m_fd)
#endif
{
    other.m_data = nullptr;
    other.m_size = 0;
#ifdef _WIN32
    other.m_file_handle    = nullptr;
    other.m_mapping_handle = nullptr;
#else
    other.m_fd = -1;
#endif
}

MMapRegion& MMapRegion::operator=(MMapRegion&& other) noexcept {
    if (this != &other) {
        close();
        m_data = other.m_data;
        m_size = other.m_size;
#ifdef _WIN32
        m_file_handle    = other.m_file_handle;
        m_mapping_handle = other.m_mapping_handle;
#else
        m_fd = other.m_fd;
#endif
        other.m_data = nullptr;
        other.m_size = 0;
#ifdef _WIN32
        other.m_file_handle    = nullptr;
        other.m_mapping_handle = nullptr;
#else
        other.m_fd = -1;
#endif
    }
    return *this;
}

}  // namespace cnnp
