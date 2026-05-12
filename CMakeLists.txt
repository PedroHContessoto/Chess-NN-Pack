cmake_minimum_required(VERSION 3.20)

project(cnnp
    VERSION 0.1.0
    DESCRIPTION "Chess-NN-Pack: binary format for NNUE training data"
    LANGUAGES CXX
)

# ─── Language settings ────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type for single-config generators
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ─── Options ──────────────────────────────────────────────────────────────────
option(CNNP_BUILD_TESTS    "Build unit tests"           ON)
option(CNNP_BUILD_CLI      "Build cnnp CLI tool"        ON)   # Phase 6
option(CNNP_BUILD_EXAMPLES "Build example programs"     ON)
option(CNNP_BUILD_PYTHON   "Build Python bindings"      OFF)  # Phase 8
option(CNNP_BUILD_BINPACK_CONVERTER
       "Build the binpack→cnnp converter (vendors nnue-pytorch GPL-3.0)" OFF)
option(CNNP_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# ─── Compiler warnings (sane defaults) ────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CNNP_WARN_FLAGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion
                        -Wsign-conversion -Wold-style-cast -Wnon-virtual-dtor
                        -Wcast-align -Wunused -Woverloaded-virtual
                        -Wnull-dereference -Wdouble-promotion -Wformat=2)
elseif(MSVC)
    set(CNNP_WARN_FLAGS /W4 /permissive-)
endif()

if(CNNP_WARNINGS_AS_ERRORS)
    if(MSVC)
        list(APPEND CNNP_WARN_FLAGS /WX)
    else()
        list(APPEND CNNP_WARN_FLAGS -Werror)
    endif()
endif()

# ─── Core library: cnnp ───────────────────────────────────────────────────────
add_library(cnnp STATIC
    src/header.cpp
    src/encoding.cpp
    src/validator.cpp
    src/mmap_region.cpp
    src/reader.cpp
    src/writer.cpp
)

target_include_directories(cnnp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(cnnp PUBLIC cxx_std_20)
target_compile_options(cnnp PRIVATE ${CNNP_WARN_FLAGS})

# Static lib that may be linked into shared libs (Python module). Harmless
# on Windows; required by GCC/Clang on Linux/macOS.
set_target_properties(cnnp PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 64-bit process requirement (per spec §12)
if(CMAKE_SIZEOF_VOID_P LESS 8)
    message(FATAL_ERROR "CNNP requires a 64-bit build. Detected pointer size: ${CMAKE_SIZEOF_VOID_P} bytes")
endif()

# ─── Tests ────────────────────────────────────────────────────────────────────
if(CNNP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ─── CLI tool ─────────────────────────────────────────────────────────────────
if(CNNP_BUILD_CLI)
    # Pull nlohmann/json (single header, MIT) for real JSON validation in
    # `cnnp validate`. Scoped to the CLI target so the core library stays
    # dependency-free per spec §12.
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
        GIT_SHALLOW    TRUE
    )
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(nlohmann_json)

    add_executable(cnnp_cli src/cli/cnnp_main.cpp)
    target_link_libraries(cnnp_cli PRIVATE cnnp nlohmann_json::nlohmann_json)
    target_compile_options(cnnp_cli PRIVATE ${CNNP_WARN_FLAGS})
    # Output binary is `cnnp` (target name kept distinct from library target).
    set_target_properties(cnnp_cli PROPERTIES OUTPUT_NAME cnnp)
endif()

# ─── Examples ─────────────────────────────────────────────────────────────────
if(CNNP_BUILD_EXAMPLES)
    add_executable(make_sample examples/make_sample.cpp)
    target_link_libraries(make_sample PRIVATE cnnp)
    target_compile_options(make_sample PRIVATE ${CNNP_WARN_FLAGS})
endif()

# ─── Binpack converter (GPL-3.0 — isolated tool) ─────────────────────────────
# Vendors nnue-pytorch's data loader headers via FetchContent. Source files
# are downloaded at build time into build/_deps/ (NOT committed to this repo);
# the resulting binary `binpack_to_cnnp` is GPL-3.0. Everything else
# (cnnp.lib, cnnp.exe, cnnp Python module) stays MIT.
#
# See tools/binpack_to_cnnp/README.md for the full licensing rationale.
if(CNNP_BUILD_BINPACK_CONVERTER)
    include(FetchContent)
    FetchContent_Declare(
        nnue_pytorch
        GIT_REPOSITORY https://github.com/official-stockfish/nnue-pytorch.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
        # Skip add_subdirectory — we only want the header files, not their build.
        SOURCE_SUBDIR  this_subdir_does_not_exist_on_purpose
    )
    FetchContent_MakeAvailable(nnue_pytorch)

    set(CNNP_BINPACK_VENDOR_DIR
        ${nnue_pytorch_SOURCE_DIR}/data_loader/cpp/lib)

    add_executable(binpack_to_cnnp
        tools/binpack_to_cnnp/binpack_to_cnnp.cpp)
    target_link_libraries(binpack_to_cnnp PRIVATE cnnp)
    target_include_directories(binpack_to_cnnp PRIVATE
        ${CNNP_BINPACK_VENDOR_DIR})
    target_compile_features(binpack_to_cnnp PRIVATE cxx_std_20)

    # nnue-pytorch headers are large and use compiler intrinsics; keep our
    # strict warning flags off this target to avoid noise from third-party code.
    if(MSVC)
        target_compile_options(binpack_to_cnnp PRIVATE /W2 /wd4244 /wd4267)
    else()
        target_compile_options(binpack_to_cnnp PRIVATE -w)
    endif()
endif()

# ─── Python bindings (pybind11) ───────────────────────────────────────────────
if(CNNP_BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(cnnp_python python/cnnp_py.cpp)
    target_link_libraries(cnnp_python PRIVATE cnnp)
    target_compile_options(cnnp_python PRIVATE ${CNNP_WARN_FLAGS})
    # Resulting file: cnnp.cpython-XYZ-<arch>.<ext>; importable as `import cnnp`.
    set_target_properties(cnnp_python PROPERTIES OUTPUT_NAME cnnp)

    # Place the .pyd / .so into build/python/ for easy PYTHONPATH setup.
    # ARCHIVE_OUTPUT_DIRECTORY for the .lib import library too, otherwise
    # MSVC tries to put it next to the static cnnp.lib and fails LNK1149.
    set_target_properties(cnnp_python PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY         ${CMAKE_BINARY_DIR}/python
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/python
        LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_BINARY_DIR}/python
        RUNTIME_OUTPUT_DIRECTORY         ${CMAKE_BINARY_DIR}/python
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/python
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_BINARY_DIR}/python
        ARCHIVE_OUTPUT_DIRECTORY         ${CMAKE_BINARY_DIR}/python
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/python
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_BINARY_DIR}/python
    )
endif()

# ─── Status banner ────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "─── Chess-NN-Pack v${PROJECT_VERSION} ───")
message(STATUS "  C++ standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler:        ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Tests:           ${CNNP_BUILD_TESTS}")
message(STATUS "  CLI:             ${CNNP_BUILD_CLI}")
message(STATUS "  Examples:        ${CNNP_BUILD_EXAMPLES}")
message(STATUS "  Python bindings: ${CNNP_BUILD_PYTHON}")
if(CNNP_BUILD_PYTHON)
    message(STATUS "  Python module → ${CMAKE_BINARY_DIR}/python")
endif()
message(STATUS "  Binpack conv:    ${CNNP_BUILD_BINPACK_CONVERTER} (GPL-3.0 isolated)")
message(STATUS "")
