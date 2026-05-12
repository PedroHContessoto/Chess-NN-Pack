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
option(CNNP_BUILD_PYTHON   "Build Python bindings"      OFF)  # Phase 7+
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
    add_executable(cnnp_cli src/cli/cnnp_main.cpp)
    target_link_libraries(cnnp_cli PRIVATE cnnp)
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
message(STATUS "")
