# Fetch GoogleTest at configure time. Pinned to a known stable release.
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)

# Avoid conflicts with parent project options on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Disable warnings-as-errors for vendored gtest sources.
if(TARGET gtest)
    if(MSVC)
        target_compile_options(gtest      PRIVATE /W0)
        target_compile_options(gtest_main PRIVATE /W0)
    else()
        target_compile_options(gtest      PRIVATE -w)
        target_compile_options(gtest_main PRIVATE -w)
    endif()
endif()

# ─── Unit test binaries ───────────────────────────────────────────────────────

add_executable(test_header test_header.cpp)
target_link_libraries(test_header PRIVATE cnnp gtest_main)
target_compile_options(test_header PRIVATE ${CNNP_WARN_FLAGS})
add_test(NAME test_header COMMAND test_header)

add_executable(test_encoding test_encoding.cpp)
target_link_libraries(test_encoding PRIVATE cnnp gtest_main)
target_compile_options(test_encoding PRIVATE ${CNNP_WARN_FLAGS})
add_test(NAME test_encoding COMMAND test_encoding)

add_executable(test_validator test_validator.cpp)
target_link_libraries(test_validator PRIVATE cnnp gtest_main)
target_compile_options(test_validator PRIVATE ${CNNP_WARN_FLAGS})
add_test(NAME test_validator COMMAND test_validator)

add_executable(test_reader test_reader.cpp)
target_link_libraries(test_reader PRIVATE cnnp gtest_main)
target_compile_options(test_reader PRIVATE ${CNNP_WARN_FLAGS})
add_test(NAME test_reader COMMAND test_reader)

add_executable(test_writer test_writer.cpp)
target_link_libraries(test_writer PRIVATE cnnp gtest_main)
target_compile_options(test_writer PRIVATE ${CNNP_WARN_FLAGS})
add_test(NAME test_writer COMMAND test_writer)

add_executable(test_golden test_golden.cpp)
target_link_libraries(test_golden PRIVATE cnnp gtest_main)
target_compile_options(test_golden PRIVATE ${CNNP_WARN_FLAGS})
add_test(NAME test_golden COMMAND test_golden)
