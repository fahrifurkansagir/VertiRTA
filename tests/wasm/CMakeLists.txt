cmake_minimum_required(VERSION 3.22)
project(vertirta_wasm_host_tests LANGUAGES C CXX)

include(CTest)

set(VERTIRTA_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH "VertiRTA repository root")

if(EXISTS "${VERTIRTA_REPO_ROOT}/build/dev-linux-x64/_deps/daqp-src/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_DAQP)
  set(FETCHCONTENT_SOURCE_DIR_DAQP "${VERTIRTA_REPO_ROOT}/build/dev-linux-x64/_deps/daqp-src" CACHE PATH "Reuse cached DAQP v0.10.0 source")
endif()

add_subdirectory("${VERTIRTA_REPO_ROOT}/wasm" "${CMAKE_CURRENT_BINARY_DIR}/wasm")

find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
  include(FetchContent)
  if(EXISTS "${VERTIRTA_REPO_ROOT}/build/dev-linux-x64/_deps/catch2-src/CMakeLists.txt")
    set(FETCHCONTENT_SOURCE_DIR_CATCH2 "${VERTIRTA_REPO_ROOT}/build/dev-linux-x64/_deps/catch2-src" CACHE PATH "")
  endif()
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(Catch2)
endif()

add_executable(test_wasm_abi_host test_wasm_abi_host.cpp)
target_link_libraries(test_wasm_abi_host PRIVATE vertirta_wasm_abi Catch2::Catch2WithMain)
target_compile_features(test_wasm_abi_host PRIVATE cxx_std_17)
target_compile_options(test_wasm_abi_host PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_test(NAME test_wasm_abi_host COMMAND test_wasm_abi_host)
