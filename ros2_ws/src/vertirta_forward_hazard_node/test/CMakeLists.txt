# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.22)
project(vertirta_forward_hazard_node_tests VERSION 1.0.0 LANGUAGES C CXX)

option(VERTIRTA_ENABLE_ASAN_UBSAN "Enable ASan + UBSan" OFF)

enable_testing()
find_package(Catch2 3 REQUIRED)
find_package(Threads REQUIRED)

get_filename_component(VERTIRTA_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)

if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_DAQP
   AND EXISTS "${VERTIRTA_REPO_ROOT}/build/dev-linux-x64/_deps/daqp-src/CMakeLists.txt")
  set(FETCHCONTENT_SOURCE_DIR_DAQP "${VERTIRTA_REPO_ROOT}/build/dev-linux-x64/_deps/daqp-src" CACHE PATH "" FORCE)
endif()

set(VERTIRTA_ENABLE_STRICT_FP ON CACHE BOOL "" FORCE)
set(VERTIRTA_STRICT_FP_FLAGS
  -fno-fast-math
  -ffp-contract=off
  -fno-associative-math
  -fno-reciprocal-math
  -frounding-math
)

if(VERTIRTA_ENABLE_ASAN_UBSAN)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_subdirectory("${VERTIRTA_REPO_ROOT}/core/flight_core" "${CMAKE_CURRENT_BINARY_DIR}/flight_core" EXCLUDE_FROM_ALL)
add_subdirectory("${VERTIRTA_REPO_ROOT}/core/rta_companion" "${CMAKE_CURRENT_BINARY_DIR}/rta_companion" EXCLUDE_FROM_ALL)

add_executable(test_forward_hazard_node
  test_forward_hazard_node.cpp
)
target_include_directories(test_forward_hazard_node PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)
target_link_libraries(test_forward_hazard_node PRIVATE
  vertirta::rta_companion
  vertirta::flight_core
  Catch2::Catch2WithMain
  Threads::Threads
)
target_compile_features(test_forward_hazard_node PRIVATE cxx_std_20)
target_compile_options(test_forward_hazard_node PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow
  ${VERTIRTA_STRICT_FP_FLAGS}
)

add_test(NAME test_forward_hazard_node COMMAND test_forward_hazard_node)
