# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.22)
project(vertirta_voice_node_tests VERSION 4.0.0 LANGUAGES C CXX)

option(VERTIRTA_ENABLE_ASAN_UBSAN "Enable ASan + UBSan" OFF)
option(VERTIRTA_ENABLE_TSAN "Enable ThreadSanitizer" OFF)

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
elseif(VERTIRTA_ENABLE_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

add_subdirectory("${VERTIRTA_REPO_ROOT}/core/flight_core" "${CMAKE_CURRENT_BINARY_DIR}/flight_core" EXCLUDE_FROM_ALL)
add_subdirectory("${VERTIRTA_REPO_ROOT}/core/rta_companion" "${CMAKE_CURRENT_BINARY_DIR}/rta_companion" EXCLUDE_FROM_ALL)
add_subdirectory("${VERTIRTA_REPO_ROOT}/core/stl" "${CMAKE_CURRENT_BINARY_DIR}/stl" EXCLUDE_FROM_ALL)
add_subdirectory("${VERTIRTA_REPO_ROOT}/core/copilot" "${CMAKE_CURRENT_BINARY_DIR}/copilot" EXCLUDE_FROM_ALL)
target_include_directories(vertirta_stl_monitor PUBLIC "${VERTIRTA_REPO_ROOT}/include")
target_include_directories(vertirta_copilot PUBLIC "${VERTIRTA_REPO_ROOT}/include")

add_library(vertirta_voice_node_core_lib STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/voice_node_core.cpp"
)
target_include_directories(vertirta_voice_node_core_lib PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  "${VERTIRTA_REPO_ROOT}/include"
)
target_link_libraries(vertirta_voice_node_core_lib PUBLIC
  vertirta::copilot
  vertirta::stl_monitor
  vertirta::rta_companion
  vertirta::flight_core
  Threads::Threads
)
target_compile_features(vertirta_voice_node_core_lib PUBLIC cxx_std_20)
target_compile_options(vertirta_voice_node_core_lib PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow
  ${VERTIRTA_STRICT_FP_FLAGS}
)

add_executable(test_voice_node_core
  test_voice_node_core.cpp
)
target_link_libraries(test_voice_node_core PRIVATE
  vertirta_voice_node_core_lib
  Catch2::Catch2WithMain
  Threads::Threads
)
target_compile_features(test_voice_node_core PRIVATE cxx_std_20)
target_compile_options(test_voice_node_core PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow
  ${VERTIRTA_STRICT_FP_FLAGS}
)

add_test(NAME test_voice_node_core COMMAND test_voice_node_core)
