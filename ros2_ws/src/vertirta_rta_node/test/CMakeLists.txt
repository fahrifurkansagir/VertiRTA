# SPDX-License-Identifier: Apache-2.0
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.25)
  project(VertiRtaNodeWiringTests VERSION 1.0.0 LANGUAGES C CXX)
  include(CTest)
  include(FetchContent)

  option(VERTIRTA_ENABLE_STRICT_FP "Enforce IEEE-754 deterministic FP flags" ON)
  set(VERTIRTA_STRICT_FP_FLAGS
    -fno-fast-math
    -ffp-contract=off
    -fno-associative-math
    -fno-reciprocal-math
    -frounding-math
  )

  if(NOT TARGET Catch2::Catch2WithMain)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.7.1
      GIT_SHALLOW TRUE
      FIND_PACKAGE_ARGS 3
    )
    FetchContent_MakeAvailable(Catch2)
  endif()
endif()

if(NOT TARGET vertirta::flight_core)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../../core/flight_core"
    "${CMAKE_CURRENT_BINARY_DIR}/_flight_core")
endif()
if(NOT TARGET vertirta::rta_companion)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../../core/rta_companion"
    "${CMAKE_CURRENT_BINARY_DIR}/_rta_companion")
endif()

add_executable(test_rta_node_wiring test_rta_node_wiring.cpp)
target_include_directories(test_rta_node_wiring PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../vertirta_px4_bridge/include"
)
target_link_libraries(test_rta_node_wiring PRIVATE
  vertirta::rta_companion
  vertirta::flight_core
  Catch2::Catch2WithMain
)
target_compile_features(test_rta_node_wiring PRIVATE cxx_std_20)
target_compile_options(test_rta_node_wiring PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow
)
add_test(NAME test_rta_node_wiring COMMAND test_rta_node_wiring)
