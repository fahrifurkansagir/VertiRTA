# SPDX-License-Identifier: Apache-2.0
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.25)
  project(VertiRTAForensicsTests VERSION 1.0.0 LANGUAGES C CXX)
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
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../core/flight_core"
    "${CMAKE_CURRENT_BINARY_DIR}/_flight_core")
endif()
if(NOT TARGET vertirta::forensics)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../core/forensics"
    "${CMAKE_CURRENT_BINARY_DIR}/_forensics")
endif()

add_executable(test_forensics_sidecar_ercs_e5x
  test_forensics_sidecar_ercs_e5x.cpp
)
target_link_libraries(test_forensics_sidecar_ercs_e5x PRIVATE
  vertirta::forensics
  Catch2::Catch2WithMain
)
target_compile_features(test_forensics_sidecar_ercs_e5x PRIVATE cxx_std_20)
target_compile_options(test_forensics_sidecar_ercs_e5x PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow
)
add_test(NAME test_forensics_sidecar_ercs_e5x COMMAND test_forensics_sidecar_ercs_e5x)
