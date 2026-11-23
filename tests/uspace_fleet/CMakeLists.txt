# SPDX-License-Identifier: Apache-2.0
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.25)
  project(VertiRTAUspaceFleetTests VERSION 3.5.0 LANGUAGES C CXX)
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
if(NOT TARGET vertirta::rta_companion)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../core/rta_companion"
    "${CMAKE_CURRENT_BINARY_DIR}/_rta_companion")
endif()
if(NOT TARGET vertirta::uspace_fleet)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../core/uspace_fleet"
    "${CMAKE_CURRENT_BINARY_DIR}/_uspace_fleet")
endif()

foreach(_uf_test
    test_vertiport_thermal_queue
    test_part_is_security
    test_cbf_orca_and_fleet_sim)
  add_executable(${_uf_test} "${_uf_test}.cpp")
  target_link_libraries(${_uf_test} PRIVATE
    vertirta::uspace_fleet
    Catch2::Catch2WithMain
  )
  target_compile_features(${_uf_test} PRIVATE cxx_std_20)
  target_compile_options(${_uf_test} PRIVATE
    -Wall -Wextra -Wpedantic -Werror -Wshadow
  )
  add_test(NAME ${_uf_test} COMMAND ${_uf_test})
endforeach()
