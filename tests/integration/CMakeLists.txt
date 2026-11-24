# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.25)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(VertiRTAIntegrationTests VERSION 4.0.0 LANGUAGES C CXX)
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
endif()

option(VERTIRTA_ENABLE_TSAN "Build integration concurrency tests with ThreadSanitizer" OFF)

find_package(Threads REQUIRED)
get_filename_component(_vertirta_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

if(NOT TARGET Catch2::Catch2WithMain)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS 3
  )
  FetchContent_MakeAvailable(Catch2)
endif()

# ---------------------------------------------------------------------------
# 1. Lock-Free 4-Thread SPSC + Seqlock Concurrency & TSan Stress Suite
# ---------------------------------------------------------------------------
add_executable(test_spsc_tsan
  test_spsc_tsan.cpp
)
target_include_directories(test_spsc_tsan PRIVATE
  "${_vertirta_repo_root}/ros2_ws/src/vertirta_rta_node/include"
)
target_link_libraries(test_spsc_tsan PRIVATE
  Catch2::Catch2WithMain
  Threads::Threads
)
target_compile_features(test_spsc_tsan PRIVATE cxx_std_20)
target_compile_options(test_spsc_tsan PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror
  -fno-fast-math
  -ffp-contract=off
)
if(VERTIRTA_ENABLE_TSAN)
  target_compile_options(test_spsc_tsan PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
  target_link_options(test_spsc_tsan PRIVATE -fsanitize=thread)
endif()
add_test(NAME test_spsc_tsan COMMAND test_spsc_tsan)

# ---------------------------------------------------------------------------
# 2. Full-Stack 5-Scenario Istanbul Compound-Fault Mission Suite
#    Built unconditionally in unified root builds (where vertirta::compound_fault_suite
#    already exists) and in standalone non-TSan builds. Skipped by default in standalone
#    VERTIRTA_ENABLE_TSAN=ON quickstart builds to keep TSan fast and free of libz3 noise.
# ---------------------------------------------------------------------------
set(_vertirta_build_compound_fault ON)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND VERTIRTA_ENABLE_TSAN)
  set(_vertirta_build_compound_fault OFF)
endif()

if(_vertirta_build_compound_fault)
  if(NOT TARGET vertirta::compound_fault_suite)
    if(NOT TARGET vertirta::flight_core)
      foreach(_dep_dir
          core/flight_core
          core/rta_companion
          core/airframes/lift_cruise
          core/wind_acoustics
          core/uspace_fleet
          core/forward_hazard
          sim/gazebo_plugins/mmwave_77ghz
          core/gnss_denied
          core/voice
          core/forensics)
        string(REPLACE "/" "_" _dep_bin "${_dep_dir}")
        add_subdirectory("${_vertirta_repo_root}/${_dep_dir}" "${CMAKE_BINARY_DIR}/deps/${_dep_bin}")
      endforeach()
    endif()
    add_subdirectory(
      "${_vertirta_repo_root}/benchmarks/compound_fault_suite"
      "${CMAKE_BINARY_DIR}/deps/benchmarks_compound_fault_suite"
    )
  endif()

  add_executable(test_compound_fault_istanbul_mission
    test_compound_fault_istanbul_mission.cpp
  )
  target_compile_features(test_compound_fault_istanbul_mission PRIVATE cxx_std_20)
  target_compile_options(test_compound_fault_istanbul_mission PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  if(VERTIRTA_ENABLE_STRICT_FP)
    target_compile_options(test_compound_fault_istanbul_mission PRIVATE ${VERTIRTA_STRICT_FP_FLAGS})
  endif()
  target_link_libraries(test_compound_fault_istanbul_mission PRIVATE
    vertirta::compound_fault_suite
    Catch2::Catch2WithMain
  )
  add_test(NAME test_compound_fault_istanbul_mission COMMAND test_compound_fault_istanbul_mission)
endif()
