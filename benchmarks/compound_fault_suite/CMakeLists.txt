# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.25)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(VertiRTACompoundFaultSuite VERSION 4.0.0 LANGUAGES C CXX)
  option(VERTIRTA_ENABLE_STRICT_FP "Enforce IEEE-754 deterministic FP flags" ON)
  set(VERTIRTA_STRICT_FP_FLAGS
    -fno-fast-math
    -ffp-contract=off
    -fno-associative-math
    -fno-reciprocal-math
    -frounding-math
  )
  get_filename_component(_vertirta_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
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

add_library(vertirta_compound_fault_suite STATIC
  src/compound_fault_runner.cpp
)
add_library(vertirta::compound_fault_suite ALIAS vertirta_compound_fault_suite)

target_include_directories(vertirta_compound_fault_suite
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(vertirta_compound_fault_suite PUBLIC cxx_std_20)
target_compile_options(vertirta_compound_fault_suite PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
if(VERTIRTA_ENABLE_STRICT_FP)
  target_compile_options(vertirta_compound_fault_suite PRIVATE ${VERTIRTA_STRICT_FP_FLAGS})
endif()

target_link_libraries(vertirta_compound_fault_suite
  PUBLIC
    vertirta::flight_core
    vertirta::rta_companion
    vertirta::airframes_lc
    vertirta::wind_acoustics
    vertirta::uspace_fleet
    vertirta::forward_hazard
    vertirta::mmwave_77ghz_physics
    vertirta::gnss_denied
    vertirta::voice
    vertirta::forensics
)

add_executable(vertirta_compound_fault_cli
  src/main.cpp
)
target_compile_features(vertirta_compound_fault_cli PRIVATE cxx_std_20)
target_compile_options(vertirta_compound_fault_cli PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
if(VERTIRTA_ENABLE_STRICT_FP)
  target_compile_options(vertirta_compound_fault_cli PRIVATE ${VERTIRTA_STRICT_FP_FLAGS})
endif()
target_link_libraries(vertirta_compound_fault_cli PRIVATE vertirta::compound_fault_suite)
