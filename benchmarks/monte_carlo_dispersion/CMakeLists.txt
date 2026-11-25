# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.25)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(VertiRTAMonteCarloDispersion VERSION 4.0.0 LANGUAGES C CXX)
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
      core/gnss_denied
      core/forensics)
    string(REPLACE "/" "_" _dep_bin "${_dep_dir}")
    add_subdirectory("${_vertirta_repo_root}/${_dep_dir}" "${CMAKE_BINARY_DIR}/deps/${_dep_bin}")
  endforeach()
endif()

find_package(Threads REQUIRED)

add_library(vertirta_monte_carlo_dispersion STATIC
  src/mc_1000_runner.cpp
)
add_library(vertirta::monte_carlo_dispersion ALIAS vertirta_monte_carlo_dispersion)

target_include_directories(vertirta_monte_carlo_dispersion
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(vertirta_monte_carlo_dispersion PUBLIC cxx_std_20)
target_compile_options(vertirta_monte_carlo_dispersion PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
if(VERTIRTA_ENABLE_STRICT_FP)
  target_compile_options(vertirta_monte_carlo_dispersion PRIVATE ${VERTIRTA_STRICT_FP_FLAGS})
endif()

target_link_libraries(vertirta_monte_carlo_dispersion
  PUBLIC
    vertirta::flight_core
    vertirta::rta_companion
    vertirta::airframes_lc
    vertirta::wind_acoustics
    vertirta::gnss_denied
    vertirta::forensics
    Threads::Threads
)

add_executable(vertirta_mc_1000_cli
  src/main.cpp
)
target_compile_features(vertirta_mc_1000_cli PRIVATE cxx_std_20)
target_compile_options(vertirta_mc_1000_cli PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
if(VERTIRTA_ENABLE_STRICT_FP)
  target_compile_options(vertirta_mc_1000_cli PRIVATE ${VERTIRTA_STRICT_FP_FLAGS})
endif()
target_link_libraries(vertirta_mc_1000_cli PRIVATE vertirta::monte_carlo_dispersion)
