# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.25)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(VertiRTAMavlinkBridge VERSION 4.0.0 LANGUAGES C CXX)
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
      core/wind_acoustics
      core/gnss_denied)
    string(REPLACE "/" "_" _dep_bin "${_dep_dir}")
    add_subdirectory("${_vertirta_repo_root}/${_dep_dir}" "${CMAKE_BINARY_DIR}/deps/${_dep_bin}")
  endforeach()
endif()

find_package(Threads REQUIRED)

add_library(vertirta_mavlink_rta_bridge STATIC
  src/mavlink_v2_codec.cpp
  src/offboard_rta_shield.cpp
  src/udp_bridge.cpp
)
add_library(vertirta::mavlink_rta_bridge ALIAS vertirta_mavlink_rta_bridge)

target_compile_features(vertirta_mavlink_rta_bridge PUBLIC cxx_std_20)
target_include_directories(vertirta_mavlink_rta_bridge
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(vertirta_mavlink_rta_bridge
  PUBLIC
    vertirta::flight_core
    vertirta::rta_companion
    vertirta::wind_acoustics
    vertirta::gnss_denied
    Threads::Threads
)
target_compile_options(vertirta_mavlink_rta_bridge PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror
  -Wshadow
  -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)

add_executable(vertirta_mavlink_rta_bridge_cli
  src/main.cpp
)
set_target_properties(vertirta_mavlink_rta_bridge_cli PROPERTIES
  OUTPUT_NAME "vertirta_mavlink_rta_bridge"
)
target_compile_features(vertirta_mavlink_rta_bridge_cli PRIVATE cxx_std_20)
target_link_libraries(vertirta_mavlink_rta_bridge_cli PRIVATE vertirta::mavlink_rta_bridge)
target_compile_options(vertirta_mavlink_rta_bridge_cli PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror
  -Wshadow
  -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
