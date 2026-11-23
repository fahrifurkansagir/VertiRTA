# SPDX-License-Identifier: Apache-2.0
option(VERTIRTA_REQUIRE_LIBSODIUM
  "Fail CMake configure if native libsodium >= 1.0.21 is not found (forced ON in M2 CI)"
  OFF)

find_package(Threads REQUIRED)
find_package(PkgConfig QUIET)

set(_vertirta_libsodium_found FALSE)
set(_vertirta_libsodium_target "")
set(_vertirta_libsodium_inc "")
set(_vertirta_libsodium_lib "")
set(_vertirta_libsodium_ver "")

if(PkgConfig_FOUND)
  pkg_check_modules(VERTIRTA_SODIUM QUIET IMPORTED_TARGET libsodium)
  if(VERTIRTA_SODIUM_FOUND)
    set(_vertirta_libsodium_ver "${VERTIRTA_SODIUM_VERSION}")
    if(_vertirta_libsodium_ver VERSION_GREATER_EQUAL "1.0.21")
      set(_vertirta_libsodium_found TRUE)
      set(_vertirta_libsodium_target PkgConfig::VERTIRTA_SODIUM)
    else()
      message(WARNING
        "Ignoring vulnerable libsodium ${_vertirta_libsodium_ver} (<= 1.0.20 Ed25519 subgroup CVE); require >= 1.0.21.")
    endif()
  endif()
endif()

if(NOT _vertirta_libsodium_found)
  find_path(_sodium_inc_dir NAMES sodium.h PATH_SUFFIXES include)
  find_library(_sodium_lib_file NAMES sodium)
  if(_sodium_inc_dir AND _sodium_lib_file AND EXISTS "${_sodium_inc_dir}/sodium/version.h")
    file(STRINGS "${_sodium_inc_dir}/sodium/version.h" _sodium_ver_line
      REGEX "#define[ \t]+SODIUM_VERSION_STRING[ \t]+\"[^\"]+\"")
    if(_sodium_ver_line MATCHES "\"([0-9]+\\.[0-9]+\\.[0-9]+)")
      set(_vertirta_libsodium_ver "${CMAKE_MATCH_1}")
      if(_vertirta_libsodium_ver VERSION_GREATER_EQUAL "1.0.21")
        set(_vertirta_libsodium_found TRUE)
        set(_vertirta_libsodium_inc "${_sodium_inc_dir}")
        set(_vertirta_libsodium_lib "${_sodium_lib_file}")
      else()
        message(WARNING
          "Ignoring vulnerable libsodium ${_vertirta_libsodium_ver} at ${_sodium_inc_dir} (<= 1.0.20 CVE); require >= 1.0.21.")
      endif()
    endif()
  endif()
endif()

if(VERTIRTA_REQUIRE_LIBSODIUM AND NOT _vertirta_libsodium_found)
  message(FATAL_ERROR
    "VERTIRTA_REQUIRE_LIBSODIUM=ON but libsodium >= 1.0.21 was not found (detected: '${_vertirta_libsodium_ver}'). "
    "Build immutable libsodium-1.0.22.tar.gz from download.libsodium.org/libsodium/releases/.")
endif()

add_library(vertirta_uspace_fleet STATIC
  src/corridor_network.cpp
  src/thermal_turnaround.cpp
  src/vertiport_queue.cpp
  src/part_is_security.cpp
  src/cbf_orca_3d.cpp
  src/fleet_sim_engine.cpp
)
add_library(vertirta::uspace_fleet ALIAS vertirta_uspace_fleet)

target_compile_features(vertirta_uspace_fleet PUBLIC cxx_std_20)
target_include_directories(vertirta_uspace_fleet
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(vertirta_uspace_fleet
  PUBLIC
    vertirta::flight_core
    vertirta::rta_companion
    Threads::Threads
)

if(_vertirta_libsodium_found)
  target_compile_definitions(vertirta_uspace_fleet PUBLIC VERTIRTA_HAS_LIBSODIUM=1)
  if(_vertirta_libsodium_target)
    target_link_libraries(vertirta_uspace_fleet PRIVATE ${_vertirta_libsodium_target})
  else()
    target_include_directories(vertirta_uspace_fleet PRIVATE "${_vertirta_libsodium_inc}")
    target_link_libraries(vertirta_uspace_fleet PRIVATE "${_vertirta_libsodium_lib}")
  endif()
  message(STATUS "vertirta_uspace_fleet: Part-IS Mode B using native libsodium ${_vertirta_libsodium_ver}")
else()
  target_compile_definitions(vertirta_uspace_fleet PUBLIC VERTIRTA_HAS_LIBSODIUM=0)
  message(STATUS "vertirta_uspace_fleet: Part-IS Mode B using SIM_HMAC_SHA256_BACKEND (libsodium >= 1.0.21 absent)")
endif()

target_compile_options(vertirta_uspace_fleet PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
