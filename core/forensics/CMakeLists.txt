# SPDX-License-Identifier: Apache-2.0
option(VERTIRTA_REQUIRE_LIBSODIUM
  "Fail CMake configure if native libsodium >= 1.0.21 is not found (forced ON in M3 CI)"
  OFF)

find_package(Threads REQUIRED)
find_package(PkgConfig QUIET)

set(_vertirta_forensics_sodium_found FALSE)
set(_vertirta_forensics_sodium_target "")
set(_vertirta_forensics_sodium_inc "")
set(_vertirta_forensics_sodium_lib "")
set(_vertirta_forensics_sodium_ver "")

if(PkgConfig_FOUND)
  pkg_check_modules(VERTIRTA_FORENSICS_SODIUM QUIET IMPORTED_TARGET libsodium)
  if(VERTIRTA_FORENSICS_SODIUM_FOUND)
    set(_vertirta_forensics_sodium_ver "${VERTIRTA_FORENSICS_SODIUM_VERSION}")
    if(_vertirta_forensics_sodium_ver VERSION_GREATER_EQUAL "1.0.21")
      set(_vertirta_forensics_sodium_found TRUE)
      set(_vertirta_forensics_sodium_target PkgConfig::VERTIRTA_FORENSICS_SODIUM)
    else()
      message(WARNING
        "vertirta_forensics: ignoring vulnerable libsodium ${_vertirta_forensics_sodium_ver} (<= 1.0.20 Ed25519 subgroup CVE); require >= 1.0.21.")
    endif()
  endif()
endif()

if(NOT _vertirta_forensics_sodium_found)
  find_path(_forensics_sodium_inc_dir NAMES sodium.h PATH_SUFFIXES include)
  find_library(_forensics_sodium_lib_file NAMES sodium)
  if(_forensics_sodium_inc_dir AND _forensics_sodium_lib_file AND EXISTS "${_forensics_sodium_inc_dir}/sodium/version.h")
    file(STRINGS "${_forensics_sodium_inc_dir}/sodium/version.h" _forensics_sodium_ver_line
      REGEX "#define[ \t]+SODIUM_VERSION_STRING[ \t]+\"[^\"]+\"")
    if(_forensics_sodium_ver_line MATCHES "\"([0-9]+\\.[0-9]+\\.[0-9]+)")
      set(_vertirta_forensics_sodium_ver "${CMAKE_MATCH_1}")
      if(_vertirta_forensics_sodium_ver VERSION_GREATER_EQUAL "1.0.21")
        set(_vertirta_forensics_sodium_found TRUE)
        set(_vertirta_forensics_sodium_inc "${_forensics_sodium_inc_dir}")
        set(_vertirta_forensics_sodium_lib "${_forensics_sodium_lib_file}")
      else()
        message(WARNING
          "vertirta_forensics: ignoring vulnerable libsodium ${_vertirta_forensics_sodium_ver} at ${_forensics_sodium_inc_dir} (<= 1.0.20 CVE); require >= 1.0.21.")
      endif()
    endif()
  endif()
endif()

if(VERTIRTA_REQUIRE_LIBSODIUM AND NOT _vertirta_forensics_sodium_found)
  message(FATAL_ERROR
    "VERTIRTA_REQUIRE_LIBSODIUM=ON but libsodium >= 1.0.21 was not found (detected: '${_vertirta_forensics_sodium_ver}'). "
    "Build immutable libsodium-1.0.22.tar.gz from download.libsodium.org/libsodium/releases/.")
endif()

add_library(vertirta_forensics STATIC
  src/vrta_events_sidecar.cpp
  src/ercs_scorer.cpp
  src/narrative_grounding.cpp
  src/eccairs2_e5x_emitter.cpp
  src/forensics_pipeline.cpp
)
add_library(vertirta::forensics ALIAS vertirta_forensics)

target_compile_features(vertirta_forensics PUBLIC cxx_std_20)
target_include_directories(vertirta_forensics
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(vertirta_forensics
  PUBLIC
    vertirta::flight_core
    Threads::Threads
)

if(_vertirta_forensics_sodium_found)
  target_compile_definitions(vertirta_forensics PUBLIC VERTIRTA_HAS_LIBSODIUM=1)
  if(_vertirta_forensics_sodium_target)
    target_link_libraries(vertirta_forensics PRIVATE ${_vertirta_forensics_sodium_target})
  else()
    target_include_directories(vertirta_forensics PRIVATE "${_vertirta_forensics_sodium_inc}")
    target_link_libraries(vertirta_forensics PRIVATE "${_vertirta_forensics_sodium_lib}")
  endif()
  message(STATUS "vertirta_forensics: ED-155A sidecar Ed25519 using native libsodium ${_vertirta_forensics_sodium_ver}")
else()
  target_compile_definitions(vertirta_forensics PUBLIC VERTIRTA_HAS_LIBSODIUM=0)
  message(STATUS "vertirta_forensics: ED-155A sidecar using SIM_SHA256_TEST_SIGNER (libsodium >= 1.0.21 absent)")
endif()

target_compile_options(vertirta_forensics PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)

add_executable(vertirta_forensics_cli
  src/forensics_cli.cpp
)
target_link_libraries(vertirta_forensics_cli PRIVATE vertirta::forensics)
target_compile_features(vertirta_forensics_cli PRIVATE cxx_std_20)
target_compile_options(vertirta_forensics_cli PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
)
