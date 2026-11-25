cmake_minimum_required(VERSION 3.25)
project(vertirta_python_sdk VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(VERTIRTA_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH "VertiRTA repository root")

if(EXISTS "${VERTIRTA_REPO_ROOT}/build/dev-linux-x64/_deps/daqp-src/CMakeLists.txt"
   AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_DAQP)
  set(FETCHCONTENT_SOURCE_DIR_DAQP
    "${VERTIRTA_REPO_ROOT}/build/dev-linux-x64/_deps/daqp-src"
    CACHE PATH "Reuse cached DAQP v0.10.0 source"
  )
endif()

if(NOT TARGET vertirta::flight_core)
  option(VERTIRTA_ENABLE_STRICT_FP "Enforce IEEE-754 deterministic FP flags" ON)
  if(MSVC)
    set(VERTIRTA_STRICT_FP_FLAGS /fp:strict)
  else()
    set(VERTIRTA_STRICT_FP_FLAGS
      -fno-fast-math
      -ffp-contract=off
      -fno-associative-math
      -fno-reciprocal-math
      -frounding-math
    )
  endif()
  add_subdirectory(
    "${VERTIRTA_REPO_ROOT}/core/flight_core"
    "${CMAKE_CURRENT_BINARY_DIR}/flight_core"
    EXCLUDE_FROM_ALL
  )
endif()

if(NOT TARGET vertirta::rta_companion
   AND EXISTS "${VERTIRTA_REPO_ROOT}/core/rta_companion/CMakeLists.txt")
  add_subdirectory(
    "${VERTIRTA_REPO_ROOT}/core/rta_companion"
    "${CMAKE_CURRENT_BINARY_DIR}/rta_companion"
    EXCLUDE_FROM_ALL
  )
endif()
if(NOT TARGET vertirta::uspace_fleet
   AND EXISTS "${VERTIRTA_REPO_ROOT}/core/uspace_fleet/CMakeLists.txt")
  set(VERTIRTA_REQUIRE_LIBSODIUM OFF CACHE BOOL "Optional libsodium in Python SDK wheel" FORCE)
  add_subdirectory(
    "${VERTIRTA_REPO_ROOT}/core/uspace_fleet"
    "${CMAKE_CURRENT_BINARY_DIR}/uspace_fleet"
    EXCLUDE_FROM_ALL
  )
endif()
if(NOT TARGET vertirta::airframes_lc
   AND EXISTS "${VERTIRTA_REPO_ROOT}/core/airframes/lift_cruise/CMakeLists.txt")
  add_subdirectory(
    "${VERTIRTA_REPO_ROOT}/core/airframes/lift_cruise"
    "${CMAKE_CURRENT_BINARY_DIR}/airframes_lc"
    EXCLUDE_FROM_ALL
  )
endif()
if(NOT TARGET vertirta::forward_hazard
   AND EXISTS "${VERTIRTA_REPO_ROOT}/core/forward_hazard/CMakeLists.txt")
  add_subdirectory(
    "${VERTIRTA_REPO_ROOT}/core/forward_hazard"
    "${CMAKE_CURRENT_BINARY_DIR}/forward_hazard"
    EXCLUDE_FROM_ALL
  )
endif()
if(NOT TARGET vertirta::gnss_denied
   AND EXISTS "${VERTIRTA_REPO_ROOT}/core/gnss_denied/CMakeLists.txt")
  add_subdirectory(
    "${VERTIRTA_REPO_ROOT}/core/gnss_denied"
    "${CMAKE_CURRENT_BINARY_DIR}/gnss_denied"
    EXCLUDE_FROM_ALL
  )
endif()
if(NOT TARGET vertirta::voice
   AND EXISTS "${VERTIRTA_REPO_ROOT}/core/voice/CMakeLists.txt")
  set(VERTIRTA_REQUIRE_WHISPER OFF CACHE BOOL "Optional whisper.cpp in Python SDK wheel" FORCE)
  set(VERTIRTA_REQUIRE_SHERPA_ONNX OFF CACHE BOOL "Optional sherpa-onnx in Python SDK wheel" FORCE)
  add_subdirectory(
    "${VERTIRTA_REPO_ROOT}/core/voice"
    "${CMAKE_CURRENT_BINARY_DIR}/voice"
    EXCLUDE_FROM_ALL
  )
endif()
if(NOT TARGET vertirta::forensics
   AND EXISTS "${VERTIRTA_REPO_ROOT}/core/forensics/CMakeLists.txt")
  set(VERTIRTA_REQUIRE_LIBSODIUM OFF CACHE BOOL "Optional libsodium in Python SDK wheel" FORCE)
  add_subdirectory(
    "${VERTIRTA_REPO_ROOT}/core/forensics"
    "${CMAKE_CURRENT_BINARY_DIR}/forensics"
    EXCLUDE_FROM_ALL
  )
endif()

find_package(Python 3.11 REQUIRED
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule
)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
  RESULT_VARIABLE _nb_rc
)
if(NOT _nb_rc EQUAL 0)
  message(FATAL_ERROR "Failed to locate nanobind CMake directory via '${Python_EXECUTABLE} -m nanobind --cmake_dir'")
endif()

find_package(nanobind CONFIG REQUIRED)

set(_nb_abi_args NB_STATIC)
if(Python_VERSION VERSION_GREATER_EQUAL 3.12 AND TARGET Python::SABIModule)
  list(PREPEND _nb_abi_args STABLE_ABI)
endif()

nanobind_add_module(_vertirta_native
  ${_nb_abi_args}
  src/vertirta_bindings.cpp
)

target_link_libraries(_vertirta_native PRIVATE vertirta::flight_core)
if(TARGET vertirta::uspace_fleet AND TARGET vertirta::airframes_lc)
  target_link_libraries(_vertirta_native PRIVATE
    vertirta::uspace_fleet
    vertirta::airframes_lc
  )
  target_compile_definitions(_vertirta_native PRIVATE VERTIRTA_SDK_HAS_M2=1)
else()
  target_compile_definitions(_vertirta_native PRIVATE VERTIRTA_SDK_HAS_M2=0)
endif()
if(TARGET vertirta::forward_hazard
   AND TARGET vertirta::gnss_denied
   AND TARGET vertirta::voice
   AND TARGET vertirta::forensics)
  target_link_libraries(_vertirta_native PRIVATE
    vertirta::forward_hazard
    vertirta::gnss_denied
    vertirta::voice
    vertirta::forensics
  )
  target_compile_definitions(_vertirta_native PRIVATE VERTIRTA_SDK_HAS_M3=1)
else()
  target_compile_definitions(_vertirta_native PRIVATE VERTIRTA_SDK_HAS_M3=0)
endif()
target_compile_features(_vertirta_native PRIVATE cxx_std_20)
if(NOT MSVC)
  target_compile_options(_vertirta_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS _vertirta_native LIBRARY DESTINATION vertirta)
