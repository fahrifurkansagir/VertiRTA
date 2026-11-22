cmake_minimum_required(VERSION 3.25)
project(vertirta_python_sdk VERSION 3.0.0 LANGUAGES C CXX)

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
target_compile_features(_vertirta_native PRIVATE cxx_std_20)
if(NOT MSVC)
  target_compile_options(_vertirta_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS _vertirta_native LIBRARY DESTINATION vertirta)
