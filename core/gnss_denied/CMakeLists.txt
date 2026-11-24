cmake_minimum_required(VERSION 3.22)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(vertirta_gnss_denied_standalone VERSION 4.0.0 LANGUAGES C CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  option(VERTIRTA_BUILD_TESTS "Build gnss_denied Catch2 tests" ON)
  option(VERTIRTA_ENABLE_STRICT_FP "Enable strict IEEE-754 floating-point flags" ON)
  set(VERTIRTA_STRICT_FP_FLAGS
    -fno-fast-math
    -ffp-contract=off
    -fno-associative-math
    -fno-reciprocal-math
    -frounding-math
  )
  get_filename_component(_VERTIRTA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  if(NOT TARGET vertirta::flight_core)
    add_subdirectory("${_VERTIRTA_ROOT}/core/flight_core" "${CMAKE_BINARY_DIR}/flight_core" EXCLUDE_FROM_ALL)
  endif()
  if(NOT TARGET vertirta::rta_companion)
    add_subdirectory("${_VERTIRTA_ROOT}/core/rta_companion" "${CMAKE_BINARY_DIR}/rta_companion" EXCLUDE_FROM_ALL)
  endif()
  enable_testing()
endif()

add_library(vertirta_gnss_denied STATIC
  src/basalt_vio_adapter.cpp
  src/gnss_spoof_guard.cpp
  src/visual_georeloc_adapter.cpp
  src/ctr_inflation_fsm.cpp
)
add_library(vertirta::gnss_denied ALIAS vertirta_gnss_denied)

target_include_directories(vertirta_gnss_denied
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(vertirta_gnss_denied
  PUBLIC
    vertirta::flight_core
    vertirta::rta_companion
)

target_compile_features(vertirta_gnss_denied PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(vertirta_gnss_denied PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  if(VERTIRTA_ENABLE_STRICT_FP)
    target_compile_options(vertirta_gnss_denied PRIVATE ${VERTIRTA_STRICT_FP_FLAGS})
  endif()
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND VERTIRTA_BUILD_TESTS)
  get_filename_component(_VERTIRTA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  if(EXISTS "${_VERTIRTA_ROOT}/tests/gnss_denied/CMakeLists.txt")
    add_subdirectory("${_VERTIRTA_ROOT}/tests/gnss_denied" "${CMAKE_BINARY_DIR}/tests_gnss_denied")
  endif()
endif()
