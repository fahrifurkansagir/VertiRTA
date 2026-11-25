cmake_minimum_required(VERSION 3.22)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(vertirta_forward_hazard_standalone VERSION 1.0.0 LANGUAGES C CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  option(VERTIRTA_BUILD_TESTS "Build forward_hazard Catch2 tests" ON)
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

add_library(vertirta_forward_hazard STATIC
  src/horizon_split_stack.cpp
  src/onnx_forward_detectors.cpp
  src/se3_pose_ring_buffer.cpp
  src/wire_span_ekf.cpp
  src/forward_hazard_fusion.cpp
)
add_library(vertirta::forward_hazard ALIAS vertirta_forward_hazard)

target_include_directories(vertirta_forward_hazard
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(vertirta_forward_hazard
  PUBLIC
    vertirta::flight_core
    vertirta::rta_companion
)

target_compile_features(vertirta_forward_hazard PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(vertirta_forward_hazard PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  if(VERTIRTA_ENABLE_STRICT_FP)
    target_compile_options(vertirta_forward_hazard PRIVATE ${VERTIRTA_STRICT_FP_FLAGS})
  endif()
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND VERTIRTA_BUILD_TESTS)
  get_filename_component(_VERTIRTA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  if(EXISTS "${_VERTIRTA_ROOT}/tests/forward_hazard/CMakeLists.txt")
    add_subdirectory("${_VERTIRTA_ROOT}/tests/forward_hazard" "${CMAKE_BINARY_DIR}/tests_forward_hazard")
  endif()
endif()
