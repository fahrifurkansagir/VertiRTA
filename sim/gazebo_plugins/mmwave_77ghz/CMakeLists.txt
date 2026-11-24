if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.25)
  project(VertiRTAMmWave77GHz VERSION 4.0.0 LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  option(VERTIRTA_BUILD_TESTS "Build standalone mmWave 77 GHz Catch2 tests" OFF)
  option(VERTIRTA_ENABLE_STRICT_FP "Enforce IEEE-754 deterministic FP flags" ON)
  if(VERTIRTA_BUILD_TESTS)
    include(CTest)
  endif()
endif()

set(_vertirta_mmwave_strict_fp
  -fno-fast-math
  -ffp-contract=off
  -fno-associative-math
  -fno-reciprocal-math
  -frounding-math
)

add_library(vertirta_mmwave_77ghz_physics STATIC
  src/mmwave_77ghz_physics.cpp
  src/mmwave_77ghz_sensor.cpp
)
add_library(vertirta::mmwave_77ghz_physics ALIAS vertirta_mmwave_77ghz_physics)

target_include_directories(vertirta_mmwave_77ghz_physics
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(vertirta_mmwave_77ghz_physics PUBLIC cxx_std_20)
target_compile_options(vertirta_mmwave_77ghz_physics
  PRIVATE
    -Wall -Wextra -Wpedantic -Werror -Wshadow
    $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${_vertirta_mmwave_strict_fp}>
)

find_package(gz-sensors8 QUIET)
find_package(gz-sim8 QUIET)
if(gz-sensors8_FOUND AND gz-sim8_FOUND)
  target_compile_definitions(vertirta_mmwave_77ghz_physics PUBLIC VERTIRTA_HAS_GZ_SENSORS=1)
  target_link_libraries(vertirta_mmwave_77ghz_physics
    PUBLIC
      gz-sensors8::gz-sensors8
      gz-sim8::gz-sim8
  )
  message(STATUS "VertiRTA mmwave_77ghz: gz-sensors8 & gz-sim8 found — Gazebo Harmonic plugin enabled")
else()
  message(STATUS "VertiRTA mmwave_77ghz: gz-sensors8/gz-sim8 absent — building pure C++20 physics bridge")
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND VERTIRTA_BUILD_TESTS)
  add_subdirectory(tests)
endif()
