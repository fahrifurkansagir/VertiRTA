cmake_minimum_required(VERSION 3.22)
project(vertirta_gz_plugins VERSION 1.0.0 LANGUAGES CXX)

option(VERTIRTA_SIM_BUILD_TESTS "Build pure-C++ sim kernel and SDF Catch2 tests" ON)

add_library(vertirta_gz_kernels INTERFACE)
add_library(vertirta::gz_kernels ALIAS vertirta_gz_kernels)
target_include_directories(vertirta_gz_kernels INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(vertirta_gz_kernels INTERFACE cxx_std_20)

# Build Gazebo Harmonic (.so) plugins whenever gz-sim8 + gz-transport13 + gz-msgs10 are installed.
find_package(gz-sim8 QUIET)
find_package(gz-transport13 QUIET)
find_package(gz-msgs10 QUIET)

if(gz-sim8_FOUND AND gz-transport13_FOUND AND gz-msgs10_FOUND)
  message(STATUS "VertiRTA sim: gz-sim8 found — building libvertirta_coaxial_propulsion.so & libvertirta_dryden_wind.so")

  add_library(vertirta_coaxial_propulsion SHARED
    src/vertirta_coaxial_propulsion.cpp
  )
  target_link_libraries(vertirta_coaxial_propulsion PRIVATE
    vertirta::gz_kernels
    gz-sim8::gz-sim8
    gz-transport13::gz-transport13
    gz-msgs10::gz-msgs10
  )
  target_compile_options(vertirta_coaxial_propulsion PRIVATE -Wall -Wextra -Wpedantic)

  add_library(vertirta_dryden_wind SHARED
    src/vertirta_dryden_wind.cpp
  )
  target_link_libraries(vertirta_dryden_wind PRIVATE
    vertirta::gz_kernels
    gz-sim8::gz-sim8
    gz-transport13::gz-transport13
    gz-msgs10::gz-msgs10
  )
  target_compile_options(vertirta_dryden_wind PRIVATE -Wall -Wextra -Wpedantic)
else()
  message(STATUS "VertiRTA sim: gz-sim8 not installed — skipping .so plugins; pure physics kernels & Catch2 tests remain enabled")
endif()

if(VERTIRTA_SIM_BUILD_TESTS)
  enable_testing()
  find_package(Catch2 3 QUIET)
  if(Catch2_FOUND)
    get_filename_component(VERTIRTA_REPO_ROOT_ABS "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    set(SIM_TEST_SRC "${VERTIRTA_REPO_ROOT_ABS}/tests/sim/test_gz_plugins.cpp")
    if(EXISTS "${SIM_TEST_SRC}")
      add_executable(test_gz_plugins "${SIM_TEST_SRC}")
      target_link_libraries(test_gz_plugins PRIVATE
        vertirta::gz_kernels
        Catch2::Catch2WithMain
      )
      target_compile_definitions(test_gz_plugins PRIVATE
        VERTIRTA_REPO_ROOT="${VERTIRTA_REPO_ROOT_ABS}"
      )
      target_compile_options(test_gz_plugins PRIVATE
        -Wall -Wextra -Wpedantic -Werror
        -fno-fast-math -ffp-contract=off
      )
      add_test(NAME test_gz_plugins COMMAND test_gz_plugins)
    endif()
  else()
    message(STATUS "VertiRTA sim: Catch2 v3 not found — skipping test_gz_plugins target")
  endif()
endif()
