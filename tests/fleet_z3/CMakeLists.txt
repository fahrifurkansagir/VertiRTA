# SPDX-License-Identifier: Apache-2.0
# Dual-mode CMakeLists.txt: supports Wave 1 standalone builds AND Wave 2 root add_subdirectory().

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.25)
  project(VertiRtaFleetZ3Tests VERSION 3.5.0 LANGUAGES C CXX)
  include(CTest)
  include(FetchContent)

  option(VERTIRTA_ENABLE_STRICT_FP "Enforce IEEE-754 deterministic FP flags" ON)
  set(VERTIRTA_STRICT_FP_FLAGS
    -fno-fast-math
    -ffp-contract=off
    -fno-associative-math
    -fno-reciprocal-math
    -frounding-math
  )

  if(NOT TARGET Catch2::Catch2WithMain)
    find_package(Catch2 3 QUIET)
    if(NOT Catch2_FOUND)
      FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.7.1
        GIT_SHALLOW TRUE
      )
      FetchContent_MakeAvailable(Catch2)
    endif()
  endif()
endif()

if(NOT TARGET vertirta::fleet_z3)
  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../../core/fleet_z3"
    "${CMAKE_CURRENT_BINARY_DIR}/vertirta_fleet_z3_build"
  )
endif()

add_executable(test_fleet_z3_scheduler test_fleet_z3_scheduler.cpp)
target_link_libraries(test_fleet_z3_scheduler PRIVATE
  vertirta::fleet_z3
  Catch2::Catch2WithMain
)
target_compile_features(test_fleet_z3_scheduler PRIVATE cxx_std_20)
target_compile_options(test_fleet_z3_scheduler PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
add_test(NAME test_fleet_z3_scheduler COMMAND test_fleet_z3_scheduler)
set_tests_properties(test_fleet_z3_scheduler PROPERTIES RUN_SERIAL TRUE)

add_executable(emit_fleet_smt2_cases emit_fleet_smt2_cases.cpp)
target_link_libraries(emit_fleet_smt2_cases PRIVATE vertirta::fleet_z3)
target_compile_features(emit_fleet_smt2_cases PRIVATE cxx_std_20)
target_compile_options(emit_fleet_smt2_cases PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
