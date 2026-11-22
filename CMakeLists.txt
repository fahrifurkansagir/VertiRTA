cmake_minimum_required(VERSION 3.22)
project(VertiRTA VERSION 1.0.0 LANGUAGES C CXX)

include(CTest)
include(FetchContent)

option(VERTIRTA_BUILD_COMPANION "Build C++20 companion RTA target" ON)
option(VERTIRTA_BUILD_TESTS "Build VertiRTA Catch2 unit tests" ON)
option(VERTIRTA_ENABLE_STRICT_FP "Enforce IEEE-754 deterministic FP flags" ON)

set(VERTIRTA_STRICT_FP_FLAGS
  -fno-fast-math
  -ffp-contract=off
  -fno-associative-math
  -fno-reciprocal-math
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/flight_core/CMakeLists.txt")
  add_subdirectory(core/flight_core)
endif()

if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/rta_companion/CMakeLists.txt")
  add_subdirectory(core/rta_companion)
endif()

if(VERTIRTA_BUILD_TESTS AND BUILD_TESTING)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/flight_core/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/rta_companion/CMakeLists.txt")
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.7.1
      GIT_SHALLOW TRUE
      FIND_PACKAGE_ARGS 3
    )
    FetchContent_MakeAvailable(Catch2)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/flight_core/CMakeLists.txt")
    add_subdirectory(tests/flight_core)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/rta_companion/CMakeLists.txt")
    add_subdirectory(tests/rta_companion)
  endif()
endif()
