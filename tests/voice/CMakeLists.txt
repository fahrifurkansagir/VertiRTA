# SPDX-License-Identifier: Apache-2.0
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.25)
  project(VertiRTAVoiceTests VERSION 4.0.0 LANGUAGES C CXX)
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
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.7.1
      GIT_SHALLOW TRUE
      FIND_PACKAGE_ARGS 3
    )
    FetchContent_MakeAvailable(Catch2)
  endif()
endif()

if(NOT TARGET vertirta::voice)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../core/voice"
    "${CMAKE_CURRENT_BINARY_DIR}/_voice")
endif()

foreach(_v_test
    test_voice_pipeline
    test_mondrian_voice_gate)
  add_executable(${_v_test} "${_v_test}.cpp")
  target_link_libraries(${_v_test} PRIVATE
    vertirta::voice
    Catch2::Catch2WithMain
  )
  target_compile_features(${_v_test} PRIVATE cxx_std_20)
  target_compile_options(${_v_test} PRIVATE
    -Wall -Wextra -Wpedantic -Werror -Wshadow
  )
  add_test(NAME ${_v_test} COMMAND ${_v_test})
endforeach()
