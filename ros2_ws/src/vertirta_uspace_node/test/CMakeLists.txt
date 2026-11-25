# SPDX-License-Identifier: Apache-2.0
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.25)
  project(VertiRTAUspaceNodeTests VERSION 1.0.0 LANGUAGES CXX)
  include(CTest)
  include(FetchContent)

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

add_executable(test_uspace_node test_uspace_node.cpp)
target_include_directories(test_uspace_node PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)
target_link_libraries(test_uspace_node PRIVATE Catch2::Catch2WithMain)
target_compile_features(test_uspace_node PRIVATE cxx_std_20)
target_compile_options(test_uspace_node PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -fno-fast-math -ffp-contract=off
)
add_test(NAME test_uspace_node COMMAND test_uspace_node)
