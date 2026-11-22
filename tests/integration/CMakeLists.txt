cmake_minimum_required(VERSION 3.22)
project(vertirta_integration_tests VERSION 1.0.0 LANGUAGES CXX)

option(VERTIRTA_ENABLE_TSAN "Build integration concurrency tests with ThreadSanitizer" OFF)

enable_testing()
find_package(Catch2 3 REQUIRED)
find_package(Threads REQUIRED)

get_filename_component(VERTIRTA_REPO_ROOT_ABS "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(test_spsc_tsan
  test_spsc_tsan.cpp
)
target_include_directories(test_spsc_tsan PRIVATE
  "${VERTIRTA_REPO_ROOT_ABS}/ros2_ws/src/vertirta_rta_node/include"
)
target_link_libraries(test_spsc_tsan PRIVATE
  Catch2::Catch2WithMain
  Threads::Threads
)
target_compile_features(test_spsc_tsan PRIVATE cxx_std_20)
target_compile_options(test_spsc_tsan PRIVATE
  -Wall -Wextra -Wpedantic -Werror
  -fno-fast-math -ffp-contract=off
)

if(VERTIRTA_ENABLE_TSAN)
  target_compile_options(test_spsc_tsan PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
  target_link_options(test_spsc_tsan PRIVATE -fsanitize=thread)
endif()

add_test(NAME test_spsc_tsan COMMAND test_spsc_tsan)
