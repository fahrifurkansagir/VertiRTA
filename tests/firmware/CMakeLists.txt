cmake_minimum_required(VERSION 3.22)
project(VertiRTA_Firmware_Host_Tests VERSION 1.0.0 LANGUAGES C CXX)

include(CTest)
include(FetchContent)

get_filename_component(VERTIRTA_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(VERTIRTA_CYPHAL_DIR "${VERTIRTA_REPO_ROOT}/firmware/renode_cyphal")

# Prefer already-cached local Catch2 v3.7.1 checkout to avoid network dependency; fallback to FetchContent.
set(VERTIRTA_LOCAL_CATCH2 "${VERTIRTA_REPO_ROOT}/build/dev-linux-x64/_deps/catch2-src")
if(EXISTS "${VERTIRTA_LOCAL_CATCH2}/CMakeLists.txt")
  add_subdirectory("${VERTIRTA_LOCAL_CATCH2}" "${CMAKE_CURRENT_BINARY_DIR}/_catch2")
else()
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

add_library(vertirta_cyphal_host STATIC
  "${VERTIRTA_CYPHAL_DIR}/src/safety_envelope_codec.c"
  "${VERTIRTA_CYPHAL_DIR}/src/o1heap_arena.cpp"
  "${VERTIRTA_CYPHAL_DIR}/src/cyphal_bus_node.cpp"
)
target_include_directories(vertirta_cyphal_host PUBLIC
  "${VERTIRTA_CYPHAL_DIR}/include"
)
target_compile_features(vertirta_cyphal_host PUBLIC c_std_11 cxx_std_17)
target_compile_options(vertirta_cyphal_host PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wvla -Wshadow
  -fno-fast-math -ffp-contract=off
)

add_executable(test_cyphal_bus_and_o1heap
  test_cyphal_bus_and_o1heap.cpp
  "${VERTIRTA_REPO_ROOT}/tests/flight_core/zero_heap_guard.cpp"
)
target_include_directories(test_cyphal_bus_and_o1heap PRIVATE
  "${VERTIRTA_CYPHAL_DIR}/include"
  "${VERTIRTA_REPO_ROOT}/tests/flight_core"
)
target_compile_features(test_cyphal_bus_and_o1heap PRIVATE cxx_std_17)
target_compile_options(test_cyphal_bus_and_o1heap PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wvla -Wshadow
  -fno-fast-math -ffp-contract=off
)
target_link_libraries(test_cyphal_bus_and_o1heap PRIVATE
  vertirta_cyphal_host
  Catch2::Catch2WithMain
)

if(NOT CMAKE_CXX_FLAGS MATCHES "fsanitize=address")
  target_compile_definitions(test_cyphal_bus_and_o1heap PRIVATE VERTIRTA_WRAP_MALLOC=1)
  target_link_options(test_cyphal_bus_and_o1heap PRIVATE
    -Wl,--wrap=malloc
    -Wl,--wrap=calloc
    -Wl,--wrap=realloc
  )
endif()

add_test(NAME test_cyphal_bus_and_o1heap COMMAND test_cyphal_bus_and_o1heap)
