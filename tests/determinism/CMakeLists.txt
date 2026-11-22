# Standalone-capable and root-includable CMakeLists for Tier-1 Determinism Gates.
# Guarded per Wave-2 Handoff Note #3: per-target -Wl,--wrap=malloc ONLY (never directory-wide).

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.22)
  project(vertirta_determinism LANGUAGES C CXX)
  include(CTest)
  set(VERTIRTA_ENABLE_STRICT_FP ON CACHE BOOL "Enforce IEEE-754 strict FP flags")
  set(VERTIRTA_STRICT_FP_FLAGS
    -fno-fast-math
    -ffp-contract=off
    -fno-associative-math
    -fno-reciprocal-math
    -frounding-math
  )
  set(VERTIRTA_CORE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../core/flight_core" CACHE PATH "Path to flight_core")
  add_subdirectory("${VERTIRTA_CORE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/flight_core")
endif()

# Do not override global operator new / wrap malloc under ASan or TSan sanitizer runtimes.
if(NOT CMAKE_CXX_FLAGS MATCHES "-fsanitize=")
  add_executable(vertirta_flight_core_noheap_test
    test_flight_core_noheap.cpp
    no_heap_link_trap.cpp
  )

  target_link_libraries(vertirta_flight_core_noheap_test PRIVATE vertirta::flight_core)
  target_compile_features(vertirta_flight_core_noheap_test PRIVATE cxx_std_17)

  target_compile_options(vertirta_flight_core_noheap_test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wvla
    -ffunction-sections
    -fdata-sections
    -fno-exceptions
    -fno-rtti
    -fno-threadsafe-statics
    -fno-lto
    -fstack-usage
    -fcallgraph-info=su,da
    -Wstack-usage=4096
    -fno-fast-math
    -ffp-contract=off
    -fno-associative-math
    -fno-reciprocal-math
    -frounding-math
  )

  # Strictly per-target link options — never poisons Catch2 tests or vertirta_replay_cli.
  target_link_options(vertirta_flight_core_noheap_test PRIVATE
    -Wl,--gc-sections
    -Wl,--wrap=malloc
    -Wl,--wrap=calloc
    -Wl,--wrap=realloc
  )

  add_test(NAME vertirta_flight_core_noheap_test COMMAND vertirta_flight_core_noheap_test)
endif()
