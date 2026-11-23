if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.25)
  project(VertiRTAWindAcousticsTests VERSION 3.5.0 LANGUAGES C CXX)
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

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS 3
  )
  FetchContent_MakeAvailable(Catch2)
endif()

get_filename_component(_vertirta_repo_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

if(NOT TARGET vertirta::flight_core)
  add_subdirectory("${_vertirta_repo_root}/core/flight_core" "${CMAKE_CURRENT_BINARY_DIR}/_flight_core")
endif()
if(NOT TARGET vertirta::rta_companion)
  add_subdirectory("${_vertirta_repo_root}/core/rta_companion" "${CMAKE_CURRENT_BINARY_DIR}/_rta_companion")
endif()
if(NOT TARGET vertirta::wind_acoustics)
  add_subdirectory("${_vertirta_repo_root}/core/wind_acoustics" "${CMAKE_CURRENT_BINARY_DIR}/_wind_acoustics")
endif()

add_executable(verify_vrtawnd1_load verify_vrtawnd1_load.cpp)
target_link_libraries(verify_vrtawnd1_load PRIVATE vertirta::wind_acoustics)
target_compile_options(verify_vrtawnd1_load PRIVATE -Wall -Wextra -Wpedantic -Werror -Wshadow)

foreach(_test_name
    test_mmap_vrta_wind_lut
    test_aeroacoustics_hemisphere
    test_noise_aware_router_lexicographic)
  add_executable(${_test_name} ${_test_name}.cpp)
  target_link_libraries(${_test_name}
    PRIVATE
      vertirta::wind_acoustics
      Catch2::Catch2WithMain
  )
  target_compile_definitions(${_test_name} PRIVATE VERTIRTA_REPO_ROOT="${_vertirta_repo_root}")
  target_compile_options(${_test_name} PRIVATE -Wall -Wextra -Wpedantic -Werror -Wshadow)
  add_test(NAME ${_test_name} COMMAND ${_test_name})
endforeach()
set_tests_properties(
  test_mmap_vrta_wind_lut
  test_aeroacoustics_hemisphere
  PROPERTIES RUN_SERIAL TRUE
)
