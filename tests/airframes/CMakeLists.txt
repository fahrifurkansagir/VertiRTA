if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.25)
  project(VertiRTA_AirframesTests VERSION 3.5.0 LANGUAGES C CXX)
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
  add_subdirectory("${_vertirta_repo_root}/core/flight_core" "${CMAKE_CURRENT_BINARY_DIR}/flight_core_dep")
endif()

if(NOT TARGET vertirta::airframes_lc)
  add_subdirectory("${_vertirta_repo_root}/core/airframes/lift_cruise" "${CMAKE_CURRENT_BINARY_DIR}/airframes_lc_dep")
endif()

add_executable(test_lift_cruise_dynamics test_lift_cruise_dynamics.cpp)
target_link_libraries(test_lift_cruise_dynamics PRIVATE vertirta::airframes_lc Catch2::Catch2WithMain)
target_compile_features(test_lift_cruise_dynamics PRIVATE cxx_std_20)
add_test(NAME test_lift_cruise_dynamics COMMAND test_lift_cruise_dynamics)

add_executable(test_transition_indi_wls test_transition_indi_wls.cpp)
target_link_libraries(test_transition_indi_wls PRIVATE vertirta::airframes_lc Catch2::Catch2WithMain)
target_compile_features(test_transition_indi_wls PRIVATE cxx_std_20)
add_test(NAME test_transition_indi_wls COMMAND test_transition_indi_wls)
