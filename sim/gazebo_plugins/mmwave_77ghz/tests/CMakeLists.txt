if(NOT TARGET Catch2::Catch2WithMain)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS 3
  )
  FetchContent_MakeAvailable(Catch2)
endif()

get_filename_component(_vertirta_repo_root "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

add_executable(test_mmwave_77ghz_sensor
  "${_vertirta_repo_root}/tests/forward_hazard/test_mmwave_77ghz_sensor.cpp"
)
target_link_libraries(test_mmwave_77ghz_sensor
  PRIVATE
    vertirta::mmwave_77ghz_physics
    Catch2::Catch2WithMain
)
target_compile_definitions(test_mmwave_77ghz_sensor
  PRIVATE
    VERTIRTA_REPO_ROOT="${_vertirta_repo_root}"
)
target_compile_options(test_mmwave_77ghz_sensor PRIVATE -Wall -Wextra -Wpedantic -Werror -Wshadow)
add_test(NAME test_mmwave_77ghz_sensor COMMAND test_mmwave_77ghz_sensor)
