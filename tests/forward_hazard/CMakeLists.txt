if(NOT TARGET Catch2::Catch2WithMain)
  find_package(Catch2 3 REQUIRED)
endif()

add_executable(test_forward_hazard_fusion
  test_forward_hazard_fusion.cpp
)
target_link_libraries(test_forward_hazard_fusion
  PRIVATE
    vertirta::forward_hazard
    Catch2::Catch2WithMain
)
target_compile_features(test_forward_hazard_fusion PRIVATE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_forward_hazard_fusion PRIVATE -Wall -Wextra -Wpedantic -Werror)
  if(VERTIRTA_ENABLE_STRICT_FP)
    target_compile_options(test_forward_hazard_fusion PRIVATE ${VERTIRTA_STRICT_FP_FLAGS})
  endif()
endif()
add_test(NAME test_forward_hazard_fusion COMMAND test_forward_hazard_fusion)

# Wave-1 race-free guard: linked cleanly in Wave 2 once sim/gazebo_plugins/mmwave_77ghz is wired in root CMakeLists.txt
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_mmwave_77ghz_sensor.cpp" AND TARGET vertirta::mmwave_77ghz_physics)
  get_filename_component(_vertirta_fh_repo_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
  add_executable(test_mmwave_77ghz_sensor
    test_mmwave_77ghz_sensor.cpp
  )
  target_compile_definitions(test_mmwave_77ghz_sensor
    PRIVATE
      VERTIRTA_REPO_ROOT="${_vertirta_fh_repo_root}"
  )
  target_link_libraries(test_mmwave_77ghz_sensor
    PRIVATE
      vertirta::mmwave_77ghz_physics
      Catch2::Catch2WithMain
  )
  target_compile_features(test_mmwave_77ghz_sensor PRIVATE cxx_std_20)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_mmwave_77ghz_sensor PRIVATE -Wall -Wextra -Wpedantic -Werror)
    if(VERTIRTA_ENABLE_STRICT_FP)
      target_compile_options(test_mmwave_77ghz_sensor PRIVATE ${VERTIRTA_STRICT_FP_FLAGS})
    endif()
  endif()
  add_test(NAME test_mmwave_77ghz_sensor COMMAND test_mmwave_77ghz_sensor)
endif()
