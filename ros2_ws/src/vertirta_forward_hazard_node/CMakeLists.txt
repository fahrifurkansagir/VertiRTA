# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.22)
project(vertirta_forward_hazard_node VERSION 1.0.0 LANGUAGES C CXX)

find_package(Threads REQUIRED)
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(geometry_msgs QUIET)
find_package(sensor_msgs QUIET)
find_package(std_msgs QUIET)
find_package(visualization_msgs QUIET)
find_package(px4_msgs QUIET)
find_package(vertirta_msgs QUIET)

get_filename_component(VERTIRTA_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(VERTIRTA_STRICT_FP_FLAGS
  -fno-fast-math
  -ffp-contract=off
  -fno-associative-math
  -fno-reciprocal-math
  -frounding-math
)
set(VERTIRTA_ENABLE_STRICT_FP ON CACHE BOOL "" FORCE)

if(NOT TARGET vertirta::flight_core)
  add_subdirectory("${VERTIRTA_REPO_ROOT}/core/flight_core" "${CMAKE_CURRENT_BINARY_DIR}/flight_core" EXCLUDE_FROM_ALL)
endif()
if(NOT TARGET vertirta::rta_companion)
  add_subdirectory("${VERTIRTA_REPO_ROOT}/core/rta_companion" "${CMAKE_CURRENT_BINARY_DIR}/rta_companion" EXCLUDE_FROM_ALL)
endif()

add_library(vertirta_forward_hazard_core INTERFACE)
add_library(vertirta::forward_hazard_node_core ALIAS vertirta_forward_hazard_core)
target_include_directories(vertirta_forward_hazard_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(vertirta_forward_hazard_core INTERFACE
  vertirta::rta_companion
  vertirta::flight_core
)
target_compile_features(vertirta_forward_hazard_core INTERFACE cxx_std_20)

if(ament_cmake_FOUND AND rclcpp_FOUND)
  add_executable(vertirta_forward_hazard_node src/forward_hazard_node.cpp)
  target_compile_definitions(vertirta_forward_hazard_node PRIVATE VERTIRTA_WITH_ROS2=1)
  target_link_libraries(vertirta_forward_hazard_node PRIVATE
    vertirta::forward_hazard_node_core
    Threads::Threads
  )
  set(_ament_deps rclcpp)
  foreach(_dep geometry_msgs sensor_msgs std_msgs visualization_msgs px4_msgs vertirta_msgs)
    if(${_dep}_FOUND)
      list(APPEND _ament_deps ${_dep})
    endif()
  endforeach()
  ament_target_dependencies(vertirta_forward_hazard_node ${_ament_deps})
  target_compile_features(vertirta_forward_hazard_node PRIVATE cxx_std_20)
  target_compile_options(vertirta_forward_hazard_node PRIVATE
    -Wall -Wextra -Wpedantic -Werror -Wshadow ${VERTIRTA_STRICT_FP_FLAGS}
  )

  install(DIRECTORY include/ DESTINATION include)
  install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
  install(TARGETS vertirta_forward_hazard_node DESTINATION lib/${PROJECT_NAME})

  if(BUILD_TESTING)
    find_package(Catch2 3 QUIET)
    if(Catch2_FOUND)
      add_executable(test_forward_hazard_node test/test_forward_hazard_node.cpp)
      target_link_libraries(test_forward_hazard_node PRIVATE
        vertirta::forward_hazard_node_core
        Catch2::Catch2WithMain
        Threads::Threads
      )
      target_compile_features(test_forward_hazard_node PRIVATE cxx_std_20)
      target_compile_options(test_forward_hazard_node PRIVATE
        -Wall -Wextra -Wpedantic -Werror -Wshadow ${VERTIRTA_STRICT_FP_FLAGS}
      )
      add_test(NAME test_forward_hazard_node COMMAND test_forward_hazard_node)
    endif()
  endif()

  ament_package()
else()
  add_executable(vertirta_forward_hazard_node_cli src/forward_hazard_node.cpp)
  target_compile_definitions(vertirta_forward_hazard_node_cli PRIVATE VERTIRTA_WITH_ROS2=0)
  target_link_libraries(vertirta_forward_hazard_node_cli PRIVATE
    vertirta::forward_hazard_node_core
    Threads::Threads
  )
  target_compile_features(vertirta_forward_hazard_node_cli PRIVATE cxx_std_20)
  target_compile_options(vertirta_forward_hazard_node_cli PRIVATE
    -Wall -Wextra -Wpedantic -Werror -Wshadow ${VERTIRTA_STRICT_FP_FLAGS}
  )
endif()
