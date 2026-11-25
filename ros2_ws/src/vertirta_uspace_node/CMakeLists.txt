# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.22)
project(vertirta_uspace_node VERSION 1.0.0 LANGUAGES CXX)

find_package(Threads REQUIRED)
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(geometry_msgs QUIET)
find_package(vertirta_msgs QUIET)

add_library(vertirta_uspace_core INTERFACE)
add_library(vertirta::uspace_core ALIAS vertirta_uspace_core)
target_include_directories(vertirta_uspace_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(vertirta_uspace_core INTERFACE cxx_std_20)

if(ament_cmake_FOUND AND rclcpp_FOUND)
  add_executable(vertirta_uspace_node src/uspace_node.cpp)
  target_compile_definitions(vertirta_uspace_node PRIVATE VERTIRTA_HAS_RCLCPP=1)
  target_link_libraries(vertirta_uspace_node PRIVATE
    vertirta::uspace_core
    Threads::Threads
  )
  set(_ament_deps rclcpp)
  if(geometry_msgs_FOUND)
    list(APPEND _ament_deps geometry_msgs)
  endif()
  if(vertirta_msgs_FOUND)
    list(APPEND _ament_deps vertirta_msgs)
  endif()
  ament_target_dependencies(vertirta_uspace_node ${_ament_deps})
  target_compile_features(vertirta_uspace_node PRIVATE cxx_std_20)
  target_compile_options(vertirta_uspace_node PRIVATE
    -Wall -Wextra -Wpedantic -Werror -Wshadow -fno-fast-math -ffp-contract=off
  )

  install(DIRECTORY include/ DESTINATION include)
  install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
  install(TARGETS vertirta_uspace_node DESTINATION lib/${PROJECT_NAME})
  ament_package()
else()
  add_executable(vertirta_uspace_node_cli src/uspace_node.cpp)
  target_compile_definitions(vertirta_uspace_node_cli PRIVATE VERTIRTA_HAS_RCLCPP=0)
  target_link_libraries(vertirta_uspace_node_cli PRIVATE
    vertirta::uspace_core
    Threads::Threads
  )
  target_compile_features(vertirta_uspace_node_cli PRIVATE cxx_std_20)
  target_compile_options(vertirta_uspace_node_cli PRIVATE
    -Wall -Wextra -Wpedantic -Werror -Wshadow -fno-fast-math -ffp-contract=off
  )
endif()
