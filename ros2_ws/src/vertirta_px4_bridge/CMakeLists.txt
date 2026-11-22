cmake_minimum_required(VERSION 3.22)
project(vertirta_px4_bridge VERSION 1.0.0 LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(px4_ros2_cpp REQUIRED)
find_package(vertirta_msgs REQUIRED)

add_executable(vertirta_px4_bridge_node
  src/vertirta_mode.cpp
  src/vertirta_executor.cpp
  src/px4_bridge_main.cpp
)
target_include_directories(vertirta_px4_bridge_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(vertirta_px4_bridge_node
  rclcpp
  rclcpp_action
  geometry_msgs
  px4_msgs
  px4_ros2_cpp
  vertirta_msgs
)
target_compile_features(vertirta_px4_bridge_node PRIVATE cxx_std_20)
target_compile_options(vertirta_px4_bridge_node PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS vertirta_px4_bridge_node DESTINATION lib/${PROJECT_NAME})

ament_package()
