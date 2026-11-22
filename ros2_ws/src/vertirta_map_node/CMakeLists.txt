cmake_minimum_required(VERSION 3.22)
project(vertirta_map_node VERSION 1.0.0 LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(vertirta_msgs REQUIRED)

add_executable(vertirta_map_node src/map_node.cpp)
target_include_directories(vertirta_map_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(vertirta_map_node
  rclcpp
  geometry_msgs
  sensor_msgs
  visualization_msgs
  px4_msgs
  vertirta_msgs
)
target_compile_features(vertirta_map_node PRIVATE cxx_std_20)
target_compile_options(vertirta_map_node PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS vertirta_map_node DESTINATION lib/${PROJECT_NAME})

ament_package()
