cmake_minimum_required(VERSION 3.22)
project(vertirta_fault_injector VERSION 1.0.0 LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(vertirta_msgs REQUIRED)

add_executable(vertirta_fault_injector_node src/fault_injector_node.cpp)
ament_target_dependencies(vertirta_fault_injector_node
  rclcpp
  std_msgs
  px4_msgs
  vertirta_msgs
)
target_compile_features(vertirta_fault_injector_node PRIVATE cxx_std_20)
target_compile_options(vertirta_fault_injector_node PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS vertirta_fault_injector_node DESTINATION lib/${PROJECT_NAME})

ament_package()
