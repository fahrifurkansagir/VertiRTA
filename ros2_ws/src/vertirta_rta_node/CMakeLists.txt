cmake_minimum_required(VERSION 3.22)
project(vertirta_rta_node VERSION 1.0.0 LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(vertirta_msgs REQUIRED)
find_package(Threads REQUIRED)

add_library(vertirta_rta_transport INTERFACE)
target_include_directories(vertirta_rta_transport INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(vertirta_rta_transport INTERFACE cxx_std_20)

add_executable(vertirta_rta_node src/rta_node.cpp)
target_link_libraries(vertirta_rta_node PRIVATE
  vertirta_rta_transport
  Threads::Threads
)
ament_target_dependencies(vertirta_rta_node
  rclcpp
  geometry_msgs
  px4_msgs
  vertirta_msgs
)
target_compile_features(vertirta_rta_node PRIVATE cxx_std_20)
target_compile_options(vertirta_rta_node PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS vertirta_rta_node DESTINATION lib/${PROJECT_NAME})

ament_package()
