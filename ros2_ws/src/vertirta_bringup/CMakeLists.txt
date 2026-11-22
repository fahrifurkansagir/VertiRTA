cmake_minimum_required(VERSION 3.22)
project(vertirta_bringup VERSION 1.0.0 LANGUAGES NONE)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
