cmake_minimum_required(VERSION 3.22)
project(vertirta_perception_node VERSION 1.0.0 LANGUAGES C CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(vertirta_msgs REQUIRED)

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
if(NOT TARGET vertirta::slz_bayes)
  add_subdirectory("${VERTIRTA_REPO_ROOT}/core/slz_bayes" "${CMAKE_CURRENT_BINARY_DIR}/slz_bayes" EXCLUDE_FROM_ALL)
endif()

add_library(vertirta_perception_core INTERFACE)
target_include_directories(vertirta_perception_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(vertirta_perception_core INTERFACE
  vertirta::slz_bayes
  vertirta::rta_companion
)
target_compile_features(vertirta_perception_core INTERFACE cxx_std_20)

add_executable(vertirta_perception_node src/perception_node.cpp)
target_compile_definitions(vertirta_perception_node PRIVATE VERTIRTA_WITH_ROS2=1)
target_link_libraries(vertirta_perception_node PRIVATE vertirta_perception_core)
ament_target_dependencies(vertirta_perception_node
  rclcpp
  geometry_msgs
  sensor_msgs
  visualization_msgs
  px4_msgs
  vertirta_msgs
)
target_compile_features(vertirta_perception_node PRIVATE cxx_std_20)
target_compile_options(vertirta_perception_node PRIVATE -Wall -Wextra -Wpedantic ${VERTIRTA_STRICT_FP_FLAGS})

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(TARGETS vertirta_perception_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(Catch2 3 QUIET)
  if(Catch2_FOUND)
    add_executable(test_perception_node test/test_perception_node.cpp)
    target_link_libraries(test_perception_node PRIVATE vertirta_perception_core Catch2::Catch2WithMain)
    target_compile_features(test_perception_node PRIVATE cxx_std_20)
    target_compile_options(test_perception_node PRIVATE -Wall -Wextra -Wpedantic -Werror ${VERTIRTA_STRICT_FP_FLAGS})
    add_test(NAME test_perception_node COMMAND test_perception_node)
  endif()
endif()

ament_package()
