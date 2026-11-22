cmake_minimum_required(VERSION 3.22)
project(vertirta_mission_node VERSION 2.0.0 LANGUAGES C CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(vertirta_msgs REQUIRED)
find_package(Threads REQUIRED)

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
if(NOT TARGET vertirta::stl_monitor)
  add_subdirectory("${VERTIRTA_REPO_ROOT}/core/stl" "${CMAKE_CURRENT_BINARY_DIR}/stl" EXCLUDE_FROM_ALL)
  target_include_directories(vertirta_stl_monitor PUBLIC "${VERTIRTA_REPO_ROOT}/include")
endif()
if(NOT TARGET vertirta::copilot)
  add_subdirectory("${VERTIRTA_REPO_ROOT}/core/copilot" "${CMAKE_CURRENT_BINARY_DIR}/copilot" EXCLUDE_FROM_ALL)
  target_include_directories(vertirta_copilot PUBLIC "${VERTIRTA_REPO_ROOT}/include")
endif()

add_library(vertirta_mission_core INTERFACE)
target_include_directories(vertirta_mission_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(vertirta_mission_core INTERFACE
  vertirta::copilot
  vertirta::stl_monitor
  vertirta::rta_companion
  Threads::Threads
)
target_compile_features(vertirta_mission_core INTERFACE cxx_std_20)

add_executable(vertirta_mission_node src/mission_node.cpp)
target_compile_definitions(vertirta_mission_node PRIVATE VERTIRTA_WITH_ROS2=1)
target_link_libraries(vertirta_mission_node PRIVATE vertirta_mission_core)
ament_target_dependencies(vertirta_mission_node
  rclcpp
  rclcpp_action
  std_msgs
  px4_msgs
  vertirta_msgs
)
target_compile_features(vertirta_mission_node PRIVATE cxx_std_20)
target_compile_options(vertirta_mission_node PRIVATE -Wall -Wextra -Wpedantic ${VERTIRTA_STRICT_FP_FLAGS})

install(DIRECTORY include/ DESTINATION include)
install(TARGETS vertirta_mission_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(Catch2 3 QUIET)
  if(Catch2_FOUND)
    add_executable(test_mission_isolation test/test_mission_isolation.cpp)
    target_link_libraries(test_mission_isolation PRIVATE vertirta_mission_core Catch2::Catch2WithMain)
    target_compile_features(test_mission_isolation PRIVATE cxx_std_20)
    target_compile_options(test_mission_isolation PRIVATE -Wall -Wextra -Wpedantic -Werror ${VERTIRTA_STRICT_FP_FLAGS})
    add_test(NAME test_mission_isolation COMMAND test_mission_isolation)
  endif()
endif()

ament_package()
