# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.22)
project(vertirta_voice_node VERSION 4.0.0 LANGUAGES C CXX)

find_package(Threads REQUIRED)
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
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
if(NOT TARGET vertirta::stl_monitor)
  add_subdirectory("${VERTIRTA_REPO_ROOT}/core/stl" "${CMAKE_CURRENT_BINARY_DIR}/stl" EXCLUDE_FROM_ALL)
  target_include_directories(vertirta_stl_monitor PUBLIC "${VERTIRTA_REPO_ROOT}/include")
endif()
if(NOT TARGET vertirta::copilot)
  add_subdirectory("${VERTIRTA_REPO_ROOT}/core/copilot" "${CMAKE_CURRENT_BINARY_DIR}/copilot" EXCLUDE_FROM_ALL)
  target_include_directories(vertirta_copilot PUBLIC "${VERTIRTA_REPO_ROOT}/include")
endif()

add_library(vertirta_voice_node_core STATIC
  src/voice_node_core.cpp
)
add_library(vertirta::voice_node_core ALIAS vertirta_voice_node_core)
target_include_directories(vertirta_voice_node_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  "${VERTIRTA_REPO_ROOT}/include"
)
target_link_libraries(vertirta_voice_node_core PUBLIC
  vertirta::copilot
  vertirta::stl_monitor
  vertirta::rta_companion
  vertirta::flight_core
  Threads::Threads
)
target_compile_features(vertirta_voice_node_core PUBLIC cxx_std_20)
target_compile_options(vertirta_voice_node_core PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow ${VERTIRTA_STRICT_FP_FLAGS}
)

if(ament_cmake_FOUND AND rclcpp_FOUND)
  add_executable(vertirta_voice_node src/voice_node.cpp)
  target_compile_definitions(vertirta_voice_node PRIVATE VERTIRTA_WITH_ROS2=1)
  target_link_libraries(vertirta_voice_node PRIVATE
    vertirta::voice_node_core
    Threads::Threads
  )
  set(_ament_deps rclcpp)
  foreach(_dep std_msgs vertirta_msgs)
    if(${_dep}_FOUND)
      list(APPEND _ament_deps ${_dep})
    endif()
  endforeach()
  ament_target_dependencies(vertirta_voice_node ${_ament_deps})
  target_compile_features(vertirta_voice_node PRIVATE cxx_std_20)
  target_compile_options(vertirta_voice_node PRIVATE
    -Wall -Wextra -Wpedantic -Werror -Wshadow ${VERTIRTA_STRICT_FP_FLAGS}
  )

  install(DIRECTORY include/ DESTINATION include)
  install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
  install(TARGETS vertirta_voice_node DESTINATION lib/${PROJECT_NAME})

  if(BUILD_TESTING)
    find_package(Catch2 3 QUIET)
    if(Catch2_FOUND)
      add_executable(test_voice_node_core test/test_voice_node_core.cpp)
      target_link_libraries(test_voice_node_core PRIVATE
        vertirta::voice_node_core
        Catch2::Catch2WithMain
        Threads::Threads
      )
      target_compile_features(test_voice_node_core PRIVATE cxx_std_20)
      target_compile_options(test_voice_node_core PRIVATE
        -Wall -Wextra -Wpedantic -Werror -Wshadow ${VERTIRTA_STRICT_FP_FLAGS}
      )
      add_test(NAME test_voice_node_core COMMAND test_voice_node_core)
    endif()
  endif()

  ament_package()
else()
  add_executable(vertirta_voice_node_cli src/voice_node.cpp)
  target_compile_definitions(vertirta_voice_node_cli PRIVATE VERTIRTA_WITH_ROS2=0)
  target_link_libraries(vertirta_voice_node_cli PRIVATE
    vertirta::voice_node_core
    Threads::Threads
  )
  target_compile_features(vertirta_voice_node_cli PRIVATE cxx_std_20)
  target_compile_options(vertirta_voice_node_cli PRIVATE
    -Wall -Wextra -Wpedantic -Werror -Wshadow ${VERTIRTA_STRICT_FP_FLAGS}
  )
endif()
