# SPDX-License-Identifier: Apache-2.0
add_library(vertirta_colab_egl STATIC
  src/gz_egl_bridge.cpp
)
add_library(vertirta::colab_egl ALIAS vertirta_colab_egl)

target_compile_features(vertirta_colab_egl PUBLIC cxx_std_20)
target_include_directories(vertirta_colab_egl
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

find_package(gz-transport13 QUIET)
find_package(gz-msgs10 QUIET)

if(gz-transport13_FOUND AND gz-msgs10_FOUND)
  message(STATUS "VertiRTA colab_egl: gz-transport13 + gz-msgs10 found — enabling VERTIRTA_HAS_GZ_TRANSPORT")
  target_compile_definitions(vertirta_colab_egl PUBLIC VERTIRTA_HAS_GZ_TRANSPORT=1)
  target_link_libraries(vertirta_colab_egl PUBLIC
    gz-transport13::gz-transport13
    gz-msgs10::gz-msgs10
  )
else()
  message(STATUS "VertiRTA colab_egl: gz-transport13 not found — building pure C++20 bridge without live Gazebo transport")
endif()

target_compile_options(vertirta_colab_egl PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
