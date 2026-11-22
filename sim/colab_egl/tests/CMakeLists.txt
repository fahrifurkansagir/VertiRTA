# SPDX-License-Identifier: Apache-2.0
add_executable(test_gz_egl_bridge
  test_gz_egl_bridge.cpp
)
target_link_libraries(test_gz_egl_bridge PRIVATE
  vertirta::colab_egl
  Catch2::Catch2WithMain
)
target_compile_options(test_gz_egl_bridge PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
add_test(NAME test_gz_egl_bridge COMMAND test_gz_egl_bridge)
