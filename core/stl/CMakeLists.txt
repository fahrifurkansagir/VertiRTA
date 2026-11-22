add_library(vertirta_stl_monitor STATIC
  stl_monitor.cpp
)
add_library(vertirta::stl_monitor ALIAS vertirta_stl_monitor)

target_compile_features(vertirta_stl_monitor PUBLIC cxx_std_20)
target_include_directories(vertirta_stl_monitor
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(vertirta_stl_monitor PUBLIC vertirta::rta_companion)
target_compile_options(vertirta_stl_monitor PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
