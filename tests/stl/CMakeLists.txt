add_executable(test_stl_monitor_zero_heap
  test_stl_monitor_zero_heap.cpp
  ${CMAKE_SOURCE_DIR}/tests/flight_core/zero_heap_guard.cpp
)
target_include_directories(test_stl_monitor_zero_heap PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/flight_core
)
target_link_libraries(test_stl_monitor_zero_heap PRIVATE
  vertirta::stl_monitor
  Catch2::Catch2WithMain
)
target_compile_options(test_stl_monitor_zero_heap PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
if(UNIX AND NOT APPLE AND NOT CMAKE_CXX_FLAGS MATCHES "fsanitize=address")
  target_compile_definitions(test_stl_monitor_zero_heap PRIVATE VERTIRTA_WRAP_MALLOC=1)
  target_link_options(test_stl_monitor_zero_heap PRIVATE
    -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc
  )
endif()
add_test(NAME test_stl_monitor_zero_heap COMMAND test_stl_monitor_zero_heap)
