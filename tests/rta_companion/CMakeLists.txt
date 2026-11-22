add_executable(test_lut_dryden test_lut_dryden.cpp)
target_link_libraries(test_lut_dryden PRIVATE vertirta::rta_companion Catch2::Catch2WithMain)
target_compile_definitions(test_lut_dryden PRIVATE VERTIRTA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_options(test_lut_dryden PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
add_test(NAME test_lut_dryden COMMAND test_lut_dryden)

add_executable(test_cbvf_safety_filter
  test_cbvf_safety_filter.cpp
  ${CMAKE_SOURCE_DIR}/tests/flight_core/zero_heap_guard.cpp
)
target_include_directories(test_cbvf_safety_filter PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/flight_core
)
target_link_libraries(test_cbvf_safety_filter PRIVATE vertirta::rta_companion Catch2::Catch2WithMain)
target_compile_definitions(test_cbvf_safety_filter PRIVATE VERTIRTA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_options(test_cbvf_safety_filter PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
if(UNIX AND NOT APPLE AND NOT CMAKE_CXX_FLAGS MATCHES "fsanitize=address")
  target_compile_definitions(test_cbvf_safety_filter PRIVATE VERTIRTA_WRAP_MALLOC=1)
  target_link_options(test_cbvf_safety_filter PRIVATE
    -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc
  )
endif()
add_test(NAME test_cbvf_safety_filter COMMAND test_cbvf_safety_filter)

add_executable(test_esdf_hazards_slz test_esdf_hazards_slz.cpp)
target_link_libraries(test_esdf_hazards_slz PRIVATE vertirta::rta_companion Catch2::Catch2WithMain)
target_compile_options(test_esdf_hazards_slz PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
add_test(NAME test_esdf_hazards_slz COMMAND test_esdf_hazards_slz)
