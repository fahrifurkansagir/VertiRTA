get_filename_component(VERTIRTA_SLZ_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(test_slz_bayes_grid
  test_slz_bayes_grid.cpp
  ${VERTIRTA_SLZ_REPO_ROOT}/tests/flight_core/zero_heap_guard.cpp
)
target_include_directories(test_slz_bayes_grid PRIVATE
  ${VERTIRTA_SLZ_REPO_ROOT}/tests/flight_core
)
target_link_libraries(test_slz_bayes_grid PRIVATE vertirta::slz_bayes Catch2::Catch2WithMain)
target_compile_options(test_slz_bayes_grid PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
if(UNIX AND NOT APPLE AND NOT CMAKE_CXX_FLAGS MATCHES "fsanitize=")
  target_compile_definitions(test_slz_bayes_grid PRIVATE VERTIRTA_WRAP_MALLOC=1)
  target_link_options(test_slz_bayes_grid PRIVATE
    -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc
  )
endif()
add_test(NAME test_slz_bayes_grid COMMAND test_slz_bayes_grid)

add_executable(test_active_inspection_policy
  test_active_inspection_policy.cpp
  ${VERTIRTA_SLZ_REPO_ROOT}/tests/flight_core/zero_heap_guard.cpp
)
target_include_directories(test_active_inspection_policy PRIVATE
  ${VERTIRTA_SLZ_REPO_ROOT}/tests/flight_core
)
target_link_libraries(test_active_inspection_policy PRIVATE vertirta::slz_bayes Catch2::Catch2WithMain)
target_compile_options(test_active_inspection_policy PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
if(UNIX AND NOT APPLE AND NOT CMAKE_CXX_FLAGS MATCHES "fsanitize=")
  target_compile_definitions(test_active_inspection_policy PRIVATE VERTIRTA_WRAP_MALLOC=1)
  target_link_options(test_active_inspection_policy PRIVATE
    -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc
  )
endif()
add_test(NAME test_active_inspection_policy COMMAND test_active_inspection_policy)
