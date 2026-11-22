find_package(Threads REQUIRED)

add_executable(vertirta_replay_cli vertirta_replay_cli.cpp)
target_link_libraries(vertirta_replay_cli PRIVATE vertirta::rta_companion)
target_compile_features(vertirta_replay_cli PRIVATE cxx_std_20)
target_compile_options(vertirta_replay_cli PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)

add_executable(vertirta_bench_cli vertirta_bench_cli.cpp)
target_link_libraries(vertirta_bench_cli PRIVATE vertirta::rta_companion)
target_compile_features(vertirta_bench_cli PRIVATE cxx_std_20)
target_compile_options(vertirta_bench_cli PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)

add_executable(vertirta_6dof_batch_cli vertirta_6dof_batch_cli.cpp)
target_link_libraries(vertirta_6dof_batch_cli PRIVATE vertirta::rta_companion Threads::Threads)
target_compile_features(vertirta_6dof_batch_cli PRIVATE cxx_std_20)
target_compile_options(vertirta_6dof_batch_cli PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
