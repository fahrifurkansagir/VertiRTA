foreach(_vertirta_copilot_test
    test_dsl_and_synthesizer
    test_z3_verifier_and_repair
    test_llama_engine_and_renderer
    test_conformal_gate)
  add_executable(${_vertirta_copilot_test} ${_vertirta_copilot_test}.cpp)
  target_link_libraries(${_vertirta_copilot_test} PRIVATE
    vertirta::copilot
    Catch2::Catch2WithMain
  )
  target_compile_definitions(${_vertirta_copilot_test} PRIVATE
    VERTIRTA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )
  target_compile_options(${_vertirta_copilot_test} PRIVATE
    -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
    $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
  )
  add_test(NAME ${_vertirta_copilot_test} COMMAND ${_vertirta_copilot_test})
endforeach()

add_executable(emit_smt2_cases emit_smt2_cases.cpp)
target_link_libraries(emit_smt2_cases PRIVATE vertirta::copilot)
target_compile_options(emit_smt2_cases PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
