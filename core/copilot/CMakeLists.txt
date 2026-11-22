add_library(vertirta_copilot STATIC
  dsl_parser.cpp
  stl_synthesizer.cpp
  z3_corridor_verifier.cpp
  unsat_repair.cpp
  llama_gbnf_engine.cpp
  bilingual_renderer.cpp
  conformal_gate.cpp
)
add_library(vertirta::copilot ALIAS vertirta_copilot)

target_compile_features(vertirta_copilot PUBLIC cxx_std_20)
target_include_directories(vertirta_copilot
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(vertirta_copilot PUBLIC vertirta::stl_monitor vertirta::rta_companion)
target_compile_options(vertirta_copilot PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)

# Optional native Z3 C++ API (falls back to exact QF_LRA evaluator + SMT-LIB2 emitter when absent)
find_package(Z3 QUIET)
if(Z3_FOUND AND TARGET z3::libz3)
  target_compile_definitions(vertirta_copilot PUBLIC VERTIRTA_HAS_Z3=1)
  target_link_libraries(vertirta_copilot PUBLIC z3::libz3)
else()
  find_path(VERTIRTA_Z3_INCLUDE_DIR z3++.h)
  find_library(VERTIRTA_Z3_LIBRARY z3)
  if(VERTIRTA_Z3_INCLUDE_DIR AND VERTIRTA_Z3_LIBRARY)
    target_compile_definitions(vertirta_copilot PUBLIC VERTIRTA_HAS_Z3=1)
    target_include_directories(vertirta_copilot PUBLIC "${VERTIRTA_Z3_INCLUDE_DIR}")
    target_link_libraries(vertirta_copilot PUBLIC "${VERTIRTA_Z3_LIBRARY}")
  endif()
endif()

# Optional native llama.cpp C API (falls back to deterministic mock mode when absent)
find_path(VERTIRTA_LLAMA_INCLUDE_DIR llama.h)
find_library(VERTIRTA_LLAMA_LIBRARY llama)
if(VERTIRTA_LLAMA_INCLUDE_DIR AND VERTIRTA_LLAMA_LIBRARY)
  target_compile_definitions(vertirta_copilot PUBLIC VERTIRTA_HAS_LLAMA=1)
  target_include_directories(vertirta_copilot PUBLIC "${VERTIRTA_LLAMA_INCLUDE_DIR}")
  target_link_libraries(vertirta_copilot PUBLIC "${VERTIRTA_LLAMA_LIBRARY}")
endif()
