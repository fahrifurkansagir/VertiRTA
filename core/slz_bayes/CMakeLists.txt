add_library(vertirta_slz_bayes STATIC
  src/slz_ort_engine.cpp
  src/slz_geometry.cpp
  src/slz_belief_grid.cpp
  src/active_inspection_policy.cpp
)
add_library(vertirta::slz_bayes ALIAS vertirta_slz_bayes)

target_compile_features(vertirta_slz_bayes PUBLIC cxx_std_20)
target_include_directories(vertirta_slz_bayes
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(vertirta_slz_bayes PUBLIC vertirta::rta_companion)
target_compile_options(vertirta_slz_bayes PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)

# Optional native ONNX Runtime + CUDA V2 Execution Providers (graceful zero-dep mock otherwise).
option(VERTIRTA_ENABLE_ORT "Enable native ONNX Runtime + CUDA V2 EP if installed" ON)
if(VERTIRTA_ENABLE_ORT)
  find_path(VERTIRTA_ORT_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
    HINTS /usr/include/onnxruntime /usr/local/include/onnxruntime
          $ENV{ONNXRUNTIME_ROOT}/include)
  find_library(VERTIRTA_ORT_LIBRARY NAMES onnxruntime
    HINTS /usr/lib /usr/local/lib $ENV{ONNXRUNTIME_ROOT}/lib)
  find_package(CUDAToolkit QUIET)
  if(VERTIRTA_ORT_INCLUDE_DIR AND VERTIRTA_ORT_LIBRARY AND CUDAToolkit_FOUND)
    target_compile_definitions(vertirta_slz_bayes PUBLIC VERTIRTA_HAS_ORT=1)
    target_include_directories(vertirta_slz_bayes PUBLIC "${VERTIRTA_ORT_INCLUDE_DIR}")
    target_link_libraries(vertirta_slz_bayes PUBLIC "${VERTIRTA_ORT_LIBRARY}" CUDA::cudart)
  endif()
endif()
