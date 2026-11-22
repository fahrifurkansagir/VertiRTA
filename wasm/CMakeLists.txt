cmake_minimum_required(VERSION 3.22)
project(vertirta_wasm_abi LANGUAGES C CXX)

set(VERTIRTA_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "VertiRTA repository root")
set(VERTIRTA_WASM_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "Output directory for Emscripten WASM bundle")

if(EXISTS "${VERTIRTA_REPO_ROOT}/build/dev-linux-x64/_deps/daqp-src/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_DAQP)
  set(FETCHCONTENT_SOURCE_DIR_DAQP "${VERTIRTA_REPO_ROOT}/build/dev-linux-x64/_deps/daqp-src" CACHE PATH "Reuse cached DAQP v0.10.0 source")
endif()

if(NOT TARGET vertirta::flight_core)
  set(VERTIRTA_ENABLE_STRICT_FP ON CACHE BOOL "" FORCE)
  set(VERTIRTA_STRICT_FP_FLAGS -fno-fast-math -ffp-contract=off)
  if(NOT EMSCRIPTEN)
    list(APPEND VERTIRTA_STRICT_FP_FLAGS -fno-associative-math -fno-reciprocal-math -frounding-math)
  endif()
  add_subdirectory("${VERTIRTA_REPO_ROOT}/core/flight_core" "${CMAKE_CURRENT_BINARY_DIR}/flight_core" EXCLUDE_FROM_ALL)
endif()

add_library(vertirta_wasm_abi STATIC
  vertirta_wasm_abi.cpp
)
target_include_directories(vertirta_wasm_abi PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(vertirta_wasm_abi PUBLIC vertirta::flight_core)
target_compile_features(vertirta_wasm_abi PUBLIC cxx_std_17)
target_compile_options(vertirta_wasm_abi PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow
  -fno-rtti -fno-exceptions -fno-threadsafe-statics
  -ffp-contract=off -fno-fast-math
)

if(EMSCRIPTEN)
  add_executable(libvertirta_flight_core_wasm vertirta_wasm_abi.cpp)
  target_include_directories(libvertirta_flight_core_wasm PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
  target_link_libraries(libvertirta_flight_core_wasm PRIVATE vertirta::flight_core)
  target_compile_features(libvertirta_flight_core_wasm PRIVATE cxx_std_17)
  target_compile_options(libvertirta_flight_core_wasm PRIVATE
    -Os -fno-rtti -fno-exceptions -fno-threadsafe-statics
    -ffp-contract=off -fno-fast-math
  )
  set_target_properties(libvertirta_flight_core_wasm PROPERTIES
    OUTPUT_NAME "libvertirta_flight_core"
    SUFFIX ".mjs"
    RUNTIME_OUTPUT_DIRECTORY "${VERTIRTA_WASM_OUT_DIR}"
  )
  target_link_options(libvertirta_flight_core_wasm PRIVATE
    -Os
    -sENVIRONMENT=worker
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sALLOW_MEMORY_GROWTH=0
    -sINITIAL_MEMORY=262144
    -sSTACK_SIZE=32768
    -sFILESYSTEM=0
    -sASSERTIONS=0
    "-sEXPORTED_FUNCTIONS=@${CMAKE_CURRENT_LIST_DIR}/exported_funcs.txt"
    "-sEXPORTED_RUNTIME_METHODS=['HEAPU8','HEAPF64']"
  )
endif()
