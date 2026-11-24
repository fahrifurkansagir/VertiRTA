# SPDX-License-Identifier: Apache-2.0
# Isolated C++20 Offline Cockpit-Noise TR/EN Voice ATC + Zero-GPL TTS + Mondrian KnowNo Gate.

option(VERTIRTA_REQUIRE_WHISPER "Fail configure if native whisper.cpp (MIT) is missing" OFF)
option(VERTIRTA_REQUIRE_SHERPA_ONNX "Fail configure if native sherpa-onnx >= 2.0.0 (Apache-2.0) is missing" OFF)

add_library(vertirta_voice STATIC
  src/voice_types.cpp
  src/silero_vad.cpp
  src/dual_tier_whisper_asr.cpp
  src/zero_gpl_tts.cpp
  src/mondrian_voice_gate.cpp
  src/voice_pipeline.cpp
)
add_library(vertirta::voice ALIAS vertirta_voice)

target_compile_features(vertirta_voice PUBLIC cxx_std_20)
target_include_directories(vertirta_voice
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(vertirta_voice PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)

# Optional native whisper.cpp (MIT)
find_path(VERTIRTA_WHISPER_INCLUDE_DIR whisper.h)
find_library(VERTIRTA_WHISPER_LIBRARY whisper)
if(VERTIRTA_WHISPER_INCLUDE_DIR AND VERTIRTA_WHISPER_LIBRARY)
  target_compile_definitions(vertirta_voice PUBLIC VERTIRTA_HAS_WHISPER=1)
  target_include_directories(vertirta_voice SYSTEM PUBLIC "${VERTIRTA_WHISPER_INCLUDE_DIR}")
  target_link_libraries(vertirta_voice PUBLIC "${VERTIRTA_WHISPER_LIBRARY}")
elseif(VERTIRTA_REQUIRE_WHISPER)
  message(FATAL_ERROR "VERTIRTA_REQUIRE_WHISPER=ON but native whisper.cpp was not found.")
endif()

# Optional native sherpa-onnx >= 2.0.0 (Apache-2.0, PR #4055 zero espeak-ng/piper-phonemize).
# Reject any sherpa-onnx < 2.0.0 to eliminate GPL-3.0 contamination.
find_path(VERTIRTA_SHERPA_INCLUDE_DIR sherpa-onnx/c-api/c-api.h)
find_library(VERTIRTA_SHERPA_LIBRARY sherpa-onnx-c-api)
if(VERTIRTA_SHERPA_INCLUDE_DIR AND VERTIRTA_SHERPA_LIBRARY)
  set(_sherpa_ver_ok OFF)
  if(DEFINED VERTIRTA_SHERPA_ONNX_VERSION AND VERTIRTA_SHERPA_ONNX_VERSION VERSION_GREATER_EQUAL "2.0.0")
    set(_sherpa_ver_ok ON)
  endif()
  if(NOT _sherpa_ver_ok)
    message(FATAL_ERROR
      "Detected sherpa-onnx headers without verified VERTIRTA_SHERPA_ONNX_VERSION >= 2.0.0. "
      "Versions < 2.0.0 bundle GPL-3.0 espeak-ng/piper-phonemize and are strictly banned."
    )
  endif()
  target_compile_definitions(vertirta_voice PUBLIC VERTIRTA_HAS_SHERPA_ONNX=1)
  target_include_directories(vertirta_voice SYSTEM PUBLIC "${VERTIRTA_SHERPA_INCLUDE_DIR}")
  target_link_libraries(vertirta_voice PUBLIC "${VERTIRTA_SHERPA_LIBRARY}")
elseif(VERTIRTA_REQUIRE_SHERPA_ONNX)
  message(FATAL_ERROR "VERTIRTA_REQUIRE_SHERPA_ONNX=ON but native sherpa-onnx >= 2.0.0 was not found.")
endif()
