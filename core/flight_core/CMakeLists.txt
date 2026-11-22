add_library(vertirta_flight_core STATIC
  src/frames.cpp
  src/tmr_voter.cpp
)
add_library(vertirta::flight_core ALIAS vertirta_flight_core)

target_compile_features(vertirta_flight_core PUBLIC cxx_std_17)
target_include_directories(vertirta_flight_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(vertirta_flight_core PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror
  -Wshadow
  -Wconversion
  -Wuninitialized
  -ffunction-sections
  -fdata-sections
  $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions -fno-rtti -fno-threadsafe-statics>
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
