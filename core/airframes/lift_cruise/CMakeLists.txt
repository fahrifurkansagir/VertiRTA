add_library(vertirta_airframes_lc STATIC
  src/coaxial_x8_adapter.cpp
  src/lift_cruise_dynamics.cpp
  src/transition_indi_wls.cpp
)
add_library(vertirta::airframes_lc ALIAS vertirta_airframes_lc)

target_compile_features(vertirta_airframes_lc PUBLIC cxx_std_20)
target_include_directories(vertirta_airframes_lc
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(vertirta_airframes_lc PUBLIC vertirta::flight_core)
target_compile_options(vertirta_airframes_lc PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
