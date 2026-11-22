add_library(vertirta_rta_companion STATIC
  src/mmap_cbvf_lut.cpp
  src/cel_2d_lut.cpp
  src/dryden_wind.cpp
  src/cbvf_safety_filter.cpp
  src/robocentric_esdf.cpp
  src/wire_capsules.cpp
  src/intruder_kf.cpp
  src/slz_reach.cpp
)
add_library(vertirta::rta_companion ALIAS vertirta_rta_companion)

target_compile_features(vertirta_rta_companion PUBLIC cxx_std_20)
target_include_directories(vertirta_rta_companion
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(vertirta_rta_companion PUBLIC vertirta::flight_core)
target_compile_options(vertirta_rta_companion PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
