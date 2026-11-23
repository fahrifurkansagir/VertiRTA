add_library(vertirta_wind_acoustics STATIC
  src/mmap_vrta_wind_lut.cpp
  src/aeroacoustics.cpp
  src/istanbul_receptors.cpp
  src/noise_aware_router.cpp
)
add_library(vertirta::wind_acoustics ALIAS vertirta_wind_acoustics)

target_compile_features(vertirta_wind_acoustics PUBLIC cxx_std_20)
target_include_directories(vertirta_wind_acoustics
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(vertirta_wind_acoustics
  PUBLIC
    vertirta::flight_core
    vertirta::rta_companion
)
target_compile_options(vertirta_wind_acoustics PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
