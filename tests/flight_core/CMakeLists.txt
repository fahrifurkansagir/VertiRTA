add_executable(test_frames_geodesy test_frames_geodesy.cpp)
target_link_libraries(test_frames_geodesy PRIVATE vertirta::flight_core Catch2::Catch2WithMain)
target_compile_features(test_frames_geodesy PRIVATE cxx_std_17)
add_test(NAME test_frames_geodesy COMMAND test_frames_geodesy)

add_executable(test_tmr_voter test_tmr_voter.cpp)
target_link_libraries(test_tmr_voter PRIVATE vertirta::flight_core Catch2::Catch2WithMain)
target_compile_features(test_tmr_voter PRIVATE cxx_std_17)
add_test(NAME test_tmr_voter COMMAND test_tmr_voter)

add_executable(test_indi_wls_thevenin test_indi_wls_thevenin.cpp)
target_link_libraries(test_indi_wls_thevenin PRIVATE vertirta::flight_core Catch2::Catch2WithMain)
target_compile_features(test_indi_wls_thevenin PRIVATE cxx_std_17)
add_test(NAME test_indi_wls_thevenin COMMAND test_indi_wls_thevenin)

add_executable(test_daqp_vrec_plant test_daqp_vrec_plant.cpp zero_heap_guard.cpp)
target_link_libraries(test_daqp_vrec_plant PRIVATE vertirta::flight_core Catch2::Catch2WithMain)
target_compile_features(test_daqp_vrec_plant PRIVATE cxx_std_17)
if(NOT CMAKE_CXX_FLAGS MATCHES "fsanitize=address")
  target_compile_definitions(test_daqp_vrec_plant PRIVATE VERTIRTA_WRAP_MALLOC=1)
  target_link_options(test_daqp_vrec_plant PRIVATE
    -Wl,--wrap=malloc
    -Wl,--wrap=calloc
    -Wl,--wrap=realloc
  )
endif()
add_test(NAME test_daqp_vrec_plant COMMAND test_daqp_vrec_plant)
