if(NOT TARGET Catch2::Catch2WithMain)
  find_package(Catch2 3 REQUIRED)
endif()

add_executable(test_gnss_spoof_mcusum_vio
  test_gnss_spoof_mcusum_vio.cpp
)
target_link_libraries(test_gnss_spoof_mcusum_vio
  PRIVATE
    vertirta::gnss_denied
    Catch2::Catch2WithMain
)
target_compile_features(test_gnss_spoof_mcusum_vio PRIVATE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_gnss_spoof_mcusum_vio PRIVATE -Wall -Wextra -Wpedantic -Werror)
  if(VERTIRTA_ENABLE_STRICT_FP)
    target_compile_options(test_gnss_spoof_mcusum_vio PRIVATE ${VERTIRTA_STRICT_FP_FLAGS})
  endif()
endif()
add_test(NAME test_gnss_spoof_mcusum_vio COMMAND test_gnss_spoof_mcusum_vio)

add_executable(test_visual_reloc_and_ctr_inflation
  test_visual_reloc_and_ctr_inflation.cpp
)
target_link_libraries(test_visual_reloc_and_ctr_inflation
  PRIVATE
    vertirta::gnss_denied
    Catch2::Catch2WithMain
)
target_compile_features(test_visual_reloc_and_ctr_inflation PRIVATE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_visual_reloc_and_ctr_inflation PRIVATE -Wall -Wextra -Wpedantic -Werror)
  if(VERTIRTA_ENABLE_STRICT_FP)
    target_compile_options(test_visual_reloc_and_ctr_inflation PRIVATE ${VERTIRTA_STRICT_FP_FLAGS})
  endif()
endif()
add_test(NAME test_visual_reloc_and_ctr_inflation COMMAND test_visual_reloc_and_ctr_inflation)
