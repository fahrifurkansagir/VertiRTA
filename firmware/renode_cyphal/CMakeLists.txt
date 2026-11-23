cmake_minimum_required(VERSION 3.22)

# Standalone OpenCyphal v1.0 + o1heap v3.0.0 + Dual-STM32H753 Lockstep Firmware CMakeLists.
# Supports two clean modes without touching root CMakeLists.txt or frozen core/flight_core/:
#   1. Host mode (-DVERTIRTA_CYPHAL_BARE_METAL=OFF): builds static lib vertirta_cyphal_core
#      for Catch2 v3.7.1 + ASan/UBSan + zero-heap verification in tests/firmware/.
#   2. Bare-metal Cortex-M7 mode (-DVERTIRTA_CYPHAL_BARE_METAL=ON): cross-compiles
#      vertirta_fcc_primary.elf (Node 10), vertirta_fcc_secondary.elf (Node 11), and
#      vertirta_cyphal_node.elf linking frozen libvertirta_flight_core.a into ITCM/DTCM.

option(VERTIRTA_CYPHAL_BARE_METAL "Build bare-metal STM32H753 Cortex-M7 dual-FCC ELFs" ON)

if(VERTIRTA_CYPHAL_BARE_METAL)
  set(CMAKE_SYSTEM_NAME Generic)
  set(CMAKE_SYSTEM_PROCESSOR arm)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  set(CMAKE_C_COMPILER arm-none-eabi-gcc)
  set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
  set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
  project(VertiRTA_Renode_Cyphal VERSION 3.0.0 LANGUAGES C CXX ASM)
else()
  project(VertiRTA_Renode_Cyphal VERSION 3.0.0 LANGUAGES C CXX)
endif()

get_filename_component(VERTIRTA_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
get_filename_component(VERTIRTA_HITL_DIR "${CMAKE_CURRENT_LIST_DIR}/../stm32h7_hitl" ABSOLUTE)

set(VERTIRTA_FLIGHT_CORE_LIB
  "${VERTIRTA_REPO_ROOT}/build/arm-cortex-m7/core/flight_core/libvertirta_flight_core.a"
  CACHE FILEPATH "Path to frozen Cortex-M7 libvertirta_flight_core.a"
)

add_library(vertirta_cyphal_core STATIC
  src/safety_envelope_codec.c
  src/o1heap_arena.cpp
  src/cyphal_bus_node.cpp
)
target_include_directories(vertirta_cyphal_core PUBLIC
  "${CMAKE_CURRENT_LIST_DIR}/include"
)
target_compile_features(vertirta_cyphal_core PUBLIC c_std_11 cxx_std_17)

if(VERTIRTA_CYPHAL_BARE_METAL)
  set(VERTIRTA_M7_FLAGS
    -mcpu=cortex-m7
    -mthumb
    -mfpu=fpv5-d16
    -mfloat-abi=hard
    -Os
    -ffunction-sections
    -fdata-sections
    -fno-fast-math
    -ffp-contract=off
    -fno-associative-math
    -fno-reciprocal-math
    -frounding-math
    -mlong-calls
  )

  target_compile_options(vertirta_cyphal_core PRIVATE
    ${VERTIRTA_M7_FLAGS}
    -Wall -Wextra -Wpedantic -Werror -Wvla -Wshadow -Wuninitialized
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-use-cxa-atexit>
    -fstack-usage -fcallgraph-info=su,da -Wstack-usage=3500
  )

  add_library(vertirta_cyphal_bsp STATIC
    "${VERTIRTA_HITL_DIR}/startup/startup_stm32h753.S"
    "${VERTIRTA_HITL_DIR}/src/stm32h7_boot.cpp"
    "${VERTIRTA_HITL_DIR}/src/dwt_profiler.cpp"
    "${VERTIRTA_HITL_DIR}/src/uart_console.cpp"
    src/fdcan_stm32h7.cpp
  )
  target_include_directories(vertirta_cyphal_bsp PUBLIC
    "${VERTIRTA_HITL_DIR}/include"
    "${CMAKE_CURRENT_LIST_DIR}/include"
  )
  target_compile_features(vertirta_cyphal_bsp PUBLIC cxx_std_17)
  target_compile_definitions(vertirta_cyphal_bsp PRIVATE VERTIRTA_RENODE_SIL=1)
  target_compile_options(vertirta_cyphal_bsp PRIVATE
    $<$<COMPILE_LANGUAGE:ASM>:-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard>
    $<$<COMPILE_LANGUAGE:CXX>:${VERTIRTA_M7_FLAGS}
      -Wall -Wextra -Wpedantic -Werror -Wvla -Wshadow -Wuninitialized
      -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-use-cxa-atexit
      -fstack-usage -fcallgraph-info=su,da -Wstack-usage=3500>
  )

  function(add_vertirta_fcc_elf target_name node_id)
    add_executable(${target_name} src/main_fcc_node.cpp)
    target_compile_definitions(${target_name} PRIVATE VERTIRTA_FCC_NODE_ID=${node_id}u)
    target_include_directories(${target_name} PRIVATE
      "${VERTIRTA_REPO_ROOT}/core/flight_core/include"
      "${VERTIRTA_HITL_DIR}/include"
      "${CMAKE_CURRENT_LIST_DIR}/include"
    )
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_compile_options(${target_name} PRIVATE
      ${VERTIRTA_M7_FLAGS}
      -Wall -Wextra -Wpedantic -Werror -Wvla -Wshadow -Wuninitialized
      -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-use-cxa-atexit
      -fstack-usage -fcallgraph-info=su,da -Wstack-usage=3500
    )
    target_link_libraries(${target_name} PRIVATE
      vertirta_cyphal_core
      vertirta_cyphal_bsp
      "${VERTIRTA_FLIGHT_CORE_LIB}"
      m
      gcc
    )
    target_link_options(${target_name} PRIVATE
      -mcpu=cortex-m7
      -mthumb
      -mfpu=fpv5-d16
      -mfloat-abi=hard
      -T "${VERTIRTA_HITL_DIR}/stm32h753_tcm.ld"
      -Wl,--gc-sections
      -Wl,--print-memory-usage
      -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/${target_name}.map
      -Wl,--wrap=malloc
      -Wl,--wrap=calloc
      -Wl,--wrap=realloc
      --specs=nano.specs
      --specs=nosys.specs
      -nostartfiles
    )
  endfunction()

  add_vertirta_fcc_elf(vertirta_fcc_primary.elf 10)
  add_vertirta_fcc_elf(vertirta_fcc_secondary.elf 11)
  add_vertirta_fcc_elf(vertirta_cyphal_node.elf 10)
else()
  target_compile_options(vertirta_cyphal_core PRIVATE
    -Wall -Wextra -Wpedantic -Werror -Wvla -Wshadow
    -fno-fast-math -ffp-contract=off
  )
endif()
