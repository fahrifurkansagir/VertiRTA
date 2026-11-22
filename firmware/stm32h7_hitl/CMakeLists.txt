cmake_minimum_required(VERSION 3.22)

# Standalone Bare-Metal ARM Cortex-M7 STM32H753 / NUCLEO-H743ZI2 HITL Harness.
# Intentionally decoupled from root CMakeLists.txt so Plan 4A never touches frozen flight_core.
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)

project(VertiRTA_STM32H7_HITL VERSION 3.0.0 LANGUAGES C CXX ASM)

get_filename_component(VERTIRTA_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

set(VERTIRTA_FLIGHT_CORE_LIB
  "${VERTIRTA_REPO_ROOT}/build/arm-cortex-m7/core/flight_core/libvertirta_flight_core.a"
  CACHE FILEPATH "Path to frozen Cortex-M7 libvertirta_flight_core.a"
)
option(VERTIRTA_ENABLE_TINYUSB "Enable optional MIT TinyUSB CDC hook" OFF)

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

add_library(vertirta_stm32h7_bsp STATIC
  startup/startup_stm32h753.S
  src/stm32h7_boot.cpp
  src/dwt_profiler.cpp
  src/uart_console.cpp
)
target_include_directories(vertirta_stm32h7_bsp PUBLIC
  "${CMAKE_CURRENT_LIST_DIR}/include"
)
target_compile_features(vertirta_stm32h7_bsp PUBLIC cxx_std_17)
target_compile_options(vertirta_stm32h7_bsp PRIVATE
  $<$<COMPILE_LANGUAGE:ASM>:-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard>
  $<$<COMPILE_LANGUAGE:CXX>:${VERTIRTA_M7_FLAGS}
    -Wall -Wextra -Wpedantic -Werror -Wvla -Wshadow -Wuninitialized
    -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-use-cxa-atexit
    -fstack-usage -fcallgraph-info=su,da -Wstack-usage=3500>
)

add_executable(vertirta_stm32h7_hitl.elf
  src/main_hitl.cpp
)
target_include_directories(vertirta_stm32h7_hitl.elf PRIVATE
  "${VERTIRTA_REPO_ROOT}/core/flight_core/include"
)
target_compile_features(vertirta_stm32h7_hitl.elf PRIVATE cxx_std_17)
target_compile_options(vertirta_stm32h7_hitl.elf PRIVATE
  ${VERTIRTA_M7_FLAGS}
  -Wall -Wextra -Wpedantic -Werror -Wvla -Wshadow -Wuninitialized
  -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-use-cxa-atexit
  -fstack-usage -fcallgraph-info=su,da -Wstack-usage=3500
)
if(VERTIRTA_ENABLE_TINYUSB)
  target_compile_definitions(vertirta_stm32h7_hitl.elf PRIVATE VERTIRTA_ENABLE_TINYUSB=1)
endif()

target_link_libraries(vertirta_stm32h7_hitl.elf PRIVATE
  vertirta_stm32h7_bsp
  "${VERTIRTA_FLIGHT_CORE_LIB}"
  m
  gcc
)
target_link_options(vertirta_stm32h7_hitl.elf PRIVATE
  -mcpu=cortex-m7
  -mthumb
  -mfpu=fpv5-d16
  -mfloat-abi=hard
  -T "${CMAKE_CURRENT_LIST_DIR}/stm32h753_tcm.ld"
  -Wl,--gc-sections
  -Wl,--print-memory-usage
  -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/vertirta_stm32h7_hitl.map
  -Wl,--wrap=malloc
  -Wl,--wrap=calloc
  -Wl,--wrap=realloc
  --specs=nano.specs
  --specs=nosys.specs
  -nostartfiles
)
