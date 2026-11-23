# SPDX-License-Identifier: Apache-2.0
# Isolated C++20 Two-Tier Strategic Fleet Z3 Scheduler (Tier-1 QF_RDL STN + Tier-2 local QF_LRA).

option(VERTIRTA_REQUIRE_Z3 "Fail configure if native libz3 (z3++.h + libz3) is missing" OFF)

add_library(vertirta_fleet_z3 STATIC
  src/qf_rdl_stn.cpp
  src/qf_lra_local_solver.cpp
  src/fleet_z3_scheduler.cpp
)
add_library(vertirta::fleet_z3 ALIAS vertirta_fleet_z3)

target_compile_features(vertirta_fleet_z3 PUBLIC cxx_std_20)
target_include_directories(vertirta_fleet_z3
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(vertirta_fleet_z3 PRIVATE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wuninitialized
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)

# Optional native Z3 C++ API (MIT-licensed libz3; zero GPL/libgmp dependency on Ubuntu 24.04).
# When VERTIRTA_REQUIRE_Z3=ON (forced in M2 Cloud CI), configure fails loudly if libz3 is absent.
set(_vertirta_fleet_z3_found OFF)
find_package(Z3 QUIET)
if(Z3_FOUND AND TARGET z3::libz3)
  target_compile_definitions(vertirta_fleet_z3 PUBLIC VERTIRTA_HAS_Z3=1)
  target_link_libraries(vertirta_fleet_z3 PUBLIC z3::libz3)
  set(_vertirta_fleet_z3_found ON)
else()
  find_path(VERTIRTA_Z3_INCLUDE_DIR z3++.h)
  find_library(VERTIRTA_Z3_LIBRARY z3)
  if(VERTIRTA_Z3_INCLUDE_DIR AND VERTIRTA_Z3_LIBRARY)
    target_compile_definitions(vertirta_fleet_z3 PUBLIC VERTIRTA_HAS_Z3=1)
    target_include_directories(vertirta_fleet_z3 SYSTEM PUBLIC "${VERTIRTA_Z3_INCLUDE_DIR}")
    target_link_libraries(vertirta_fleet_z3 PUBLIC "${VERTIRTA_Z3_LIBRARY}")
    set(_vertirta_fleet_z3_found ON)
  endif()
endif()

if(VERTIRTA_REQUIRE_Z3 AND NOT _vertirta_fleet_z3_found)
  message(FATAL_ERROR
    "VERTIRTA_REQUIRE_Z3=ON but native libz3 (z3++.h + libz3) was not found. "
    "Install libz3-dev or disable VERTIRTA_REQUIRE_Z3 for local fallback builds."
  )
endif()
