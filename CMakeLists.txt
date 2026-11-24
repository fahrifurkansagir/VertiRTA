cmake_minimum_required(VERSION 3.25)
project(VertiRTA VERSION 3.0.0 LANGUAGES C CXX)

include(CTest)
include(FetchContent)

option(VERTIRTA_BUILD_COMPANION "Build C++20 companion RTA target" ON)
option(VERTIRTA_BUILD_TESTS "Build VertiRTA Catch2 unit tests" ON)
option(VERTIRTA_ENABLE_STRICT_FP "Enforce IEEE-754 deterministic FP flags" ON)
option(VERTIRTA_INSTALL "Generate relocatable VertiRTA CMake install & export rules" ON)
option(VERTIRTA_BUILD_PYTHON "Build nanobind Python SDK extension module" OFF)

set(VERTIRTA_STRICT_FP_FLAGS
  -fno-fast-math
  -ffp-contract=off
  -fno-associative-math
  -fno-reciprocal-math
  -frounding-math
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/flight_core/CMakeLists.txt")
  add_subdirectory(core/flight_core)
endif()

if(VERTIRTA_BUILD_COMPANION)
  foreach(_vertirta_opt_dir
      core/rta_companion
      core/stl
      core/copilot
      core/slz_bayes
      core/airframes/lift_cruise
      core/wind_acoustics
      core/uspace_fleet
      core/fleet_z3
      core/forward_hazard
      sim/gazebo_plugins/mmwave_77ghz
      core/gnss_denied
      core/voice
      core/forensics
      tools/cli
      sim/colab_egl
      benchmarks/compound_fault_suite)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_vertirta_opt_dir}/CMakeLists.txt")
      add_subdirectory(${_vertirta_opt_dir})
    endif()
  endforeach()
endif()

if(VERTIRTA_BUILD_TESTS AND BUILD_TESTING)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/flight_core/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/rta_companion/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/determinism/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/stl/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/copilot/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/airframes/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/wind_acoustics/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/uspace_fleet/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/fleet_z3/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/forward_hazard/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/gnss_denied/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/voice/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/forensics/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/slz_bayes/tests/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sim/colab_egl/tests/CMakeLists.txt"
     OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/CMakeLists.txt")
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.7.1
      GIT_SHALLOW TRUE
      FIND_PACKAGE_ARGS 3
    )
    FetchContent_MakeAvailable(Catch2)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/flight_core/CMakeLists.txt")
    add_subdirectory(tests/flight_core)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/rta_companion/CMakeLists.txt")
    add_subdirectory(tests/rta_companion)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/stl/CMakeLists.txt")
    add_subdirectory(tests/stl)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/copilot/CMakeLists.txt")
    add_subdirectory(tests/copilot)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/airframes/CMakeLists.txt")
    add_subdirectory(tests/airframes)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/wind_acoustics/CMakeLists.txt")
    add_subdirectory(tests/wind_acoustics)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/uspace_fleet/CMakeLists.txt")
    add_subdirectory(tests/uspace_fleet)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/fleet_z3/CMakeLists.txt")
    add_subdirectory(tests/fleet_z3)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/forward_hazard/CMakeLists.txt")
    add_subdirectory(tests/forward_hazard)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/gnss_denied/CMakeLists.txt")
    add_subdirectory(tests/gnss_denied)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/voice/CMakeLists.txt")
    add_subdirectory(tests/voice)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/forensics/CMakeLists.txt")
    add_subdirectory(tests/forensics)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/slz_bayes/tests/CMakeLists.txt")
    add_subdirectory(core/slz_bayes/tests)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sim/colab_egl/tests/CMakeLists.txt")
    add_subdirectory(sim/colab_egl/tests)
  endif()
  if(VERTIRTA_BUILD_COMPANION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/CMakeLists.txt")
    add_subdirectory(tests/integration)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/determinism/CMakeLists.txt")
    add_subdirectory(tests/determinism)
  endif()
endif()

if(VERTIRTA_BUILD_PYTHON AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sdk/python/CMakeLists.txt")
  add_subdirectory(sdk/python)
endif()

if(VERTIRTA_INSTALL AND NOT CMAKE_CROSSCOMPILING AND TARGET vertirta_flight_core)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  install(TARGETS vertirta_flight_core
    EXPORT VertiRTAFlightCoreTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    FILE_SET HEADERS DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )

  install(EXPORT VertiRTAFlightCoreTargets
    FILE VertiRTAFlightCoreTargets.cmake
    NAMESPACE VertiRTA::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VertiRTA"
  )

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VertiRTAConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/VertiRTAConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VertiRTA"
  )

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/VertiRTAConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/VertiRTAConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/VertiRTAConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VertiRTA"
  )
endif()
