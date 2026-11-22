############################################################################
# VertiRTA-X8-240 400 Hz PX4-Internal TMR + INDI + f64/QR WLS Module
# Linked via PX4_EXTERNAL_MODULES_LOCATION against freestanding C++17 vertirta_flight_core
############################################################################

set(VERTIRTA_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH "Path to VertiRTA repository root")

if(NOT TARGET vertirta_flight_core)
	add_subdirectory("${VERTIRTA_REPO_ROOT}/core/flight_core" "${CMAKE_CURRENT_BINARY_DIR}/vertirta_flight_core" EXCLUDE_FROM_ALL)
endif()

px4_add_module(
	MODULE modules__mc_indi_wls_control
	MAIN mc_indi_wls_control
	COMPILE_FLAGS
		-fno-exceptions
		-fno-rtti
		-fno-fast-math
		-ffp-contract=off
	SRCS
		mc_indi_wls_control.hpp
		mc_indi_wls_control.cpp
	DEPENDS
		vertirta_flight_core
		px4_work_queue
)
