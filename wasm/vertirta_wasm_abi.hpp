// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Fault injection identifiers for VertiGCS Sandbox & Live Fault-Injection Panel
enum VertirtaWasmFaultKind {
  VERTIRTA_FAULT_NONE = 0U,
  VERTIRTA_FAULT_SINGLE_MOTOR = 1U,
  VERTIRTA_FAULT_SAME_ARM_DUAL = 2U,
  VERTIRTA_FAULT_IMU_BIAS = 3U,
  VERTIRTA_FAULT_WIND_GUST = 4U,
};

// Initialize static zero-heap simulation arena anchored to Kartal ENU datum.
// Returns 0 on success, -1 on invalid datum/geoid parameters (guards against N0 = 0.0 m).
int32_t vertirta_init(
    double datum_lat_deg,
    double datum_lon_deg,
    double datum_amsl_m,
    double egm2008_n_m);

// Deterministically resets arena to initial datum/hover state (bit-exact replay parity).
int32_t vertirta_reset(void);

// Step closed-loop 400 Hz CoaxialPlant6Dof + TmrVoter + MotorFdiObserver + IndiWlsController
// for n_ticks_400hz ticks and finalize latest 256-byte VRTA0001 frame with CRC-32C.
int32_t vertirta_step_batch(
    uint32_t n_ticks_400hz,
    double a_nom_e_mps2,
    double a_nom_n_mps2,
    double a_nom_u_mps2);

// Inject runtime fault (single motor, same-arm dual-motor CEL, IMU bias, or coastal wind gust).
int32_t vertirta_inject_fault(uint32_t fault_kind, uint32_t target_idx);

// Zero-copy pointer to static 256-byte VrtaFrame100Hz in linear memory.
const uint8_t* vertirta_get_vrta_ptr(void);

// Canonical VrtaFrame100Hz byte size (always 256).
uint32_t vertirta_get_vrta_size(void);

// Validate and decode an external 256-byte VRTA0001 frame (checks Castagnoli CRC-32C).
// Returns 0 if valid, -1 on size mismatch or CRC failure.
int32_t vertirta_parse_vrta_frame(const uint8_t* bytes, uint32_t len);

// Embedded <1 KB 5'x5' Istanbul EGM2008 bilinear geoid undulation [m] anchored to +36.7 m at Kartal.
double vertirta_egm2008_n_m(double lat_deg, double lon_deg);

// Conservative local ENU safety margin [m] against LTFJ (incl. NW J2 pinch), LTBX, and 120 m ceiling.
double vertirta_eval_cbvf_margin(double east_m, double north_m, double agl_m, uint8_t cel_latched);

#ifdef __cplusplus
}
#endif
