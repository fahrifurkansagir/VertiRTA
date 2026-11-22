// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "vertirta/core/vrta_recorder.hpp"
#include "vertirta_wasm_abi.hpp"

TEST_CASE("WASM flat-C ABI runs deterministic 400 Hz control + fault injection with zero heap") {
  REQUIRE(vertirta_init(40.8860, 29.1880, 12.0, 36.7) == 0);
  REQUIRE(vertirta_get_vrta_size() == 256U);

  // Reject invalid geoid undulation (guards against silent 0.0 m fallback)
  REQUIRE(vertirta_init(40.8860, 29.1880, 12.0, 0.0) != 0);
  REQUIRE(vertirta_init(40.8860, 29.1880, 12.0, 36.7) == 0);

  // Step 400 ticks (1.0 s) nominal hover
  REQUIRE(vertirta_step_batch(400U, 0.0, 0.0, 0.0) == 0);
  const auto* frame_ptr = reinterpret_cast<const vertirta::core::VrtaFrame100Hz*>(vertirta_get_vrta_ptr());
  REQUIRE(frame_ptr != nullptr);
  REQUIRE(vertirta::core::verify_vrta_frame(*frame_ptr));
  REQUIRE(frame_ptr->active_motor_mask == 0xFFU);
  REQUIRE(frame_ptr->cel_latched == 0U);
  REQUIRE(frame_ptr->tmr_state == 0U);
  REQUIRE(frame_ptr->v_term_v > 350.0f);
  REQUIRE(frame_ptr->ltfj_margin_m > 0.0f);

  // Inject IMU #1 gyro bias -> TmrVoter isolates within 20 ms to DUPLEX_2OF2
  REQUIRE(vertirta_inject_fault(VERTIRTA_FAULT_IMU_BIAS, 1U) == 0);
  REQUIRE(vertirta_step_batch(40U, 0.0, 0.0, 0.0) == 0);
  REQUIRE(vertirta::core::verify_vrta_frame(*frame_ptr));
  REQUIRE(frame_ptr->tmr_state == 1U);

  // Inject same-arm dual-motor loss (Arm 0: rotors 0 & 4) -> Option B1b flat spin CEL latch
  REQUIRE(vertirta_inject_fault(VERTIRTA_FAULT_SAME_ARM_DUAL, 0U) == 0);
  REQUIRE(vertirta_step_batch(400U, 0.0, 0.0, 0.0) == 0);
  REQUIRE(vertirta::core::verify_vrta_frame(*frame_ptr));
  REQUIRE(frame_ptr->cel_latched == 1U);
  REQUIRE((frame_ptr->active_motor_mask & 0x11U) == 0U);
  REQUIRE(std::isfinite(frame_ptr->v_term_v));
  REQUIRE(frame_ptr->v_term_v > 300.0f);

  // Geoid lookup anchored to Kartal EGM2008 +36.7 m
  REQUIRE_THAT(vertirta_egm2008_n_m(40.8860, 29.1880), Catch::Matchers::WithinAbs(36.7, 0.05));
  REQUIRE(vertirta_eval_cbvf_margin(0.0, 0.0, 35.0, 0U) > 0.0);
  REQUIRE(vertirta_eval_cbvf_margin(5000.0, 0.0, 35.0, 0U) < 0.0);
}

TEST_CASE("WASM ABI .vrta parser validates Castagnoli CRC32C and rejects corrupted frames") {
  REQUIRE(vertirta_init(40.8860, 29.1880, 12.0, 36.7) == 0);
  REQUIRE(vertirta_step_batch(4U, 0.0, 0.0, 0.0) == 0);
  uint8_t buf[256]{};
  std::memcpy(buf, vertirta_get_vrta_ptr(), sizeof(buf));
  REQUIRE(vertirta_parse_vrta_frame(buf, sizeof(buf)) == 0);
  REQUIRE(vertirta_parse_vrta_frame(buf, 128U) != 0);
  buf[42] ^= 0x5AU;  // Flip payload bits -> CRC mismatch
  REQUIRE(vertirta_parse_vrta_frame(buf, sizeof(buf)) != 0);
}

TEST_CASE("WASM ABI deterministic reset reproduces bit-exact VRTA0001 frames") {
  REQUIRE(vertirta_init(40.8860, 29.1880, 12.0, 36.7) == 0);
  REQUIRE(vertirta_inject_fault(VERTIRTA_FAULT_SINGLE_MOTOR, 2U) == 0);
  REQUIRE(vertirta_step_batch(200U, 1.0, -0.5, 0.0) == 0);
  uint8_t run1[256]{};
  std::memcpy(run1, vertirta_get_vrta_ptr(), sizeof(run1));

  REQUIRE(vertirta_reset() == 0);
  REQUIRE(vertirta_inject_fault(VERTIRTA_FAULT_SINGLE_MOTOR, 2U) == 0);
  REQUIRE(vertirta_step_batch(200U, 1.0, -0.5, 0.0) == 0);
  uint8_t run2[256]{};
  std::memcpy(run2, vertirta_get_vrta_ptr(), sizeof(run2));

  REQUIRE(std::memcmp(run1, run2, sizeof(run1)) == 0);
}
