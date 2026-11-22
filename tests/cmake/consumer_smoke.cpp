// SPDX-License-Identifier: Apache-2.0
// Standalone downstream CMake find_package(VertiRTA CONFIG REQUIRED COMPONENTS flight_core) smoke test.
#include <cmath>
#include <cstdint>
#include <iostream>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/core/indi_wls.hpp"
#include "vertirta/core/sim_plant_6dof.hpp"
#include "vertirta/core/vrta_recorder.hpp"

int main() {
  using namespace vertirta::core;

  static_assert(sizeof(VrtaFileHeader) == 128U, "VrtaFileHeader must remain 128 B");
  static_assert(sizeof(VrtaFrame100Hz) == 256U, "VrtaFrame100Hz must remain 256 B");

  VrtaFileHeader hdr = make_default_vrta_header();
  finalize_vrta_header(hdr);
  if (!verify_vrta_header(hdr)) {
    std::cerr << "FAIL: VrtaFileHeader CRC32C check failed\n";
    return 1;
  }

  CoaxialPlant6Dof plant{};
  IndiWlsController indi{true};
  plant.reset({0.0, 0.0, -35.0}, {0.0, 0.0, 0.0}, 0.80f, params::kOmegaHover);
  indi.reset(params::kOmegaHover);

  float omega_cmd[8]{};
  for (int tick = 0; tick < 40; ++tick) {
    const PlantState ps = plant.state();
    IndiInput in{};
    in.q_frd_ned = ps.q_frd_ned;
    in.gyro_frd = ps.omega_frd;
    in.accel_des_ned = {0.0, 0.0, 0.0};
    in.v_term_v = ps.battery.v_term_v;
    (void)indi.step_400hz(in, omega_cmd);
    (void)plant.step_400hz(omega_cmd, 0x00U, {0.0, 0.0, 0.0});
  }

  const PlantState final_ps = plant.state();
  if (!std::isfinite(final_ps.battery.v_term_v) || final_ps.battery.v_term_v < 320.0f) {
    std::cerr << "FAIL: Unexpected v_term_v=" << final_ps.battery.v_term_v << "\n";
    return 2;
  }

  VrtaFrame100Hz frame{};
  frame.seq = 10U;
  frame.t_sim_us = 100000U;
  frame.v_term_v = final_ps.battery.v_term_v;
  frame.soc_pct = final_ps.battery.soc * 100.0f;
  frame.agl_m = static_cast<float>(-final_ps.pos_ned.z);
  finalize_vrta_frame(frame);
  if (!verify_vrta_frame(frame)) {
    std::cerr << "FAIL: VrtaFrame100Hz CRC32C check failed\n";
    return 3;
  }

  std::cout << "PASS: VertiRTA::flight_core relocatable consumer smoke ok (v_term="
            << frame.v_term_v << " V, agl=" << frame.agl_m << " m)\n";
  return 0;
}
