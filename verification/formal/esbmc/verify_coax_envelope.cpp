// SPDX-License-Identifier: Apache-2.0
// ESBMC v8.7 Harness 4: Coaxial X8 Capability Envelope & 1-RC Thevenin Battery (--unwind 10)
// Verifies DR-1 voltage-headroom monotonicity (358.4 V -> T/W >= 1.25, 338.0 V -> T/W >= 1.12),
// FOC 240 rad/s cap, and finite TheveninBattery1RC outputs under extreme/non-finite loads.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "esbmc_compat.h"
#include "vertirta/core/coax_envelope.hpp"

namespace {

using vertirta::core::BatteryState;
using vertirta::core::CapabilityEnvelope;
using vertirta::core::CoaxEffectiveness;
using vertirta::core::TheveninBattery1RC;

void verify_coax_and_battery(float v1, float v2, float dt_s, float p_elec_w, uint8_t fault_mask) {
  // 1) Monotonicity & DR-1 voltage-headroom locks on two_arm_tw_at_vterm
  const float tw1 = CoaxEffectiveness::two_arm_tw_at_vterm(v1);
  const float tw2 = CoaxEffectiveness::two_arm_tw_at_vterm(v2);
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_f(tw1), "two_arm_tw_at_vterm(v1) must be finite");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_f(tw2), "two_arm_tw_at_vterm(v2) must be finite");
  if (v1 <= v2) {
    VERTIRTA_FORMAL_ASSERT(tw1 <= tw2 + 1e-5f, "two_arm_tw_at_vterm must be monotone in v_term");
  }

  const float tw_358 = CoaxEffectiveness::two_arm_tw_at_vterm(358.4f);
  const float tw_338 = CoaxEffectiveness::two_arm_tw_at_vterm(338.0f);
  VERTIRTA_FORMAL_ASSERT(tw_358 >= 1.249f, "DR-1 lock: 2-arm T/W >= 1.25 at 358.4 V");
  VERTIRTA_FORMAL_ASSERT(tw_338 >= 1.120f, "DR-1 lock: 2-arm T/W >= 1.12 at 338.0 V");

  const float omega_lim = CoaxEffectiveness::omega_lim_from_vterm(v1);
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_f(omega_lim), "omega_lim must be finite");
  VERTIRTA_FORMAL_ASSERT(omega_lim >= 0.0f && omega_lim <= 240.001f, "omega_lim capped at 240 rad/s");

  // 2) TheveninBattery1RC finite-output & bounded SoC guard
  TheveninBattery1RC batt{0.60f};
  const BatteryState st = batt.step(dt_s, p_elec_w);
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_f(st.v_oc_v), "v_oc_v must be finite");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_f(st.v_rc_v), "v_rc_v must be finite");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_f(st.v_term_v), "v_term_v must be finite");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_f(st.i_pack_a), "i_pack_a must be finite");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_f(st.soc), "soc must be finite");
  VERTIRTA_FORMAL_ASSERT(st.soc >= 0.0f && st.soc <= 1.0f, "soc must remain inside [0, 1]");

  // 3) CapabilityEnvelope finite outputs across all 256 fault masks
  const CapabilityEnvelope env = CoaxEffectiveness::compute_envelope(
      fault_mask, 0u, 0x07u, st.v_term_v, st.i_pack_a, st.soc, 0.0f, 0.0f, 0.0f, 0u);
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_f(env.t_max_n), "env.t_max_n must be finite");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_f(env.a_xy_max), "env.a_xy_max must be finite");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_f(env.tilt_max_rad), "env.tilt_max_rad must be finite");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_f(env.p_hover_deg_w), "env.p_hover_deg_w must be finite");
}

}  // namespace

int main() {
#ifdef __ESBMC__
  const float v1 = nondet_float();
  const float v2 = nondet_float();
  const float dt_s = nondet_float();
  const float p_elec = nondet_float();
  const uint8_t mask = nondet_uchar();

  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(v1) && v1 >= 320.0f && v1 <= 420.0f);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(v2) && v2 >= 320.0f && v2 <= 420.0f);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(dt_s) && dt_s >= 0.0f && dt_s <= 0.1f);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(p_elec) && p_elec >= 0.0f && p_elec <= 180000.0f);

  verify_coax_and_battery(v1, v2, dt_s, p_elec, mask);
#else
  verify_coax_and_battery(338.0f, 358.4f, 0.0025f, 40500.0f, 0x00u);
  verify_coax_and_battery(358.4f, 403.2f, 0.01f, 95000.0f, 0x03u);
  verify_coax_and_battery(320.0f, 420.0f, 0.0025f, 180000.0f, 0x11u);
  verify_coax_and_battery(340.0f, 340.0f, 0.0f, 0.0f, 0xFFu);
#endif
  return 0;
}
