#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace vertirta_gz {

inline constexpr double kPi = 3.14159265358979323846;

struct Vec3D {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

// Frozen VertiRTA-X8-240 propulsion + 1-RC Thevenin battery parameters
// Mirrors config/vehicle_x8_240.yaml (Spec v1.2 + DR-1 Forensic Lock).
struct CoaxParams {
  double rho_air{1.225};            // [kg/m^3] ISA sea level
  double k_t0{0.0138091};           // [N/(rad/s)^2] Upper/isolated thrust coeff
  double k_q0{0.00104876};          // [N*m/(rad/s)^2] Upper/isolated torque coeff
  double eta_coax_hover{0.85};      // Lower-rotor hover thrust factor (0.85 * k_T0)
  double swirl_recovery{0.96};      // Lower-rotor hover torque factor (0.96 * k_Q0, 4% swirl recovery)
  double j_r{0.135};                // [kg*m^2] Rotor + BLDC bell inertia
  double rotor_radius_m{0.75};      // [m]
  double coax_spacing_m{0.35};      // [m] Vertical separation between coaxial pair
  double tau_up_s{0.085};           // [s] Powered spin-up time constant
  double tau_down_s{0.120};         // [s] Powered spin-down time constant
  double tau_coast_s{0.424};        // [s] Unpowered coast-down time constant (hover linearized)
  double alpha_max_rad_s2{487.7};   // [rad/s^2] FOC torque-limited angular acceleration ceiling
  double omega_max_nom{240.0};      // [rad/s] Software FOC speed cap

  // BLDC voltage-headroom lock (DR-1): holds 2-arm T/W=1.25 to 358.4 V, >=1.12 to 338 V
  double ke_bus_vs_rad{1.36};       // [V*s/rad]
  double r_eff_hot_ohm{0.72};       // [Ohm]

  // 112S12P 22 kWh 1-RC Thevenin ECM (320 Wh/kg, 68 kg pack)
  double eta_drive{0.88};           // Motor x ESC x harness efficiency
  double p_avionics_w{450.0};       // [W] Constant avionics/companion bus load
  double capacity_ah{54.0};         // [Ah]
  double r0_ohm{0.1363};            // [Ohm] Ohmic series resistance
  double r1_ohm{0.0840};            // [Ohm] RC polarization resistance
  double c1_f{300.0};               // [F] RC polarization capacitance (tau1 = 25.2 s)
  double i_max_cont_a{540.0};       // [A] Continuous pack current limit
  double soc_init{0.85};            // Initial State of Charge [0..1]
};

struct RotorState {
  int pair_upper_idx{-1};  // -1 for upper rotors (0..3), 0..3 for lower rotors (4..7)
  int spin_sign{+1};       // PX4 FRD spin sign: +1 CCW (+tau_z FRD / -tau_z FLU), -1 CW
  double omega{0.0};       // [rad/s] Actual rotor angular velocity
  double omega_cmd{0.0};   // [rad/s] Commanded angular velocity
  bool fault_off{false};   // True when ESC/motor power is cut
};

struct CoaxStepOutput {
  std::array<double, 8> omega_rad_s{};
  std::array<double, 8> dw_rad_s2{};
  std::array<double, 8> thrust_n{};
  std::array<double, 8> torque_z_flu{};      // Reaction torque on airframe in FLU (+Z Up)
  std::array<double, 8> eta_t{};
  std::array<double, 8> eta_q{};
  std::array<double, 4> v_iu_mps{};          // Upper rotor Glauert induced velocity [m/s]
  std::array<double, 4> v_axial_down_mps{};  // Total downward axial flow v_iu + v_air.z [m/s]
  std::array<double, 4> wake_skew_chi_rad{}; // Wake skew angle chi [rad]
  double omega_cap_rad_s{240.0};
  double v_term_v{403.2};
  double i_pack_a{0.0};
  double soc{0.85};
  double p_elec_w{0.0};
};

// Pure C++ coaxial X8 aerodynamics, asymmetric BLDC lag, voltage-ceiling, and 1-RC Thevenin kernel.
// Zero Gazebo dependency so Catch2 unit tests validate the exact physics run by gz-sim8.
class CoaxialPhysicsKernel {
 public:
  explicit CoaxialPhysicsKernel(const CoaxParams& params = CoaxParams{}) noexcept : params_(params) {
    reset();
  }

  void reset() noexcept {
    // Locked spin signs from config/vehicle_x8_240.yaml:
    // Upper 0..3: FR(+1), RL(+1), FL(-1), RR(-1); Lower 4..7: counter-rotating
    static constexpr std::array<int, 8> kSpinSigns = {+1, +1, -1, -1, -1, -1, +1, +1};
    for (int i = 0; i < 8; ++i) {
      rotors_[i] = RotorState{};
      rotors_[i].pair_upper_idx = (i < 4) ? -1 : (i - 4);
      rotors_[i].spin_sign = kSpinSigns[i];
    }
    soc_ = std::clamp(params_.soc_init, 0.0, 1.0);
    v_rc_ = 0.0;
    v_term_ = open_circuit_voltage(soc_);
    v_term_override_enabled_ = false;
    v_term_override_ = v_term_;
  }

  void set_params(const CoaxParams& params) noexcept { params_ = params; }
  [[nodiscard]] const CoaxParams& params() const noexcept { return params_; }
  [[nodiscard]] const RotorState& rotor(int idx) const noexcept { return rotors_[idx]; }
  [[nodiscard]] double soc() const noexcept { return soc_; }
  [[nodiscard]] double v_term() const noexcept { return v_term_; }

  void set_v_term_override(double v_term_v) noexcept {
    v_term_override_enabled_ = true;
    v_term_override_ = std::max(0.0, v_term_v);
    v_term_ = v_term_override_;
  }

  void clear_v_term_override() noexcept { v_term_override_enabled_ = false; }

  void set_motor_commands(const std::array<double, 8>& cmds_rad_s) noexcept {
    for (int i = 0; i < 8; ++i) {
      rotors_[i].omega_cmd = std::isfinite(cmds_rad_s[i]) ? std::max(0.0, cmds_rad_s[i]) : 0.0;
    }
  }

  void set_rotor_speeds_direct(const std::array<double, 8>& omegas_rad_s) noexcept {
    for (int i = 0; i < 8; ++i) {
      rotors_[i].omega = std::isfinite(omegas_rad_s[i]) ? std::max(0.0, omegas_rad_s[i]) : 0.0;
    }
  }

  void set_motor_fault(int idx, bool fault_off) noexcept {
    if (idx >= 0 && idx < 8) {
      rotors_[idx].fault_off = fault_off;
    }
  }

  void set_motor_fault_mask(uint32_t mask) noexcept {
    for (int i = 0; i < 8; ++i) {
      rotors_[i].fault_off = ((mask & (1u << i)) != 0u);
    }
  }

  // Piecewise-linear NMC811 112S OCV curve anchored to DR-1 Section 4.1-4.2:
  // 100% -> 470.4 V, 50% -> 409.9 V, 20% -> 384.2 V, 0% -> 336.0 V.
  [[nodiscard]] static double open_circuit_voltage(double soc) noexcept {
    const double s = std::clamp(std::isfinite(soc) ? soc : 0.0, 0.0, 1.0);
    if (s >= 0.5) {
      return 409.9 + (470.4 - 409.9) * ((s - 0.5) / 0.5);
    }
    if (s >= 0.2) {
      return 384.2 + (409.9 - 384.2) * ((s - 0.2) / 0.3);
    }
    return 336.0 + (384.2 - 336.0) * (s / 0.2);
  }

  // DC-bus-equivalent BLDC speed ceiling:
  // V_term = K_e,bus * omega + (R_eff / K_e,bus) * k_Q0 * omega^2
  [[nodiscard]] double solve_bldc_voltage_limit(double v_term_v) const noexcept {
    if (!std::isfinite(v_term_v) || v_term_v <= 0.0) {
      return 0.0;
    }
    const double b = std::max(params_.ke_bus_vs_rad, 1e-6);
    const double a = (params_.r_eff_hot_ohm * params_.k_q0) / b;
    if (a <= 1e-12) {
      return v_term_v / b;
    }
    const double disc = b * b + 4.0 * a * v_term_v;
    return (-b + std::sqrt(std::max(0.0, disc))) / (2.0 * a);
  }

  // Glauert momentum-theory induced velocity v_iu for climb velocity v_climb_flu_z (+Z Up).
  // In FLU climb (+v_air.z > 0), ambient air enters downward through the disk so axial flow is
  // v_axial_down = v_iu + v_climb_flu_z.
  [[nodiscard]] double solve_glauert_vi(
      double thrust_u_n,
      double disk_area_m2,
      double v_horiz_mps,
      double v_climb_flu_z_mps) const noexcept {
    if (!std::isfinite(thrust_u_n) || thrust_u_n <= 1e-9 || disk_area_m2 <= 1e-9) {
      return 0.0;
    }
    const double v_h = std::sqrt(thrust_u_n / (2.0 * params_.rho_air * disk_area_m2));
    if (v_h < 1e-6) {
      return 0.0;
    }
    // Clamp severe descent outside VRS singularity for smooth monotonic wake estimation
    const double v_c = std::max(std::isfinite(v_climb_flu_z_mps) ? v_climb_flu_z_mps : 0.0, -0.5 * v_h);
    const double v_xy = std::isfinite(v_horiz_mps) ? std::max(0.0, v_horiz_mps) : 0.0;
    const double v_xy2 = v_xy * v_xy;
    const double vh2 = v_h * v_h;

    double vi = v_h;
    for (int iter = 0; iter < 12; ++iter) {
      const double ax = v_c + vi;
      const double denom = std::sqrt(v_xy2 + ax * ax);
      if (denom < 1e-6) {
        break;
      }
      const double f = vi * denom - vh2;
      const double df = denom + (vi * ax) / denom;
      if (df <= 1e-6) {
        break;
      }
      vi = std::max(0.05 * v_h, vi - f / df);
      if (std::abs(f) < 1e-9) {
        break;
      }
    }
    return vi;
  }

  // Step 8-rotor coaxial physics & coupled 1-RC Thevenin battery by dt [s] with body FLU airspeed.
  [[nodiscard]] CoaxStepOutput step(double dt_s, Vec3D v_air_flu) noexcept {
    CoaxStepOutput out{};
    const double dt = (std::isfinite(dt_s) && dt_s > 0.0) ? dt_s : 0.0;
    const double vx = std::isfinite(v_air_flu.x) ? v_air_flu.x : 0.0;
    const double vy = std::isfinite(v_air_flu.y) ? v_air_flu.y : 0.0;
    const double vz = std::isfinite(v_air_flu.z) ? v_air_flu.z : 0.0;
    const double v_horiz = std::hypot(vx, vy);
    const double disk_area = kPi * params_.rotor_radius_m * params_.rotor_radius_m;

    const double v_bus = v_term_override_enabled_ ? v_term_override_ : v_term_;
    const double w_cap = std::min(params_.omega_max_nom, solve_bldc_voltage_limit(v_bus));
    out.omega_cap_rad_s = w_cap;

    double p_mech_total = 0.0;

    // Pass 1: upper rotors 0..3 (clean freestream inflow, eta_T = 1.0, eta_Q = 1.0)
    // Pass 2: lower rotors 4..7 (coaxial wake interaction from updated upper rotor w_u)
    for (int i = 0; i < 8; ++i) {
      auto& r = rotors_[i];

      // Pre-compute coaxial wake interaction factors for lower rotors using current upper speed
      double eta_t = 1.0;
      double eta_q = 1.0;
      if (r.pair_upper_idx >= 0) {
        const int arm = r.pair_upper_idx;
        const double w_u = rotors_[arm].omega;
        if (w_u > 5.0) {
          const double t_u = params_.k_t0 * w_u * w_u;
          const double v_iu = solve_glauert_vi(t_u, disk_area, v_horiz, vz);
          // DR-1 erratum fix: climbing (+Z FLU) increases downward axial inflow to v_iu + v_air.z
          const double v_axial_down = std::max(0.5, v_iu + vz);
          const double chi = std::atan2(v_horiz, v_axial_down);
          const double xi = std::min(params_.coax_spacing_m * std::tan(chi) / (2.0 * params_.rotor_radius_m), 1.0);
          const double f_ov = (2.0 / kPi) * (std::acos(xi) - xi * std::sqrt(std::max(0.0, 1.0 - xi * xi)));
          const double ratio = std::clamp(w_u / std::max(r.omega, 10.0), 0.0, 1.2);

          // Lower hover thrust = 0.85 * k_T0 (unblanks +17.6% when upper rotor fails)
          // Lower hover torque = 0.96 * k_Q0 (4% swirl recovery)
          eta_t = 1.0 - (1.0 - params_.eta_coax_hover) * f_ov * ratio;
          eta_q = 1.0 - (1.0 - params_.swirl_recovery) * f_ov * ratio;

          out.v_iu_mps[arm] = v_iu;
          out.v_axial_down_mps[arm] = v_axial_down;
          out.wake_skew_chi_rad[arm] = chi;
        }
      }

      // Asymmetric powered lag vs wired tau_coast_ unpowered aerodynamic coast-down
      double dw = 0.0;
      if (r.fault_off) {
        const double dw_aero = -(eta_q * params_.k_q0 * r.omega * r.omega) / std::max(params_.j_r, 1e-6);
        const double dw_coast = -r.omega / (2.0 * std::max(params_.tau_coast_s, 1e-3));
        dw = std::min(dw_aero, dw_coast);
      } else {
        const double w_target = std::clamp(r.omega_cmd, 0.0, w_cap);
        const double tau = (w_target >= r.omega) ? params_.tau_up_s : params_.tau_down_s;
        dw = std::clamp((w_target - r.omega) / std::max(tau, 1e-4), -params_.alpha_max_rad_s2, params_.alpha_max_rad_s2);
      }

      r.omega = std::max(0.0, r.omega + dw * dt);

      // Re-evaluate lower-rotor ratio post-integration so steady-state hover tests are exact
      if (r.pair_upper_idx >= 0) {
        const int arm = r.pair_upper_idx;
        const double w_u = rotors_[arm].omega;
        if (w_u > 5.0) {
          const double t_u = params_.k_t0 * w_u * w_u;
          const double v_iu = solve_glauert_vi(t_u, disk_area, v_horiz, vz);
          const double v_axial_down = std::max(0.5, v_iu + vz);
          const double chi = std::atan2(v_horiz, v_axial_down);
          const double xi = std::min(params_.coax_spacing_m * std::tan(chi) / (2.0 * params_.rotor_radius_m), 1.0);
          const double f_ov = (2.0 / kPi) * (std::acos(xi) - xi * std::sqrt(std::max(0.0, 1.0 - xi * xi)));
          const double ratio = std::clamp(w_u / std::max(r.omega, 10.0), 0.0, 1.2);
          eta_t = 1.0 - (1.0 - params_.eta_coax_hover) * f_ov * ratio;
          eta_q = 1.0 - (1.0 - params_.swirl_recovery) * f_ov * ratio;
          out.v_iu_mps[arm] = v_iu;
          out.v_axial_down_mps[arm] = v_axial_down;
          out.wake_skew_chi_rad[arm] = chi;
        } else {
          eta_t = 1.0;
          eta_q = 1.0;
          out.v_iu_mps[arm] = 0.0;
          out.v_axial_down_mps[arm] = std::max(0.0, vz);
          out.wake_skew_chi_rad[arm] = 0.0;
        }
      }

      const double t_i = eta_t * params_.k_t0 * r.omega * r.omega;
      const double q_aero = eta_q * params_.k_q0 * r.omega * r.omega;
      // FLU (+Z Up): CCW rotor (spin_sign = +1) exerts -Z_FLU reaction torque on airframe
      const double q_z_flu = -static_cast<double>(r.spin_sign) * (q_aero + params_.j_r * dw);

      if (!r.fault_off) {
        const double shaft_torque = std::max(0.0, q_aero + params_.j_r * std::max(0.0, dw));
        p_mech_total += shaft_torque * r.omega;
      }

      out.omega_rad_s[i] = r.omega;
      out.dw_rad_s2[i] = dw;
      out.thrust_n[i] = t_i;
      out.torque_z_flu[i] = q_z_flu;
      out.eta_t[i] = eta_t;
      out.eta_q[i] = eta_q;
    }

    // Coupled 1-RC Thevenin battery step
    const double p_elec = (p_mech_total / std::max(params_.eta_drive, 0.1)) + params_.p_avionics_w;
    const double v_oc = open_circuit_voltage(soc_);
    const double v_eff = std::max(1.0, v_oc - v_rc_);
    const double disc = std::max(0.0, v_eff * v_eff - 4.0 * params_.r0_ohm * p_elec);
    const double i_pack = std::clamp(
        (v_eff - std::sqrt(disc)) / (2.0 * std::max(params_.r0_ohm, 1e-6)),
        0.0,
        params_.i_max_cont_a);

    if (!v_term_override_enabled_ && dt > 0.0) {
      const double tau1 = std::max(params_.r1_ohm * params_.c1_f, 1e-3);
      const double decay = std::exp(-dt / tau1);
      v_rc_ = decay * v_rc_ + params_.r1_ohm * (1.0 - decay) * i_pack;
      soc_ = std::clamp(soc_ - (dt * i_pack) / (3600.0 * std::max(params_.capacity_ah, 1e-3)), 0.0, 1.0);
      v_term_ = std::max(0.0, open_circuit_voltage(soc_) - i_pack * params_.r0_ohm - v_rc_);
    } else if (v_term_override_enabled_) {
      v_term_ = v_term_override_;
    }

    out.p_elec_w = p_elec;
    out.i_pack_a = i_pack;
    out.soc = soc_;
    out.v_term_v = v_term_;
    return out;
  }

 private:
  CoaxParams params_{};
  std::array<RotorState, 8> rotors_{};
  double soc_{0.85};
  double v_rc_{0.0};
  double v_term_{403.2};
  bool v_term_override_enabled_{false};
  double v_term_override_{403.2};
};

}  // namespace vertirta_gz
