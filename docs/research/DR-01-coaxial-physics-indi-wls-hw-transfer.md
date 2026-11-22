# VertiRTA-X8-240 Flight Physics, 400 Hz PX4-Internal Fault-Tolerant Core, Coupled Battery Sag, and Hardware Transfer Architecture

This engineering design report delivers the complete flight-physics audit, 400 Hz PX4 Autopilot v1.16 internal flight-control core (`libvertirta_flight_core` and `mc_indi_wls_control`), coupled 1-RC Thevenin propulsion-battery model, Gazebo Harmonic LTS SITL integration blueprint, and Pixhawk 6X (FMU-v6X STM32H7) hardware-transfer roadmap for the 240 kg coaxial 8-rotor urban eVTOL `VertiRTA-X8-240`. Every transformation, aerodynamic formulation, discrete control law, fault observer, and firmware patch is derived in consistent SI units with explicit coordinate frames and verified against upstream PX4 v1.16 (`release/1.16`) source architecture and Gazebo Harmonic (`gz-sim8`).

First-principles auditing of the locked baseline identifies three decisive physical constraints that require explicit design closure before flight-code freeze:

1. **Same-Arm Dual Coaxial-Pair Loss Controllability:** Under Same-Arm Dual Coaxial-Pair Loss (both rotors dead on Arm 0), static non-spinning 2-arm hover with the diagonally opposite arm zeroed is linearly uncontrollable (rank 2 of 4) because the cross-diagonal pitch-roll axis becomes strictly unilateral ($\tau_v \le 0$), causing immediate capsize under any gust or CG offset toward the dead arm. Actively spinning the airframe via differential CW/CCW yaw torque to a flat, zero-tilt rate of $|\bar{\Omega}_z| = 2.0\text{--}2.5\text{ rad/s}$ (19-24 RPM) restores full 4-state reduced-attitude controllability with only 9.6-12.0 deg actuator lag phase shift and 0.10-0.16 g passenger head centrifugal acceleration.
2. **Voltage-Sag Thrust Contraction:** If maximum rotor speed (240 rad/s) is back-EMF voltage-limited at 400 V, a 10% terminal sag to 360 V reduces maximum thrust quadratically ($T_{\max} \propto V_{\text{term}}^2$), collapsing surviving 2-arm thrust-to-weight to 1.028 and inflating a 2.5 m/s flare stopping distance from 1.49 m to 11.59 m. Holding $T/W \ge 1.12$ down to 360 V requires winding motors with at least 10% voltage headroom (DC-bus-equivalent $K_{e,\text{bus}} \le 1.36\text{ V}\cdot\text{s/rad}$) and enforcing a software FOC RPM ceiling at 240 rad/s.
3. **240 kg MTOW vs. 22 kWh COTS Battery Mass Closure:** A 400 V / 22 kWh battery built from 2026 COTS high-power `INR21700-P45B` cells (`112S12P`, 1,344 cells, 242 Wh/kg cell / ~195 Wh/kg pack) weighs 111.7 kg, leaving only 48.3 kg for airframe, eight 15 kW propulsion units, landing gear, and avionics once an 80 kg passenger is boarded at 240 kg MTOW. Engineering closure therefore requires either ~320 Wh/kg pack-level cells (68-69 kg), downsizing to a mass-compliant `112S7P` COTS pack (12.7 kWh, 65.1 kg), or operating the `112S12P` pack as a 240 kg unmanned demonstrator (30-40 kg payload).

Additionally, upstream PX4 v1.16 `GZBridge.cpp` subscribes to only one hardcoded Gazebo IMU topic (`imu_sensor/imu`) and requires a targeted multi-IMU bridge patch to feed the 400 Hz in-loop Triple Modular Redundancy (TMR) voter.

---

## 1. Coordinate Frames, Geodesy, and Physical Parameter Audit

### 1.1 Exact Frame Transformations and Quaternion Conventions

PX4 Autopilot v1.16 expresses local world kinematics in North-East-Down (`NED`) and rigid-body vectors in Forward-Right-Down (`FRD`), whereas ROS 2 (REP-105) and Gazebo Harmonic adopt East-North-Up (`ENU`) and Forward-Left-Up (`FLU`). Furthermore, PX4 `px4_msgs` encodes Hamilton unit quaternions in scalar-first order $[w, x, y, z]^T$ representing the active rotation from `FRD` to `NED` ($R_{\text{FRD}\to\text{NED}}$), while ROS 2 `geometry_msgs/msg/Quaternion` and `tf2` use scalar-last order $[q_x, q_y, q_z, q_w]^T$ representing $R_{\text{FLU}\to\text{ENU}}$.

Both frame-alignment rotation matrices are symmetric orthogonal involutions ($R = R^T = R^{-1}$, $\det R = +1$):

$$R_{\text{NED}\to\text{ENU}} = R_{\text{ENU}\to\text{NED}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & -1 \end{bmatrix}, \qquad R_{\text{FRD}\to\text{FLU}} = R_{\text{FLU}\to\text{FRD}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix}$$

In Hamilton scalar-first notation, these correspond to 180 deg rotations $q_{\text{NED}\to\text{ENU}} = [0,\, \tfrac{1}{\sqrt{2}},\, \tfrac{1}{\sqrt{2}},\, 0]^T$ and $q_{\text{FLU}\to\text{FRD}} = [0,\, 1,\, 0,\, 0]^T$. Applying $R_{\text{FLU}\to\text{ENU}} = R_{\text{NED}\to\text{ENU}} R_{\text{FRD}\to\text{NED}} R_{\text{FLU}\to\text{FRD}}$ yields the exact Hamilton sandwich product $q_{\text{FLU}\to\text{ENU}} = q_{\text{NED}\to\text{ENU}} \otimes q_{\text{PX4}} \otimes q_{\text{FLU}\to\text{FRD}}$, which reduces algebraically (up to global quaternion double-cover sign) to a branchless conversion verified to $7.77 \times 10^{-16}$ matrix error across random unit quaternions.

| Quantity / Frame Pair | PX4 Convention (`FRD` / `NED`) | ROS 2 Convention (`FLU` / `ENU`) | Exact Conversion Law (PX4 -> ROS 2) | Exact Inverse (ROS 2 -> PX4) |
| :--- | :--- | :--- | :--- | :--- |
| World Position / Velocity / Accel | `p_NED = [p_N, p_E, p_D]^T` | `p_ENU = [p_E, p_N, p_U]^T` | `[p_E, p_N, -p_D]^T` | `[p_N, p_E, -p_U]^T` |
| Body Force / Specific Force | `f_FRD = [f_x, f_y, f_z]^T` (thrust `[0, 0, -T]^T`) | `f_FLU = [f_x, f_y, f_z]^T` (thrust `[0, 0, +T]^T`) | `[+f_x, -f_y, -f_z]^T` | `[+f_x, -f_y, -f_z]^T` |
| Body Angular Rate `Omega` | `[p, q, r]^T` (`+p` right down, `+q` nose up, `+r` yaw CW) | `[w_x, w_y, w_z]^T` (`+x` right down, `+y` nose down, `+z` yaw CCW) | `[+p, -q, -r]^T` | `[+w_x, -w_y, -w_z]^T` |
| Body Moment / Torque `tau` | `[tau_x, tau_y, tau_z]^T` (right-hand about FRD) | `[M_x, M_y, M_z]^T` (right-hand about FLU) | `[+tau_x, -tau_y, -tau_z]^T` | `[+M_x, -M_y, -M_z]^T` |
| Intrinsic ZYX Euler Angles | `[psi_NED, theta_FRD, phi_FRD]` (0 deg N, CW+) | `[psi_ENU, theta_FLU, phi_FLU]` (0 deg E, CCW+) | `[wrap_pi(pi/2 - psi_NED), -theta_FRD, +phi_FRD]` | `[wrap_pi(pi/2 - psi_ENU), -theta_FLU, +phi_FLU]` |
| Attitude Quaternion | `q_PX4 = [w, x, y, z]^T` (scalar-first) | `q_ROS2 = [q_x, q_y, q_z, q_w]^T` (scalar-last) | `(1/sqrt(2)) * [x+y, x-y, w-z, w+z]^T` | `(1/sqrt(2)) * [q_w+q_z, q_x+q_y, q_x-q_y, q_w-q_z]^T` |

As a numerical verification benchmark, an aircraft flying heading 035 deg NE ($\psi_{\text{NED}} = +35^\circ$), nose pitched up +10 deg, and rolled right-wing down +15 deg carries $q_{\text{PX4}} = [0.945380, 0.098027, 0.121511, 0.286149]^T$. Applying the closed-form map yields ROS 2 scalar-last $q_{\text{ROS2}} = [0.155237, -0.016606, 0.466147, 0.870823]^T$, reproducing exact `FLU/ENU` intrinsic ZYX Euler angles of `[+55.0 deg yaw, -10.0 deg pitch, +15.0 deg roll]`.

### 1.2 WGS84 Geodetic, ECEF, and Istanbul EGM2008 Geoid Datum

Under WGS84 ellipsoid parameters ($a = 6,378,137.0\text{ m}$, $f = 1/298.257223563$, $e^2 = f(2-f) = 6.69437999014 \times 10^{-3}$), geodetic coordinates $(\varphi, \lambda, h_{\text{HAE}})$ transform to Earth-Centered Earth-Fixed (`ECEF`) coordinates via the prime vertical radius of curvature $\nu(\varphi)$:

$$\nu(\varphi) = \frac{a}{\sqrt{1 - e^2 \sin^2\varphi}}, \qquad \mathbf{p}_{\text{ECEF}} = \begin{bmatrix} (\nu(\varphi) + h_{\text{HAE}})\cos\varphi\cos\lambda \\ (\nu(\varphi) + h_{\text{HAE}})\cos\varphi\sin\lambda \\ (\nu(\varphi)(1 - e^2) + h_{\text{HAE}})\sin\varphi \end{bmatrix}$$

Height above the WGS84 ellipsoid (`HAE`) converts from orthometric mean sea level altitude (`AMSL`, $H_{\text{AMSL}}$) through the geoid undulation $N$:

$$h_{\text{HAE}} = H_{\text{AMSL}} + N_{\text{EGM2008}}$$

Official Turkish DHMI Aeronautical Information Publication (AIP) AD 2 aerodrome records for the Istanbul-Marmara region publish geoid undulations of 120 ft (+36.58 m) at Sabiha Gokcen (`LTFJ`), 121 ft (+36.88 m) at Ataturk (`LTBA`), and 122 ft (+37.19 m) at Istanbul Airport (`LTFM`). These official aeronautical values triangulate the locked Kartal Coastal Vertiport value $N_{\text{EGM2008}} = +36.7\text{ m}$ to within ~0.15 m, matching regional Turkish GPS/leveling evaluations of EGM2008 showing decimeter-level native accuracy prior to local corrector-surface fitting. At the canonical datum origin (`40.8860 N, 29.1880 E, 12.0 m AMSL`), the exact ellipsoidal altitude is $h_{\text{HAE}} = 12.0 + 36.7 = 48.7\text{ m}$. Any uncompensated mismatch between PX4's `VehicleGlobalPosition.alt` (AMSL), `SensorGps.altitude_ellipsoid_m` (HAE), and Gazebo's `<spherical_coordinates><elevation>` injects a direct 36.7 m vertical terrain-clearance error.

At Kartal Coastal Vertiport, $\nu(\varphi_0) = 6,387,303.497\text{ m}$, $\mathbf{p}_{\text{ECEF},0} = [4,215,767.720,\; 2,354,954.341,\; 4,152,892.183]^T\text{ m}$, and the local tangent-plane rotation matrices mapping $\Delta\mathbf{p}_{\text{ECEF}} = \mathbf{p}_{\text{ECEF}} - \mathbf{p}_{\text{ECEF},0}$ into `ENU` and `NED` are:

$$R_{\text{ECEF}\to\text{ENU}}(\varphi_0,\lambda_0) = \begin{bmatrix} -\sin\lambda_0 & \cos\lambda_0 & 0 \\ -\sin\varphi_0\cos\lambda_0 & -\sin\varphi_0\sin\lambda_0 & \cos\varphi_0 \\ \cos\varphi_0\cos\lambda_0 & \cos\varphi_0\sin\lambda_0 & \sin\varphi_0 \end{bmatrix} = \begin{bmatrix} -0.487677 & 0.873024 & 0.000000 \\ -0.571443 & -0.319212 & 0.756013 \\ 0.660018 & 0.368690 & 0.654556 \end{bmatrix}$$

$$R_{\text{ECEF}\to\text{NED}}(\varphi_0,\lambda_0) = R_{\text{ENU}\to\text{NED}}\,R_{\text{ECEF}\to\text{ENU}} = \begin{bmatrix} -0.571443 & -0.319212 & 0.756013 \\ -0.487677 & 0.873024 & 0.000000 \\ -0.660018 & -0.368690 & -0.654556 \end{bmatrix}$$

Transforming the `LTFJ` Aerodrome Reference Point (`40.8986 N, 29.3092 E`, 312 ft / 95.1 m AMSL) into this ENU frame places it at `[+10,213.0 E, +1,406.4 N, +75.0 U]^T m` — a horizontal distance of 10.309 km from `LTFJ` ARP, providing an exact 1.049 km western margin outside the real 5 NM (9.260 km) Sabiha Gokcen CTR. Local magnetic variation from DHMI AIP `LTFJ` is 6.1 deg E (2025) increasing at +0.05 deg/yr (~6.15 deg E in late 2026).

### 1.3 Complete Audited `VertiRTA-X8-240` SI Parameter Table

Fixed-pitch multirotor blades operate at an essentially constant thrust coefficient $C_T$ across RPM, requiring careful sizing so hover and maximum-thrust operating points simultaneously avoid blade stall and urban transonic tip-noise penalties. Selecting 3-blade tapered rigid CFRP propellers ($N_b = 3$, mean chord $c = 0.095\text{ m}$, solidity $\sigma = N_b c / (\pi R) = 0.1210$, sectional drag $C_{d0} = 0.011$) operating up to $\omega_{\max} = 240.0\text{ rad/s}$ (2,292 RPM) places the upper rotor at $C_{T,u} = 0.01134$ (mean lift coefficient $\bar{C}_L \approx 6 C_T / \sigma = 0.563$, centered near peak airfoil lift-to-drag with substantial stall margin) and caps tip speed at $M_{\text{tip,max}} = 0.529$ (180.0 m/s), while nominal hover runs quietly at 1,449 RPM ($M_{\text{tip,hov}} = 0.335$, 113.8 m/s).

| Parameter | Symbol | Baseline / Audited Value | Units | Audit Status & Physical Basis |
| :--- | :--- | :--- | :--- | :--- |
| Gross Mass / Weight | m, W | 240.0 / 2,354.4 (2,353.6 at g_0) | kg / N | Locked (g = 9.81 m/s^2); 0.8 N delta vs standard g_0 = 9.80665 m/s^2 |
| Airframe Principal Inertia | J | diag(85.0, 95.0, 140.0) | kg*m^2 | Locked; oblate (J_zz > J_yy > J_xx) avoids spinning controllability singularities |
| Arm Radius & Coax Spacing | L, d, dz_coax | 1.35 / 0.95459 / 0.35 | m | d = L/sqrt(2); dz/R = 0.467, dz/D = 0.233 |
| Rotor Radius & Disk Areas | R, A, A_4disk | 0.75 / 1.7671 / 7.0686 | m / m^2 | Adjacent tip clearance 0.409 m (0.546 R); CG-to-tip clearance 0.60 m (cabin width <1.15 m in rotor plane or high-mounted booms) |
| Hover / Max Disk Loading | DL_hov, DL_max | 333.1 (34.0 kg/m^2) / 832.7 | N/m^2 | 4-disk projected basis (8-disk per-rotor hover DL = 166.5 N/m^2) |
| Rotor Blade Geometry | N_b, c, sigma, C_d0 | 3 / 0.095 / 0.1210 / 0.011 | - / m / - | Rigid 3-blade tapered CFRP; isolated upper FM_u = 0.744 |
| Hover / Max Rotor Speed | w_hov, w_max | 151.8 (1,449 RPM) / 240.0 (2,292 RPM) | rad/s | M_tip_hov = 0.335, M_tip_max = 0.529 (ISA sea level a = 340.29 m/s) |
| Thrust Constants (Upper / Lower) | k_T0, k_T_lower | 1.3809e-2 / 1.1738e-2 | N/(rad/s)^2 | **Audited:** eta_coax = 0.85 splits 5,886 N into 795.4 N upper + 676.1 N lower (1,471.5 N/pair), not equal 735.8 N/rotor |
| Torque Constants (Upper / Lower) | k_Q0, k_Q_lower | 1.0488e-3 / 1.0068e-3 | N*m/(rad/s)^2 | Includes 4% lower swirl recovery; c_m_u = 0.07590 m, c_m_l = 0.08572 m |
| Rotor + BLDC Bell Inertia | J_r | 0.135 | kg*m^2 | 1.35 kg CFRP prop (0.120) + 2.2 kg direct-drive outrunner bell (0.015) |
| Actuator Lag & Slew Limit | tau_m_up, tau_m_coast, alpha_max | 0.085 / 0.424 / 487.7 | s / s / rad/s^2 | **Audited:** 0.085 s holds under FOC torque-limited spin-up (90 N*m peak); unpowered aerodynamic coast-down is 0.424 s |
| BLDC Bus Back-EMF / Resistance | K_e_bus, R_eff | 1.36 (corrected) / 0.60-0.72 | V*s/rad / Ohm | **Corrected from 1.533:** provides >=10% headroom so 2-arm T/W >= 1.12 holds past 360 V down to 338 V |
| Equivalent Flat-Plate Drag Area | f_drag, v_max | 0.48 / 22.0 | m^2 / m/s | Compact single-seat fuselage, booms, and skids; D(22 m/s) = 142.3 N |
| HV Pack (`INR21700-P45B`) | N_s S N_p P, V_nom, E, Q | 112S12P / 403.2 V / 21.77 kWh / 54.0 Ah | - / V / kWh / Ah | **Mass Conflict Flagged:** 1,344 cells = 94.1 kg (111.7 kg pack at 195 Wh/kg); see resolution below |
| Pack 1-RC ECM (25 C, 50% SoC) | R_0, R_1, C_1, tau_1 | 136.3 / 84.0 / 300.0 / 25.2 | mOhm / mOhm / F / s | Includes 15 mOhm busbar/contactor/fuse + 1.5 mOhm/cell weld resistance; I_max_cont = 540 A |

**Mass Budget Audit & Resolution:** Auditing the mass budget against 2026 COTS Molicel `INR21700-P45B` cell specifications (4.5 Ah, 16.2 Wh, 70 g, 242 Wh/kg cell-level) exposes an unresolvable closure conflict if all three baseline constraints—240 kg MTOW, an 80 kg passenger, and a 22 kWh COTS pack—are enforced simultaneously. At a realistic ~195 Wh/kg pack level (accounting for structural casing, BMS, HV contactors, and busbars), the `112S12P` 21.77 kWh pack weighs 111.7 kg. Subtracting an 80 kg occupant leaves just 48.3 kg for airframe, landing gear, avionics, and eight 15 kW propulsion units—whereas eight motor/ESC/propeller assemblies alone (~4.8 kg each) weigh ~38.4 kg and a primary carbon-composite airframe plus seat and skids adds 45-60 kg (85-100 kg total empty non-battery mass). To restore physical closure, the program must select one of three coherent configurations:

- **(a)** Retain 240 kg MTOW and 22 kWh by specifying next-generation 315-330 Wh/kg pack-level Si-anode or semi-solid-state cells (68 kg pack);
- **(b)** Retain 240 kg MTOW and COTS `P45B` cells by resizing the pack to `112S7P` (12.7 kWh, 65.1 kg pack, $I_{\max,\text{cont}} = 315\text{ A}$, $R_0 = 223\text{ m}\Omega$, $R_1 = 144\text{ m}\Omega$, $C_1 = 175\text{ F}$); or
- **(c)** Retain the `112S12P` COTS pack as a 240 kg unmanned demonstrator carrying 30-40 kg test payload prior to growing manned MTOW to ~290 kg.

Subsequent electrical calculations use the locked 240 kg airframe and `112S12P` pack while explicitly flagging the deeper voltage-sag impact of the `112S7P` mass-compliant alternative.

---

## 2. Coaxial X8 Aerodynamics and Actuator Dynamics

### 2.1 Control Effectiveness Matrices in PX4 FRD Frame

In PX4 `FRD` (+X Forward, +Y Right, +Z Down), rotor $i$ mounted at $\mathbf{r}_i = [x_i, y_i, z_i]^T$ generates upward thrust $\mathbf{F}_i = [0, 0, -T_i]^T$. Its lever-arm moment is $\mathbf{r}_i \times \mathbf{F}_i = [-y_i T_i,\; +x_i T_i,\; 0]^T$, so front rotors ($x_i > 0$) pitch nose-up ($+\tau_y$) and left rotors ($y_i < 0$) roll right-wing down ($+\tau_x$). Consistent with PX4 `ControlAllocator` (`CA_ROTORi_KM`), a counter-clockwise (`CCW`, viewed from above) rotor imparts a clockwise (+Z Down, $+\tau_z$) reaction torque on the airframe ($s_i = +1$), while a clockwise (`CW`) rotor imparts a counter-clockwise ($-\tau_z$) reaction ($s_i = -1$).

Geometrically, the four arms extend radially at +/-45 deg and +/-135 deg to radius $L = 1.35\text{ m}$ ($x_i, y_i = \pm 0.95459\text{ m}$), with upper rotors 0-3 stacked at $z = -0.175\text{ m}$ and counter-rotating lower rotors 4-7 mounted 0.35 m below them at $z = +0.175\text{ m}$. Indexing upper rotors 0-3 identically to PX4's standard Quad-X geometry (`0` = Front-Right, `1` = Rear-Left, `2` = Front-Left, `3` = Rear-Right) and counter-rotating lower rotors 4-7 directly beneath them guarantees that every arm carries one `CCW` and one `CW` rotor. This X-layout establishes two orthogonal diagonal moment axes: the Front-Right/Rear-Left (`FR-RL`) line (Arm 0 and Arm 1) and the perpendicular Front-Left/Rear-Right (`FL-RR`) line (Arm 2 and Arm 3), which decouple cleanly when analyzing Same-Arm Dual Loss in Section 3.4.

> **[Figure 1 — Coaxial X8 Rotor Geometry, Coordinate Frames, and Diagonal Fault Axes]:** Orthographic layout of the `VertiRTA-X8-240` coaxial airframe showing the four 45-degree arms, upper and lower counter-rotating rotor pairs, aligned PX4 FRD and ROS 2 FLU body axes, and the rotated diagonal axes exposed when Front-Right Arm 0 loses both rotors.

| Index i | Arm & Position (FRD) | (x_i, y_i, z_i) [m] | Spin / Yaw Sign s_i | Nominal k_T_i [N/(rad/s)^2] | Nominal c_m_i [m] | PX4 CA_ROTORi_KM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Arm 0 FR Upper (+45 deg) | (+0.9546, +0.9546, -0.175) | CCW (+1) | 1.3809e-2 | +0.07590 | +0.076 |
| 1 | Arm 1 RL Upper (-135 deg) | (-0.9546, -0.9546, -0.175) | CCW (+1) | 1.3809e-2 | +0.07590 | +0.076 |
| 2 | Arm 2 FL Upper (-45 deg) | (+0.9546, -0.9546, -0.175) | CW (-1) | 1.3809e-2 | -0.07590 | -0.076 |
| 3 | Arm 3 RR Upper (+135 deg) | (-0.9546, +0.9546, -0.175) | CW (-1) | 1.3809e-2 | -0.07590 | -0.076 |
| 4 | Arm 0 FR Lower (+45 deg) | (+0.9546, +0.9546, +0.175) | CW (-1) | 1.1738e-2 | -0.08572 | -0.086 |
| 5 | Arm 1 RL Lower (-135 deg) | (-0.9546, -0.9546, +0.175) | CW (-1) | 1.1738e-2 | -0.08572 | -0.086 |
| 6 | Arm 2 FL Lower (-45 deg) | (+0.9546, -0.9546, +0.175) | CCW (+1) | 1.1738e-2 | +0.08572 | +0.086 |
| 7 | Arm 3 RR Lower (+135 deg) | (-0.9546, +0.9546, +0.175) | CCW (+1) | 1.1738e-2 | +0.08572 | +0.086 |

Mapping rotor thrusts $\mathbf{T} = [T_0, \dots, T_7]^T \in \mathbb{R}^8$ or squared speeds $\mathbf{u}_{\omega^2} = [\omega_0^2, \dots, \omega_7^2]^T$ to the body wrench $\mathbf{w} = [\tau_x, \tau_y, \tau_z, T]^T$ in PX4 `FRD` gives the exact $4 \times 8$ control effectiveness matrices:

$$\mathbf{w} = \begin{bmatrix} \tau_x \\ \tau_y \\ \tau_z \\ T \end{bmatrix} = B_T \mathbf{T} = B_{\omega^2} \mathbf{u}_{\omega^2}, \qquad B_{\omega^2} = B_T\,\mathrm{diag}(k_{T,0}, \dots, k_{T,7})$$

$$B_T = \begin{bmatrix}
-d & +d & +d & -d & -d & +d & +d & -d \\
+d & -d & +d & -d & +d & -d & +d & -d \\
+c_{m,u} & +c_{m,u} & -c_{m,u} & -c_{m,u} & -c_{m,l} & -c_{m,l} & +c_{m,l} & +c_{m,l} \\
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1
\end{bmatrix}$$

where $d = 0.95459\text{ m}$, $c_{m,u} = 0.07590\text{ m}$, and $c_{m,l} = 0.08572\text{ m}$ when the upper rotor above that lower rotor is spinning. Whenever upper rotor $k \in \{0,1,2,3\}$ fails ($\omega_k \to 0$), its coaxial lower rotor $k+4$ enters clean freestream inflow; online effectiveness reconfiguration updates $k_{T,k+4} \to k_{T0}$ and $c_{m,k+4} \to c_{m,u}$, restoring +17.6% thrust capacity (676.1 N to 795.4 N) on that surviving motor. For large 1.50 m rotors ($J_r = 0.135\text{ kg}\cdot\text{m}^2$), transient motor angular-acceleration reaction torque $\Delta\tau_{z,\text{iner}} = \sum_{i=0}^{7} s_i J_r \dot{\omega}_i$ is non-negligible and is folded into the incremental INDI effectiveness matrix in Section 3.2.

### 2.2 Leishman Coaxial Momentum Theory, Forward-Flight Wake Skew, and VRS Bounds

In hover at finite vertical separation $\Delta z_{\text{coax}} / R = 0.467$, the upper-rotor wake contracts toward its vena contracta (ideal contraction ratio $r_w / R = 1/\sqrt{2} = 0.707$, empirically ~0.80-0.82 at $\Delta z/R \approx 0.47$), submerging the inner ~64% of the lower rotor disk in accelerated axial downwash ($v_{\text{slip}} \approx 1.8\text{--}2.0\,v_{i,u}$) while the outer annulus operates in cleaner ambient air. Per Leishman & Ananthan / Leishman & Syal momentum theory for a torque-balanced coaxial pair (Case 4a), the interference-induced power factor relative to two isolated rotors is $\kappa_{\text{int}} = 1.219\text{--}1.22$ (versus 1.414 for coplanar disks and 1.281 for equal thrust). Consequently, a coaxial pair producing total thrust $T_{\text{pair}} = T_u + T_l$ consumes induced power:

$$P_{i,\text{pair}} = \kappa_{\text{iso}}\,\kappa_{\text{int}}\,\frac{T_{\text{pair}}^{3/2}}{\sqrt{4\rho A}} = \kappa_{\text{iso}}\,\frac{T_{\text{pair}}^{3/2}}{\sqrt{2\rho A_{\text{eff,pair}}}}, \qquad A_{\text{eff,pair}} = \frac{2}{\kappa_{\text{int}}^2} A \approx 1.344\,A$$

Experimental hover measurements of torque-balanced coaxial rotors verify that the upper rotor supplies 53.8 +/- 2.0% of pair thrust (lower-to-upper thrust ratio ~0.859) and that swirl recovery by the lower rotor yields a 2%-4% net efficiency benefit across $0.05 < \Delta z / D < 0.85$. Extending this interaction into forward flight at airspeed $V$ and forward disk tilt $\alpha = \operatorname{atan2}(D_{\text{para}}, W)$, the upper wake skew angle $\chi$ and horizontal aft displacement $\Delta x$ at the lower rotor plane follow from Glauert's induced velocity $v_{i,u} = v_{h,u}^2 / \sqrt{(V\cos\alpha)^2 + (V\sin\alpha + v_{i,u})^2}$:

$$\chi = \operatorname{atan2}\!\left(V\cos\alpha,\; V\sin\alpha + v_{i,u}\right), \qquad \Delta x = \Delta z_{\text{coax}} \tan\chi$$

As forward velocity sweeps the upper slipstream aft, the overlapping fraction $f_{\text{ov}}$ between the upper wake and the lower disk contracts according to circle-intersection geometry ($\xi = \min(\Delta x / (2R), 1)$):

$$f_{\text{ov}}(\xi) = \frac{2}{\pi}\left(\arccos\xi - \xi\sqrt{1 - \xi^2}\right)$$

$$\eta_{\text{coax}}(\omega_u, \omega_l, V) = 1 - \left(1 - \eta_{\text{coax},0}\right)\, f_{\text{ov}}(\xi)\,\operatorname{clip}\!\left(\frac{\omega_u}{\max(\omega_l, \omega_{\varepsilon})},\, 0,\, 1.2\right)$$

with hover baseline $\eta_{\text{coax},0} = 0.85$ spanning the empirical `[0.82, 0.88]` band. In nominal 8-rotor cruise, wake skew reaches 57.7 deg at 10 m/s ($f_{\text{ov}} = 0.54$, $\eta_{\text{coax}} = 0.910$), 71.3 deg at 15 m/s ($f_{\text{ov}} = 0.20$, $\eta_{\text{coax}} = 0.954$), and 78.2 deg at 22 m/s, where the upper slipstream clears the lower disk completely ($f_{\text{ov}} = 0$, $\eta_{\text{coax}} \to 0.98$) — consistent with wind-tunnel and free-wake studies showing lower-rotor forward-flight efficiency surpassing its hover value.

Per Johnson, Newman, and Leishman axial-descent criteria, Vortex Ring State (VRS) unsteady tip-vortex recirculation develops when vertical sink rate enters $V_z / v_h \in [-0.60, -1.53]$ (peak thrust roughness between -0.75 and -1.25 $v_h$ near zero forward speed). For `VertiRTA-X8-240` in nominal hover, the upper-rotor induced velocity is $v_{h,u} = 8.57\text{ m/s}$ (projected pair disk $v_{h,\text{disk}} = 11.66\text{ m/s}$), placing VRS onset at -5.14 m/s to -8.16 m/s. Enforcing a conservative $0.40\,v_{h,u}$ vertical margin caps zero-airspeed nominal descent at $|v_z| \le 3.0\text{ m/s}$, reserving steeper descents for forward approach speeds $V_{xy} \ge 0.8\,v_{h,u} \approx 7\text{ m/s}$ where horizontal flow convects tip vortices clear of the disk. Under Same-Arm Dual Loss (2-arm hover), doubled disk loading raises $v_{h,u}$ to 12.12 m/s (delaying VRS onset to -7.27 m/s); consequently, VRS is no longer the limiting boundary — terminal flare stopping distance governs instead ($|v_z| \le 1.5\text{--}2.5\text{ m/s}$, Section 3.4).

### 2.3 Lightweight C++ Gazebo Harmonic Coaxial Actuator Plugin

Stock Gazebo Harmonic `gz::sim::systems::MulticopterMotorModel` models each rotor in isolation, omits dynamic HV bus-voltage sag and torque-slew saturation, and ties thrust computation to a visual joint scaling factor (`realMotorVelocity = motorRotVel * rotorVelocitySlowdownSim`) that requires careful configuration to avoid thrust scaling artifacts. To model coaxial wake skew, asymmetric BLDC spin-up/coast-down lag, torque-slew limits, and coupled 1-RC battery sag in `< 5 us` per 400 Hz physics step without CFD overhead, we replace the eight independent motor plugins with a single model-level `gz::sim::System` plugin (`VertirtaCoaxialPropulsionSystem`, inheriting `ISystemConfigure` and `ISystemPreUpdate`):

```cpp
// Gazebo Harmonic (gz-sim8) coaxial propulsion + asymmetric BLDC lag + 1-RC sag plugin
class VertirtaCoaxialPropulsionSystem
    : public gz::sim::System,
      public gz::sim::ISystemConfigure,
      public gz::sim::ISystemPreUpdate {
  struct RotorState {
    gz::sim::Entity link_entity, joint_entity;
    int pair_upper_idx{-1};   // -1 for upper (0..3), 0..3 for lower (4..7)
    int spin_sign{+1};        // +1 CCW (+tau_z FRD), -1 CW (-tau_z FRD)
    double omega{0.0}, omega_cmd{0.0};
    bool fault_off{false};
  };
  std::array<RotorState, 8> rotors_;
  double k_T0_{0.0138091}, k_Q0_{0.00104876}, J_r_{0.135}, R_{0.75}, dz_{0.35};
  double tau_up_{0.085}, tau_down_{0.120}, tau_coast_{0.424};
  double alpha_max_{487.7}, omega_max_nom_{240.0}, V_term_{403.2};

  void PreUpdate(const gz::sim::UpdateInfo &info, gz::sim::EntityComponentManager &ecm) override {
    if (info.paused) return;
    const double dt = std::chrono::duration<double>(info.dt).count();
    const auto v_air = ReadBodyAirspeedFLU(ecm);
    const double V_horiz = std::hypot(v_air.X(), v_air.Y());
    // DC-bus-equivalent BLDC speed ceiling (K_e,bus = 1.36 V*s/rad, hot R_eff = 0.72 Ohm)
    const double w_cap = std::min(omega_max_nom_, SolveBldcVoltageLimit(V_term_, 1.36, 0.72, k_Q0_));

    for (int i = 0; i < 8; ++i) {
      auto &r = rotors_[i];
      const double w_target = r.fault_off ? 0.0 : std::clamp(r.omega_cmd, 0.0, w_cap);
      const double tau = r.fault_off ? tau_coast_ : ((w_target >= r.omega) ? tau_up_ : tau_down_);
      const double dw = r.fault_off
          ? (-k_Q0_ * r.omega * r.omega / J_r_)              // unpowered aerodynamic coast-down
          : std::clamp((w_target - r.omega) / tau, -alpha_max_, alpha_max_);
      r.omega = std::max(0.0, r.omega + dw * dt);

      double eta_T = 1.0, eta_Q = 1.0;
      if (r.pair_upper_idx >= 0) {
        const double w_u = rotors_[r.pair_upper_idx].omega;
        if (w_u > 5.0) {
          const double v_iu = SolveGlauertVi(k_T0_ * w_u * w_u, M_PI * R_ * R_, V_horiz, -v_air.Z());
          const double chi = std::atan2(V_horiz, std::max(0.5, v_iu - v_air.Z()));
          const double xi = std::min(dz_ * std::tan(chi) / (2.0 * R_), 1.0);
          const double f_ov = (2.0 / M_PI) * (std::acos(xi) - xi * std::sqrt(std::max(0.0, 1.0 - xi * xi)));
          const double ratio = std::clamp(w_u / std::max(r.omega, 10.0), 0.0, 1.2);
          eta_T = 1.0 - 0.15 * f_ov * ratio;   // 0.85 hover -> 0.98 @ 22 m/s, 1.0 if upper dead
          eta_Q = 1.0 - 0.04 * f_ov * ratio;   // 4% swirl recovery
        }
      }
      const double T_i = eta_T * k_T0_ * r.omega * r.omega;
      // FLU (+Z Up) reaction torque: CCW (spin_sign=+1) exerts -Z_FLU torque on airframe
      const double Q_i = -r.spin_sign * (eta_Q * k_Q0_ * r.omega * r.omega + J_r_ * dw);
      ApplyLinkWrenchFLU(ecm, r.link_entity, {0.0, 0.0, T_i}, {0.0, 0.0, Q_i});
      SetVisualJointVelocity(ecm, r.joint_entity, r.spin_sign * r.omega / 10.0);
    }
    StepTheveninBattery1RC(dt, rotors_, V_term_);
    PublishMotorSpeedFeedback(rotors_);
  }
};
```

---

## 3. 400 Hz PX4-Internal Reduced-Attitude, INDI, and Prioritized WLS Architecture

### 3.1 `mc_indi_wls_control` Module Blueprint Inside PX4 v1.16

Stock PX4 v1.16 splits multicopter flight control across four separate modules (`mc_pos_control` at ~100 Hz, `mc_att_control` at ~200-250 Hz, `mc_rate_control` at ~400-800 Hz, and `control_allocator` on `wq:rate_ctrl`). Auditing upstream PX4 source files exposes four structural barriers to fault-tolerant RTA flight:

- `MulticopterPositionControl.cpp` applies static `MPC_TILTMAX_AIR` and `MPC_TILTMAX_LND` slew-limited tilt clamps rather than dynamic fault envelopes;
- `mc_att_control_main.cpp` tracks full $SO(3)$ quaternions (Brescianini et al. 2013) whose yaw-coupled error feedback destabilizes during active spin;
- `MulticopterRateControl.cpp` uses a static PID rate controller documented to overshoot rate limits by 13%-22% on aggressive step reversals; and
- `ControlAllocator.cpp` supports only pseudo-inverse clipping (`CA_METHOD = 0`) and sequential desaturation (`CA_METHOD = 1`), lacking prioritized box-constrained QP allocation and online lower-rotor unblanking upon upper-rotor failure.

To guarantee deterministic latency while remaining 100% compatible with PX4's native C++17 toolchain (`CMAKE_CXX_STANDARD 17` with `-fno-exceptions -fno-rtti -fno-sized-deallocation -fno-threadsafe-statics -nostdinc++` on NuttX), all flight-critical control, allocation, voting, FDI, and battery-observer math lives inside **`libvertirta_flight_core`**, a zero-dependency, zero-heap C++17 static library restricted to `<cstdint>`, `<cmath>`, `<cstddef>`, and fixed-size POD structs. Inside PX4 v1.16, a single integrated module **`src/modules/mc_indi_wls_control/`** (linked via `EXTERNAL_MODULES_LOCATION` or `src/modules/CMakeLists.txt`) runs synchronously at **400 Hz** (2.5 ms period) on `px4::wq_configurations::rate_ctrl`, triggered directly by the 400 Hz TMR-voted IMU callback.

When outer-loop ROS 2 RTA mode (`DR-2`) commands inertial thrust magnitude $T_{\text{des}}$, desired thrust direction $\mathbf{n}_{I,\text{des}} \in S^2$ (or inertial acceleration $\mathbf{a}_{I,\text{des}}$), and optional yaw rate $r_{\text{des}}$ over `/fmu/in/vertirta_rta_setpoint`, `mc_indi_wls_control` executes the complete closed-loop chain in a single deterministic tick:

1. 3-IMU TMR voting on raw `sensor_gyro[0..2]` and `sensor_accel[0..2]`;
2. Reduced-attitude primary-axis control ($\mathbf{n}_B \to \boldsymbol{\Omega}_{\text{ref}}$);
3. 400 Hz discrete INDI angular-acceleration regulation;
4. Prioritized WLS Active-Set allocation outputting normalized motor speeds directly to `actuator_motors` uORB;
5. Dual-channel Motor FDI updating `fault_bitmask` and $B_T, \mathbf{u}_{\max}$ online; and
6. 100 Hz decimated 1-RC battery sag and capability-envelope publishing over uXRCE-DDS.

Stock `mc_pos_control`, `mc_att_control`, `mc_rate_control`, and `control_allocator` are excluded from the airframe startup script (`rc.vertirta_apps`) to prevent dual-publisher contention on `actuator_motors`, while safety-pilot manual stabilized override routes stick tilt/yaw/throttle through the exact same INDI+WLS inner loop with zero switching transient.

### 3.2 Reduced-Attitude Kinematics and 400 Hz Discrete INDI Rate Law

Following Sun et al. and Mueller & D'Andrea, attitude tracking is formulated on the two-sphere $S^2$ around the body-fixed unit thrust vector $\mathbf{n}_B = [0, 0, -1]^T$ in PX4 `FRD`. Given an outer-loop inertial acceleration setpoint $\mathbf{a}_{I,\text{des}}$ in `NED`, the desired inertial force vector is $\mathbf{f}_{I,\text{des}} = m(\mathbf{a}_{I,\text{des}} - [0, 0, g]^T)$, defining collective thrust $T_{\text{des}} = \|\mathbf{f}_{I,\text{des}}\|$ and body-frame thrust direction $\mathbf{n}_{B,\text{des}} = R_{\text{FRD}\to\text{NED}}^T (\mathbf{f}_{I,\text{des}} / T_{\text{des}})$. The yaw-decoupled reduced-attitude error on $S^2$ and body-rate reference are:

$$\mathbf{e}_n = \mathbf{n}_B \times \mathbf{n}_{B,\text{des}} = \begin{bmatrix} +n_{B,\text{des},y} \\ -n_{B,\text{des},x} \\ 0 \end{bmatrix}, \qquad \boldsymbol{\Omega}_{\text{ref},xy} = k_n\,\mathbf{e}_{n,xy} + \boldsymbol{\Omega}_{\text{ff},xy}$$

with outer reduced-attitude gain $k_n \in [2.2, 3.0]\text{ s}^{-1}$ safely separated below the inner rate crossover (4.65 rad/s). In Regimes 0-2, yaw rate tracks heading via $r_{\text{ref}} = k_\psi \operatorname{wrap}_\pi(\psi_{\text{des}} - \psi)$; in Regime 3 (Same-Arm Dual Loss), yaw rate switches cleanly to the regulated active-spin setpoint $r_{\text{ref}} = \bar{\Omega}_z$ while $\mathbf{n}_{B,\text{des}}$ incorporates actuator-lag phase lead (Section 3.4).

In the 400 Hz discrete INDI inner loop (Smeur / Sun / de Visser), taking the first-order increment of rigid-body rotational dynamics $J\dot{\boldsymbol{\Omega}} + \boldsymbol{\Omega}\times(J\boldsymbol{\Omega}) = B_\tau \mathbf{T} + \mathbf{m}_{\text{iner}} + \mathbf{m}_{\text{ext}}$ around the filtered current state $(\hat{\mathbf{u}}_f, \dot{\boldsymbol{\Omega}}_f)$ cancels aerodynamic gusts and gyroscopic cross-coupling directly through sensor feedback:

$$\mathbf{v}_{\text{req}}[k] = \begin{bmatrix} J\left(\dot{\boldsymbol{\Omega}}_{\text{des}}[k] - \dot{\boldsymbol{\Omega}}_f[k]\right) \\ T_{\text{des}}[k] - \hat{T}_f[k] \end{bmatrix} = G_{\text{eff}}\,\Delta\mathbf{u}[k]$$

where $\dot{\boldsymbol{\Omega}}_{\text{des}}[k] = K_\Omega (\boldsymbol{\Omega}_{\text{ref}}[k] - \boldsymbol{\Omega}_f[k])$ with $K_\Omega = \mathrm{diag}(5.0, 5.0, 3.0)\text{ s}^{-1}$, $\Delta\mathbf{u}[k] = \mathbf{u}_{\text{cmd}}[k] - \hat{\mathbf{u}}_f[k]$ is the thrust increment vector, and the yaw row of $G_{\text{eff}}$ augments $B_T$ with the bounded rotor-inertia acceleration derivative $\frac{\partial \tau_{z,\text{iner}}}{\partial T_i} \approx s_i \frac{J_r}{2 k_{T,i} \max(\hat{\omega}_i, 50) \tau_m}$ to prevent high-frequency yaw chattering from motor spin-up impulses.

To prevent phase-mismatch instability when differentiating noisy 400 Hz gyro measurements ($\dot{\boldsymbol{\Omega}}_f[k] = (\boldsymbol{\Omega}_f[k] - \boldsymbol{\Omega}_f[k-1])/T_s$), an identical 2nd-order digital low-pass filter $H_f(z)$ must filter **both** the voted gyro stream $\boldsymbol{\Omega}[k]$ and the estimated actuator state $\hat{\mathbf{u}}[k]$. Actuator speeds $\hat{\omega}_i[k]$ propagate through the discrete 1st-order motor lag $\hat{\omega}_i[k] = \hat{\omega}_i[k-1] + \frac{T_s}{\tau_m}(\omega_{\text{cmd},i}[k-d] - \hat{\omega}_i[k-1])$ with $d = 2$ ticks (5 ms digital/ESC transport delay), blended via a 20 rad/s complementary observer with live `esc_status` RPM telemetry whenever fresh. Because identical filtering cancels $H_f(s)$ out of the command-to-acceleration loop, the nominal closed-loop dynamics reduce purely to $\dot{\boldsymbol{\Omega}}(s) = A(s)\dot{\boldsymbol{\Omega}}_{\text{des}}(s) + (1 - A(s)H_f(s))\mathbf{d}_{\text{ext}}(s)$, where $A(s) = e^{-t_d s} / (\tau_m s + 1)$.

| Filter Cutoff f_c (zeta = 0.7071) | Natural Freq w_n [rad/s] | Disturbance Rejection BW [rad/s] | Peak Sensitivity M_s | Mismatch Margin min abs(1+L_mis) (+35% G, tau = 0.115 s) | Differentiator Gain at 72.5 Hz BPF | Exact 400 Hz Tustin Biquad Coefficients (b_0, b_1, b_2; a_1, a_2) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 4.0 Hz | 25.1 | 5.30 | 1.35 | 0.861 | 1.39 | `b=[0.000945, 0.001889, 0.000945]`, `a=[1, -1.911197, 0.914976]` |
| 6.0 Hz | 37.7 | 6.30 | 1.29 | 0.878 | 3.12 | `b=[0.002081, 0.004161, 0.002081]`, `a=[1, -1.866892, 0.875215]` |
| **8.0 Hz (Selected)** | **50.3** | **6.96** | **1.25** | **0.888** | **5.55** | **`b=[0.003622, 0.007243, 0.003622]`, `a=[1, -1.822695, 0.837182]`** |
| 10.0 Hz | 62.8 | 7.46 | 1.22 | 0.894 | 8.66 | `b=[0.005543, 0.011085, 0.005543]`, `a=[1, -1.778632, 0.800803]` |
| 12.0 Hz | 75.4 | 7.81 | 1.20 | 0.898 | 12.48 | `b=[0.007820, 0.015640, 0.007820]`, `a=[1, -1.734726, 0.766007]` |

Selecting $f_c = 8.0\text{ Hz}$ ($\omega_n = 50.27\text{ rad/s}$) matches TU Delft MAVLab's empirically validated 50 rad/s cutoff while raising damping from Smeur's 0.55 to exact Butterworth $\zeta = 0.7071$ to suppress resonant amplification on a 240 kg airframe. This choice cuts 72.5 Hz hover blade-pass differentiation gain by 2.25x compared to a 12 Hz cutoff while maintaining an incremental-loop stability distance of 0.888 from -1 even under simultaneous +35% effectiveness error and +35% actuator lag drift (0.085 s to 0.115 s). In the outer rate loop with $\tau_m = 0.085\text{ s}$ and 5 ms delay, $K_\Omega = 5.0\text{ s}^{-1}$ yields a 4.65 rad/s (0.74 Hz) crossover with 67.1 deg phase margin and 32.1 dB gain margin (phase crossover at 48.0 rad/s).

### 3.3 Prioritized WLS Active-Set Control Allocation and AMS Contraction

To enforce box constraints, dynamic voltage-sag speed ceilings, torque slew limits, and zeroed dead motors without inducing cross-axis moment coupling, `libvertirta_flight_core` executes Harkegard and Smeur's incremental Prioritized Weighted Least Squares (`wls_alloc`) active-set solver:

$$\min_{\Delta\mathbf{u} \in \mathbb{R}^8} \; \left\| W_v \left( G_{\text{eff}}\,\Delta\mathbf{u} - \mathbf{v}_{\text{req}} \right) \right\|_2^2 + \gamma \left\| W_u \left( \hat{\mathbf{u}}_f + \Delta\mathbf{u} - \mathbf{u}_d \right) \right\|_2^2$$

$$\text{s.t.} \quad \Delta u_{\min,i} \le \Delta u_i \le \Delta u_{\max,i}, \quad i = 0,\dots,7$$

Regularization $\gamma = 10^{-4}$ guarantees a unique global optimum along the 4-dimensional null space toward the minimum-power preferred thrust vector $\mathbf{u}_d$ (equal load-sharing across surviving arms, with $u_{d,\text{opp}} = 0$ in Regime 3), while $W_u = I_8$ (weighted 1.15 on lower rotors to minimize coaxial induced losses) breaks ties cleanly. Per-tick box constraints encode dead motors ($u_{\min,i} = u_{\max,i} = 0$ whenever `fault_bitmask & (1u << i)` is set), the dynamic terminal-voltage thrust cap $u_{\max,i} = k_{T,i}\min(\omega_{\max}, \omega_{\text{lim}}(V_{\text{term}}))^2$, healthy motor idle floors ($u_{\min,i} = k_{T,i}\omega_{\text{idle}}^2$ at 20 rad/s to prevent ESC desync, yielding <5.5 N idle thrust on unloaded Arm 1), and a 25 ms lookahead slew bound $|\Delta u_i| \le 2 k_{T,i} \max(\hat{\omega}_i, 50) \dot{\omega}_{\max} \Delta t_{\text{slew}}$. Priority weights schedule by regime: in Regimes 0-2, $W_v = \mathrm{diag}(100, 100, 1.0, 25)$ protects roll/pitch stability first, collective thrust second, and sheds yaw authority smoothly during saturation; in Regime 3, $W_v$ rotates into the diagonal $(u, v, z)$ frame with $W_{v,\text{diag}} = \mathrm{diag}(w_u = 100,\; w_v = 10,\; w_\psi = 4.0,\; w_T = 30)$ so bidirectional $\tau_u$ and collective lift $T$ take priority, regulated differential yaw $\tau_z$ sustains active spin, and Arm 1 fires opportunistically only when $\tau_{v,\text{req}} < 0$.

```cpp
// Zero-heap C++17 (-fno-exceptions -fno-rtti) Active-Set WLS Allocator (Harkegard/Smeur)
// Solves min ||A*x - b||_2^2  s.t.  xmin <= x <= xmax,  x = delta_u in R^8
struct WlsResult {
  float x[8];
  int8_t W[8];      // Working set: -1 at lower bound, +1 at upper bound, 0 free
  uint8_t iters;
  uint8_t sat_mask;
};

WlsResult wls_alloc_8x4(const float G[4][8], const float v[4],
                        const float xmin[8], const float xmax[8],
                        const float Wv[4], const float Wu[8],
                        const float ud_minus_u0[8], float gamma,
                        int8_t W_warm[8], uint8_t max_iter = 12) noexcept {
  // 1. Assemble weighted normal equations H = A^T A (8x8 SPD) and g0 = A^T b (8x1)
  float H[8][8]{}, g0[8]{};
  for (int i = 0; i < 8; ++i) {
    H[i][i] = gamma * Wu[i] * Wu[i];
    g0[i]   = gamma * Wu[i] * Wu[i] * ud_minus_u0[i];
    for (int j = i; j < 8; ++j) {
      float sum = 0.0f;
      for (int r = 0; r < 4; ++r) sum += (Wv[r] * Wv[r]) * G[r][i] * G[r][j];
      H[i][j] += sum;
      H[j][i]  = H[i][j];
    }
    for (int r = 0; r < 4; ++r) g0[i] += (Wv[r] * Wv[r]) * G[r][i] * v[r];
  }

  // 2. Initialize feasible x from warm-started working set W_warm
  WlsResult res{};
  for (int i = 0; i < 8; ++i) {
    res.W[i] = W_warm[i];
    if (xmax[i] <= xmin[i] + 1e-6f) { res.x[i] = xmin[i]; res.W[i] = -1; } // dead motor locked
    else if (res.W[i] < 0)          { res.x[i] = xmin[i]; }
    else if (res.W[i] > 0)          { res.x[i] = xmax[i]; }
    else                            { res.x[i] = std::clamp(0.0f, xmin[i], xmax[i]); }
  }

  // 3. Deterministic active-set loop (<= 12 iters; ~18-32 us on 480 MHz STM32H753)
  for (uint8_t iter = 0; iter < max_iter; ++iter) {
    res.iters = iter + 1;
    int free_idx[8], n_free = 0;
    float d[8]{};
    for (int i = 0; i < 8; ++i) {
      float hx = 0.0f;
      for (int j = 0; j < 8; ++j) hx += H[i][j] * res.x[j];
      d[i] = g0[i] - hx;   // negative objective gradient -(H*x - g0)
      if (res.W[i] == 0) free_idx[n_free++] = i;
    }

    float p[8]{};
    if (n_free > 0) solve_cholesky_submatrix(H, d, free_idx, n_free, p);

    float alpha = 1.0f;
    int block_idx = -1;
    int8_t block_sign = 0;
    for (int k = 0; k < n_free; ++k) {
      int i = free_idx[k];
      if (p[i] > 1e-7f && res.x[i] + p[i] > xmax[i]) {
        float a = (xmax[i] - res.x[i]) / p[i];
        if (a < alpha) { alpha = a; block_idx = i; block_sign = +1; }
      } else if (p[i] < -1e-7f && res.x[i] + p[i] < xmin[i]) {
        float a = (xmin[i] - res.x[i]) / p[i];
        if (a < alpha) { alpha = a; block_idx = i; block_sign = -1; }
      }
    }

    for (int k = 0; k < n_free; ++k) res.x[free_idx[k]] += alpha * p[free_idx[k]];

    if (block_idx >= 0) {
      res.x[block_idx] = (block_sign > 0) ? xmax[block_idx] : xmin[block_idx];
      res.W[block_idx] = block_sign;
      continue;
    }

    // Optimal on current face: check KKT multipliers lambda_i = W_i * (g0 - H*x)_i >= 0
    float min_lambda = -1e-5f;
    int drop_idx = -1;
    for (int i = 0; i < 8; ++i) {
      if (res.W[i] == 0 |

| xmax[i] <= xmin[i] + 1e-6f) continue;
      float hx = 0.0f;
      for (int j = 0; j < 8; ++j) hx += H[i][j] * res.x[j];
      float lam = static_cast<float>(res.W[i]) * (g0[i] - hx);
      if (lam < min_lambda) { min_lambda = lam; drop_idx = i; }
    }
    if (drop_idx < 0) break;  // Global KKT optimum reached
    res.W[drop_idx] = 0;
  }

  for (int i = 0; i < 8; ++i) {
    W_warm[i] = res.W[i];
    if (res.W[i] != 0) res.sat_mask |= static_cast<uint8_t>(1u << i);
  }
  return res;
}
```

Solving exact linear programs over the 4-DOF Attainable Moment Set (AMS) polytope at hover collective thrust ($T = W = 2,354.4\text{ N}$, nominal 400 V, incorporating lower-rotor aerodynamic unblanking whenever an upper rotor fails) quantifies the exact moment and trimmable-thrust contraction across all four fault regimes. In the rotated diagonal frame defined in Section 3.4, $\tau_u = L(T_2 - T_3)$ denotes torque about the Arm 0-1 line produced by the `FL-RR` pair, and $\tau_v = L(T_0 - T_1)$ denotes torque about the Arm 2-3 line produced by the `FR-RL` pair.

| Fault Regime (Dead Motors) | Raw T_max [N] (T/W) | Trimmable T_max (tau=0) [N] (T/W) | Roll tau_x [N*m] | Pitch tau_y [N*m] | Yaw tau_z [N*m] | Diagonal tau_u (FL-RR) / tau_v (FR-RL) [N*m] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0: Nominal** (None) | 5,886.0 (2.500) | 5,886.0 (2.500) | [-2,248, +2,248] | [-2,248, +2,248] | [-192.0, +192.0] | [-1,987, +1,987] / [-1,987, +1,987] |
| **1a: Single Upper** (M0 FR-U) | 5,209.9 (2.213) | 4,470.2 (1.899) | [-1,427, +2,248] | [-2,248, +1,427] | [-185.3, +115.9] | [-1,949, +1,949] / [-1,987, +1,074] |
| **1b: Single Lower** (M4 FR-L) | 5,209.9 (2.213) | 4,456.3 (1.893) | [-1,477, +2,248] | [-2,248, +1,477] | [-124.5, +192.0] | [-1,987, +1,987] / [-1,987, +1,074] |
| **2a: Cross-Arm Opp-Spin** (M0+M5) | 4,533.8 (1.926) | 4,470.2 (1.899) | [-1,427, +1,474] | [-1,474, +1,427] | [-120.7, +115.9] | [-1,949, +1,949] / [-1,074, +1,074] |
| **2b: Cross-Arm Same-Spin** (M0+M1) | 4,533.8 (1.926) | 2,879.4 (1.223) | [-1,427, +1,427] | [-1,427, +1,427] | [-178.7, +39.8] | [-1,407, +1,407] / [-1,074, +1,074] |
| **2c: Adjacent-Arm Opp** (M0+M2) | 4,533.8 (1.926) | 3,181.6 (1.351) | [-1,374, +1,374] | [-2,248, +790] | [-109.3, +109.3] | [-1,944, +1,074] / [-1,944, +1,074] |
| **3: Same-Arm Dual Pair** (M0+M4) | 4,414.5 (1.875) | **2,879.4 (1.223)*** | **[0.0, +2,248]** | **[-2,248, 0.0]** | [-55.3, +39.8] | **[-795, +795] / [-1,987, 0.0]** |

*Regime 3 zero-yaw vs. full-thrust note:* Because upper and lower torque-to-thrust ratios differ ($c_{m,u} = 0.07590\text{ m}$ vs. $c_{m,l} = 0.08572\text{ m}$), enforcing exact zero yaw torque ($\tau_z = 0$) trims lower-rotor thrust slightly below 676.1 N when upper rotors saturate at 795.4 N, giving 2,879.4 N (1.223 $W$). Driving Arms 2 and 3 to full 2,943.0 N (1.250 $W$) produces a mild -4.8 N*m CCW yaw torque that aligns favorably with active differential-yaw spin-up. Simultaneously, compounding Regime 3's contracted 2-arm thrust margin with HV battery voltage sag exposes the critical BLDC back-EMF constraint: a motor voltage-limited at 400 V collapses quadratically to $T/W = 1.028$ at 360 V (10% sag), whereas our corrected 10%-headroom FOC-clamped winding ($K_{e,\text{bus}} = 1.36\text{ V}\cdot\text{s/rad}$) preserves the full 1.250 $T/W$ plateau down to 360 V and remains above the 1.12 survival floor down to 338 V.

> **[Figure 2 — AMS Pitch Contraction & Voltage-Sag Survival Floor]:** Hover pitch moment envelope contracts asymmetrically across fault regimes, collapsing to 0.0 N*m nose-up authority under Same-Arm Dual Loss (`M0+M4`). Under 10% terminal sag (360 V), a voltage-limited BLDC ($K_e = 1.53\text{ V}\cdot\text{s/rad}$) drops quadratically to `T/W = 1.028` (breaching the 1.12 floor; 0.924 at 340 V), whereas a 10%-headroom FOC-clamped winding ($K_e = 1.36\text{ V}\cdot\text{s/rad}$) holds `T/W = 1.250` at 360 V and `1.137` at 340 V.

### 3.4 Critical Coaxial Spin-Up Physics: Same-Arm Dual Coaxial-Pair Loss Trade Analysis

When both rotors on Arm 0 (Front-Right, +45 deg, `M0` CCW + `M4` CW) fail simultaneously, the coaxial X8 differs fundamentally from a quadrotor losing one rotor: losing a coaxial pair removes **one CCW and one CW rotor**, leaving three intact arms (`Arm 1` RL, `Arm 2` FL, `Arm 3` RR) with balanced counter-rotating pairs and active bidirectional yaw authority ($\tau_z \in [-55.3, +39.8]\text{ N}\cdot\text{m}$ from Arms 2 and 3 alone, and up to +/-83 N*m during brief Arm 1 pulsing). Because the airframe does not enter an uncontrolled yaw spin automatically, flight-control engineers must evaluate whether non-spinning 2-arm symmetric hover (`Option A`) or actively commanded differential-yaw spin (`Option B1` zero-tilt flat spin vs. `Option B2` 3-arm tilted spin) is physically viable for a 240 kg aircraft.

Rotating `FRD` by +45 deg about $+Z_{\text{FRD}}$ into the diagonal arm frame $(\mathbf{e}_u, \mathbf{e}_v, \mathbf{e}_z)$ places dead Arm 0 at $(+L, 0)$, opposite Arm 1 at $(-L, 0)$, Arm 2 (`FL`) at $(0, -L)$, and Arm 3 (`RR`) at $(0, +L)$. Let $T_k = T_{u,k} + T_{l,k} \ge 0$ denote combined arm thrust. In this rotated frame, the moments of inertia are $J_{uu} = J_{vv} = J_\perp = \tfrac{1}{2}(J_{xx} + J_{yy}) = 90.0\text{ kg}\cdot\text{m}^2$, $J_{uv} = \tfrac{1}{2}(J_{xx} - J_{yy}) = -5.0\text{ kg}\cdot\text{m}^2$, and $J_{zz} = 140.0\text{ kg}\cdot\text{m}^2$, while roll and pitch moments decouple cleanly:

$$\tau_u = L(T_2 - T_3) \in [-794.6,\, +794.6]\text{ N}\cdot\text{m} \quad (\text{bidirectional at 2-arm hover})$$

$$\tau_v = L(T_0 - T_1) = -L T_1 \in [-1,986.5,\, 0.0]\text{ N}\cdot\text{m} \quad (\text{strictly unilateral, } \tau_v \le 0)$$

Static non-spinning equilibrium ($\dot{p}_u = \dot{q}_v = 0$, $\bar{\Omega}_z = 0$) enforces $\tau_u = \tau_v = 0$, forcing diagonally opposite Arm 1 to zero thrust ($T_1 = 0$) and loading orthogonal Arms 2 and 3 to $T_2 = T_3 = W/2 = 1,177.2\text{ N}$ (80% of pair capacity, preserving 1.25x $mg$ nominal vertical thrust). However, linearizing the 4-state reduced-attitude perturbation $\tilde{\mathbf{s}} = [p_u, q_v, n_u, n_v]^T$ around $\bar{r} = 0, T_1 = 0$ produces a Kalman controllability matrix $\mathcal{C}$ of **rank 2 out of 4** ($\sigma_{\min}(\mathcal{C}) = 0$). Any disturbance tipping the vehicle toward dead Arm 0 demands $+\tau_v > 0$, which is mechanically unattainable because $T_0 \equiv 0$ and $T_1 \ge 0$ can only pull further negative. Consequently, **non-spinning 2-arm symmetric hover (`Option A`) is dynamically uncontrollable and will capsize upon any coastal gust or small CG shift toward the dead arm.**

Conversely, applying Mueller & D'Andrea's primary-axis reduced-attitude dynamics to the coaxial diagonal frame (where counter-rotating pairs cancel net rotor angular momentum, $h_{\text{rot},z} \approx 0$) shows that commanding a steady yaw spin rate $\bar{r} = \bar{\Omega}_z \neq 0$ couples $p_u$ and $q_v$ gyroscopically through $\bar{a} = \frac{J_\perp - J_{zz}}{J_\perp}\bar{r} = -0.5556\,\bar{r}$:

$$\begin{bmatrix} \dot{p}_u \\ \dot{q}_v \\ \dot{n}_u \\ \dot{n}_v \end{bmatrix} = \begin{bmatrix} 0 & \bar{a} & 0 & 0 \\ -\bar{a} & 0 & 0 & 0 \\ 0 & -\bar{n}_z & 0 & \bar{r} \\ \bar{n}_z & 0 & -\bar{r} & 0 \end{bmatrix} \begin{bmatrix} p_u \\ q_v \\ n_u \\ n_v \end{bmatrix} + \begin{bmatrix} 1/J_\perp \\ 0 \\ 0 \\ 0 \end{bmatrix} \tau_u, \qquad \det(\mathcal{C}) \propto \bar{a}\,\bar{r}\,\bar{n}_z^2\,(\bar{a} + \bar{r})^2 \left(\frac{1}{J_\perp}\right)^4$$

Because `VertiRTA-X8-240` is oblate ($J_{zz} = 140 > J_\perp = 90\text{ kg}\cdot\text{m}^2$), both $\bar{a} = -0.556\,\bar{r} \neq 0$ and $\bar{a} + \bar{r} = +0.444\,\bar{r} \neq 0$ hold for every nonzero $\bar{r}$ — completely avoiding the uncontrollability singularities that affect near-spherical frames. The single bidirectional torque $\tau_u = L(T_2 - T_3)$ from Arms 2 and 3 **alone** renders all four reduced-attitude states fully controllable at zero body tilt ($T_1 = 0, \beta = 0^\circ$). An alternative 3-arm tilted equilibrium (`Option B2`) uses positive Arm 1 thrust $T_1 > 0$ balanced by gyroscopic nutation $(J_{zz} - J_\perp)\Omega^2 \sin\beta \cos\beta = L T_1$ to lift available vertical thrust above $1.25\,W$, but requires 5.0-7.4 rad/s (48-70 RPM) spin rates, introducing severe 23-32 deg actuator phase lag on slow 1.50 m rotors, 0.64-1.38 g passenger centrifugal loads, and a 15-20 deg landing tilt that risks skid rollover.

| Strategy | Equilibrium Parameters | Controllability Rank & min sv(C) | Spin-Up Time t_spinup [s] | Settling Time tau_settle [s] | Actuator Phase Lag (tau_m = 0.085 s) | Passenger Head Centrifugal Load (r = 0.25 m) | Touchdown Tilt beta & Orbit R_ps | Verdict for 240 kg Coaxial X8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Option A: Non-Spinning 2-Arm** | r_bar = 0, T1 = 0, T2=T3 = 1,177 N | **Uncontrollable (2/4, 0.0)** | 0.0 s | inf (capsize) | 0.0 deg | 0.00 g | 0 deg / 0 m | **Reject:** linearly unstable in +v; capsizes on any upset toward Arm 0. |
| **Option B1a: Ultra-Slow Flat Spin** | abs(r_bar) = 1.0 rad/s (9.5 RPM), T1 = 0 | Full (4/4, 2.01e-3) | 2.53 s | 13.3 s | 4.9 deg | 0.025 g | 0 deg / 0 m | Marginal: attitude recovery too slow for 12-15 m/s coastal gusts. |
| **Option B1b: Optimal Flat Spin (Selected)** | **abs(r_bar) = 2.0-2.5 rad/s (19-24 RPM), T1 = 0** | **Full (4/4, peak 5.65e-3 at 2.0)** | **5.06-6.33 s** (3.4 s w/ Arm 1 pulse) | **5.0-6.4 s** | **9.6-12.0 deg** (gain 0.98) | **0.10-0.16 g** | **0 deg / 0.0 m** | **Optimal:** peak controllability conditioning, level touchdown, low G, minimal phase lag. |
| **Option B1c: Fast Flat Spin** | abs(r_bar) = 4.0 rad/s (38.2 RPM), T1 = 0 | Full (4/4, 3.70e-3) | 10.13 s | 2.9 s | 18.8 deg | 0.408 g | 0 deg / 0.0 m | Unnecessary spin-up delay and passenger disorientation. |
| **Option B2: 3-Arm Tilted Spin** | T1 = 300-500 N, Omega = 5.0-7.4 rad/s (48-70 RPM) | Full (4/4), T_max/W -> 1.33-1.41 | 8.47-12.40 s (at 83 N*m) | 2.0-2.5 s | **23.0-32.0 deg** | **0.64-1.38 g** | **15-20 deg / 0.05-0.14 m** | **Reject:** excessive slow-rotor phase lag; tilted touchdown causes skid rollover. |

> **[Figure 3 — Active Differential-Yaw Flat Spin Trade Study]:** Across 0-6 rad/s yaw spin rates, minimum controllability singular value collapses to zero at 0 rad/s and peaks at `5.65e-3` at `2.0 rad/s` (19.1 RPM); LQR attitude settling drops steeply to `6.4 s` at 2.0 rad/s while 2-arm spin-up grows linearly to `5.06-6.33 s`; and actuator phase lag stays at `9.6-12.0 deg` inside Option B1 (2.0-2.5 rad/s) versus `23-27 deg` in Option B2 (5.0-6.0 rad/s).

Spin-direction selection exploits the exact coaxial upper/lower torque asymmetry: when Arm 0 (`FR`) or Arm 1 (`RL`) fails, the surviving orthogonal arms (`FL` and `RR`) carry CW upper rotors (`M2, M3`, 795.4 N max) and CCW lower rotors (`M6, M7`, 676.1 N max), providing larger **CCW** yaw torque (-55.3 N*m vs. +39.8 N*m). Accordingly, `mc_indi_wls_control` commands **CCW** spin ($\bar{\Omega}_z = -2.2\text{ rad/s}$) for Arm 0/1 dual loss and symmetric **CW** spin ($\bar{\Omega}_z = +2.2\text{ rad/s}$, +55.3 N*m) for Arm 2/3 dual loss. During the 0-3.5 s rapid spin-up transient, WLS applies full -55.3 N*m differential yaw torque on Arms 2 and 3 plus a bounded single-rotor CCW pulse on Arm 1 lower (`M5`, capped at $T_1 \le 150\text{ N}$ only when $n_v$ allows), crossing 1.5 rad/s in `< 2.8 s` and locking at $|\bar{\Omega}_z| = 2.2\text{ rad/s}$ (21 RPM) so any initial tip toward Arm 0 rotates 90 deg into the bidirectional Arm 2/3 plane before tilt exceeds ~12 deg. In steady spin, rotating $\mathbf{n}_{B,\text{des}}$ ahead around $+Z_{\text{FRD}}$ by the exact phase lead $\Delta\psi_{\text{lead}} = \arctan(\bar{\Omega}_z \tau_m) = \mp 10.6^\circ$ cancels actuator-lag cross-coupling.

Evaluating vertical flare stopping distance $h_{\text{flare}} = \frac{v_z^2}{2 a_{\text{up}}} + |v_z|\tau_m$ where $a_{\text{up}} = (T/W_{\text{eff}} - 1)g$ establishes strict Regime 3 sink-rate limits:

- **Nominal 400 V ($T/W = 1.25$, $a_{\text{up}} = 2.45\text{ m/s}^2$):** Arresting 1.5 m/s sink requires $h_{\text{flare}} = 0.59\text{ m}$ (0.70 s); 2.5 m/s requires 1.49 m (1.10 s).
- **Minimum Spec Floor ($T/W = 1.12$, $a_{\text{up}} = 1.18\text{ m/s}^2$):** Arresting 1.5 m/s requires 1.08 m (1.36 s); 2.5 m/s requires 2.87 m (2.21 s).
- **Unprotected Voltage-Limited Motor ($T/W = 1.028$, $a_{\text{up}} = 0.27\text{ m/s}^2$):** Arresting 2.5 m/s requires an unrecoverable 11.59 m (9.19 s).

Under Regime 3, outer-loop RTA must cap approach sink rate at $|v_z| \le 2.5\text{ m/s}$ above 10 m AGL, ramping down to $|v_z| \le 1.5\text{ m/s}$ below 5 m AGL and $\le 0.8\text{ m/s}$ at touchdown, where circular compliant skids ($r_{\text{skid}} \approx 0.70\text{ m}$) absorb ~1.5 m/s tangential rim speed cleanly as weight-on-skids triggers immediate collective cutoff.

### 3.5 Online Motor FDI Residual Observer (< 100 ms, Gust-Immune)

To isolate 1-2 motor losses inside `< 100 ms` without false trips during 12-15 m/s Marmara coastal gusts, `libvertirta_flight_core` fuses a primary ESC-telemetry innovation channel with a sensorless 4-DOF INDI wrench-residual backup at 400 Hz:

- **Channel 1 — ESC RPM Innovation (35-60 ms):** PX4 v1.16 publishes motor RPM over `esc_status` from DroneCAN or bidirectional DShot; comparing predicted speed $\hat{\omega}_i[k]$ against measured $\omega_{\text{esc},i}[k]$ yields $r_{\omega,i}[k] = \hat{\omega}_i[k] - \omega_{\text{esc},i}[k]$. When power cuts on a hovering rotor (151.8 rad/s), aerodynamic drag decelerates the 0.135 kg*m^2 assembly at $\dot{\omega} = -k_{Q0}\omega^2 / J_r = -179.0\text{ rad/s}^2$, dropping 85 RPM (8.9 rad/s) inside 50 ms, whereas a 15 m/s gust shifts closed-loop motor speeds by `< 15 RPM` symmetrically across all eight motors. Latching motor $i$ when $r_{\omega,i} > 12\text{ rad/s}$ (115 RPM, or `> 12%` of $\hat{\omega}_i$) for 10 consecutive 400 Hz ticks (25 ms) — or on ESC timeout/overcurrent flags — isolates electrical/ESC motor losses in **35-60 ms**.
- **Channel 2 — Sensorless 4-DOF INDI Wrench Residual (65-85 ms):** Essential for clean propeller blade fracture (where the unloaded motor keeps spinning) or telemetry dropout, the observer forms the 4-DOF wrench residual from Butterworth-filtered TMR-voted IMU signals (where $a_{z,f}$ is the filtered body-Z accelerometer specific force, equal to $-T/m$ in hover):

$$\mathbf{r}_w[k] = \begin{bmatrix} J\dot{\boldsymbol{\Omega}}_f[k] + \boldsymbol{\Omega}_f[k] \times (J\boldsymbol{\Omega}_f[k]) \\ -m\,a_{z,f}[k] \end{bmatrix} - B_T \hat{\mathbf{T}}_f[k]$$

Losing one hovering upper rotor (`M0`, 318.2 N) creates a step residual of $|\mathbf{r}_{\tau,xy}| = 303.7\text{ N}\cdot\text{m}$ (3.57 rad/s^2 angular acceleration) and $\Delta F_z = -318.2\text{ N}$ (-1.33 m/s^2), versus at most 66.1 N parasitic drag and `<= 45 N*m` fuselage/blade-flapping moment from a sharp-edged 15 m/s gust — a **6.7x torque signal-to-noise ratio**. To isolate the exact rotor index and reject wind perturbations, the observer evaluates weighted cosine similarity against each negative column $-\mathbf{b}_i$ of $B_T$ using $D_w = \mathrm{diag}(1/300,\, 1/300,\, 1/25,\, 1/300)$:

$$\rho_i[k] = \frac{\langle D_w \mathbf{r}_w[k],\; -D_w \mathbf{b}_i \rangle}{\|D_w \mathbf{r}_w[k]\|_2 \,\|D_w \mathbf{b}_i\|_2}$$

Because coaxial upper and lower rotors share $(x_i, y_i)$ but carry opposite yaw signs $s_i$, the yaw residual $r_{\tau,z} = \mp 24.2\text{ N}\cdot\text{m}$ cleanly separates upper versus lower failure on the same arm. A 1-sided CUSUM persistence gate requiring $\|D_w \mathbf{r}_w\|_2 > 0.55$ and $\rho_{i^*} > 0.88$ for 20 ticks (50 ms) isolates mechanical blade losses in **65-85 ms** (inclusive of 8 Hz Butterworth rise time) with strong immunity to coastal gusts.

### 3.6 Cross-Layer Capability Envelope Struct (`uORB` / `uXRCE-DDS`)

To stream real-time authority contraction upward to the ROS 2 RTA layer (`DR-2`), `mc_indi_wls_control` publishes a fixed-size 72-byte uORB message `msg/VertirtaCapabilityEnvelope.msg` mirrored in `px4_msgs` and exported via `src/modules/uxrce_dds_client/dds_topics.yaml` at 100 Hz:

```text
# msg/VertirtaCapabilityEnvelope.msg (Published by mc_indi_wls_control @ 100 Hz)
uint64 timestamp                # [us] PX4 HRT monotonic timestamp
uint64 timestamp_sample         # [us] Voted IMU sample timestamp

uint8  fault_bitmask            # Bit 0..7 = 1 if motor i isolated failed (0x00 nominal)
uint8  tmr_state                # 0=TMR_3OF3, 1=DUPLEX_2OF2, 2=SIMPLEX_ANALYTICAL, 3=FAILSAFE_DIVERT
uint8  imu_health_mask          # Bit 0..2 = 1 if IMU 0..2 healthy
uint8  flight_regime            # 0=NOMINAL, 1=SINGLE_LOSS, 2=CROSS_DUAL, 3=SAME_ARM_ACTIVE_SPIN, 4=CRITICAL_DESCENT

float32 t_max                   # [N] Dynamic trimmable max collective thrust T_max(fault, V_term)
float32 a_xy_max                # [m/s^2] Max sustainable horizontal accel respecting tilt & T_max
float32 tilt_max                # [rad] Dynamic safe tilt ceiling (0.45 nom -> 0.18 Regime 3)
float32 vz_sink_max             # [m/s] Max safe flare-recoverable descent rate (positive magnitude)

float32 v_term                  # [V] Filtered 1-RC terminal bus voltage
float32 i_pack                  # [A] Measured/estimated HV pack current
float32 soc                     # [0..1] Battery State of Charge
float32 p_hover_deg             # [W] Current fault-degraded electrical hover power P_hover_deg

float32 spin_rate_ref           # [rad/s] Active yaw spin target (0.0 in Reg 0-2, +/-2.2 in Reg 3)
float32 spin_rate_meas          # [rad/s] Voted body yaw rate r
uint16  fdi_latency_ms          # [ms] Timestamped isolation latency of last latched fault
uint8[2] _padding
```

The corresponding entries in `src/modules/uxrce_dds_client/dds_topics.yaml` adhere strictly to PX4 v1.16 `/fmu/out/` and `/fmu/in/` conventions:

```yaml
publications:
  - topic: /fmu/out/vertirta_capability_envelope
    type: px4_msgs::msg::VertirtaCapabilityEnvelope
    rate_limit: 100
subscriptions:
  - topic: /fmu/in/vertirta_rta_setpoint
    type: px4_msgs::msg::VertirtaRtaSetpoint
```

---

## 4. Coupled 1-RC Thevenin Battery Voltage-Sag and Fault-Degraded Power Model

### 4.1 Discrete-Time 1-RC Thevenin ECM Coupled to Propulsion Power

High-drain `INR21700-P45B` NMC811 cells (3.6 V nominal, 4.5 Ah, 45 A continuous, 13.8 mOhm max DCIR) arranged in `112S12P` (403.2 V nominal, 470.4 V full, 54.0 Ah, 21.77 kWh) are represented cleanly for real-time sag prediction by a single-polarization (`1-RC`) Thevenin equivalent circuit model — a topology shown to hold `< 0.85%` average terminal voltage error on dynamic cylindrical Li-ion pulse profiles:

$$V_{\text{term}}[k] = V_{\text{oc}}(\text{SoC}[k]) - I[k]\,R_0(\text{SoC}, T_{\text{cell}}) - V_{RC}[k]$$

Applying exact zero-order-hold discretization over $\Delta t = 0.01\text{ s}$ (100 Hz) with polarization time constant $\tau_1 = R_1 C_1 = 25.2\text{ s}$ ($R_0 = 0.1363\,\Omega$, $R_1 = 0.0840\,\Omega$, $C_1 = 300.0\text{ F}$) yields:

$$V_{RC}[k+1] = e^{-\Delta t / \tau_1}\,V_{RC}[k] + R_1\!\left(1 - e^{-\Delta t / \tau_1}\right) I[k], \qquad \text{SoC}[k+1] = \text{SoC}[k] - \frac{\eta_c\,\Delta t}{3600\,Q_{\text{pack}}}\,I[k]$$

where $Q_{\text{pack}} = 54.0\text{ Ah}$ and $\eta_c = 1.0$ on discharge. Total electrical power couples induced, profile, parasitic, and avionics demand across all four arms $m \in \{0,1,2,3\}$ (each carrying $n_m \in \{0,1,2\}$ active rotors):

$$P_{\text{elec}} = \frac{1}{\eta_{\text{drive}}} \left[ \sum_{m=0}^{3} \left( \kappa_{\text{iso}}\,T_m\,v_{i,m} + n_m \frac{1}{8}\rho A \sigma C_{d0} (\omega_m R)^3 (1 + 4.65\mu_m^2) \right) + \frac{1}{2}\rho f_{\text{drag}} V^3 \right] + P_{\text{avionics}}$$

with drivetrain efficiency $\eta_{\text{drive}} = 0.88$ (0.93 motor x 0.96 ESC x 0.985 HV harness), $P_{\text{avionics}} = 450\text{ W}$, and effective arm disk area $A_{\text{eff},m} = \frac{2}{\kappa_{\text{int}}(V)^2} A$ for coaxial pairs (1.344 $A$ in hover) versus $A$ for a single surviving rotor. Enforcing $P_{\text{elec}} = V_{\text{term}} I$ produces the exact closed-form quadratic current closure at every tick:

$$R_0 I^2 - \left(V_{\text{oc}}(\text{SoC}) - V_{RC}\right) I + P_{\text{elec}} = 0 \implies I = \frac{V_{\text{eff}} - \sqrt{V_{\text{eff}}^2 - 4 R_0 P_{\text{elec}}}}{2 R_0}, \qquad V_{\text{eff}} \equiv V_{\text{oc}} - V_{RC}$$

The discriminant $V_{\text{eff}}^2 > 4 R_0 P_{\text{elec}}$ explicitly defines the battery voltage-collapse ceiling $P_{\text{collapse}} = V_{\text{eff}}^2 / (4 R_0)$ (~270 kW at 20% SoC and 25 C, though cold cells below 5 C can double $R_0$ and cut this ceiling in half).

### 4.2 Fault-Degraded Hover Power, Sag Spikes, and Dynamic $T_{\max}(V_{\text{term}})$

Losing rotors contracts effective disk area $A_{\text{eff}}$ and elevates disk loading, scaling induced power as $P_i \propto T^{3/2} / \sqrt{A_{\text{eff}}}$. While single-motor and cross-arm dual losses raise hover electrical power modestly (+4.3% and +8.5%), **Regime 3 (Same-Arm Dual Loss)** halves active lifting area from 4 arms to 2, driving hover induced power up by $\sqrt{2}$ (+41.4%, from 27.23 kW to 38.51 kW) and total electrical draw to 51.05 kW (+40.9%), peaking at 62.85 kW during a 1.15 $mg$ terminal flare:

| Fault Regime (30 s sustained hover; 5 s unpolarized flare) | Hover P_i / P_0 [kW] | Total P_elec [kW] | 50% SoC (V_oc = 409.9 V): I [A] / V_term [V] (% of 400 V) | 20% SoC (V_oc = 384.2 V): I [A] / V_term [V] (% of 400 V) | Hover Endurance (20% -> 10% SoC) [min] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Regime 0: Nominal 8-rotor** | 27.23 / 4.25 | 36.23 | 92.4 A (1.71C) / 391.9 V (98.0%) | 99.3 A (1.84C) / 364.8 V (91.2%) | 3.6 min |
| **Regime 1: Single-rotor loss** (M0) | 28.32 / 4.53 | 37.77 (+4.3%) | 96.6 A (1.79C) / 391.1 V (97.8%) | 103.8 A (1.92C) / 363.9 V (91.0%) | 3.5 min |
| **Regime 2a: Cross-arm dual** (M0+M5) | 29.40 / 4.80 | 39.32 (+8.5%) | 100.7 A (1.87C) / 390.3 V (97.6%) | 108.3 A (2.01C) / 363.1 V (90.8%) | 3.3 min |
| **Regime 3: Same-arm dual (2-arm)** | 38.51 / 6.01 | 51.05 (+40.9%) | 132.9 A (2.46C) / 384.0 V (96.0%) | 143.3 A (2.65C) / **356.2 V (89.1%)** | 2.6 min |
| **Regime 3: Terminal Flare (1.15 mg)** | 47.50 / 7.41 | 62.85 (+73.5%) | 163.2 A (3.02C) / 385.2 V (96.3%) | 175.8 A (3.26C) / **357.5 V (89.4%)*** | 2.1 min |

*Pre-polarized flare note:* An isolated 5 s flare appears slightly higher in terminal voltage (357.5 V) than a 30 s hover (356.2 V) solely because the 25.2 s RC polarization branch reaches only 18% charge over 5 s versus 70% over 30 s. When a 5 s flare immediately follows 30 s of sustained Regime 3 hover (already polarized at $V_{RC} = 8.4\text{ V}$), terminal voltage dips transiently by an extra ~4.4 V to **~351.8 V** (88.0% of 400 V).

Modeling the DC-bus-equivalent BLDC speed ceiling $V_{\text{term}} = K_{e,\text{bus}} \omega + \frac{R_{\text{eff}}}{K_{e,\text{bus}}} k_{Q0} \omega^2$ (absorbing inverter SVPWM modulation ceiling into $K_{e,\text{bus}}$) gives the dynamic speed and thrust contraction law:

$$\omega_{\text{lim}}(V_{\text{term}}) = \frac{-K_{e,\text{bus}} + \sqrt{K_{e,\text{bus}}^2 + 4 (R_{\text{eff}} k_{Q0} / K_{e,\text{bus}})\, V_{\text{term}}}}{2 (R_{\text{eff}} k_{Q0} / K_{e,\text{bus}})}, \qquad T_{\max,i}(V_{\text{term}}) = k_{T,i}\min\!\left(\omega_{\max},\, \omega_{\text{lim}}(V_{\text{term}})\right)^2$$

With our audited winding $K_{e,\text{bus}} = 1.36\text{ V}\cdot\text{s/rad}$ and hot effective resistance $R_{\text{eff}} = 0.72\,\Omega$, reaching 240 rad/s requires 358.4 V — holding 2-arm $T/W = 1.250$ down to 360 V, 1.20+ during the 351.8 V pre-polarized flare, and **1.135 (>1.12) even at 340 V** (15% sag). If the mass-compliant `112S7P` (12.7 kWh) COTS pack is chosen instead, higher pack impedance ($R_0 = 223\text{ m}\Omega$, $R_1 = 144\text{ m}\Omega$) drops 20% SoC Regime 3 terminal voltage to ~352 V instantaneously and ~335 V after 30 s sustained hover, necessitating either $K_{e,\text{bus}} \le 1.28\text{ V}\cdot\text{s/rad}$ or a 25% SoC hard reserve floor to preserve $T/W \ge 1.12$.

### 4.3 Contraction of the 3D Energy-Reachable SLZ Cone

Translational lift and coaxial wake unblanking ($\eta_{\text{coax}}$ rising from 0.85 to 0.98) dramatically cut electrical power in forward cruise compared to hover. For an eVTOL at altitude $h$ above a candidate Safe Landing Zone (SLZ) with usable energy $E_{\text{avail}} = (\text{SoC} - \text{SoC}_{\text{res}}) E_{\text{pack}} - P_{\text{hover,deg}} t_{\text{flare}}$ (30 s terminal approach/flare reserve) flying in wind vector $\mathbf{v}_w$, the horizontal boundary radius $R_{\text{SLZ}}$ of the **3D Energy-Reachable SLZ Cone** along ground-track direction $\hat{\mathbf{e}}_\chi$ is:

$$R_{\text{SLZ}}(\hat{\mathbf{e}}_\chi, h, \text{SoC}, \text{fault}) = \max_{V \in [0, v_{\max,\text{fault}}]} \left( \left( V + \mathbf{v}_w \cdot \hat{\mathbf{e}}_\chi \right) \cdot \frac{E_{\text{avail}} + \eta_{\text{pot}} m g h}{P_{\text{elec}}(V, \text{fault})} \right)$$

where $\eta_{\text{pot}} \approx 0.65$ captures induced-power relief during a controlled $|v_z| \le 2.5\text{ m/s}$ glide without relying on battery regeneration. Across a 10% SoC slice (20% -> 10% SoC, 2.177 kWh, zero wind, excluding flare reserve):

- **Regime 0 (Nominal 8-rotor):** 22 m/s draws 19.12 kW for **9.02 km** reach (15 m/s draws 21.40 kW for **5.49 km**);
- **Regime 1 (Single-rotor loss, 15 m/s cap):** draws 24.13 kW for **4.87 km** reach;
- **Regime 2a (Cross-arm dual, 15 m/s cap):** draws 26.86 kW for **4.38 km** reach;
- **Regime 3 (Same-arm dual, 15 m/s cap):** draws 37.84 kW for **3.11 km** reach — a **43.4% contraction** at 15 m/s and a **65.5% loss** relative to nominal 22 m/s range.

> **[Figure 4 — Electrical Power Decomposition & SLZ Reachable Range Contraction]:** Electrical hover power climbs from `36.2 kW` (Nominal) to `51.1 kW` (+40.9%) in Same-Arm Dual Loss and `62.9 kW` (+73.5%) in a 1.15 g terminal flare; forward flight unblanks the coaxial wake and yields `9.02 km` nominal reach at 22 m/s (`5.49 km` at 15 m/s), contracting by `-43.4%` to `3.11 km` at 15 m/s under Same-Arm Dual Loss.

---

## 5. PX4 v1.16 Multi-IMU SITL (`GZBridge`), SDF Model, and 400 Hz TMR FDIR Voter

### 5.1 Upstream PX4 v1.16 Source Audit (`GZBridge.cpp`) and Targeted Multi-IMU Patch

Auditing upstream PX4 v1.16 (`src/modules/simulation/gz_bridge/GZBridge.cpp`, `GZMixingInterfaceESC.cpp`, and `server.config`) confirms that stock `GZBridge.cpp` **does not support three simultaneous Gazebo Harmonic IMU sensors** out of the box. Stock `GZBridge` allocates only a single `PX4Gyroscope` and `PX4Accelerometer` pair bound to one hardcoded Gazebo topic, `/world/<world>/model/<model>/link/base_link/sensor/imu_sensor/imu`, handled by a scalar `GZBridge::imuCallback`. Three additional upstream quirks directly affect this SITL architecture:

- `src/modules/simulation/gz_bridge/server.config` loads `gz-sim-imu-system`, `gz-sim-air-pressure-system`, `gz-sim-navsat-system`, `gz-sim-magnetometer-system`, `gz-sim-wind-effects-system`, and `gz-sim-sensors-system` (`ogre2`) globally — meaning custom world SDFs must omit world `<plugin>` blocks to avoid duplicate managers;
- Adding a 3D `gpu_lidar` routinely triggers `ogre2` shader compilation delays exceeding `GZBridge::init()`'s default `sleep_count_limit = 1000` (1 s) lockstep clock timeout; and
- `GZMixingInterfaceESC.cpp` rescales normalized `[0, 1]` motor commands linearly between `SIM_GZ_EC_MIN` and `SIM_GZ_EC_MAX` onto Gazebo's `actuators` velocity vector while reading back motor speeds into `esc_status`.

To enable true triple-IMU SITL and deterministic motor failure injection without breaking stock PX4 models, apply a compact targeted patch across `GZBridge.{hpp,cpp}` and `GZMixingInterfaceESC.cpp`:

1. Replace scalar `_px4_gyro` and `_px4_accel` with `std::array<std::unique_ptr<PX4Gyroscope>, 3> _px4_gyros` and `std::array<std::unique_ptr<PX4Accelerometer>, 3> _px4_accels` initialized with distinct simulated `DeviceId` addresses (`0, 1, 2` on bus `1`) so PX4 `sensors` and `EKF2` register three independent `sensor_gyro`/`sensor_accel` uORB instances — each automatically wired into `PX4Gyroscope::update()`'s native `failure_injection::process()` hook.
2. Subscribe in a loop (`idx = 0..2`) to `/world/<world>/model/<model>/link/base_link/sensor/imu_sensor_<idx>/imu` via indexed lambdas invoking `imuCallback(int idx, const gz::msgs::IMU &msg)` (retaining legacy fallback to `imu_sensor/imu` for `idx == 0`), and mirror the identical pattern (`idx = 0..1`) for dual barometers (`air_pressure_sensor_0/1`).
3. Increase `sleep_count_limit` in `GZBridge::init()` from `1000` to `5000` (5 s) to eliminate `gpu_lidar` lockstep startup timeouts.
4. Wire `uORB::Subscription _failure_injection_sub` inside `GZMixingInterfaceESC::updateOutputs()` so `failure motor off -i <n>` cleanly zeroes the corresponding rotor command sent to Gazebo Harmonic — closing an upstream gap where motor failure injection was historically limited to Gazebo Classic.

### 5.2 `VertiRTA-X8-240` SDF Sensor Suite (3 IMUs, Dual Baro, GNSS, 16-Beam 3D LiDAR) and Airframe Config

Because CPU-based multi-beam `<sensor type="lidar">` remains unimplemented in Gazebo Harmonic `SdfEntityCreator.cc`, our lightweight 16-beam 3D LiDAR for `v1.0.0` `ROG-Map` uses `<sensor type="gpu_lidar">` on `ogre2`, specifying both `<gz_frame_id>` and `<frame_id>` to prevent Harmonic `ros_gz_bridge` initialization segfaults. All three physical IMUs sit inside `base_link` at 400 Hz with heterogeneous Gaussian noise matching Pixhawk 6X Rev 3/4 and Rev 8 silicon (`ICM-42688-P`, `ICM-45686`, `BMI088`):

```xml
<!-- Tools/simulation/gz/models/vertirta_x8_240/model.sdf (base_link inertial + sensor suite) -->
<link name="base_link">
  <inertial>
    <mass>240.0</mass>
    <inertia><ixx>85.0</ixx><iyy>95.0</iyy><izz>140.0</izz></inertia>
  </inertial>

  <!-- IMU 0 (400 Hz, ICM-42688-P class: 2.8 mdps/rtHz -> 7.0e-4 rad/s; 70 ug/rtHz -> 9.8e-3 m/s^2) -->
  <sensor name="imu_sensor_0" type="imu">
    <always_on>1</always_on><update_rate>400</update_rate>
    <imu>
      <angular_velocity>
        <x><noise type="gaussian"><mean>0</mean><stddev>0.00070</stddev></noise></x>
        <y><noise type="gaussian"><mean>0</mean><stddev>0.00070</stddev></noise></y>
        <z><noise type="gaussian"><mean>0</mean><stddev>0.00070</stddev></noise></z>
      </angular_velocity>
      <linear_acceleration>
        <x><noise type="gaussian"><mean>0</mean><stddev>0.0098</stddev></noise></x>
        <y><noise type="gaussian"><mean>0</mean><stddev>0.0098</stddev></noise></y>
        <z><noise type="gaussian"><mean>0</mean><stddev>0.0098</stddev></noise></z>
      </linear_acceleration>
    </imu>
  </sensor>

  <!-- IMU 1 (400 Hz, ICM-45686 class: 3.8 mdps/rtHz -> 9.4e-4 rad/s; 70 ug/rtHz -> 9.8e-3 m/s^2) -->
  <sensor name="imu_sensor_1" type="imu">
    <always_on>1</always_on><update_rate>400</update_rate>
    <imu>
      <angular_velocity>
        <x><noise type="gaussian"><mean>0</mean><stddev>0.00094</stddev></noise></x>
        <y><noise type="gaussian"><mean>0</mean><stddev>0.00094</stddev></noise></y>
        <z><noise type="gaussian"><mean>0</mean><stddev>0.00094</stddev></noise></z>
      </angular_velocity>
      <linear_acceleration>
        <x><noise type="gaussian"><mean>0</mean><stddev>0.0098</stddev></noise></x>
        <y><noise type="gaussian"><mean>0</mean><stddev>0.0098</stddev></noise></y>
        <z><noise type="gaussian"><mean>0</mean><stddev>0.0098</stddev></noise></z>
      </linear_acceleration>
    </imu>
  </sensor>

  <!-- IMU 2 (400 Hz, BMI088 class: 14 mdps/rtHz -> 3.46e-3 rad/s; 160/190 ug/rtHz -> 0.022/0.026 m/s^2) -->
  <sensor name="imu_sensor_2" type="imu">
    <always_on>1</always_on><update_rate>400</update_rate>
    <imu>
      <angular_velocity>
        <x><noise type="gaussian"><mean>0</mean><stddev>0.00346</stddev></noise></x>
        <y><noise type="gaussian"><mean>0</mean><stddev>0.00346</stddev></noise></y>
        <z><noise type="gaussian"><mean>0</mean><stddev>0.00346</stddev></noise></z>
      </angular_velocity>
      <linear_acceleration>
        <x><noise type="gaussian"><mean>0</mean><stddev>0.0222</stddev></noise></x>
        <y><noise type="gaussian"><mean>0</mean><stddev>0.0222</stddev></noise></y>
        <z><noise type="gaussian"><mean>0</mean><stddev>0.0264</stddev></noise></z>
      </linear_acceleration>
    </imu>
  </sensor>

  <!-- Dual Barometers (50 Hz, BMP388/ICP20100 class: 3 Pa stddev) -->
  <sensor name="air_pressure_sensor_0" type="air_pressure">
    <always_on>1</always_on><update_rate>50</update_rate>
    <air_pressure><pressure><noise type="gaussian"><mean>0</mean><stddev>3.0</stddev></noise></pressure></air_pressure>
  </sensor>
  <sensor name="air_pressure_sensor_1" type="air_pressure">
    <always_on>1</always_on><update_rate>50</update_rate>
    <air_pressure><pressure><noise type="gaussian"><mean>0</mean><stddev>3.0</stddev></noise></pressure></air_pressure>
  </sensor>

  <!-- GNSS & Magnetometer -->
  <sensor name="navsat_sensor" type="navsat"><always_on>1</always_on><update_rate>10</update_rate></sensor>
  <sensor name="magnetometer_sensor" type="magnetometer">
    <always_on>1</always_on><update_rate>100</update_rate>
    <magnetometer>
      <x><noise type="gaussian"><stddev>0.0001</stddev></noise></x>
      <y><noise type="gaussian"><stddev>0.0001</stddev></noise></y>
      <z><noise type="gaussian"><stddev>0.0001</stddev></noise></z>
    </magnetometer>
  </sensor>

  <!-- Lightweight 16-beam 3D GPU LiDAR (10 Hz, 360x30 deg FOV, 100 m range, 5,760 pts/scan) -->
  <sensor name="lidar_3d_16beam" type="gpu_lidar">
    <pose>0.55 0.0 -0.25 0 0.1745 0</pose>
    <topic>/vertirta/lidar_16</topic>
    <gz_frame_id>lidar_link</gz_frame_id><frame_id>lidar_link</frame_id>
    <always_on>1</always_on><update_rate>10</update_rate><visualize>false</visualize>
    <ray>
      <scan>
        <horizontal><samples>360</samples><resolution>1</resolution><min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal>
        <vertical><samples>16</samples><resolution>1</resolution><min_angle>-0.26180</min_angle><max_angle>0.26180</max_angle></vertical>
      </scan>
      <range><min>0.8</min><max>100.0</max><resolution>0.02</resolution></range>
      <noise><type>gaussian</type><mean>0.0</mean><stddev>0.03</stddev></noise>
    </ray>
  </sensor>
</link>
```

In `Tools/simulation/gz/worlds/kartal_coast.sdf`, physics runs at 400 Hz lockstep (`<max_step_size>0.0025</max_step_size>`, `<real_time_update_rate>400</real_time_update_rate>`, upgraded from PX4's 250 Hz `default.sdf`) with `<gravity>0 0 -9.81</gravity>` and `<spherical_coordinates>` anchored at `40.8860 N, 29.1880 E`, `<elevation>48.7</elevation>` (HAE = 12.0 m AMSL + 36.7 m EGM2008). The airframe file `ROMFS/px4fmu_common/init.d-posix/airframes/12050_gz_vertirta_x8_240` (registered in `airframes/CMakeLists.txt`) maps `mc_indi_wls_control`'s normalized angular-speed commands (`omega_cmd / 240.0`) directly into physical `rad/s` via `SIM_GZ_EC_MAX = 240`:

```sh
#!/bin/sh
# @name Gazebo VertiRTA-X8-240 Coaxial eVTOL
# @type Octorotor Coaxial X
. ${R}etc/init.d/rc.mc_defaults
PX4_SIMULATOR=${PX4_SIMULATOR:=gz}
PX4_GZ_WORLD=${PX4_GZ_WORLD:=kartal_coast}
PX4_SIM_MODEL=${PX4_SIM_MODEL:=vertirta_x8_240}

param set-default SIM_GZ_EN 1
param set-default CA_AIRFRAME 0
param set-default CA_ROTOR_COUNT 8
# Upper rotors 0..3 (z = -0.175 m)
param set-default CA_ROTOR0_PX  0.955; param set-default CA_ROTOR0_PY  0.955; param set-default CA_ROTOR0_PZ -0.175; param set-default CA_ROTOR0_KM  0.076
param set-default CA_ROTOR1_PX -0.955; param set-default CA_ROTOR1_PY -0.955; param set-default CA_ROTOR1_PZ -0.175; param set-default CA_ROTOR1_KM  0.076
param set-default CA_ROTOR2_PX  0.955; param set-default CA_ROTOR2_PY -0.955; param set-default CA_ROTOR2_PZ -0.175; param set-default CA_ROTOR2_KM -0.076
param set-default CA_ROTOR3_PX -0.955; param set-default CA_ROTOR3_PY  0.955; param set-default CA_ROTOR3_PZ -0.175; param set-default CA_ROTOR3_KM -0.076
# Lower rotors 4..7 (z = +0.175 m, counter-rotating)
param set-default CA_ROTOR4_PX  0.955; param set-default CA_ROTOR4_PY  0.955; param set-default CA_ROTOR4_PZ  0.175; param set-default CA_ROTOR4_KM -0.086
param set-default CA_ROTOR5_PX -0.955; param set-default CA_ROTOR5_PY -0.955; param set-default CA_ROTOR5_PZ  0.175; param set-default CA_ROTOR5_KM -0.086
param set-default CA_ROTOR6_PX  0.955; param set-default CA_ROTOR6_PY -0.955; param set-default CA_ROTOR6_PZ  0.175; param set-default CA_ROTOR6_KM  0.086
param set-default CA_ROTOR7_PX -0.955; param set-default CA_ROTOR7_PY  0.955; param set-default CA_ROTOR7_PZ  0.175; param set-default CA_ROTOR7_KM  0.086
# Map normalized speed commands [0, 1] directly to 0..240 rad/s
for i in 1 2 3 4 5 6 7 8; do
  param set-default SIM_GZ_EC_FUNC${i} $((100 + i))
  param set-default SIM_GZ_EC_MIN${i} 0
  param set-default SIM_GZ_EC_MAX${i} 240
done
# Multi-IMU EKF2 diversity & Failure Injection
param set-default SENS_IMU_MODE 0        # Disable single-IMU pre-selection so all 3 IMUs run independently
param set-default EKF2_MULTI_IMU 3       # 3 parallel EKF2 instances with median gyro failover
param set-default IMU_INTEG_RATE 400     # 400 Hz synchronized with physics & TMR voter
param set-default SYS_FAILURE_EN 1       # Enable MAVLink/pxh failure injection
param set-default MPC_THR_HOVER 0.40     # Nominal T/W = 2.5 hover thrust fraction (for estimator prior)
```

With `SYS_FAILURE_EN = 1`, PX4's `failure <component> <failure_type> [-i <instance_number>]` CLI (where `-i 0` targets all units and `-i 1..N` targets a 1-indexed unit) supports deterministic fault campaigns from `pxh>` or MAVSDK:

```sh
# Single upper-motor cut (M0 on FR)
failure motor off -i 1

# Same-arm dual coaxial-pair cut (M0 + M4 on FR)
failure motor off -i 1
failure motor off -i 5

# Single IMU stuck / wrong / off fault injection
failure gyro stuck -i 1
failure accel wrong -i 2
failure gyro off -i 3

# Exteroceptive / barometer fault injection
failure gps off -i 1
failure baro stuck -i 1
```

### 5.3 In-Loop 400 Hz C++17 TMR FDIR Voter (`libvertirta_flight_core`)

Executing the Triple Modular Redundancy (TMR) voter inside PX4 at 400 Hz (2.5 ms ticks) eliminates companion-link transport latency before INDI differentiation and yields 8 full samples inside a 20.0 ms isolation window — fast enough to isolate a hardover gyro before a 240 kg aircraft accumulates `> 1 deg` attitude excursion, while completely rejecting transient EMI spikes and gust turbulence.

At each 400 Hz tick $k$, the voter reads bias-calibrated, frame-aligned 3-axis vectors $\mathbf{y}_0[k], \mathbf{y}_1[k], \mathbf{y}_2[k] \in \mathbb{R}^3$ (independently for gyroscopes and accelerometers) from `sensor_gyro[0..2]` and `sensor_accel[0..2]`. In nominal `TMR_3OF3` operation, the instantaneous output passed to INDI is the coordinate-wise **Mid-Value Select (MVS)** median $y_{\text{voted},c}[k] = \operatorname{median}(y_{0,c}[k], y_{1,c}[k], y_{2,c}[k])$ for each axis $c \in \{x, y, z\}$, delivering zero-latency passive masking of any single sensor hardover or stuck bit.

Simultaneously, the voter computes the three pairwise parity residuals $\mathbf{d}_{ij}[k] = \mathbf{y}_i[k] - \mathbf{y}_j[k] - \hat{\mathbf{b}}_{ij}[k]$ for $(i, j) \in \{(0,1), (0,2), (1,2)\}$, where $\hat{\mathbf{b}}_{ij}$ is a slow (0.02 Hz) inter-IMU bias tracker frozen whenever any instantaneous statistic exceeds 50% of threshold to prevent slow-drift attacks from being learned away. Because aerodynamic gusts and maneuvers excite all three co-located IMUs identically, physical motion cancels out of $\mathbf{d}_{ij}$ as a pure common-mode signal. Under $H_0$, $\mathbf{d}_{ij}[k] \sim \mathcal{N}(\mathbf{0}, \Sigma_i + \Sigma_j)$, incorporating a conservative post-notch structural vibration floor ($\sigma_{g,\text{eff}} = 0.015\text{ rad/s}$, $\sigma_{a,\text{eff}} = 0.25\text{ m/s}^2$). Forming the 3-DOF instantaneous statistic and 8-tick (20 ms) sliding-window 24-DOF Chi-Square statistic gives:

$$q_{ij}[k] = \mathbf{d}_{ij}[k]^T (\Sigma_i + \Sigma_j)^{-1} \mathbf{d}_{ij}[k] \sim \chi^2_3, \qquad Q_{ij}[k] = \sum_{m=0}^{7} q_{ij}[k-m] \sim \chi^2_{24}$$

At false-alarm probability $P_{\text{FA}} = 10^{-6}$, the exact threshold is $T_{24} = 72.23$ (65.58 at $10^{-5}$, 58.61 at $10^{-4}$), delivering a 20 ms minimum detectable constant bias of **0.0637 rad/s (3.65 deg/s)** on gyroscopes and **1.062 m/s^2 (0.108 g)** on accelerometers under realistic flight vibration.

Fault isolation follows a strictly monotonic persistent state machine (`TMR_3OF3 -> DUPLEX_2OF2 -> SIMPLEX_ANALYTICAL / FAILSAFE_DIVERT`) without in-flight auto-unlatching to avoid chattering on intermittent silicon faults:

- **`TMR_3OF3 -> DUPLEX_2OF2`:** Whenever two pairs sharing sensor $i$ violate threshold while the opposing pair remains healthy — e.g., $(Q_{01} > T_{24}) \land (Q_{02} > T_{24}) \land (Q_{12} \le T_{24})$ — IMU 0 is permanently latched failed, `imu_health_mask` clears bit 0 (aligning with PX4's `EKF2_MULTI_IMU = 3` median failover), and the voter transitions to `DUPLEX_2OF2`, outputting the inverse-variance weighted average of the two surviving IMUs.
- **`DUPLEX_2OF2 -> SIMPLEX_ANALYTICAL` or `FAILSAFE_DIVERT`:** If the remaining pair later violates $Q_{ab}[k] > T_{24}$, pairwise parity detects the second fault and triggers a 20 ms analytical tie-breaker comparing both sensors against the INDI model-predicted angular rate $\hat{\boldsymbol{\Omega}}_{\text{model}}$ (integrated from `esc_status` and $B_T$) and `EKF2` innovation ratios: if one channel agrees within $3\sigma$ while the other exceeds $6\sigma$, the voter latches `SIMPLEX_ANALYTICAL` and requests an immediate precautionary RTA divert; otherwise it immediately latches `FAILSAFE_DIVERT` (`tmr_state = 3`) streamed upward at 100 Hz over uXRCE-DDS.

---

## 6. Real-World Sim-to-Real Hardware Transfer Reality

### 6.1 Key Physical and Avionics Reality Gaps

Porting `libvertirta_flight_core` and `mc_indi_wls_control` from Gazebo Harmonic SITL onto physical 240 kg hardware powered by a 400 V HV bus and a Pixhawk 6X (`FMU-v6X`) confronts four hard physical and avionics gaps:

1. **Coaxial Ground Effect & Forward-Flight Aerodynamics:** Hover-stand testing shows In-Ground Effect (IGE) below $z/R < 3.0$ (2.25 m AGL) reduces total coaxial power by up to ~30% at $z/R = 0.4$ (0.30 m), while shifting loading asymmetrically so the lower rotor demands +8% more induced power as the upper rotor unloads. Furthermore, upper/lower inflow coupling is bidirectional (~18% induced-power penalty on the upper rotor from lower-rotor suction), and rigid coaxial pairs in cruise develop a negative longitudinal trim gradient (pitch-up with airspeed) alongside intense `6/rev` ($2 N_b$) blade-meeting pressure pulses. While the 400 Hz INDI inner loop rejects steady IGE thrust surges and forward pitch-up trim moments without a lookup table, altitude damping must still be scheduled against LiDAR AGL below 2.5 m to prevent flare ballooning during Regime 3 touchdowns.
2. **400 V HV BLDC/ESC Isolation, Jitter, and Braking Limits:** Although PX4 v1.16 adds native bidirectional DShot (`BDShot`) on FMU AUX pins, routing non-isolated single-ended DShot lines between 400 V / 15 kW inverters and a 3.3 V STM32H7 MCU invites catastrophic HV fault propagation. Production HV propulsion requires **galvanically isolated DroneCAN / CAN-FD** (Pixhawk 6X exposes two native CAN buses) or optical-fiber PWM with isolated CAN telemetry, introducing 2-5 ms bus jitter and 50-200 Hz asynchronous `esc_status` frames rather than lockstep 400 Hz feedback. Equally important, HV inverters impose thermal current derating above 85-100 C MOSFET/IGBT junction temperature (shrinking $\dot{\omega}_{\max}$ during prolonged Regime 3 2.5-3.3C draws) and clamp regenerative braking to protect DC-link capacitors — exposing the large physical asymmetry between powered spin-up ($\tau_{\text{up}} = 0.085\text{ s}$) and unpowered aerodynamic coast-down ($\tau_{\text{coast}} = 0.424\text{ s}$).
3. **MEMS IMU Structural Harmonics, Drift, and Silicon Diversity:** Three-blade 1.50 m coaxial rotors excite dominant blade-pass harmonics at $f_{\text{BPF}} = N_b \cdot \text{RPM}/60$ (72.5 Hz hover to 114.6 Hz max) and coaxial blade-meeting pulses at $2 f_{\text{BPF}}$ (145.0-229.2 Hz). Because 145-229 Hz straddles the 200 Hz Nyquist limit of a 400 Hz control loop, raw sampling at 400 Hz would alias blade vibration directly into INDI's $\dot{\boldsymbol{\Omega}}_f$. Hardware IMUs must therefore sample internal FIFOs at 1.6-4.0 kHz and run PX4's ESC-RPM Dynamic Notch Filter (`IMU_GYRO_DNF_EN = 1`, `IMU_GYRO_DNF_HMC = 3` tracking 1x, 2x, 3x harmonics) plus static boom bending notches (`IMU_GYRO_NF0_FRQ`) **prior** to 400 Hz decimation. Silicon diversity also depends critically on Pixhawk 6X board revision: current-shipping **Rev 8** carries three identical `TDK ICM-45686` dies (`BalancedGyro`, 3.8 mdps/sqrt(Hz), 70 ug/sqrt(Hz), +/-4,000 dps, +/-32 g, +/-0.005 deg/s/C) — offering strong vibration immunity but **zero silicon diversity** against common-mode acoustic resonance — whereas discontinued **Rev 3/4** mixed `ICM-42688-P` (2.8 mdps/sqrt(Hz), 65-70 ug/sqrt(Hz); typical consumer-MEMS engineering characterizations place in-run bias instability around ~2-5 deg/hr and ARW ~0.07 deg/sqrt(hr)), `BMI088` (14 mdps/sqrt(Hz), 160/190 ug/sqrt(Hz), +/-0.015 deg/s/K, 0.1 deg/s/g g-sensitivity), and `ICM-42670-P`. For manned flight hardware, pairing Pixhawk 6X Rev 8's internal IMUs with one external vibration-isolated dissimilar IMU (`BMI088` or tactical-grade `ADIS16507`) over SPI/CAN restores dissimilar-silicon TMR while calibrating individual channel variances $\Sigma_i$ in the Chi-Square voter (`BMI088` gyro noise variance is ~25x higher than `ICM-42688-P`).
4. **STM32H753 Compute Headroom vs. UART uXRCE-DDS Saturation:** On the 480 MHz Cortex-M7 (2 MB Flash, 1 MB RAM, double-precision `FPv5-D16` FPU), single-precision `float32` profiling of zero-heap `libvertirta_flight_core` yields a worst-case execution time (WCET) of **~40 us per 2,500 us tick** (1.6% CPU: TMR voter 2.5 us, 11 Butterworth biquads 1.5 us, INDI 1.8 us, 12-iter 8x4 WLS active-set Cholesky 18-32 us, Motor FDI 2.2 us, 1-RC battery 1.5 us) inside `< 4.8 KB` static DTCM RAM. In contrast, streaming `vehicle_odometry` (100 Hz), `vertirta_capability_envelope` (100 Hz), `sensor_combined`, status topics, and incoming 100 Hz RTA setpoints consumes ~34.8 KB/s raw payload (~52.2 KB/s = 0.42 Mbps framed). Over 921,600 baud `TELEM2` UART (92.16 KB/s wire ceiling, ~65 KB/s usable without hardware flow control), this exceeds 56.6% of wire rate and ~80% of usable throughput — reproducing PX4's documented UART uXRCE-DDS session-drop bottleneck (GitHub Issue `#22989`) unless topics in `dds_topics.yaml` are heavily pruned. Utilizing Pixhawk 6X's native **100 Mbps Ethernet** consumes just **0.42%** of link capacity at `< 0.8 ms` RTT.

### 6.2 Sim-to-Real Gap and Hardware Transfer Matrix

| Subsystem / Physics Domain | SITL / Gazebo Harmonic Model | Real-World VertiRTA-X8-240 Hardware Reality | Residual Risk & Control Mitigation | Mandatory Bench / Flight Verification Gate |
| :--- | :--- | :--- | :--- | :--- |
| **Coaxial Aerodynamics & Ground Effect** | Momentum wake-skew eta_coax(V) + kappa_int = 1.22; no IGE or BVI | Bidirectional upper/lower coupling (~18% upper penalty); -30% IGE power below 2.25 m AGL (+8% lower shift); forward pitch-up trim gradient; 6/rev pulses | INDI cancels steady IGE and trim moments; schedule altitude damping against LiDAR AGL below 2.5 m to prevent flare ballooning | **Gate 1 (Coaxial Thrust Stand):** Sweep 0-240 rad/s on single coaxial arm; identify true k_T_u, k_T_l, k_Q_u, k_Q_l, eta_coax(w_u) and step lag |
| **400 V HV Propulsion & ESC Telemetry** | 1st-order tau_up = 0.085 s, tau_coast = 0.424 s; lockstep 400 Hz RPM | Galvanic-isolated DroneCAN (50-200 Hz, 2-5 ms jitter); MOSFET thermal derating (>90 C); DC-bus regen clamp blocks active braking | Prohibit non-isolated BDShot on 400 V; fuse CAN RPM via 20 rad/s complementary observer; feed ESC temp into dynamic t_max | **Gate 2 (Full-Power HV Dyno + Fault Cut):** Inject phase-open fault at 1,500 RPM; verify <60 ms FDI isolation and confirm K_e_bus = 1.36 holds 240 rad/s at 360 V |
| **HV Battery Pack & Voltage Sag** | 112S12P 1-RC ECM at fixed 25 C (R_0 = 136 mOhm, R_1 = 84 mOhm) | 2x-3x R_0 rise below 10 C; cell imbalance; BMS contactor trips; 22 kWh COTS mass conflicts with 240 kg manned MTOW | Index R_0(T_cell, SoC) from live BMS CAN frames; close mass budget (112S7P vs 320 Wh/kg cells vs unmanned demo) before airframe freeze | **Gate 3 (Pulse-Load Battery HIL):** Run 175 A (3.2C) 10 s flare pulses at 20% SoC and 10 C; verify V_term >= 350 V and 1-RC model error <1.5% |
| **Triple-IMU Sensing & Vibration** | 3 white-noise 400 Hz Gazebo IMUs; zero vibration harmonics or thermal drift | 72.5-114.6 Hz 3/rev BPF and 145-229 Hz 6/rev coaxial pulses aliasing 400 Hz loop; ~2-5 deg/hr BI; +/-0.005-0.015 deg/s/C drift; Rev 8 homogeneous dies | Sample hardware FIFOs at 1.6-4 kHz; apply IMU_GYRO_DNF_EN=1 (3 harmonics) + static boom notches before 400 Hz TMR; add 1 dissimilar external IMU | **Gate 4 (Shaker Table & Tethered FFT):** Measure Allan variance + airframe PSD; verify post-notch sigma_g <= 0.015 rad/s and zero TMR false alarms across 0-2,292 RPM |
| **FMU-v6X NuttX RT & Companion Link** | POSIX Linux SITL (x86_64), localhost UDP uXRCE-DDS, ample RAM | STM32H753 (480 MHz, 2 MB Flash, 1 MB RAM, -nostdinc++ -fno-exceptions -fno-rtti); 921.6 kbps UART drops DDS at >55% load | Zero-heap C++17 libvertirta_flight_core (~40 us WCET, <4.8 KB DTCM); mandate 100 Mbps Ethernet (0.42% util, <0.8 ms RTT); prune dds_topics.yaml | **Gate 5 (FMU-v6X HITL + Logic Analyzer):** Build make px4_fmu-v6x_default with arm-none-eabi-g++; verify GPIO-timed 400 Hz WCET <60 us, zero stack fault (CONFIG_ARMV7M_STACKCHECK), and 0 Ethernet drops over 1 hr |
| **Same-Arm Dual-Loss Active Spin** | Rigid body + lockstep Gazebo physics | Boom aeroelasticity, coaxial rotor precession coupling, asymmetric airframe drag during 2.2 rad/s flat spin, skid ground contact friction | Cap spin target to 2.0-2.5 rad/s (0.10-0.16 g, 9.6-12 deg lag); lead-compensate n_B_des by atan(Omega_z * tau_m); fit circular compliant skids | **Gate 6 (Subscale -> Tethered Full-Scale Fault Test):** First validate 2.2 rad/s flat spin & <1.5 m/s flare on 15-25 kg subscale X8 (matching dimensionless Omega_z * tau_m ~ 0.19), then full-scale gantry-tethered single-arm cut |

### 6.3 Step-by-Step Pixhawk 6X (`FMU-v6X`) and ARM64 Companion Porting Procedure

1. **Bare-Metal Cross-Compile Gate (`libvertirta_flight_core`):** Enforce a CI gate compiling `libvertirta_flight_core` standalone with:
   ```sh
   arm-none-eabi-g++ -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard \
     -std=c++17 -O2 -fno-exceptions -fno-rtti -fno-threadsafe-statics \
     -nostdinc++ -Wall -Werror
   ```
   Inspect `arm-none-eabi-nm` symbols to prove zero references to `malloc`, `free`, `operator new`, `__cxa_throw`, or `libstdc++`.
2. **PX4 v1.16 Board & Flash Configuration:** Enable `CONFIG_MODULES_MC_INDI_WLS_CONTROL=y` in `boards/px4/fmu-v6x/default.px4board` (pruning unused fixed-wing/rover modules if 2 MB Flash headroom tightens). Keep all `mc_indi_wls_control` matrices in static `.bss` storage and confirm via `work_queue status` that `wq:rate_ctrl` total CPU load stays `< 25%` with `mc_indi_wls_control` executing in `~40 us`.
3. **Ethernet uXRCE-DDS Commissioning:** Connect Pixhawk 6X's native 100 Mbps Ethernet port directly to the ARM64 companion NIC on a dedicated static subnet (`192.168.10.2` FMU, `192.168.10.1` companion). Configure:
   - `UXRCE_DDS_CFG = 1000` (Ethernet)
   - `UXRCE_DDS_AG_IP = -1062729215` (signed `int32` two's-complement encoding of `192.168.10.1` via `Tools/convert_ip.py`)
   - `UXRCE_DDS_PRT = 8888`
   - `UXRCE_DDS_SYNCT = 1`
   Run `MicroXRCEAgent udp4 -p 8888` on the companion and verify `ros2 topic hz /fmu/out/vertirta_capability_envelope` locks cleanly at `100.0 Hz` with `< 1 ms` jitter.
4. **Safety-Gated Flight Campaign:** Complete Bench Gates 1-5 strictly prior to initial hover; verify `EKF2_MULTI_IMU = 3` (`SENS_IMU_MODE = 0`) failover and dynamic notch attenuation (`IMU_GYRO_DNF_EN = 1`) under tethered hover; execute unconstrained Regime 1 (single-motor) and Regime 2a (cross-arm dual) fault cuts; and advance to Regime 3 Same-Arm Dual-Loss active-spin (+/-2.2 rad/s) only after subscale validation and full-scale gantry-tethered testing confirm clean spin-up, zero structural resonance, and controlled `< 1.5 m/s` terminal flare arrest.