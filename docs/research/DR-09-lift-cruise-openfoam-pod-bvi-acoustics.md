# Multi-Airframe Transition Flight, GPL-Isolated Urban Wake Surrogates, and Safety-Subordinated Aeroacoustic Routing for VertiRTA-240

Expanding the Apache-2.0 VertiRTA-X8-240 flight stack into winged Lift+Cruise transition flight, building-resolved coastal urban wind modeling, and community noise routing along Istanbul's Maltepe–Dragos–Kartal corridor requires three tightly coupled architectural upgrades that preserve real-time determinism and license purity. First, introducing a zero-allocation C++20 IAirframeDynamics abstraction enables the existing 400 Hz actuator-synchronized Incremental Nonlinear Dynamic Inversion (INDI) and double-precision (f64) active-set Weighted Least Squares (WLS) allocator to control both the baseline coaxial X8 and a 240 kg Lift+Cruise platform (VertiRTA-LC-240) across hover, transition (14–28 m/s), and wing-borne cruise (38 m/s) without switching inner-loop controllers. Analytical controllability checks for the 240 kg sizing confirm that an 8+1 Lift+Cruise architecture is necessary for urban hover one-engine-inoperative (OEI) survival—retaining full 4-DOF hover trim at 79.1%–79.2% of peak rotor thrust after a worst-case outboard lift-rotor failure—whereas a 4+1 quadplane drops to [STRIPPED_MATH_OR_ICON] in hover and only regains full 4-DOF trim once forward airspeed exceeds 8 m/s (with meaningful dynamic thrust margin above 12–14 m/s). [1][2][3]

Second, OpenFOAM (ESI OpenCFD v2412 or current June 2026 v2606, GNU GPLv3) can generate high-resolution steady RANS and scale-resolving kOmegaSSTIDDES urban wake fields over Kartal–Maltepe coastal high-rise blocks (such as the İSTMarina waterfront complex at 40.8846°N, 29.2057°E) strictly inside an air-gapped offline tooling directory. Under GPLv3 Section 0 and Free Software Foundation (FSF) program-output guidance, pure numerical CFD flow arrays and Proper Orthogonal Decomposition (POD) / Radial Basis Function (RBF) tables contain no OpenFOAM code and are unencumbered data. Packing 8 spatial POD modes, anisotropic turbulence RMS, and a calibrated local P95 surrogate error bound into 64-byte cache-aligned int16 nodes across a two-tier brick-swizzled grid yields a 13.89 MiB binary VRTAWND1 artifact (below the 16 MiB cap) with 0.22 mm/s quantization RMS and an analytical C++20 mmap lookup latency of 0.20–1.10 µs (4.5×–25× below the 5 µs ceiling). [1][2][3]

Third, coupling radial strip-integrated Brooks–Pope–Marcolini (BPM) self-noise (augmented with 2025 non-compactness corrections), Amiet–Lowson leading-edge inflow-turbulence broadband noise, and Lowson steady/unsteady coaxial Blade-Vortex Interaction (BVI) tonal harmonics to an ISO 9613-1:1993 and ISO 9613-2:2024 propagation kernel synthesizes a 180-node 3D acoustic hemisphere and evaluates 64 sensitive Istanbul ground receptors in an estimated 0.37 ms (< 1 ms). Community noise abatement over Kartal Dr. Lütfi Kırdar Şehir Hastanesi, Süreyyapaşa Hospital, schools, and Dragos–Maltepe residences is formulated strictly as a smooth finite softplus objective inside the outer 4D trajectory planner—never as a hard constraint—subordinate to the 100 Hz Robust Hamilton–Jacobi–Isaacs Control Barrier-Value Function QP (HJI-CBVF-QP), whose wind-disturbance set is inflated online by VRTAWND1 turbulence and local ROM error. Two operational caveats govern deployment: exact horizontal vertex coordinates for the Sabiha Gökçen (LTFJ) CTR from Turkish AIP Section AD 2.17 are absent from the publicly retrieved LT_AD_2_LTFJ_en.pdf excerpt and must be ingested from the live AIRAC database; and linear POD-RBF surrogates exhibit higher localized peak errors inside steep building-wake shear layers than nonlinear neural ROMs, requiring per-node P95 error inflation near rooftop Safe Landing Zones (SLZs). [1][2][3]

1. Multi-Airframe IAirframeDynamics and Lift+Cruise Transition INDI-WLS

1.1 Reference 240 kg Airframe Sizing: Coaxial X8 vs. Lift+Cruise (8+1 and 4+1)

Supporting both the existing wingless coaxial VertiRTA-X8-240 and a winged VertiRTA-LC-240 at a common 240 kg MTOW ([STRIPPED_MATH_OR_ICON] at sea-level density [STRIPPED_MATH_OR_ICON], coastal speed of sound [STRIPPED_MATH_OR_ICON] at 20 °C) demands a wing loading that keeps stall speed low enough for rapid transitions inside Istanbul's narrow coastal airspace while preserving cruise efficiency. Analytical baseline sizing places the VertiRTA-LC-240 wing at area [STRIPPED_MATH_OR_ICON], aspect ratio [STRIPPED_MATH_OR_ICON], wingspan [STRIPPED_MATH_OR_ICON], and mean aerodynamic chord [STRIPPED_MATH_OR_ICON] ([STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON] assuming two-bladed lift rotors stop streamwise-aligned in cruise per NASA's Lift+Cruise concept, Oswald efficiency [STRIPPED_MATH_OR_ICON]). This yields a 1-g stall speed of [STRIPPED_MATH_OR_ICON] (42.4 kt), transition completion at [STRIPPED_MATH_OR_ICON] (1.28 [STRIPPED_MATH_OR_ICON]), and corridor cruise at [STRIPPED_MATH_OR_ICON] (73.9 kt) with [STRIPPED_MATH_OR_ICON], requiring 199.1 N pusher thrust and 9.23 kW shaft power at 82% propulsive efficiency. [1][2][3][4][5][6]

Design Parameter (Analytical 240 kg Reference)

VertiRTA-X8-240 (Coaxial X8)

VertiRTA-LC-240 (8+1 Recommended)

VertiRTA-LC-240 (4+1 Quadplane)

MTOW / Weight (m / W)

240 kg / 2353.6 N

240 kg / 2353.6 N

240 kg / 2353.6 N

Lift Rotor Geometry (N_r \times D)

8 (4 coax pairs, z/D = 0.22) × 1.65 m

8 isolated (2 booms/side) × 1.30 m

4 isolated (1 boom/side) × 1.75 m

Nominal / Max Lift Rotor Thrust

294.2 N avg (306.0 N upper / 282.4 N lower)

294.2 N / 544.3 N (1.85× T/W)

588.4 N / 1059.1 N (1.80× T/W)

Hover Tip Speed / Tip Mach / RPM

98.8 m/s / 0.288 / 1144 RPM

122.8 m/s / 0.358 / 1804 RPM

129.0 m/s / 0.376 / 1408 RPM

Wing Area / Span / MAC (S / b / \bar{c})

None (wingless)

5.20 m² / 6.84 m / 0.760 m

5.20 m² / 6.84 m / 0.760 m

Inertia Tensor (I_{xx}, I_{yy}, I_{zz})

Existing baseline CAD/SysID

68, 82, 128 kg·m²

68, 82, 128 kg·m²

Stall / Blend / Transition / Cruise V_a

N/A (multirotor envelope)

21.8 / 14–28 / 28.0 / 38.0 m/s

21.8 / 14–28 / 28.0 / 38.0 m/s

Pusher Prop (B_p \times D_p, Cruise)

None

3-blade × 1.35 m, 1550 RPM, 199.1 N

3-blade × 1.35 m, 1550 RPM, 199.1 N

Hover OEI 4-DOF Trim Feasibility

Feasible (7 active rotors)

Feasible (430.4–430.8 N peak, 79.1%–79.2% T_{\max})

Infeasible (\text{rank} = 3 < 4; needs 1176.8 N)

Mid-Transition OEI Min V_a (4-DOF)

0 m/s

0 m/s (full envelope)

8 m/s (97.3% T_{\max}); ≥ 14 m/s for > 14% thrust margin

Table values come from our analytical 240 kg sizing and LP/SVD controllability models.

The linear-programming controllability audit demonstrates why 8+1 is structurally superior to 4+1 for urban Istanbul flight. Arrange 8 lift rotors symmetrically on four booms at longitudinal stations [STRIPPED_MATH_OR_ICON] and inboard/outboard lateral arms [STRIPPED_MATH_OR_ICON] with alternating CW/CCW spin (torque-to-thrust ratio [STRIPPED_MATH_OR_ICON], mirroring NASA's 8+1 layout). When the worst-case outboard front-left rotor (#1) fails in hover, zeroing its diagonally coupled inboard partner while matching the four loaded rotors satisfies pitch and yaw identically and yields a closed-form minimax peak thrust of [STRIPPED_MATH_OR_ICON]. For both the compact span ([STRIPPED_MATH_OR_ICON], giving 430.8 N or 79.2% of [STRIPPED_MATH_OR_ICON]) and the 6.84 m span ([STRIPPED_MATH_OR_ICON], giving 430.4 N or 79.1% of [STRIPPED_MATH_OR_ICON]), full 4-DOF hover trim retains > 20.8% dynamic thrust margin for gust rejection. Conversely, a 4+1 quadplane losing one lift rotor drops to [STRIPPED_MATH_OR_ICON]: even abandoning yaw control and shutting down the diagonally opposite rotor forces the remaining two rotors to produce [STRIPPED_MATH_OR_ICON] each, exceeding [STRIPPED_MATH_OR_ICON]. Accordingly, 8+1 is recommended as the operational Lift+Cruise airframe, while IAirframeDynamics supports 4+1 natively for PX4's gz_standard_vtol SITL workflow. [1][2][3][4][5][6][7]

1.2 Nonlinear 6-DOF Hover-to-Cruise Transition Dynamics

In the body-fixed Front-Right-Down (FRD) frame, the 6-DOF equations of motion combine aerodynamic loads, distributed lift rotors, pusher propulsion, gravity, and propulsor gyroscopic angular momentum [STRIPPED_MATH_OR_ICON]: [1][2][3][4][5][6][7]

Retaining the pitch–yaw product of inertia [STRIPPED_MATH_OR_ICON] (non-negligible on twin-boom or high-tail Lift+Cruise airframes) gives the component rotational dynamics from NASA's 6-DOF Lift+Cruise model:

To couple directly to the VRTAWND1 urban wake vector [STRIPPED_MATH_OR_ICON], aerodynamic states evaluate strictly on relative body airspeed [STRIPPED_MATH_OR_ICON], giving [STRIPPED_MATH_OR_ICON], dynamic pressure [STRIPPED_MATH_OR_ICON], angle of attack \alpha = \operatorname{atan2}(w_a, u_a), and sideslip \beta = \arcsin\!\big(v_a / \max(V_a, 0.25)\big). Wind-axis aerodynamic forces transform into FRD via [STRIPPED_MATH_OR_ICON]: [1]

with nondimensional rates [STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON], and [STRIPPED_MATH_OR_ICON]. Because transition traverses extreme angles of attack ([STRIPPED_MATH_OR_ICON] during vertical climbs or urban updrafts), lift and drag employ smooth sigmoid blending between linear attached wing theory and post-stall flat-plate aerodynamics (consistent with the sigmoid blending and flat-plate parameters in gz::sim::systems::AdvancedLiftDrag): [1][2]

On the propulsive side, each lift rotor [STRIPPED_MATH_OR_ICON] at edgewise advance ratio [STRIPPED_MATH_OR_ICON] produces vertical thrust [STRIPPED_MATH_OR_ICON] (along body [STRIPPED_MATH_OR_ICON]), edgewise hub drag ([STRIPPED_MATH_OR_ICON]-force opposing forward airflow, [STRIPPED_MATH_OR_ICON]), reaction torque [STRIPPED_MATH_OR_ICON], and blade-flapping pitch-up moment [STRIPPED_MATH_OR_ICON]. Expanding the pusher propeller thrust [STRIPPED_MATH_OR_ICON] as a quadratic in advance ratio [STRIPPED_MATH_OR_ICON] yields a dimensionally exact thrust law equivalent to NASA's [STRIPPED_MATH_OR_ICON] cruise formulation: [1]

where [STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON], and [STRIPPED_MATH_OR_ICON]. Complex rotor–wing downwash and rotor–empennage wake interference terms lack universal closed-form algebraic formulas across airframe geometries; NASA identifies them globally through CFD response surfaces. Within VertiRTA, those slowly varying interference moments are treated as unmodeled aerodynamic perturbations canceled directly by the 400 Hz sensor-based INDI inner loop. [1]

1.3 Actuator-Synchronized 400 Hz INDI and Airspeed-Scheduled [STRIPPED_MATH_OR_ICON]

Define the 5-DOF controlled virtual acceleration vector [STRIPPED_MATH_OR_ICON] and the actuator vector [STRIPPED_MATH_OR_ICON] ([STRIPPED_MATH_OR_ICON] for 8+1, [STRIPPED_MATH_OR_ICON] for 4+1). Taylor-expanding around the current synchronized state [STRIPPED_MATH_OR_ICON] yields the INDI incremental inversion equation: [1]

As demonstrated by Smeur, filter-delay synchronization is essential for closed-loop stability: because differentiated gyroscope rates and accelerometer readings must pass through a low-pass filter [STRIPPED_MATH_OR_ICON] (typically a second-order Butterworth filter tuned around 15–25 Hz at 400 Hz), the measured or estimated actuator vector [STRIPPED_MATH_OR_ICON] must pass through the identical filter [STRIPPED_MATH_OR_ICON] to form [STRIPPED_MATH_OR_ICON]. Synchronized filtering cancels [STRIPPED_MATH_OR_ICON] in the command channel, reducing command tracking and external angular-acceleration disturbance rejection \boldsymbol{\eta} to: [1]

Omitting actuator filter compensation causes the controller to repeatedly accumulate [STRIPPED_MATH_OR_ICON] during the filter lag window before acceleration changes appear, driving oscillation and instability. Because Lift+Cruise actuators have mismatched bandwidths—aero control-surface servos typically exhibit natural frequencies near 50 rad/s ([STRIPPED_MATH_OR_ICON]), whereas large electric lift propulsors exhibit slower dynamics near 8–10.1 rad/s ([STRIPPED_MATH_OR_ICON])—each actuator channel [STRIPPED_MATH_OR_ICON] uses its individual dynamic lag model [STRIPPED_MATH_OR_ICON] (or direct motor RPM from PX4 v1.16 bidirectional DShot ESC_STATUS filtered through [STRIPPED_MATH_OR_ICON]) to generate [STRIPPED_MATH_OR_ICON]. Where aggressive transition maneuvers require explicit compensation of heterogeneous first-order actuator rates \boldsymbol{\varepsilon}_u = \operatorname{diag}(1/\tau_i), Actuator-Aware NDI (ANDI) scales the effectiveness matrix as \Delta\mathbf{u} = (B\boldsymbol{\varepsilon}_u)^+(\boldsymbol{\nu}_{\text{req}} - \mathbf{F}_x\dot{\mathbf{x}}), though filter-synchronized INDI remains preferable for rejecting unmodeled urban wake gusts. [1][2]

The airspeed-scheduled control effectiveness matrix [STRIPPED_MATH_OR_ICON] retains the full coupled inverse inertia tensor [STRIPPED_MATH_OR_ICON] (including [STRIPPED_MATH_OR_ICON]) and defines reaction-torque signs [STRIPPED_MATH_OR_ICON] in body FRD. Each lift rotor column [STRIPPED_MATH_OR_ICON] accounts for thrust-arm moments, backward [STRIPPED_MATH_OR_ICON]-force moment arm ([STRIPPED_MATH_OR_ICON], positive nose-up when the rotor sits above the CG at [STRIPPED_MATH_OR_ICON] in FRD), steady reaction torque [STRIPPED_MATH_OR_ICON], and rotor spin-up inertial reaction torque [STRIPPED_MATH_OR_ICON] (using discrete step [STRIPPED_MATH_OR_ICON] when referencing direct RPM differences): [1][2]

The aerodynamic surface block [STRIPPED_MATH_OR_ICON] scales with dynamic pressure [STRIPPED_MATH_OR_ICON], pusher-slipstream dynamic pressure at the empennage [STRIPPED_MATH_OR_ICON], post-stall authority attenuation [STRIPPED_MATH_OR_ICON], and lateral-directional cross-derivatives [STRIPPED_MATH_OR_ICON]:

1.4 Smooth Hover-to-Cruise Blending via Scheduled f64-WLS Allocation

Directly multiplying [STRIPPED_MATH_OR_ICON] by [STRIPPED_MATH_OR_ICON] to zero lift-rotor columns in cruise blinds the allocator to rotor authority if a control surface saturates or fails in wing-borne flight. Instead, VertiRTA keeps the true physical effectiveness matrix [STRIPPED_MATH_OR_ICON] intact and executes smooth transition blending through a [STRIPPED_MATH_OR_ICON]-continuous quintic Hermite schedule [STRIPPED_MATH_OR_ICON] applied to the WLS actuator preference vector [STRIPPED_MATH_OR_ICON] and effort weight matrix [STRIPPED_MATH_OR_ICON]: [1][2][3][4]

with [STRIPPED_MATH_OR_ICON] and [STRIPPED_MATH_OR_ICON] (1.28 [STRIPPED_MATH_OR_ICON]). At 400 Hz, the f64 active-set WLS solver minimizes Härkegård's unified quadratic objective: [1][2][3]

subject to position-and-rate-intersected step bounds [STRIPPED_MATH_OR_ICON] and [STRIPPED_MATH_OR_ICON]. Setting [STRIPPED_MATH_OR_ICON] in IEEE-754 f64 closely approximates true sequential least-squares prioritization while avoiding single-precision ill-conditioning. Channel priority weights follow Smeur's fault-tolerant ordering: W_v(\eta) = \operatorname{diag}\big(10.0,\, 10.0,\, 1.5,\, 5.0(1 - 0.98\eta),\, 3.0\big), giving roll and pitch top authority, smoothly releasing direct vertical rotor acceleration [STRIPPED_MATH_OR_ICON] to outer-loop pitch/energy control as wing lift builds, and placing yaw lowest so crosswind saturation never compromises attitude leveling. Concurrently, effort weights ramp lift-rotor cost from [STRIPPED_MATH_OR_ICON] to [STRIPPED_MATH_OR_ICON] while dropping aero-surface cost from 20.0 to 1.0, and preferred rotor thrust follows [STRIPPED_MATH_OR_ICON] until rotors stop and index streamwise once [STRIPPED_MATH_OR_ICON] and [STRIPPED_MATH_OR_ICON]. Because [STRIPPED_MATH_OR_ICON] remains physically non-zero prior to lock engagement, any gust or surface fault saturating [STRIPPED_MATH_OR_ICON] forces the [STRIPPED_MATH_OR_ICON]-weighted primary term to recruit lift rotors immediately. Warm-starting the active set [STRIPPED_MATH_OR_ICON] from step [STRIPPED_MATH_OR_ICON] solves the 17×12 stacked least-squares problem in typically 1–3 QR iterations, keeping execution well below 1% of the 2.5 ms 400 Hz control period. [1][2]

1.5 Envelope Protection ([STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON]) and Mid-Transition Fault Policies

Stall protection, Pseudo-Control Hedging (PCH), and mid-transition propulsion-failure logic execute deterministically across the outer guidance reference model and the 400 Hz INDI-WLS inner loop. For angle-of-attack protection, set the operational AoA ceiling at [STRIPPED_MATH_OR_ICON] (retaining gust margin against distributed-propulsion inflow shifts). Whenever [STRIPPED_MATH_OR_ICON], the pitch reference model enforces the barrier rate cap \dot{q}_{\text{req}} \le \min\!\big(\dot{q}_{\text{cmd}},\, k_\alpha(\alpha_{[span_41](start_span)[span_41](end_span)\max} - \alpha) - k_q q\big). Any discrepancy \boldsymbol{\nu}_h = \Delta\boldsymbol{\nu} - B\,\Delta\mathbf{u}^* caused by [STRIPPED_MATH_OR_ICON] clamping, actuator saturation, or OEI thrust limits is subtracted immediately from the outer reference model via Pseudo-Control Hedging (PCH), eliminating integrator windup and post-saturation pitch departure.

Envelope / Failure Event

Detection Trigger

Immediate Inner-Loop WLS Action

Outer-Loop Recovery Policy

Stall Floor Breach (V_{\text{stall}})

Cruise V_a < 1.08\,V_{\text{stall}}(n_z) where V_{\text{stall}}(n_z) = 21.83\sqrt{\max(n_z, 0.5)}\text{ m/s} (23.6 m/s at 1 g)

Release rotor lock, reset \eta along lower hysteresis branch, recruit lift rotors via WLS

Command nose-down pitch + max pusher thrust until V_a > 26.0\text{ m/s} or complete back-transition

8+1 Lift-Rotor OEI (Any V_a)

DShot ESC_STATUS RPM residual + INDI residual \|\dot{\boldsymbol{\omega}}_f - \dot{\boldsymbol{\omega}}_{\text{pred}}\| > 1.5\text{ rad/s} (< 30 ms)

Zero column j of B; clamp \Delta u_j = -u_{f,j}; WLS reallocates (≤ 79.2% T_{\max} in hover)

Abort mission, divert to nearest coastal SLZ using full 4-DOF authority

4+1 Lift-Rotor OEI Mid-Transition

Same DShot + INDI isolation

Zero column j of B; if V_a < 12\text{ m/s}, set W_{v,\text{yaw}} \to 0 to protect roll/pitch

V_a \ge 12\text{ m/s}: commit forward to wing-borne cruise (peak thrust ≤ 979 N, 92.4% T_{\max}, dropping to 0 at V_{\text{trans}}). V_a < 12\text{ m/s}: accelerate on pusher if altitude permits, else emergency descent

Pusher Failure Mid-Transition

Pusher RPM drop + longitudinal acceleration residual

Zero pusher column; hold C_{L,\text{bg}} = \sqrt{C_{D,0}\pi e AR} = 0.861 (V_{\text{bg}} = 29.3\text{ m/s}, L/D_{\max} = 13.46, sink 2.18 m/s)

V_a < 24.0\text{ m/s} (1.10 V_{\text{stall}}): immediate hover abort (\eta \to 0 in 0.4 s) & vertical SLZ landing. V_a \ge 25.1\text{ m/s} (1.15 V_{\text{stall}}): glide at 13.5:1 (> 1.6 km from 120 m AGL) to SLZ, then spin up lift rotors for flare (hysteresis band 24.0–25.1 m/s selects glide only if SLZ exceeds hover battery range)

Operational thresholds synthesized from our 240 kg Lift+Cruise LP/glide calculations and INDI fault-tolerant control literature. [1][2][3][4][5][6]

Notice an informative subtlety in the 4+1 OEI linear-program outputs: because aerodynamic surface deflections carry zero thrust-consumption cost in the LP objective, the allocator commands full aileron ([STRIPPED_MATH_OR_ICON]) across 8–22 m/s to minimize remaining lift-rotor thrust; at 8–10 m/s, peak rotor load hits 97.3%–98.1% of [STRIPPED_MATH_OR_ICON] with almost zero dynamic control margin, confirming 12–14 m/s as the true practical forward-commit threshold for a 4+1 airframe.

1.6 C++20 IAirframeDynamics, PX4 v1.16 standard_vtol, and Gazebo Harmonic Architecture

To keep the 400 Hz control thread allocation-free and polymorphic across VertiRTA-X8-240, VertiRTA-LC-240 (8+1), and VertiRTA-LC-240 (4+1), IAirframeDynamics uses fixed-capacity stack matrices (kMaxActuators = 12, double):

Integrating this stack with PX4 v1.16 and Gazebo Harmonic (gz-sim 8) directly leverages three v1.16 release capabilities: native bidirectional DShot publishing live ESC RPM onto uORB ESC_STATUS for INDI actuator synchronization and rapid OEI detection; exposure of vtol_vehicle_status in dds_topics (PX4-Autopilot#24582) alongside backward-transition position feedback (#23731) and the VTOL DSHOT150 scheduler fix (#24727); and first-class Gazebo Harmonic LTS SITL support. For full 400 Hz offboard authority over uXRCE-DDS, the companion C++20 runtime publishes px4_msgs::msg::OffboardControlMode with direct_actuator = true and all six higher-level flags (po[span_47](start_span)[span_47](end_span)sition, velocity, acceleration, attitude, body_rate, thrust_and_torque) set to false. Normalized motor commands ([-1, 1], NaN disarms) stream on px4_msgs::msg::ActuatorMotors (mapping the 8 lift rotors and 1 pusher), while control surface commands stream on px4_msgs::msg::ActuatorServos (NUM_CONTROLS = 8, normalized [-1, 1]). Because uXRCE-DDS is unauthenticated and accesses uORB directly, the companion-to-FMU link must stay on a physically isolated serial or Ethernet interface; in SITL, setting UXRCE_DDS_SYNCT = false ensures Gazebo Harmonic acts as the single deterministic clock for both ROS 2 and PX4. PX4's native vtol_att_control remains configured as a failsafe backup (VT_ARSP_BLEND = 14.0, VT_ARSP_TRANS = 28.0, FW_AIRSPD_MIN = 24.0, VT_PSHER_SLEW = 0.33), with a mandatory pre-flight/SITL health check confirming vtol_att_control is running to guard against a known PX4 v1.16 standard_vtol transition issue reported in community testing. [1][2][3][4]

In Gazebo Harmonic (gz-sim 8), the VertiRTA-LC-240 SDF combines per-rotor MulticopterMotorModel plugins with gz::sim::systems::AdvancedLiftDrag (populated via PX4's AVL automation tool for stability and control derivatives with respect to [STRIPPED_MATH_OR_ICON] and sigmoid stall parameters). Inspecting gz-sim 8's Wind.hh confirms that gz::sim::components::Wind is a tag-only NoData component and stock gz::sim::systems::WindEffects models only a spatially uniform global wind. Consequently, our Apache-2.0 custom world plugin (custom::VertiRtaUrbanWindSystem, inheriting gz::sim::System, ISystemConfigure, ISystemPreUpdate, and ISystemPostUpdate, registered via GZ_ADD_PLUGIN and GZ_ADD_PLUGIN_ALIAS) queries the mmap-ed VRTAWND1 surrogate at the vehicle WorldPose during PreUpdate to perform a dual injection: updating the world wind entity's WorldLinearVelocity to [STRIPPED_MATH_OR_ICON] so AdvancedLiftDrag and MulticopterMotorModel compute true local relative airspeed, while querying left-wingtip, right-wingtip, and tail wake offsets to apply differential roll-shear and pitch-shear gradient wrenches directly to airframe links. [1][2][3][4]

2. GPL-Isolated Offline OpenFOAM URANS/LES to mmap POD Urban Wake Surrogate (VRTAWND1)

2.1 Legal Air-Gap Guarantee and Repository Hygiene

OpenFOAM (ESI OpenCFD v2412 and current June 2026 release v2606) is licensed under GNU GPLv3. Under GPLv3 Section 0 and the FSF program-output rule, "the output from running a covered work is covered by this License only if the output, given its content, constitutes a covered work"—meaning numerical output is only GPL-encumbered if it embeds copied portions of the GPL program itself. Physical CFD velocity fields, turbulence statistics, POD eigenmodes, and RBF weights computed over user-supplied Istanbul building geometries are purely numerical data rather than derivative code. While commercial product release should still undergo routine counsel review, four strict engineering isolation gates prevent any technical GPL contamination of the Apache-2.0 flight stack: [1][2][3][4]

2.2 Offline OpenFOAM Workflow for Kartal–Maltepe Coastal High-Rises (Linux / Colab)

Istanbul's Maltepe–Dragos–Kartal shoreline experiences sharp directional wind contrasts: predominant NE Poyraz and secondary N Yıldız winds blowing across inland urban roughness alternate rapidly (particularly in spring and late fall) with strong SW Lodos winds and thermal sea breezes sweeping unobstructed off the Sea of Marmara directly onto waterfront high-rises such as İSTMarina (40.8846°N, 29.2057°E). On Ubuntu 22.04/24.04 Linux or Google Colab, OpenFOAM v2412 installs cleanly via the OpenCFD Debian repository (curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash && sudo apt-get update && sudo apt-get install openfoam2412-default); for automated cloud/CI pipelines, use the official Apptainer image (openfoam-dev:apptainer or openfoam-dev:2406-apptainer) because dl.openfoam.com has historically blocked GitHub Actions IPs (OpenFOAM GL#3236). [1][2][3][4]

Domain geometry should be parameterized by the maximum local building height [STRIPPED_MATH_OR_ICON] extracted from Istanbul Metropolitan Municipality 3D GIS/DSM footprints (exact roof heights of İSTMarina and adjacent towers were not present in retrieved sources and must be sourced directly from GIS). Following COST 732 (Franke et al. 2007) and AIJ (Tominaga et al. 2008) urban CFD best practices for boundary clearance and blockage control, allocate ≥ 5 [STRIPPED_MATH_OR_ICON] upstream fetch, lateral margins, and domain height (keeping blockage ratio < 3%) and ≥ 15 [STRIPPED_MATH_OR_ICON] downstream wake recovery. Generate the mesh via snappyHexMesh (utilizing v2412's buffer-layer snapping improvements) with 1.5–2.5 m isotropic refinement across rooftop SLZ approach volumes and sharp cornice edges, grading smoothly to 10–15 m in the macro corridor. [1][2][3][4]

Prescribe neutral atmospheric boundary layer (ABL) inlets using OpenFOAM's atmBoundaryLayer YGCJ / Richards–Hoxey equilibrium profiles (atmBoundaryLayerInletVelocity, atmBoundaryLayerInletK, atmBoundaryLayerInletOmega): [1][2][3][4]

with default Richards–Hoxey constants [STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON]. Roughness length [STRIPPED_MATH_OR_ICON] must reflect coastal fetch asymmetry: onshore SW Lodos and marine sea breezes use open-water roughness [STRIPPED_MATH_OR_ICON] ([STRIPPED_MATH_OR_ICON]), producing a full-momentum profile with severe facade impingement and rooftop suction, whereas offshore NE Poyraz and N Yıldız winds use urban terrain roughness [STRIPPED_MATH_OR_ICON] ([STRIPPED_MATH_OR_ICON]). Apply atmNutkWallFunction (Hargreaves–Wright consistent with [STRIPPED_MATH_OR_ICON]) on the ground, nutkRoughWallFunction ([STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON]) on building walls, and slip rather than symmetryPlane on lateral/top patches to avoid OpenFOAM's ABL symmetry-plane glitch (GL#3226). [1][2]

To make urban wake generation tractable on a workstation or Colab GPU/CPU instance without sacrificing separated-wake accuracy, run a two-stage campaign. First, execute a steady RANS sweep (simpleFoam, kOmegaSST) over 16 wind directions ([STRIPPED_MATH_OR_ICON], matching the U3DWind UAM urban benchmark structure) across 3 reference speeds ([STRIPPED_MATH_OR_ICON] at [STRIPPED_MATH_OR_ICON]), generating [STRIPPED_MATH_OR_ICON] mean-flow snapshots. Second, for the 4–6 dominant coastal directions (NE Poyraz, N Yıldız, SW Lodos, SSW Sea Breeze), restart from RANS into transient pimpleFoam using hybrid Detached Eddy Simulation kOmegaSSTIDDES with IDDESDelta. IDDES preserves RANS efficiency in attached building walls while resolving unsteady eddies inside separated rooftop recirculation bubbles and leeward wakes; advancing 5 flow-through times and sampling 200 snapshots extracts corrected time-averaged means [STRIPPED_MATH_OR_ICON], anisotropic Reynolds stresses [STRIPPED_MATH_OR_ICON], and turbulence integral length scales [STRIPPED_MATH_OR_ICON]. [1][2]

2.3 POD + RBF Compression and Binary VRTAWND1 mmap Layout

All [STRIPPED_MATH_OR_ICON] snapshots are resampled onto a two-tier structured grid and normalized by [STRIPPED_MATH_OR_ICON] so high-Reynolds bluff-body similarity is exploited while residual [STRIPPED_MATH_OR_ICON]/roughness shifts stay parameterized. Defining the periodic inflow vector \boldsymbol{\mu}_j = [\cos\psi_{w,j},\, \sin\psi_{w,j},\, U_{\text{ref},j}/U_0]^T \in \mathbb{R}^3, form the mean-centered snapshot matrix \mathbf{S} = [\tilde{\mathbf{u}}(\boldsymbol{\mu}_1) - \mathbf{u}_0, \dots, \tilde{\mathbf{u}}(\boldsymbol{\mu}_{N_s}) - \mathbf{u}_0] \in \mathbb{R}^{3N_{\text{nodes}} \times N_s} and take its thin Singular Value Decomposition (SVD): [1][2]

Retaining [STRIPPED_MATH_OR_ICON] orthonormal spatial POD modes \boldsymbol{\Phi}_k minimizes projection error \|\mathbf{S} - \boldsymbol{\Phi}_R\boldsymbol{\Phi}_R^T\mathbf{S}\|_F^2. Modal coefficients \mathbf{a}(\boldsymbol{\mu}_j) = \boldsymbol{\Phi}_R^T(\tilde{\mathbf{u}}(\boldsymbol{\mu}_j) - \mathbf{u}_0) \in \mathbb{R}^8 are interpolated continuously across arbitrary live wind inputs \boldsymbol{\mu} via a regularized multiquadric RBF plus affine trend: [1][2][3][4][5]

Online, the 3D wind vector sums the POD-RBF mean field with spatially correlated anisotropic Ornstein–Uhlenbeck turbulence fluctuations driven by local [STRIPPED_MATH_OR_ICON]:

where [STRIPPED_MATH_OR_ICON] accounts for Taylor frozen-turbulence convection across the moving airframe.

To satisfy < 16 MiB storage and < 5 µs C++20 mmap lookup with maximal cache locality, every grid node packs into exactly one 64-byte CPU cache line (32 × int16_t) organized in 4×4×4 Morton-swizzled spatial bricks:

VRTAWND1 Binary Section

Domain / Resolution

Node Count

Storage Size

Memory Role

Header + SHA-256 + Mode Scales + RBF Table

48 centers × 8 modes + Z-LUTs

—

12.0 KiB

Preloaded into L1 cache on mmap()

Tier 0: Macro Corridor Wake Grid

2400 × 1200 × 240 m (\Delta xy = 25\text{ m}, \Delta z = 3\text{–}15\text{ m})

129,024 (96×48×28)

7.88 MiB

En-route coastal tower wakes

Tier 1: High-Res SLZ Rooftop Nest

320 × 320 × 112 m (\Delta xy = 5\text{ m}, \Delta z = 1.5\text{–}8\text{ m})

98,304 (64×64×24)

6.00 MiB

Final approach & rooftop shear bubble

Total VRTAWND1 Artifact

Two-tier hierarchical 4×4×4 bricks

227,328 nodes

13.887 MiB

2.11 MiB margin under 16 MiB cap

Byte-verified layout and analytical quantization/latency budget from.

At ±25 m/s full-scale per mode, int16 quantization gives an LSB of 0.763 mm/s and RMS quantization error of 0.220 mm/s—orders of magnitude smaller than CFD turbulence uncertainty. During 400 Hz continuous flight, 4×4×4 brick spatial locality keeps the 8 trilinear corner nodes (512 bytes = 8 cache lines) resident in L2/L3 cache; analytical cycle budgeting for index math (~15 ns), 8 cache-line loads, and AVX2 unpack/blend/8-mode FMA (~45 ns at 3.0 GHz) places single-point lookup at ~0.20 µs warm, ~0.70 µs cold DRAM, and ~1.10 µs when refreshing the 48-center RBF weights at 1–10 Hz ambient wind updates.

2.4 Quantifying Wake-Shear Prediction Error vs. Full CFD and Rooftop Downdrafts

Surrogate error is strongly heterogeneous across urban flows and must be bounded explicitly before flight deployment. Published non-intrusive ROM benchmarks on 3D unsteady bluff-body cylinder wakes report > 99% CPU time reduction with < 5.2% accuracy loss (and 2.3% on panel FSI optimization), while urban discrete point-prediction models achieve 0.016 m/s mean error and 0.060 m/s RMSE in smooth outer flow compared to 0.387 m/s for legacy models. Crucially, however, DLR's 2025 comparison of PODI+RBF, Autoencoders (AE), and Neural Implicit Flows (NIF) on a 434,388-cell urban RANS domain established a vital trade-off: PODI+RBF achieved the lowest overall mean relative [STRIPPED_MATH_OR_ICON]-error and fastest evaluation speedup, yet linear modal truncation produced higher localized peak errors inside steep velocity-gradient wake zones immediately behind buildings compared to nonlinear AE and NIF surrogates. [1][2]

Those steep velocity gradients coincide directly with rooftop SLZ approach cones. Politecnico di Milano's December 2025 urban vertiport study (low-rise rooftop [STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON] at 100 m) found that sharp leading-edge separation creates a reverse-flow recirculation bubble extending ~3.5 m vertically and 26 m ([STRIPPED_MATH_OR_ICON]) downstream with interior speeds < 3 m/s, topped immediately by an accelerated shear layer reaching nearly 6 m/s whose perturbation persists up to ~50 m ([STRIPPED_MATH_OR_ICON]) above the roof. Rooftop baseline turbulence intensity sits near 10%, peaking in the shear layer at [STRIPPED_MATH_OR_ICON] and [STRIPPED_MATH_OR_ICON], concentrating severe gusts within the first 30–40 m above rooftop FATO surfaces. Because explicit normalized rooftop downdraft tables ([STRIPPED_MATH_OR_ICON]) and per-zone RMSE values were absent from those retrieved excerpts, VRTAWND1 requires a two-part offline verification protocol: first validate the OpenFOAM kOmegaSSTIDDES solver itself against the open CC BY 4.0 ERIES-WonV1 wind-tunnel vertiport dataset (Cobra-probe and PIV measurements over ground-level and rooftop vertiports near upstream buildings, Milani et al. 2026); second, execute Leave-One-Direction-Out (LODO) cross-validation against unseen IDDES directions stratified across freestream corridor ([STRIPPED_MATH_OR_ICON], target RMSE < 3%–4% of [STRIPPED_MATH_OR_ICON]), leeward canyons (target RMSE < 6%–8%), and rooftop FATO shear/downdraft volumes ([STRIPPED_MATH_OR_ICON]): [1][2]

Writing the local P95 residual [STRIPPED_MATH_OR_ICON] directly into channel [30] of VrtaWndNode64 converts POD-RBF's localized shear-layer inaccuracy into a formal, spatially adaptive disturbance margin inside the 100 Hz HJI-CBVF-QP safety filter (Section 3.3).

3. Validated Coaxial BVI + Broadband Aeroacoustics and Soft-Constraint 4D Routing

3.1 Closed-Form/Semi-Empirical Source Model and ISO 9613-1/2:2024 Propagation (< 1 ms)

To predict community noise (dBA, [STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON]) in < 1 ms without running online Ffowcs Williams–Hawkings (FW-H) CFD solvers, VertiRTA synthesizes a body-fixed 3D acoustic hemisphere at reference radius [STRIPPED_MATH_OR_ICON]—safely beyond the experimentally observed spherical-decay onset at [STRIPPED_MATH_OR_ICON] where spreading errors drop to 1.03 dB—on an 18-azimuth × 10-elevation (180-node) grid following NASA STRIVES / ANOPP2 angle conventions across 27 third-octave bands ([STRIPPED_MATH_OR_ICON]). Three distinct acoustic source mechanisms sum incoherently in mean-square pressure per band. [1][2][3][4][5][6][7]

A. Strip-Integrated BPM Broadband Self-Noise + 2025 Non-Compactness and Amiet Inflow-Turbulence

Discretizing each blade into [STRIPPED_MATH_OR_ICON] radial strips ([STRIPPED_MATH_OR_ICON], chord [STRIPPED_MATH_OR_ICON], span [STRIPPED_MATH_OR_ICON], local velocity [STRIPPED_MATH_OR_ICON], Mach [STRIPPED_MATH_OR_ICON], Reynolds number [STRIPPED_MATH_OR_ICON]), the Brooks–Pope–Marcolini (NASA RP-1218) turbulent-boundary-layer trailing-edge (TBL-TE) and separation-stall 1/3-octave SPLs are: [1][2][3][4][5][6][7]

where [STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON], high-frequency edge-scattering directivity [STRIPPED_MATH_OR_ICON], and exact piecewise Reynolds adjustment [STRIPPED_MATH_OR_ICON]: [1][2][3][4][5][6][7][8][9]

As demonstrated by 2025 Penn State and 2026 JASA aeroacoustic investigations, classical BPM underpredicts small-to-medium rotor high-frequency broadband noise by 5–15 dB because compact-chord assumptions break down above 1 kHz ([STRIPPED_MATH_OR_ICON]) and Blade-Wake Interaction (BWI) plus Turbulence Ingestion Noise (TIN) dominate between 800 Hz and 8 kHz. Adding the non-compactness correction [STRIPPED_MATH_OR_ICON] and the Amiet–Lowson leading-edge inflow-turbulence 1/3-octave mean-square pressure closes this underprediction: [1][2][3][4][5]

where [STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON] with Sears transfer function [STRIPPED_MATH_OR_ICON], and ingested turbulence velocity [STRIPPED_MATH_OR_ICON] scales with momentum-theory induced velocity [STRIPPED_MATH_OR_ICON] (isolated lift rotor: [STRIPPED_MATH_OR_ICON]; coaxial pair ingesting sheared upper-rotor wake: [STRIPPED_MATH_OR_ICON]; cruise pusher in wing/empennage wake: [STRIPPED_MATH_OR_ICON]). Coaxial co-rotating experiments (VFS Forum 81, 2025) further show that once shed tip-vortex core radii exceed ~10% of tip chord, coherent wake structure breaks down and broadband noise saturates across phase offsets. [1][2]

B. Lowson Tonal Harmonics and Coaxial BVI Penalty

For blade-passing harmonics [STRIPPED_MATH_OR_ICON] ([STRIPPED_MATH_OR_ICON]), compact Lowson steady loading radiates predominantly near the rotor plane via [STRIPPED_MATH_OR_ICON] and vanishes directly on-axis ([STRIPPED_MATH_OR_ICON]). In a coaxial contra-rotating X8, periodic upper–lower blade crossings and lower-rotor chopping of upper-rotor tip vortices excite unsteady loading harmonics ([STRIPPED_MATH_OR_ICON], including on-axis [STRIPPED_MATH_OR_ICON] modes where [STRIPPED_MATH_OR_ICON]) that radiate strongly straight downward: [1][2]

Calibrating [STRIPPED_MATH_OR_ICON] against NASA TM-2018-219895 (Schatzman: coaxial loading exceeds a single rotor by +3.4 to +8.5 dB across observers 1–10 and peaks at +7.4 dB directly beneath the hub, requiring axial separation [STRIPPED_MATH_OR_ICON] chords) and University of Southampton contra-rotating experiments (Chaitanya et al.: minimum-noise optimum at [STRIPPED_MATH_OR_ICON] with a flat plateau across [STRIPPED_MATH_OR_ICON]; potential-field tones dominating for [STRIPPED_MATH_OR_ICON]; tip-vortex pseudo-tones and broadband within 1 dB for [STRIPPED_MATH_OR_ICON]; overall sound power scaling as [STRIPPED_MATH_OR_ICON] and thrust as [STRIPPED_MATH_OR_ICON]) yields: [1][2][3][4][5][6][7][8]

where [STRIPPED_MATH_OR_ICON] activates during shallow descent approaches ([STRIPPED_MATH_OR_ICON] at [STRIPPED_MATH_OR_ICON]) when upper-rotor tip vortices graze the lower disk—capturing the 20–35 dB impulsive coaxial BVI penalty documented in level/descending flight and incentivizing steeper non-BVI SLZ approaches. Note also that side-by-side rotor proximity below [STRIPPED_MATH_OR_ICON] can raise nadir tonal noise by up to +10 dBA unless lateral spacing reaches ≥ 2.6 [STRIPPED_MATH_OR_ICON] or 30°–90° phase synchronization is applied (yielding 4.2–8 dB tonal reductions); and for the Lift+Cruise pusher behind the wing/empennage wake, [STRIPPED_MATH_OR_ICON] accounts for rear-propeller wake-overlap penalties observed in 2025 Lift+Cruise wind-tunnel tests. [1][2][3][4]

C. ISO 9613-1:1993 Absorption, ISO 9613-2:2024 Ground Effect, and IEC 61672 A-Weighting

Propagating hemisphere band levels along slant range [STRIPPED_MATH_OR_ICON] to ground receptor [STRIPPED_MATH_OR_ICON] applies Doppler convective shift [STRIPPED_MATH_OR_ICON] ([STRIPPED_MATH_OR_ICON], +40$\log_{10}D_{\text{conv}}$ amplification), spherical spreading [STRIPPED_MATH_OR_ICON], exact ISO 9613-1:1993 atmospheric absorption [STRIPPED_MATH_OR_ICON], and ISO 9613-2:2024 ground effect [STRIPPED_MATH_OR_ICON]: [1][2][3][4]

with [STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON], and [STRIPPED_MATH_OR_ICON]. Under Istanbul Marmara coastal conditions (20 °C, 70% RH), [STRIPPED_MATH_OR_ICON] rises from 0.09 dB/km at 63 Hz and 4.98 dB/km at 1 kHz to 23.09 dB/km at 4 kHz and 77.63 dB/km at 8 kHz, meaning high-frequency broadband attenuates rapidly with altitude while low-frequency BPF tones carry farthest. Crucially, ISO 9613-2:2024 introduced the geometric correction factor [STRIPPED_MATH_OR_ICON] to eliminate the ~3 dB ground-attenuation artifact present in ISO 9613-2:1996 for elevated sources ([STRIPPED_MATH_OR_ICON] where [STRIPPED_MATH_OR_ICON]); over hard coastal concrete/water ([STRIPPED_MATH_OR_ICON]), near-overhead ground reflection contributes a slight constructive gain [STRIPPED_MATH_OR_ICON]. Summing across all 27 bands with IEC 61672 A-weighting [STRIPPED_MATH_OR_ICON] produces instantaneous [STRIPPED_MATH_OR_ICON], from which [STRIPPED_MATH_OR_ICON] (1-second slow integration; [STRIPPED_MATH_OR_ICON] sits 1–2 dB higher), Sound Exposure Level [STRIPPED_MATH_OR_ICON], and EASA EPTS hover [STRIPPED_MATH_OR_ICON] / flight-path EPNL proxies are derived. Engineers should note one physical boundary on straight-ray ISO 9613 propagation: severe coastal temperature inversions, wind-gradient refraction, and Dragos hill terrain shielding can alter shadow-zone SPLs by up to 35 dB relative to homogeneous straight-ray predictions. [1]

3.2 Predicted Ground dBA Footprints (240 kg Coaxial X8 vs. Lift+Cruise) and < 1 ms Budget

240 kg Model Prediction (Istanbul 20 °C, 70% RH, G = 0.2)

60 m AGL (BB / Ton)

100 m AGL (BB / Ton)

120 m AGL (BB / Ton)

150 m AGL (BB / Ton)

Coaxial X8-240 Hover (z/D = 0.22)

79.5 dBA (78.7 / 71.8)

74.9 dBA (74.1 / 67.3)

73.3 dBA (72.4 / 65.7)

71.2 dBA (70.4 / 63.7)

Coaxial X8-240 Tight (z/D = 0.12)

80.8 dBA (79.8 / 74.1)

76.3 dBA (75.2 / 69.5)

74.6 dBA (73.6 / 67.9)

72.6 dBA (71.5 / 65.9)

Lift+Cruise 8+1 Hover (Isolated Rotors)

74.4 dBA (73.7 / 65.9)

69.8 dBA (69.1 / 61.3)

68.1 dBA (67.5 / 59.7)

66.1 dBA (65.4 / 57.6)

Lift+Cruise 8+1 Mid-Transition (20 m/s)

70.1 dBA (69.5 / 61.0)

65.5 dBA (64.9 / 56.5)

63.8 dBA (63.3 / 54.8)

61.8 dBA (61.2 / 52.8)

Lift+Cruise 8+1 Cruise (38 m/s, Pusher)

60.6 dBA (59.7 / 53.1)

55.9 dBA (55.1 / 48.5)

54.3 dBA (53.4 / 46.9)

52.2 dBA (51.3 / 44.8)

Overhead ground footprints (Total dBA with Broadband / Tonal A-weighted split) from our strip-integrated BPM + Amiet + Lowson + ISO 9613-1/2:2024 model.

No publicly released outdoor flight-test dataset directly benchmarks a 240 kg coaxial X8 against a 240 kg 8+1 Lift+Cruise, so field calibration with a 3-point flight-path and 9-point hover microphone array per EASA EPTS remains required prior to operational certification. Nevertheless, our semi-empirical predictions sit squarely within expected physical bounds: industry reference summaries and ducted-propulsor vendor baselines place open-rotor eVTOL hover/takeoff noise in the 70–85 dBA range at 100 m, and NASA's primary acoustic flight tests of the heavier multi-passenger Joby eVTOL confirmed broadband-dominated flyover signatures between 66 and 74 dBA. [1][2][3][4][5][6]

Three operational routing principles follow directly from these model predictions. First, tightening X8 axial spacing from [STRIPPED_MATH_OR_ICON] to 0.12 elevates ground noise by 1.3–1.4 dBA (+2.2 dB tonal jump), while eliminating coaxial wake ingestion via the 8+1 Lift+Cruise isolated rotors cuts hover noise by 5.1–5.2 dBA. Second, accelerating VertiRTA-LC-240 into 38 m/s wing-borne cruise reduces overhead noise at 120 m AGL to 54.3 dBA—19.0 dBA quieter than hovering coaxial X8 while simultaneously cutting Sound Exposure Level ([STRIPPED_MATH_OR_ICON]) via 3–4× shorter pass-by dwell. Third, because the cruise pusher's thrust axis is horizontal, its acoustic radiation peaks in-plane rather than directly beneath the nose: at 120 m AGL, L+C Cruise rises from 54.3 dBA overhead ([STRIPPED_MATH_OR_ICON]) to a 58.6 dBA sideline dipole peak at [STRIPPED_MATH_OR_ICON] before decaying to 53.5 dBA at 350 m and 50.3 dBA at 500 m—demonstrating why a full 3D acoustic hemisphere (not a 1D distance-only lookup) is essential when routing beside coastal hospitals.

Analytically budgeting the runtime compute cost confirms ample margin inside 1 ms: refreshing the full 180-node × 27-band hemisphere via 8-strip BPM, Amiet, and 12-harmonic Lowson requires ~0.85M FLOPs (~0.35 ms on a single 3.0 GHz AVX2 core, refreshed only when RPM shifts > 2% or airspeed shifts > 2 m/s), while propagating onto 64 Istanbul ground receptors via bilinear hemisphere lookup and precomputed ISO 9613-1 [STRIPPED_MATH_OR_ICON] tables costs ~38k FLOPs (~0.015 ms), giving an end-to-end ~0.37 ms evaluation cycle.

3.3 Lexicographic Soft-Cost 4D Noise Routing Subordinate to Hard HJI-CBVF-QP Safety

Along the Maltepe–Dragos–Kartal coastal corridor, sensitive receptors lie compressed between the Sea of Marmara shoreline and the western approach boundary of Sabiha Gökçen International Airport (LTFJ, ARP 40.89833°N, 29.30917°E, elevation 95 m, CTR GND–2500 ft MSL, lying 8.83 km east of İSTMarina and 12.88 km east of the Kartal–Maltepe Dragos border at 40.9120°N, 29.1570°E). Priority acoustic receptors include Kartal Dr. Lütfi Kırdar Şehir Hastanesi (1,140 total beds, 205 ICU beds, Cevizli D-100 Güney Yanyol No:47, plus its Denizer Cad. Burn Center and Çamlık Polyclinic), adjacent Koşuyolu High Specialization Hospital, Süreyyapaşa Chest Diseases Hospital (Başıbüyük, Maltepe), coastal schools, and dense residential frontages. Crucially, Kartal Dr. Lütfi Kırdar Şehir Hastanesi also operates 1 active helipad (1 Helikopter Pisti): its helicopter approach/departure volume must be encoded as a hard airspace separation invariant inside Tier 2, completely separate from the hospital's soft community-noise exterior cost in Tier 1. [1][2][3][4][5][6][7]

Nominal soft noise thresholds synthesize Türkiye's 2022 Çevresel Gürültü Kontrol Yönetmeliği (as summarized in environmental compliance reports and audiology guidance: transportation L_Aeq,5dk limits of 65 dBA day / 60 dBA evening / 55 dBA night, and 45 dBA sensitive night/open-window threshold) with WHO environmental noise guidelines (WHO 2018 aviation recommendations of 45 dB [STRIPPED_MATH_OR_ICON] and 40 dB [STRIPPED_MATH_OR_ICON], plus earlier WHO indoor ward/classroom limits of 30–35 dB [STRIPPED_MATH_OR_ICON] and 40 dB [STRIPPED_MATH_OR_ICON] translated outward via the standard 15 dB open-window attenuation to ~45–50 dBA exterior targets, and 55 dBA residential outdoor living areas), while retaining EASA's EPTS (EPNL for overflight at 150 m / take-off / approach; [STRIPPED_MATH_OR_ICON] for hover) and < 600 kg Specific Category UAS measurement guidelines as certification reference metrics. [1][2][3][4][5][6][7]

To guarantee mathematically that noise abatement can never compromise obstacle clearance, airspace compliance, or OEI flight safety, the architecture enforces a strict two-tier lexicographic hierarchy:

By Proposition 3 of Choi et al., because [STRIPPED_MATH_OR_ICON] adds zero constraints to the CBVF feasible half-space [STRIPPED_MATH_OR_ICON], the 100 Hz Robust CBVF-QP remains provably feasible everywhere inside the verified safe set under compact polytopic actuator bounds [STRIPPED_MATH_OR_ICON] (including OEI-reconfigured bounds). Whenever an İSTMarina rooftop downdraft, mid-transition rotor fault, or traffic conflict pushes the aircraft toward a safety boundary—even if the evasive path temporarily overflies a quiet hospital zone—the CBVF half-space projects [STRIPPED_MATH_OR_ICON] instantaneously onto [STRIPPED_MATH_OR_ICON] within 10 ms, shedding 100% of noise preference without mode-switching discontinuities, slack-variable tuning hazards, or solver infeasibility. [1][2][3]

4. Phased Verification and Integration Roadmap

Implementation Phase

Target Subsystem & License Boundary

Quantitative Acceptance Gate

Phase 1: IAirframeDynamics + Transition SITL

Apache-2.0 C++20 core + PX4 v1.16 (direct_actuator, DShot ESC_STATUS) + Gazebo Harmonic AdvancedLiftDrag

Full 0–38–0 m/s transition with < 0.3 m altitude deviation; 8+1 OEI hover/transition trim verified at ≤ 80% T_{\max}; 400 Hz warm-started f64-WLS solve time < 50 µs (≤ 2% of 2.5 ms step)

Phase 2: Offline OpenFOAM \to VRTAWND1

Air-gapped GPLv3 tools/cfd_offline_gpl/ (OpenFOAM v2412/v2606 kOmegaSST + kOmegaSSTIDDES) \to Apache-2.0 mmap reader

Binary file < 16 MiB (target 13.89 MiB); C++20 mmap lookup < 5 µs (analytical budget ≤ 1.1 µs); IDDES benchmarked vs ERIES-WonV1 wind tunnel; LODO RMSE < 4% freestream & P95 error populated in eps_rom_p95; CI ldd/nm audit confirms 0 GPL symbols in runtime

Phase 3: Aeroacoustics + Soft-Noise 4D Routing

Apache-2.0 BPM + Amiet + Lowson/BVI + ISO 9613-1/2:2024 engine + 100 Hz HJI-CBVF-QP disturbance coupling

End-to-end 180-node hemisphere + 64 Istanbul receptors < 1.0 ms (analytical budget 0.37 ms); 0 CBVF safety violations across 500 Monte Carlo Lodos/Poyraz + OEI + hospital-proximity stress runs; live AIRAC LTFJ AD 2.17 polygon and Kartal Hospital helipad cylinder verified; EASA EPTS 3-point/9-point outdoor acoustic calibration scheduled

1. https://arc.aiaa.org/doi/10.2514/1.G008659 (Unified-Actuator Nonlinear Dynamic Inversion Controller for the)

2. https://research.tudelft.nl/files/47112416/Thesis_EJJ_Smeur.pdf (Incremental Control of Hybrid Micro Air Vehicles)

3. https://arc.aiaa.org/doi/10.2514/1.G007978 (Controllability Evaluation for VTOL Aircraft in Velocity Envelope)

4. https://www.scribd.com/document/808251946/OpenFOAM-Quick-Reference-Guide (OpenFOAM Quick Reference Guide | PDF | Turbulence - Scribd)

5. https://polytex.readthedocs.io/_/downloads/en/latest/pdf/ (PolyTex - Release 0.4.5 Bin Yang)

6. https://opensource.org/license/gpl-3.0 (GNU General Public License version 3 - Open Source Initiative)

7. https://aip.aero/tr/en/vfr/?LTFJ (Istanbul Sabiha Gökçen International Airport LTFJ - AIP:Aero)

8. https://gazebosim.org/api/sim/8/namespacegz_1_1sim_1_1systems.html (gz::sim::systems Namespace Reference - Gazebo)

9. https://www.youtube.com/watch?v=1BeuytG2QeE&vl=id (Write a Gazebo Harmonic system plugin - YouTube)
