# VertirtaCapabilityEnvelope.msg
# Published by mc_indi_wls_control @ 100 Hz over uORB / uXRCE-DDS (/fmu/out/vertirta_capability_envelope)
# Exact packed size: 72 bytes (8-byte aligned, zero implicit padding, DR-1 Erratum #4/#5 lock)

uint8 TMR_3OF3 = 0
uint8 TMR_DUPLEX_2OF2 = 1
uint8 TMR_SIMPLEX_ANALYTICAL = 2
uint8 TMR_FAILSAFE_DIVERT = 3

uint8 REGIME_NOMINAL = 0
uint8 REGIME_SINGLE_LOSS = 1
uint8 REGIME_CROSS_DUAL = 2
uint8 REGIME_SAME_ARM_ACTIVE_SPIN = 3
uint8 REGIME_CRITICAL_DESCENT = 4

uint64 timestamp                # [us] PX4 HRT monotonic timestamp (offset 0..7)
uint64 timestamp_sample         # [us] Voted IMU sample timestamp (offset 8..15)

uint8 fault_bitmask             # Bit 0..7 = 1 if motor i isolated failed (offset 16)
uint8 tmr_state                 # 0=TMR_3OF3, 1=DUPLEX_2OF2, 2=SIMPLEX_ANALYTICAL, 3=FAILSAFE_DIVERT (offset 17)
uint8 imu_health_mask           # Bit 0..2 = 1 if IMU 0..2 healthy (offset 18)
uint8 flight_regime             # 0=NOMINAL .. 4=CRITICAL_DESCENT (offset 19)

float32 t_max                   # [N] Dynamic trimmable max collective thrust T_max(fault, V_term) (offset 20..23)
float32 a_xy_max                # [m/s^2] Max sustainable horizontal accel respecting tilt & T_max (offset 24..27)
float32 tilt_max                # [rad] Dynamic safe tilt ceiling (0.45 nom -> 0.18 Regime 3) (offset 28..31)
float32 vz_sink_max             # [m/s] Max safe flare-recoverable descent rate magnitude (offset 32..35)

float32 v_term                  # [V] Filtered 1-RC terminal bus voltage (offset 36..39)
float32 i_pack                  # [A] Measured/estimated HV pack current (offset 40..43)
float32 soc                     # [0..1] Battery State of Charge (offset 44..47)
float32 p_hover_deg             # [W] Current fault-degraded electrical hover power (offset 48..51)

float32 spin_rate_ref           # [rad/s] Active yaw spin target (0.0 in Reg 0-2, +/-2.2 in Reg 3) (offset 52..55)
float32 spin_rate_meas          # [rad/s] Voted body yaw rate r (offset 56..59)
float32 chi2_residual           # Max pairwise 24-DOF chi^2 TMR parity statistic Q_ij (offset 60..63)

uint16 fdi_latency_ms           # [ms] Timestamped isolation latency of last latched fault (offset 64..65)
uint8[2] reserved_align         # Explicit 2-byte ROS 2 / uORB compliant alignment padding (offset 66..67)
uint32 reserved_pad             # Explicit 4-byte tail padding to exact 72 bytes (offset 68..71)
