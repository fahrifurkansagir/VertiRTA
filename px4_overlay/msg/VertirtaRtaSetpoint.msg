# VertirtaRtaSetpoint.msg
# Subscribed by mc_indi_wls_control @ 100 Hz over uXRCE-DDS (/fmu/in/vertirta_rta_setpoint)
# Exact packed size: 56 bytes (8-byte aligned, zero implicit padding)

uint8 MODE_ACCEL_NED = 0
uint8 MODE_DIRECT_THRUST_ATT = 1
uint8 MODE_CEL_CONTROLLED_LANDING = 2
uint8 MODE_HOLD_HOVER = 3

uint64 timestamp                # [us] PX4 HRT receive timestamp (offset 0..7)
uint64 timestamp_sample         # [us] Companion RTA filter compute timestamp (offset 8..15)

float32[3] accel_ned            # [m/s^2] Safe inertial acceleration setpoint [a_N, a_E, a_D] in NED (offset 16..27)
float32 thrust_des_n            # [N] Direct collective thrust setpoint (used in CEL / direct-thrust mode) (offset 28..31)
float32 yaw_sp                  # [rad] Desired heading in NED (Regimes 0..2) (offset 32..35)
float32 yawspeed_sp             # [rad/s] Feedforward yaw rate (overridden by Option B1b +/-2.2 in Regime 3) (offset 36..39)
float32 vz_limit_mps            # [m/s] Max sink-rate magnitude cap (2.5 >10m, 1.5 <5m, 0.8 touchdown) (offset 40..43)

uint8 control_mode              # 0=ACCEL_NED, 1=DIRECT_THRUST_ATT, 2=CEL_CONTROLLED_LANDING, 3=HOLD_HOVER (offset 44)
uint8 cel_latched               # 1 if outer RTA irreversibly latched CEL (offset 45)
uint8 fault_override_mask       # Optional SITL/HIL coordinated fault override bitmask (0 in flight) (offset 46)
uint8 valid                     # 1 if companion setpoint is valid & fresh (<250 ms) (offset 47)

uint32 seq                      # Monotonic 100 Hz companion setpoint sequence counter (offset 48..51)
uint32 reserved_pad             # Explicit 4-byte ROS 2 / uORB compliant tail padding to 56 bytes (offset 52..55)
