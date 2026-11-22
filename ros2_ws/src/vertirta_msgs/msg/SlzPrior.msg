# Canonical Safe Landing Zone (SLZ) candidate prior (mirrors vertirta::rta::SlzPrior)
# Consumed by Plan 1A RTA, Plan 2 TR/EN Co-Pilot, and Plan 3 Active Visual SLZ.

uint32 REASON_NONE = 0
uint32 REASON_CTR_VIOLATION = 1
uint32 REASON_ROUGHNESS_EXCEEDED = 2
uint32 REASON_SLOPE_EXCEEDED = 4
uint32 REASON_RADIUS_TOO_SMALL = 8
uint32 REASON_ENERGY_UNREACHABLE = 16
uint32 REASON_OBSTACLE_CLEARANCE = 32
uint32 REASON_STEP_EXCEEDED = 64
uint32 REASON_OOD_UNCERTAIN = 128
uint32 REASON_MONO_ALT_VETO = 256
uint32 REASON_ORBIT_TIMEOUT = 512
uint32 REASON_DYNAMIC_HAZARD = 1024

string id
float64 lat_deg
float64 lon_deg
float64 elev_amsl_m
geometry_msgs/Point center_enu
float32 p_safe
float32 slz_ood_variance
float32 theta_max_deg
float32 sigma_z_m
float32 step_max_m
float32 clearance_r_m
float32 h_dyn_max
float32 eta_s
float32 soc_margin_pct
uint8 rank
uint32 reason_mask
