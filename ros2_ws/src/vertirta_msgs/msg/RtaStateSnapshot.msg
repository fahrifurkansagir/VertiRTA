# Frozen 400 V / 22 kWh RTA State Snapshot (/vertirta/rta_state_snapshot @ 1 Hz + on-fault)
# Consumed asynchronously by Plan 2 vertirta_mission_node (EASA AI Level 1/2A isolation).

std_msgs/Header header
uint64 stamp_ns
uint8 flight_regime
uint8 fault_bitmask
uint8 tmr_state
uint8 imu_health_mask
bool cel_latched
bool shield_active

geometry_msgs/Point pos_enu
geometry_msgs/Vector3 vel_enu
float32 agl_m

# Locked 400 V / 22 kWh HV battery & authority contract
float32 pack_nominal_volts
float32 pack_energy_kwh
float32 v_term_volts
float32 i_pack_amps
float32 soc_pct
float32 soc_reserve_floor_pct

float32 t_max_n
float32 a_xy_max_mps2
float32 tilt_max_rad
float32 vz_sink_max_mps
float32 p_hover_deg_w
float32 spin_rate_ref_rad_s
float32 spin_rate_meas_rad_s
float32 chi2_residual

# Safety filter, airspace CTR margins, and staleness indicators
float32 cbvf_min_margin_m
float32 ltfj_margin_m
float32 ltbx_margin_m
float32 esdf_age_ms
float32 intruder_age_ms
bool perception_stale

# Active dynamic NFZs (max 4) & ranked SLZ priors
uint8 num_dyn_nfz
string[] dyn_nfz_ids
geometry_msgs/Point[] dyn_nfz_centers_enu
float32[] dyn_nfz_radii_m
float32[] dyn_nfz_floor_amsl_m
float32[] dyn_nfz_ceil_amsl_m
vertirta_msgs/SlzPrior[] slz_priors
