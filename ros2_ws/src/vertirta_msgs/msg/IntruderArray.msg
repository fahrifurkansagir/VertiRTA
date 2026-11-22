# ADS-B / Remote-ID constant-velocity KF intruder tracks in Kartal local ENU (/vertirta/intruders, <= 16 tracks)

uint8 SOURCE_ADSB = 0
uint8 SOURCE_REMOTE_ID = 1
uint8 SOURCE_TRACKER_FUSED = 2

std_msgs/Header header
uint64 stamp_ns
bool stale
uint32[] track_ids
geometry_msgs/Point[] pos_enu
geometry_msgs/Vector3[] vel_enu
float32[] radius_horiz_m
float32[] half_height_m
float32[] age_s
uint8[] source_type
