# Analytic thin-wire obstacle capsules in Kartal local ENU (/vertirta/wire_capsules, <= 64 spans)

uint8 CLASS_DISTRIBUTION = 0
uint8 CLASS_TRANSMISSION = 1
uint8 CLASS_CRANE_CABLE = 2

std_msgs/Header header
uint64 stamp_ns
bool stale
uint32[] ids
geometry_msgs/Point[] p0_enu
geometry_msgs/Point[] p1_enu
float32[] radius_m
uint8[] voltage_class
