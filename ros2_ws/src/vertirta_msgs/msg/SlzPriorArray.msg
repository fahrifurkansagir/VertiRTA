# Ranked SLZ prior array + active inspection policy state (/vertirta/slz_priors @ 20-30 Hz)
# Mirrors vertirta::rta::SlzProviderSnapshot across Plan 1A & Plan 3.

uint8 STATE_IDLE = 0
uint8 STATE_APPROACH_ORBIT = 1
uint8 STATE_ACTIVE_INSPECT_ORBIT = 2
uint8 STATE_COMMIT_TOUCHDOWN = 3
uint8 STATE_REJECT_DIVERT_NEXT = 4

std_msgs/Header header
uint64 stamp_ns
bool stale
uint8 active_decision_state
float32 gimbal_pitch_rad
float32 gimbal_yaw_rad
vertirta_msgs/SlzPrior[] priors
