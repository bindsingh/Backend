# in state_manager.py
import time

# This dictionary will be imported by other files to read/write the current state.
traffic_state = {
    "current_phase": 0, # Which lane is green (e.g., 0 for lane_1)
    "last_update_time": time.time(),
    "last_decision_reason": "System Initialized",
    "lane_counts": {"lane_1": 0, "lane_2": 0, "lane_3": 0, "lane_4": 0},
    "pedestrian_count": 0,
    "ai_status": "DISCONNECTED"
}