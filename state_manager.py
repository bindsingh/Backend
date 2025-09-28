import time

# Use consistent lane naming with frontend
traffic_state = {
    "current_phase": 0,
    "last_update_time": time.time(),
    "last_decision_reason": "System Initialized",
    "lane_counts": {
        "Northbound": 0,
        "Southbound": 0, 
        "Eastbound": 0,
        "Westbound": 0
    },
    "pedestrian_count": 0,
    "ai_status": "DISCONNECTED"
}
