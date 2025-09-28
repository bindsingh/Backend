# in business_logic.py
import time
from state_manager import traffic_state
from models import DecisionPayload

MIN_GREEN_TIME_SECONDS = 10

def apply_ai_decision(data: DecisionPayload):
    """
    This is the brain. It takes the AI's validated decision and applies it
    to the shared state, enforcing our rules.
    """
    # Rule 1: Check if enough time has passed since the last light change.
    time_since_last_change = time.time() - traffic_state["last_update_time"]
    proposed_phase = data.decision.get("action_code")

    # If the AI wants to change the light too soon, we ignore it for now.
    if proposed_phase != traffic_state["current_phase"] and time_since_last_change < MIN_GREEN_TIME_SECONDS:
        print(f"IGNORING AI: Tried to change light after {time_since_last_change:.1f}s. Min time is {MIN_GREEN_TIME_SECONDS}s.")
        # We don't change the phase, but we still update the counts.
        traffic_state["lane_counts"] = data.lane_counts
        traffic_state["pedestrian_count"] = data.pedestrian_count
        return # Stop processing here

    # If the rule passes, we update the state fully.
    # If the phase is new, reset the timer.
    if proposed_phase != traffic_state["current_phase"]:
        traffic_state["last_update_time"] = time.time()

    traffic_state["current_phase"] = proposed_phase
    traffic_state["last_decision_reason"] = data.decision.get("reason")
    traffic_state["lane_counts"] = data.lane_counts
    traffic_state["pedestrian_count"] = data.pedestrian_count

    # This is where you would collaborate with Dev B.
    # After updating the state, you need to trigger his broadcast function.
    # For now, we can just print.
    print(f"STATE UPDATED: Phase -> {traffic_state['current_phase']}, Reason: {traffic_state['last_decision_reason']}")