import cv2
import numpy as np
import time
import json
import os
import sys 
from ultralytics import YOLO
import asyncio
import websockets
from q_learning_agent import AdaptiveQLearningAgent
from optimization_engine import OptimizationEngine

# =================================================================================
# === CONFIGURATION                                                             ===
# =================================================================================
IS_RAILWAY = bool(os.environ.get("RAILWAY_ENVIRONMENT"))

# --- File and Model Paths ---
BASE_DIR = os.path.dirname(__file__)
VIDEO_FILE = os.path.join(BASE_DIR, "my_video.mp4")
SAVED_AGENT_MODEL_PATH = os.path.join(BASE_DIR, "traffic_agent")

# --- Lanes, Crosswalks, and Directions ---
LANE_POLYGONS = {
    "Northbound": np.array([[2124, 487], [2830, 514], [2103, 1657], [2829, 1592]], np.int32),
    "Southbound": np.array([[966, 1568], [1380, 1574], [1467, 2048], [830, 2085]], np.int32),
    "Eastbound": np.array([[0,0], [1,1], [2,2], [3,3]], np.int32),
    "Westbound": np.array([[0,0], [1,1], [2,2], [3,3]], np.int32),
}
LANE_NAMES_ORDER = ["Northbound", "Southbound", "Eastbound", "Westbound"] 
CROSSWALK_POLYGONS = {
    "crosswalk_1": np.array([[1900, 1000], [2100, 1000], [2100, 1200], [1900, 1200]], np.int32)
}
TRAFFIC_LIGHT_POSITIONS = {
    "Northbound": (150, 250), "Southbound": (150, 450),
    "Eastbound": (150, 650), "Westbound": (150, 850)
}

# --- Traffic Logic and Timers ---
GREEN_LIGHT_DURATION = 8.0
YELLOW_LIGHT_DURATION = 1.5
EMERGENCY_GREEN_DURATION = 5.0
EMERGENCY_CLEARING_TIME = 2.0
STARVATION_THRESHOLD = 30.0
MAX_QUEUE_LENGTH = 30
PEDESTRIAN_THRESHOLD = 10
PEDESTRIAN_CROSSING_DURATION = 15

# --- Detection and RL Settings ---
YOLO_MODEL = 'yolov8s.pt'
CONF_THRESHOLD = 0.3
PERSON_CLASS_ID = 0
VEHICLE_CLASSES = [2, 3, 5, 7]
ALL_DETECTABLE_CLASSES = [PERSON_CLASS_ID] + VEHICLE_CLASSES
MAX_VEHICLES_PER_LANE = 40

# Railway backend WebSocket
WEBSOCKET_URI = "wss://backend-production-039d.up.railway.app/ws/ai"

# =================================================================================
# === HELPER FUNCTIONS                                                          ===
# =================================================================================
def draw_single_traffic_light(frame, position, status):
    x, y = position
    radius = 35
    color = (0, 0, 255)
    if status == "GREEN":
        color = (0, 255, 0)
    elif status == "YELLOW":
        color = (0, 255, 255)
    cv2.circle(frame, (x, y), radius, color, -1)
    cv2.circle(frame, (x, y), radius, (50, 50, 50), 3)

async def send_to_backend(data):
    try:
        async with websockets.connect(WEBSOCKET_URI) as websocket:
            await websocket.send(json.dumps(data))
    except Exception:
        print(f"[ERROR] Could not connect to backend.")

# =================================================================================
# === MAIN SCRIPT                                                               ===
# =================================================================================
async def run_live_inference():
    agent = AdaptiveQLearningAgent(action_size=4)
    agent.load_model(SAVED_AGENT_MODEL_PATH)
    engine = OptimizationEngine(starvation_threshold=STARVATION_THRESHOLD)
    model = YOLO(YOLO_MODEL)

    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        sys.exit(f"\n[ERROR] Could not open video file '{VIDEO_FILE}'.")

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

    signal_state = "GREEN"
    current_green_lane_index = 0
    state_timer = 0.0
    emergency_override_state = None
    lane_to_clear_index = -1
    emergency_target_lane = -1

    print("\n[INFO] LIVE MODE started...")
    frame_count = 0  # put this before the loop

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Video ended, restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"[DEBUG] Processed {frame_count} frames from {VIDEO_FILE}")

        # 👇 correctly indented inside while loop
        print(f"[INFO] Processing frame at {cap.get(cv2.CAP_PROP_POS_MSEC)/1000:.2f} seconds")

        results = model(frame, verbose=False)
        lane_counts = {name: 0 for name in LANE_NAMES_ORDER}
        pedestrian_count = 0

        # --- Object detection ---
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            if class_id not in ALL_DETECTABLE_CLASSES or box.conf[0].item() < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
            center_point = ((x1 + x2) // 2, (y1 + y2) // 2)

            if class_id == PERSON_CLASS_ID:
                for poly in CROSSWALK_POLYGONS.values():
                    if cv2.pointPolygonTest(poly, center_point, False) >= 0:
                        pedestrian_count += 1
                        break
            else:
                for name, poly in LANE_POLYGONS.items():
                    if cv2.pointPolygonTest(poly, center_point, False) >= 0:
                        lane_counts[name] += 1
                        break

        # --- GUI handling ---
        if IS_RAILWAY:
            key = -1   # no GUI possible
        else:
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow('AI Traffic System - Final Demo', display_frame)
            key = cv2.waitKey(1) & 0xFF

        # --- Key handling ---
        manual_emergency_lane = -1
        if key == ord('1'): manual_emergency_lane = 0
        elif key == ord('2'): manual_emergency_lane = 1
        elif key == ord('3'): manual_emergency_lane = 2
        elif key == ord('4'): manual_emergency_lane = 3
        if key == ord('q'):
            break

        state_timer += 1/fps
        decision_reason, send_update_to_backend = "Observing", False

        # =================================================================================
        # === DECISION MAKING LOGIC (Emergency + Normal)                                  ===
        # =================================================================================
        if manual_emergency_lane != -1 and emergency_override_state is None:
            if not (current_green_lane_index == manual_emergency_lane and signal_state == "GREEN"):
                emergency_override_state = "CLEARING_YELLOW"
                lane_to_clear_index = current_green_lane_index
                state_timer = 0.0
                emergency_target_lane = manual_emergency_lane
                decision_reason = f"MANUAL EMERGENCY: Clearing for {LANE_NAMES_ORDER[manual_emergency_lane]}"
                send_update_to_backend = True

        if emergency_override_state is not None:
            if emergency_override_state == "CLEARING_YELLOW":
                signal_state = "YELLOW"
                if state_timer >= YELLOW_LIGHT_DURATION:
                    emergency_override_state, state_timer = "CLEARING_ALL_RED", 0.0
                    decision_reason = "EMERGENCY: All Red Clearance"
                    send_update_to_backend = True
            elif emergency_override_state == "CLEARING_ALL_RED":
                signal_state = "ALL_RED"
                if state_timer >= EMERGENCY_CLEARING_TIME:
                    emergency_override_state = "ACTIVE"
                    current_green_lane_index = emergency_target_lane
                    signal_state, state_timer = "GREEN", 0.0
                    decision_reason = f"EMERGENCY: Green for {LANE_NAMES_ORDER[emergency_target_lane]}"
                    send_update_to_backend = True
            elif emergency_override_state == "ACTIVE":
                signal_state = "GREEN"
                if state_timer >= EMERGENCY_GREEN_DURATION:
                    emergency_override_state = None
                    signal_state, state_timer = "YELLOW", 0.0
                    decision_reason = "Emergency cleared, returning to normal operation"
                    send_update_to_backend = True
        else:
            # --- Normal cycle ---
            if signal_state == "GREEN" and (
                state_timer >= GREEN_LIGHT_DURATION or
                any(lane_counts[name] >= MAX_QUEUE_LENGTH for i, name in enumerate(LANE_NAMES_ORDER) if i != current_green_lane_index)
            ):
                signal_state, state_timer = "YELLOW", 0.0
            elif signal_state == "YELLOW" and state_timer >= YELLOW_LIGHT_DURATION:
                counts_ordered = [lane_counts[n] for n in LANE_NAMES_ORDER]
                norm_counts = [c/MAX_VEHICLES_PER_LANE for c in counts_ordered]
                observation = np.array(norm_counts + [current_green_lane_index / 3, 0])
                agent_recommendation = agent.choose_action(observation, training=False)
                context = {'emergency_active': False, 'emergency_lane': None, 'pedestrian_count': pedestrian_count}
                final_action, decision_reason, _ = engine.optimize_action(
                    lane_counts, LANE_NAMES_ORDER[current_green_lane_index],
                    agent_recommendation, context
                )
                current_green_lane_index, signal_state, state_timer, send_update_to_backend = final_action, "GREEN", 0.0, True
                engine.last_service_times[LANE_NAMES_ORDER[current_green_lane_index]] = time.time()
                engine.update_performance(final_action, -sum(lane_counts.values()), False, context)

        # =================================================================================
        # === SEND UPDATE TO BACKEND                                                      ===
        # =================================================================================
        if send_update_to_backend:
            output_data = {
                "timestamp": time.time(),
                "lane_counts": lane_counts,
                "pedestrian_count": pedestrian_count,
                "decision": {"reason": decision_reason},
                "signal_state": {
                    "active_direction": LANE_NAMES_ORDER[current_green_lane_index],
                    "state": signal_state,
                    "timer": int(state_timer)
                }
            }
            await send_to_backend(output_data)

    print("[INFO] Cleaning up...")
    cap.release()
    if not IS_RAILWAY:
        cv2.destroyAllWindows()

# =================================================================================
# === ENTRY POINT                                                               ===
# =================================================================================
if __name__ == '__main__':
    try:
        asyncio.run(run_live_inference())
    except FileNotFoundError:
        print(f"[FATAL ERROR] The model file '{SAVED_AGENT_MODEL_PATH}_qtable.pkl' was not found!")
    except ConnectionRefusedError:
        print(f"[FATAL ERROR] Connection to backend at {WEBSOCKET_URI} was refused.")
    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user.")
