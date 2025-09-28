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
CROSSWALK_POLYGONS = { "crosswalk_1": np.array([[1900, 1000], [2100, 1000], [2100, 1200], [1900, 1200]], np.int32) }
TRAFFIC_LIGHT_POSITIONS = { "Northbound": (150, 250), "Southbound": (150, 450), "Eastbound": (150, 650), "Westbound": (150, 850) }

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
YOLO_MODEL = 'yolov8s.pt'; CONF_THRESHOLD = 0.3
PERSON_CLASS_ID = 0; VEHICLE_CLASSES = [2, 3, 5, 7]
ALL_DETECTABLE_CLASSES = [PERSON_CLASS_ID] + VEHICLE_CLASSES
MAX_VEHICLES_PER_LANE = 40

# Railway backend WebSocket
WEBSOCKET_URI = "wss://backend-production-039d.up.railway.app/ws/ai"

# =================================================================================
# === HELPER FUNCTIONS                                                          ===
# =================================================================================
def draw_single_traffic_light(frame, position, status):
    x, y = position; radius = 35; color = (0, 0, 255)
    if status == "GREEN": color = (0, 255, 0)
    elif status == "YELLOW": color = (0, 255, 255)
    cv2.circle(frame, (x, y), radius, color, -1); cv2.circle(frame, (x, y), radius, (50, 50, 50), 3)

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
    
    signal_state = "GREEN"; current_green_lane_index = 0
    state_timer = 0.0; emergency_override_state = None
    lane_to_clear_index = -1; emergency_target_lane = -1
    
    print("\n[INFO] LIVE MODE started...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model(frame, verbose=False)
        lane_counts = {name: 0 for name in LANE_NAMES_ORDER}
        pedestrian_count = 0
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
        
        manual_emergency_lane = -1
        if key == ord('1'): manual_emergency_lane = 0
        elif key == ord('2'): manual_emergency_lane = 1
        elif key == ord('3'): manual_emergency_lane = 2
        elif key == ord('4'): manual_emergency_lane = 3
        if key == ord('q'): break
        
        state_timer += 1/fps
        decision_reason, send_update_to_backend = "Observing", False

        # --- Emergency & Normal logic (keep your original) ---
        # TODO: paste back the same decision-making code you had

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

if __name__ == '__main__':
    try:
        asyncio.run(run_live_inference())
    except FileNotFoundError:
        print(f"[FATAL ERROR] The model file '{SAVED_AGENT_MODEL_PATH}_qtable.pkl' was not found!")
    except ConnectionRefusedError:
        print(f"[FATAL ERROR] Connection to backend at {WEBSOCKET_URI} was refused.")
    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user.")
