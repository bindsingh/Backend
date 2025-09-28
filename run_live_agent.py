import cv2
import numpy as np
import time
import json
import os
import sys 
import asyncio
import websockets
import logging
from ultralytics import YOLO

# Reduce logging spam
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# =================================================================================
# === CONFIGURATION (MINIMAL FOR RAILWAY)                                       ===
# =================================================================================
IS_RAILWAY = bool(os.environ.get("RAILWAY_ENVIRONMENT"))

# Use internal WebSocket URL for Railway
if IS_RAILWAY:
    WEBSOCKET_URI = "ws://localhost:8000/ws/ai"
else:
    WEBSOCKET_URI = "wss://backend-production-039d.up.railway.app/ws/ai"

# --- File Paths ---
BASE_DIR = os.path.dirname(__file__)
VIDEO_FILE = os.path.join(BASE_DIR, "my_video.mp4")

# --- Simple Lane Configuration ---
LANE_POLYGONS = {
    "Northbound": np.array([[2124, 487], [2830, 514], [2103, 1657], [2829, 1592]], np.int32),
    "Southbound": np.array([[966, 1568], [1380, 1574], [1467, 2048], [830, 2085]], np.int32),
    "Eastbound": np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.int32),
    "Westbound": np.array([[300, 100], [400, 100], [400, 200], [300, 200]], np.int32),
}
LANE_NAMES_ORDER = ["Northbound", "Southbound", "Eastbound", "Westbound"]

# --- Minimal Detection Settings ---
YOLO_MODEL = 'yolov8n.pt'  # Use nano model for speed
CONF_THRESHOLD = 0.4
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
PROCESS_EVERY_N_FRAMES = 5  # Process every 5th frame only
SEND_DATA_EVERY_N_SECONDS = 5  # Send data every 5 seconds

# =================================================================================
# === HELPER FUNCTIONS                                                          ===
# =================================================================================
async def send_to_backend(data):
    """Send data to backend via WebSocket with error handling"""
    try:
        async with websockets.connect(WEBSOCKET_URI, ping_timeout=10) as websocket:
            await websocket.send(json.dumps(data))
            print(f"[SUCCESS] Data sent: {data['signal_state']['active_direction']}")
    except Exception as e:
        print(f"[ERROR] Backend connection failed: {e}")

# =================================================================================
# === MAIN SIMPLIFIED SCRIPT                                                    ===
# =================================================================================
async def run_live_inference():
    """Simplified AI inference with minimal resource usage"""
    
    print("[INFO] Loading minimal YOLO model...")
    try:
        model = YOLO(YOLO_MODEL)
        model.overrides['verbose'] = False
        model.overrides['conf'] = CONF_THRESHOLD
        model.overrides['device'] = 'cpu'  # Force CPU to save memory
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        return

    print("[INFO] Opening video...")
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {VIDEO_FILE}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

    # Simple state variables
    current_lane_index = 0
    frame_count = 0
    last_data_send_time = 0
    
    # Scale down polygons for processing
    scaled_lane_polygons = {}
    
    print("[INFO] Starting simplified AI agent...")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video ended, restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1
            
            # Skip frames for performance
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                continue
            
            current_time = time.time()
            
            # Only process and send data every N seconds
            if current_time - last_data_send_time < SEND_DATA_EVERY_N_SECONDS:
                continue
                
            last_data_send_time = current_time

            try:
                # Resize frame for faster processing
                original_height, original_width = frame.shape[:2]
                processed_frame = cv2.resize(frame, (320, 240))  # Very small for Railway
                
                # Scale polygons if not done
                if not scaled_lane_polygons:
                    scale_x = 320 / original_width
                    scale_y = 240 / original_height
                    
                    for name, polygon in LANE_POLYGONS.items():
                        scaled_polygon = polygon.copy().astype(np.float32)
                        scaled_polygon[:, 0] *= scale_x
                        scaled_polygon[:, 1] *= scale_y
                        scaled_lane_polygons[name] = scaled_polygon.astype(np.int32)

                # Run YOLO detection
                results = model(processed_frame, verbose=False)
                lane_counts = {name: 0 for name in LANE_NAMES_ORDER}
                
                # Count vehicles in lanes
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        class_id = int(box.cls[0].item())
                        if class_id in VEHICLE_CLASSES and box.conf[0].item() > CONF_THRESHOLD:
                            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
                            center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                            
                            for name, poly in scaled_lane_polygons.items():
                                if cv2.pointPolygonTest(poly, center_point, False) >= 0:
                                    lane_counts[name] += 1
                                    break

                # Simple logic: rotate through lanes
                if sum(lane_counts.values()) > 0:  # Only change if there are vehicles
                    max_lane = max(lane_counts, key=lane_counts.get)
                    current_lane_index = LANE_NAMES_ORDER.index(max_lane)

                # Send data to backend
                output_data = {
                    "timestamp": current_time,
                    "lane_counts": lane_counts,
                    "pedestrian_count": 0,  # Simplified
                    "decision": {"reason": f"AI Decision: {LANE_NAMES_ORDER[current_lane_index]} has most traffic"},
                    "signal_state": {
                        "active_direction": LANE_NAMES_ORDER[current_lane_index],
                        "state": "GREEN",
                        "timer": 10
                    }
                }
                
                await send_to_backend(output_data)
                print(f"[INFO] Frame {frame_count}: {sum(lane_counts.values())} vehicles total")
                
            except Exception as e:
                print(f"[ERROR] Processing error: {e}")
                continue

    except Exception as e:
        print(f"[FATAL] AI agent crashed: {e}")
    finally:
        print("[INFO] Cleaning up AI agent...")
        cap.release()

# =================================================================================
# === ENTRY POINT                                                               ===
# =================================================================================
if __name__ == '__main__':
    try:
        asyncio.run(run_live_inference())
    except KeyboardInterrupt:
        print("[INFO] AI agent stopped by user")
    except Exception as e:
        print(f"[ERROR] AI agent failed: {e}")
'''import cv2
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
            print(f"[SUCCESS] Data sent to backend")
    except Exception as e:
        print(f"[ERROR] Backend connection failed: {e}")

# =================================================================================
# === MAIN SCRIPT                                                               ===
# =================================================================================
async def run_live_inference():
    print("[INFO] Initializing Q-Learning agent...")
    agent = AdaptiveQLearningAgent(action_size=4)
    agent.load_model(SAVED_AGENT_MODEL_PATH)
    
    print("[INFO] Initializing optimization engine...")
    engine = OptimizationEngine(starvation_threshold=STARVATION_THRESHOLD)
    
    print("[INFO] Loading YOLO model...")
    model = YOLO(YOLO_MODEL)
    
    # Set YOLO optimizations AFTER model is loaded
    model.overrides['verbose'] = False
    model.overrides['conf'] = CONF_THRESHOLD
    model.overrides['half'] = True

    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        sys.exit(f"\n[ERROR] Could not open video file '{VIDEO_FILE}'.")

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

    # Initialize state variables
    signal_state = "GREEN"
    current_green_lane_index = 0
    state_timer = 0.0
    emergency_override_state = None
    lane_to_clear_index = -1
    emergency_target_lane = -1
    frame_count = 0

    # Scale polygons for processing at startup
    scaled_lane_polygons = {}
    scaled_crosswalk_polygons = {}
    
    print("\n[INFO] LIVE MODE started...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Video ended, restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        
        # OPTIMIZATION 1: Skip frames for performance
        if frame_count % 3 != 0:
            continue
        
        if frame_count % 100 == 0:
            print(f"[DEBUG] Processed {frame_count} frames from {VIDEO_FILE}")

        print(f"[INFO] Processing frame at {cap.get(cv2.CAP_PROP_POS_MSEC)/1000:.2f} seconds")

        # OPTIMIZATION 2: Scale frame for processing
        original_height, original_width = frame.shape[:2]
        processed_frame = cv2.resize(frame, (640, 480))
        
        # Scale polygons if not already done
        if not scaled_lane_polygons:
            scale_x = 640 / original_width
            scale_y = 480 / original_height
            
            for name, polygon in LANE_POLYGONS.items():
                scaled_polygon = polygon.copy().astype(np.float32)
                scaled_polygon[:, 0] *= scale_x
                scaled_polygon[:, 1] *= scale_y
                scaled_lane_polygons[name] = scaled_polygon.astype(np.int32)
            
            for name, polygon in CROSSWALK_POLYGONS.items():
                scaled_polygon = polygon.copy().astype(np.float32)
                scaled_polygon[:, 0] *= scale_x
                scaled_polygon[:, 1] *= scale_y
                scaled_crosswalk_polygons[name] = scaled_polygon.astype(np.int32)

        # Use processed frame for YOLO
        results = model(processed_frame, verbose=False)
        lane_counts = {name: 0 for name in LANE_NAMES_ORDER}
        pedestrian_count = 0

        # Object detection with scaled coordinates
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            if class_id not in ALL_DETECTABLE_CLASSES or box.conf[0].item() < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
            center_point = ((x1 + x2) // 2, (y1 + y2) // 2)

            if class_id == PERSON_CLASS_ID:
                for poly in scaled_crosswalk_polygons.values():
                    if cv2.pointPolygonTest(poly, center_point, False) >= 0:
                        pedestrian_count += 1
                        break
            else:
                for name, poly in scaled_lane_polygons.items():
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
        print("\n[INFO] Program interrupted by user.")'''
