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
'''import os
import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
import json

from state_manager import traffic_state

app = FastAPI()

# Track connected clients
dashboard_clients: List[WebSocket] = []

@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket endpoint for frontend dashboards"""
    await websocket.accept()
    dashboard_clients.append(websocket)
    print(f"Dashboard client connected. Total clients: {len(dashboard_clients)}")
    
    try:
        # Send initial state immediately
        frontend_payload = transform_to_frontend_format(traffic_state)
        await websocket.send_json(frontend_payload)
        
        while True:
            # Send updates every 3 seconds
            frontend_payload = transform_to_frontend_format(traffic_state)
            await websocket.send_json(frontend_payload)
            await asyncio.sleep(3)
            
    except WebSocketDisconnect:
        print("Dashboard client disconnected")
    except Exception as e:
        print(f"Dashboard WebSocket error: {e}")
    finally:
        if websocket in dashboard_clients:
            dashboard_clients.remove(websocket)
        print(f"Dashboard client removed. Total clients: {len(dashboard_clients)}")

@app.websocket("/ws/ai")
async def websocket_ai_agent(websocket: WebSocket):
    """WebSocket endpoint for AI agent to send data"""
    await websocket.accept()
    traffic_state["ai_status"] = "CONNECTED"
    print("AI agent connected")
    
    try:
        while True:
            # Receive data from AI agent
            data = await websocket.receive_text()
            ai_data = json.loads(data)
            print(f"Received AI data: {ai_data.get('signal_state', {}).get('active_direction', 'Unknown')}")
            
            # Update traffic state
            update_traffic_state(ai_data)
            
            # Broadcast to all dashboard clients immediately
            await broadcast_to_dashboards()
            
    except WebSocketDisconnect:
        print("AI agent disconnected")
        traffic_state["ai_status"] = "DISCONNECTED"
    except Exception as e:
        print(f"AI WebSocket error: {e}")
        traffic_state["ai_status"] = "DISCONNECTED"

def transform_to_frontend_format(state):
    """Transform backend state to frontend expected format"""
    lane_names = ["Northbound", "Southbound", "Eastbound", "Westbound"]
    current_phase = state.get("current_phase", 0)
    
    # Ensure current_phase is within valid range
    if current_phase >= len(lane_names):
        current_phase = 0
    
    active_direction = lane_names[current_phase]
    
    # Get lane counts with fallback to empty dict
    lane_counts = state.get("lane_counts", {})
    
    # Ensure all expected lanes exist in counts
    for lane in lane_names:
        if lane not in lane_counts:
            lane_counts[lane] = 0
    
    return {
        "main_dashboard": {
            "signal_state": {
                "active_direction": active_direction,
                "state": "GREEN",  # You might want to make this dynamic
                "timer": 10
            },
            "vehicle_counters": lane_counts,
            "total_vehicles": sum(lane_counts.values())
        },
        "performance_metrics": [
            {
                "title": "AI Status",
                "value": state.get("ai_status", "DISCONNECTED"),
                "status": "GOOD" if state.get("ai_status") == "CONNECTED" else "POOR",
                "details": f"Last decision: {state.get('last_decision_reason', 'N/A')}"
            },
            {
                "title": "Queue Efficiency",
                "value": "85%",
                "status": "GOOD",
                "details": "Average queue reduction"
            },
            {
                "title": "Response Time",
                "value": "2.3s",
                "status": "EXCELLENT",
                "details": "Average AI decision time"
            }
        ],
        "emergency_mode": None,
        "timestamp": time.time()
    }

def update_traffic_state(ai_data):
    """Update traffic state from AI agent data"""
    try:
        if "lane_counts" in ai_data:
            traffic_state["lane_counts"].update(ai_data["lane_counts"])
        
        if "signal_state" in ai_data:
            # Map active_direction to phase index
            direction_to_phase = {
                "Northbound": 0, "Southbound": 1, 
                "Eastbound": 2, "Westbound": 3
            }
            active_dir = ai_data["signal_state"].get("active_direction")
            if active_dir in direction_to_phase:
                traffic_state["current_phase"] = direction_to_phase[active_dir]
        
        if "decision" in ai_data:
            traffic_state["last_decision_reason"] = ai_data["decision"].get("reason", "")
        
        if "pedestrian_count" in ai_data:
            traffic_state["pedestrian_count"] = ai_data["pedestrian_count"]
        
        traffic_state["last_update_time"] = time.time()
        print(f"State updated - Phase: {traffic_state['current_phase']}, Lanes: {traffic_state['lane_counts']}")
        
    except Exception as e:
        print(f"Error updating traffic state: {e}")

async def broadcast_to_dashboards():
    """Broadcast updated data to all dashboard clients"""
    if not dashboard_clients:
        return
        
    payload = transform_to_frontend_format(traffic_state)
    
    disconnected_clients = []
    for client in dashboard_clients:
        try:
            await client.send_json(payload)
        except Exception as e:
            print(f"Failed to send to dashboard client: {e}")
            disconnected_clients.append(client)
    
    # Remove disconnected clients
    for client in disconnected_clients:
        if client in dashboard_clients:
            dashboard_clients.remove(client)

@app.get("/status")
async def get_status():
    """Returns the current traffic state"""
    return traffic_state

@app.get("/debug")
async def get_debug_info():
    """Debug endpoint to check system status"""
    return {
        "traffic_state": traffic_state,
        "dashboard_clients": len(dashboard_clients),
        "ai_connected": traffic_state.get("ai_status") == "CONNECTED",
        "last_update": traffic_state.get("last_update_time", 0),
        "time_since_update": time.time() - traffic_state.get("last_update_time", 0)
    }

@app.on_event("startup")
async def startup_event():
    print("Starting AI agent in background...")
    # Import here to avoid circular import
    from run_live_agent import run_live_inference
    asyncio.create_task(run_live_inference())

@app.get("/")
async def root():
    return {
        "message": "Traffic AI Backend running", 
        "timestamp": int(time.time()),
        "ai_status": traffic_state.get("ai_status", "DISCONNECTED"),
        "dashboard_clients": len(dashboard_clients)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)'''


