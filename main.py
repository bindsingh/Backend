'''import os
import asyncio
import time
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
import json
import logging

# Configure logging to reduce spam
logging.getLogger("ultralytics").setLevel(logging.WARNING)

from state_manager import traffic_state

app = FastAPI()

# Track connected clients
dashboard_clients: List[WebSocket] = []
ai_agent_running = False

@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket endpoint for frontend dashboards"""
    await websocket.accept()
    dashboard_clients.append(websocket)
    print(f"Dashboard client connected. Total: {len(dashboard_clients)}")
    
    try:
        # Send initial state immediately
        frontend_payload = transform_to_frontend_format(traffic_state)
        await websocket.send_json(frontend_payload)
        
        while True:
            # Send updates every 5 seconds (reduced frequency)
            frontend_payload = transform_to_frontend_format(traffic_state)
            await websocket.send_json(frontend_payload)
            await asyncio.sleep(5)  # Increased from 3 to 5 seconds
            
    except WebSocketDisconnect:
        print("Dashboard client disconnected")
    except Exception as e:
        print(f"Dashboard WebSocket error: {e}")
    finally:
        if websocket in dashboard_clients:
            dashboard_clients.remove(websocket)

@app.websocket("/ws/ai")
async def websocket_ai_agent(websocket: WebSocket):
    """WebSocket endpoint for AI agent to send data"""
    await websocket.accept()
    traffic_state["ai_status"] = "CONNECTED"
    print("AI agent WebSocket connected")
    
    try:
        while True:
            # Receive data from AI agent
            data = await websocket.receive_text()
            ai_data = json.loads(data)
            
            # Update traffic state
            update_traffic_state(ai_data)
            
            # Broadcast to dashboards (but don't wait)
            asyncio.create_task(broadcast_to_dashboards())
            
    except WebSocketDisconnect:
        print("AI agent WebSocket disconnected")
        traffic_state["ai_status"] = "DISCONNECTED"
    except Exception as e:
        print(f"AI WebSocket error: {e}")
        traffic_state["ai_status"] = "DISCONNECTED"

def transform_to_frontend_format(state):
    """Transform backend state to frontend expected format"""
    lane_names = ["Northbound", "Southbound", "Eastbound", "Westbound"]
    current_phase = state.get("current_phase", 0)
    
    if current_phase >= len(lane_names):
        current_phase = 0
    
    active_direction = lane_names[current_phase]
    lane_counts = state.get("lane_counts", {})
    
    # Ensure all lanes exist
    for lane in lane_names:
        if lane not in lane_counts:
            lane_counts[lane] = 0
    
    return {
        "main_dashboard": {
            "signal_state": {
                "active_direction": active_direction,
                "state": "GREEN",
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
        except Exception:
            disconnected_clients.append(client)
    
    for client in disconnected_clients:
        if client in dashboard_clients:
            dashboard_clients.remove(client)

def run_ai_agent_in_thread():
    """Run AI agent in a separate thread with error handling"""
    global ai_agent_running
    
    def ai_worker():
        global ai_agent_running
        ai_agent_running = True
        print("Starting AI agent in background thread...")
        
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Import and run the AI agent
            from run_live_agent import run_live_inference
            loop.run_until_complete(run_live_inference())
            
        except Exception as e:
            print(f"AI agent crashed: {e}")
            ai_agent_running = False
            
            # Wait before potential restart
            time.sleep(30)
    
    # Start in daemon thread so it doesn't block server shutdown
    thread = threading.Thread(target=ai_worker, daemon=True)
    thread.start()

@app.get("/status")
async def get_status():
    """Returns the current traffic state"""
    return {
        **traffic_state,
        "server_status": "running",
        "ai_agent_thread": ai_agent_running
    }

@app.get("/debug")
async def get_debug_info():
    """Debug endpoint to check system status"""
    return {
        "traffic_state": traffic_state,
        "dashboard_clients": len(dashboard_clients),
        "ai_connected": traffic_state.get("ai_status") == "CONNECTED",
        "ai_thread_running": ai_agent_running,
        "last_update": traffic_state.get("last_update_time", 0),
        "time_since_update": time.time() - traffic_state.get("last_update_time", 0),
        "server_memory_usage": "unknown"  # Railway doesn't provide this easily
    }

@app.get("/health")
async def health_check():
    """Simple health check for Railway"""
    return {"status": "healthy", "timestamp": time.time()}

# Start AI agent in background thread (NOT asyncio task)
@app.on_event("startup")
async def startup_event():
    print("FastAPI server starting...")
    print("Starting AI agent in background thread...")
    
    # Use threading instead of asyncio to avoid blocking the server
    run_ai_agent_in_thread()

@app.get("/")
async def root():
    return {
        "message": "Traffic AI Backend running",
        "timestamp": int(time.time()),
        "ai_status": traffic_state.get("ai_status", "DISCONNECTED"),
        "dashboard_clients": len(dashboard_clients),
        "ai_thread": ai_agent_running
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    
    # Use single worker and reduced settings for Railway
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False,
        access_log=False,  # Reduce logging
        workers=1  # Single worker to save memory
    )'''
import os
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
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)




