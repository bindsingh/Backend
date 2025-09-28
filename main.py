import os
import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
import json

from state_manager import traffic_state
from run_live_agent import run_live_inference

app = FastAPI()

# Track connected clients
dashboard_clients: List[WebSocket] = []

@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket endpoint for frontend dashboards"""
    await websocket.accept()
    dashboard_clients.append(websocket)
    
    try:
        # Send initial state
        frontend_payload = transform_to_frontend_format(traffic_state)
        await websocket.send_json(frontend_payload)
        
        while True:
            # Send updates every second
            frontend_payload = transform_to_frontend_format(traffic_state)
            await websocket.send_json(frontend_payload)
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        print("Dashboard client disconnected")
    finally:
        if websocket in dashboard_clients:
            dashboard_clients.remove(websocket)

@app.websocket("/ws/ai")
async def websocket_ai_agent(websocket: WebSocket):
    """WebSocket endpoint for AI agent to send data"""
    await websocket.accept()
    traffic_state["ai_status"] = "CONNECTED"
    
    try:
        while True:
            # Receive data from AI agent
            data = await websocket.receive_text()
            ai_data = json.loads(data)
            
            # Update traffic state
            update_traffic_state(ai_data)
            
            # Broadcast to all dashboard clients
            await broadcast_to_dashboards()
            
    except WebSocketDisconnect:
        print("AI agent disconnected")
        traffic_state["ai_status"] = "DISCONNECTED"

def transform_to_frontend_format(state):
    """Transform backend state to frontend expected format"""
    lane_names = ["Northbound", "Southbound", "Eastbound", "Westbound"]
    active_direction = lane_names[state.get("current_phase", 0)]
    
    return {
        "main_dashboard": {
            "signal_state": {
                "active_direction": active_direction,
                "state": "GREEN",
                "timer": 10
            },
            "vehicle_counters": state.get("lane_counts", {}),
            "total_vehicles": sum(state.get("lane_counts", {}).values())
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
    if "lane_counts" in ai_data:
        traffic_state["lane_counts"] = ai_data["lane_counts"]
    
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
    
    traffic_state["last_update_time"] = time.time()

async def broadcast_to_dashboards():
    """Broadcast updated data to all dashboard clients"""
    if not dashboard_clients:
        return
        
    payload = transform_to_frontend_format(traffic_state)
    
    for client in dashboard_clients[:]:  # Copy list to avoid modification during iteration
        try:
            await client.send_json(payload)
        except Exception:
            dashboard_clients.remove(client)

@app.get("/status")
async def get_status():
    return traffic_state

@app.on_event("startup")
async def startup_event():
    print("Starting AI agent loop...")
    asyncio.create_task(run_live_inference())

@app.get("/")
async def root():
    return {"message": "Traffic AI Backend running", "timestamp": int(time.time())}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
