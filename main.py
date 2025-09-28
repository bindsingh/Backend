import os
import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from typing import List

from models import DecisionPayload
from state_manager import traffic_state
from business_logic import apply_ai_decision

# Import the AI loop from run_live_agent
from run_live_agent import run_live_inference

app = FastAPI()

# --- Track connected dashboard clients ---
dashboard_clients: List[WebSocket] = []

# --- WebSocket endpoint for AI agent to push data ---
@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Safely handle current_phase (fallback to 0 if None)
            phase = traffic_state.get("current_phase")
            if phase is None:
                phase = 0

            # Safely handle AI status
            ai_status = traffic_state.get("ai_status", "DISCONNECTED")

            # Build frontend payload
            frontend_payload = {
                "main_dashboard": {
                    "signal_state": {
                        "active_direction": f"lane_{phase+1}",
                        "state": "GREEN",  # TODO: update with your actual signal logic
                        "timer": 10        # TODO: replace with actual timer value
                    },
                    "vehicle_counters": traffic_state.get("lane_counts", {}),
                    "total_vehicles": sum(traffic_state.get("lane_counts", {}).values())
                },
                "performance_metrics": [
                    {
                        "title": "AI Status",
                        "value": ai_status,
                        "status": "GOOD" if ai_status == "CONNECTED" else "POOR",
                        "details": f"Last decision: {traffic_state.get('last_decision_reason', 'N/A')}"
                    },
                    {
                        "title": "Pedestrians",
                        "value": str(traffic_state.get("pedestrian_count", 0)),
                        "status": "AVERAGE",
                        "details": "Number of pedestrians detected"
                    }
                ],
                "emergency_mode": {
                    "priority_direction": "None",
                    "delayed_vehicles": 0,
                    "total_vehicles": sum(traffic_state.get("lane_counts", {}).values())
                }
            }

            await websocket.send_json(frontend_payload)
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("📴 Dashboard client disconnected")


# --- WebSocket endpoint for frontend dashboards ---

@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            frontend_payload = {
                "main_dashboard": {
                    "signal_state": {
                        "active_direction": f"lane_{traffic_state['current_phase']+1}",
                        "state": "GREEN",  # TODO: replace with actual light state
                        "timer": 10        # TODO: replace with your real timer
                    },
                    "vehicle_counters": traffic_state["lane_counts"],
                    "total_vehicles": sum(traffic_state["lane_counts"].values())
                },
                "performance_metrics": [
                    {
                        "title": "AI Status",
                        "value": traffic_state["ai_status"],
                        "status": "GOOD" if traffic_state["ai_status"] == "CONNECTED" else "POOR",
                        "details": f"Last decision: {traffic_state['last_decision_reason']}"
                    },
                    {
                        "title": "Pedestrians",
                        "value": str(traffic_state["pedestrian_count"]),
                        "status": "AVERAGE",
                        "details": "Number of pedestrians detected"
                    }
                ],
                "emergency_mode": {
                    "priority_direction": "None",
                    "delayed_vehicles": 0,
                    "total_vehicles": sum(traffic_state["lane_counts"].values())
                }
            }

            await websocket.send_json(frontend_payload)
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("Dashboard client disconnected")



    # Send initial state immediately
    try:
        await websocket.send_json(traffic_state)
    except Exception:
        print("⚠️ Failed to send initial state")

    try:
        while True:
            # keep the connection alive with a small ping
            await asyncio.sleep(5)
            if websocket.application_state != websocket.application_state.CONNECTED:
                break
    except WebSocketDisconnect:
        print("🔌 Dashboard client disconnected")
    finally:
        if websocket in dashboard_clients:
            dashboard_clients.remove(websocket)




async def broadcast_to_dashboard(data):
    print(f"📤 Broadcasting to {len(dashboard_clients)} clients")
    for ws in dashboard_clients[:]:
        try:
            await ws.send_json(data)
        except Exception:
            dashboard_clients.remove(ws)



# --- REST API Endpoints ---
@app.get("/status")
async def get_status():
    """Returns the current, live state of the intersection."""
    return traffic_state

@app.get("/metrics")
async def get_metrics():
    """Returns simple performance metrics. (This can be expanded later)."""
    total_vehicles_waiting = sum(traffic_state["lane_counts"].values())
    return {
        "ai_status": traffic_state["ai_status"],
        "total_vehicles_waiting": total_vehicles_waiting,
        "last_decision_reason": traffic_state["last_decision_reason"],
    }


# --- Background Startup Task: Run AI Agent ---
@app.on_event("startup")
async def startup_event():
    print("🚦 Starting AI agent loop (Railway or Local)...")
    asyncio.create_task(run_live_inference())

   




# --- Optional root endpoint ---
@app.get("/")
async def root():
    return {"message": "Traffic AI Backend running", "timestamp": int(time.time())}


# --- Proper Railway Entry Point ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Railway injects PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)














