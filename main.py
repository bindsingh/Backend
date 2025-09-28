import os
import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from models import DecisionPayload
from state_manager import traffic_state
from business_logic import apply_ai_decision

# Import the AI loop from run_live_agent
from run_live_agent import run_live_inference

app = FastAPI()

# --- WebSocket endpoint for AI agent to push data ---
@app.websocket("/ws/ai")
async def websocket_ai_endpoint(websocket: WebSocket):
    await websocket.accept()
    traffic_state["ai_status"] = "CONNECTED"
    print("✅ AI Agent Connected")

    try:
        while True:
            # 1. Receive data from the AI agent
            data = await websocket.receive_json()

            # 2. Validate and apply decision
            try:
                validated_data = DecisionPayload(**data)
                print(f"📊 Received decision: {validated_data.decision['reason']}")

                apply_ai_decision(validated_data)  # update shared state

            except ValidationError as e:
                print(f"❌ Invalid data received from AI: {e}")

    except WebSocketDisconnect:
        traffic_state["ai_status"] = "DISCONNECTED"
        print("🔴 AI Agent Disconnected")


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
#@app.on_event("startup")
#async def startup_event():
 #   """
  #  When FastAPI boots, also start the AI agent loop in the background.
   # This ensures Railway runs both backend and inference together.
    #"""
    #print("🚦 Starting AI agent loop...")
    #asyncio.create_task(run_live_inference())
@app.on_event("startup")
async def startup_event():
    if not os.environ.get("RAILWAY_ENVIRONMENT"):
        print("🚦 Starting AI agent loop locally...")
        asyncio.create_task(run_live_inference())
    else:
        print("⚠️ Skipping run_live_inference on Railway (headless mode)")


# --- Optional root endpoint ---
@app.get("/")
async def root():
    return {"message": "Traffic AI Backend running", "timestamp": int(time.time())}


# --- Proper Railway Entry Point ---
# Commenting this in ONLY for standalone run (Docker CMD: python main.py)
# On Railway this ensures the PORT is picked from env correctly.
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

