# in main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from models import DecisionPayload
from state_manager import traffic_state # Import the shared state
from business_logic import apply_ai_decision # You will create this next

app = FastAPI()

@app.websocket("/ws/ai")
async def websocket_ai_endpoint(websocket: WebSocket):
    await websocket.accept()
    traffic_state["ai_status"] = "CONNECTED"
    print("✅ AI Agent Connected")
    try:
        while True:
            # 1. Receive data from the AI agent
            data = await websocket.receive_json()

            # 2. Validate the data using your Pydantic model
            try:
                validated_data = DecisionPayload(**data)
                print(f"Received decision: {validated_data.decision['reason']}")

                # 3. Apply the decision using your business logic
                apply_ai_decision(validated_data)

            except ValidationError as e:
                # If data is bad, log it and ignore it
                print(f"❌ Invalid data received from AI: {e}")

    except WebSocketDisconnect:
        traffic_state["ai_status"] = "DISCONNECTED"
        print("🔴 AI Agent Disconnected")


        # in main.py, add these below your websocket endpoint

@app.get("/status")
async def get_status():
    """Returns the current, live state of the intersection."""
    return traffic_state

@app.get("/metrics")
async def get_metrics():
    """Returns simple performance metrics. (This can be expanded later)"""
    # This is a placeholder for more complex logic you might build.
    total_vehicles_waiting = sum(traffic_state["lane_counts"].values())
    return {
        "ai_status": traffic_state["ai_status"],
        "total_vehicles_waiting": total_vehicles_waiting,
        "last_decision_reason": traffic_state["last_decision_reason"]
    }