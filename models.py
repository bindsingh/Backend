# in models.py
from pydantic import BaseModel, Field
from typing import Dict

class DecisionPayload(BaseModel):
    timestamp: float
    lane_counts: Dict[str, int]
    pedestrian_count: int
    decision: Dict[str, str | int]

    # Example of the 'decision' dict:
    # {"action_code": 1, "reason": "AI Agent Decision: Green for lane_2"}