import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import heapq

class PriorityLevel(Enum):
    """Priority levels for lane service"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    EMERGENCY = 3

@dataclass
class EfficiencyMetrics:
    """Traffic flow efficiency metrics"""
    throughput: float = 0.0
    average_delay: float = 0.0
    queue_length: float = 0.0
    fairness_index: float = 0.0
    starvation_incidents: int = 0

class OptimizationEngine:
    """Main optimization engine with simplified functionality"""
    
    def __init__(self, starvation_threshold: float = 120.0):
        self.starvation_threshold = starvation_threshold
        self.last_service_times = {}
        self.optimization_history = deque(maxlen=1000)
        print("Optimization Engine initialized")
    
    def optimize_action(self, lane_counts: Dict[str, int], current_phase: str, 
                       agent_recommendation: int, context: Dict[str, Any]) -> Tuple[int, str, Dict[str, Any]]:
        """Optimize action selection"""
        
        # Emergency override
        if context.get('emergency_active', False):
            emergency_lane = context.get('emergency_lane')
            if emergency_lane:
                phase_mapping = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
                if emergency_lane in phase_mapping:
                    action = phase_mapping[emergency_lane]
                    return action, f"Emergency vehicle override for {emergency_lane}", {'override_type': 'emergency'}
        
        # Check starvation
        current_time = time.time()
        for lane_name, count in lane_counts.items():
            if count > 0:
                last_service = self.last_service_times.get(lane_name, current_time)
                wait_time = current_time - last_service
                
                if wait_time > self.starvation_threshold:
                    phase_mapping = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
                    if lane_name in phase_mapping:
                        action = phase_mapping[lane_name]
                        return action, f"Starvation override for {lane_name}", {'override_type': 'starvation'}
        
        # Default: use agent recommendation
        return agent_recommendation, "Agent recommendation", {'override_type': 'none'}
    
    def update_performance(self, action: int, reward: float, was_exploration: bool, context: Dict[str, Any]):
        """Update performance tracking"""
        decision = {
            'timestamp': time.time(),
            'action': action,
            'reward': reward,
            'was_exploration': was_exploration
        }
        self.optimization_history.append(decision)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        recent_decisions = list(self.optimization_history)[-50:]
        
        if recent_decisions:
            avg_reward = np.mean([d['reward'] for d in recent_decisions])
            exploration_rate = len([d for d in recent_decisions if d['was_exploration']]) / len(recent_decisions)
        else:
            avg_reward = 0.0
            exploration_rate = 0.0
        
        return {
            'average_reward': f"{avg_reward:.1f}",
            'exploration_rate': f"{exploration_rate:.3f}",
            'total_decisions': len(self.optimization_history)
        }

def test_optimization_engine():
    """Test the optimization engine"""
    print("Testing Optimization Engine...")
    
    # Create engine
    engine = OptimizationEngine(starvation_threshold=60.0)
    
    # Test 1: Normal operation
    print("\n1. Testing normal optimization:")
    lane_counts = {'north': 8, 'south': 3, 'east': 12, 'west': 5}
    context = {'emergency_active': False, 'traffic_level': 'normal'}
    
    action, reason, info = engine.optimize_action(lane_counts, 'north', 2, context)
    print(f"Action: {action}, Reason: {reason}")
    
    # Test 2: Emergency override
    print("\n2. Testing emergency override:")
    emergency_context = {'emergency_active': True, 'emergency_lane': 'south'}
    
    action, reason, info = engine.optimize_action(lane_counts, 'north', 2, emergency_context)
    print(f"Action: {action}, Reason: {reason}")
    
    # Test 3: Starvation detection
    print("\n3. Testing starvation detection:")
    engine.last_service_times['west'] = time.time() - 120  # 2 minutes ago
    
    action, reason, info = engine.optimize_action(lane_counts, 'north', 2, context)
    print(f"Action: {action}, Reason: {reason}")
    
    # Test 4: Performance tracking
    print("\n4. Testing performance tracking:")
    for i in range(5):
        reward = np.random.normal(-15, 5)
        engine.update_performance(i % 4, reward, i % 3 == 0, context)
    
    stats = engine.get_optimization_stats()
    print("Statistics:", stats)
    
    print("\nOptimization Engine test completed!")

if __name__ == "__main__":
    test_optimization_engine()