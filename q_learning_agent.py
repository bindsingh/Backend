"""
Improved Q-Learning Agent for Traffic Signal Control

Enhanced with better state discretization, adaptive learning,
and more robust exploration strategies.
"""

import numpy as np
import pickle
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import time

class ImprovedStateDiscretizer:
    """
    Enhanced state discretization with adaptive and relative binning
    """
    
    def __init__(self, max_vehicles_per_lane: int = 20, max_time: int = 30):
        """
        Initialize improved state discretization
        
        Args:
            max_vehicles_per_lane: Maximum vehicles per lane
            max_time: Maximum time since last change
        """
        self.max_vehicles_per_lane = max_vehicles_per_lane
        self.max_time = max_time
        
        # Adaptive lane count bins based on traffic density patterns
        self.lane_bins = [0, 2, 5, 10, 15, max_vehicles_per_lane]  # 5 states per lane
        
        # Signal phase (4 discrete values)
        self.phase_bins = [0, 1, 2, 3]
        
        # More granular time bins for better temporal resolution
        self.time_bins = [0, 3, 8, 15, 25, max_time]  # 5 time states
        
        # Calculate state space dimensions
        self.num_lane_states = len(self.lane_bins) - 1  # 5 states per lane
        self.num_phase_states = len(self.phase_bins)    # 4 states
        self.num_time_states = len(self.time_bins) - 1  # 5 states
        
        self.total_states = (self.num_lane_states ** 4) * self.num_phase_states * self.num_time_states
        
        # Track state visitation for adaptive binning
        self.state_visits = defaultdict(int)
        
        print(f"🔢 Improved State discretizer initialized:")
        print(f"   Lane states per lane: {self.num_lane_states}")
        print(f"   Total discrete states: {self.total_states:,}")
    
    def discretize_value(self, value: float, bins: List[float]) -> int:
        """Convert continuous value to discrete bin index with boundary handling"""
        value = max(0, min(value, bins[-1]))  # Clamp to valid range
        
        for i in range(len(bins) - 1):
            if value <= bins[i + 1]:
                return i
        return len(bins) - 2  # Last valid bin
    
    def discretize_state(self, observation: np.ndarray, 
                        max_vehicles: int = None, max_time: int = None) -> Tuple[int, ...]:
        """
        Enhanced state discretization with relative traffic levels
        
        Args:
            observation: [lane1, lane2, lane3, lane4, phase, time] (normalized)
            max_vehicles: Override maximum vehicles per lane
            max_time: Override maximum time value
            
        Returns:
            Tuple of discrete state indices
        """
        if max_vehicles is None:
            max_vehicles = self.max_vehicles_per_lane
        if max_time is None:
            max_time = self.max_time
            
        # Denormalize the observation
        lane_counts = observation[:4] * max_vehicles
        current_phase = int(observation[4] * 3)  # 0-3
        time_since_change = observation[5] * max_time
        
        # Use relative traffic levels for better generalization
        total_traffic = np.sum(lane_counts)
        if total_traffic > 0:
            # Create relative traffic features
            relative_levels = lane_counts / total_traffic
            # Scale relative levels to absolute bins for discretization
            scaled_levels = relative_levels * max_vehicles
        else:
            scaled_levels = lane_counts
        
        # Discretize each component
        discrete_lanes = tuple(
            self.discretize_value(count, self.lane_bins) 
            for count in scaled_levels
        )
        
        discrete_phase = current_phase
        discrete_time = self.discretize_value(time_since_change, self.time_bins)
        
        # Construct discrete state
        discrete_state = discrete_lanes + (discrete_phase, discrete_time)
        
        # Track state visitation for analysis
        self.state_visits[discrete_state] += 1
        
        return discrete_state
    
    def get_state_density_info(self) -> Dict[str, Any]:
        """Analyze state visitation patterns"""
        if not self.state_visits:
            return {"total_visited": 0, "coverage": 0.0}
            
        total_visits = sum(self.state_visits.values())
        unique_states = len(self.state_visits)
        coverage = unique_states / self.total_states
        
        # Find most and least visited states
        most_visited = max(self.state_visits.items(), key=lambda x: x[1])
        visit_distribution = list(self.state_visits.values())
        
        return {
            "total_visited": unique_states,
            "coverage": coverage,
            "total_visits": total_visits,
            "avg_visits_per_state": total_visits / unique_states,
            "most_visited_state": most_visited[0],
            "most_visited_count": most_visited[1],
            "visit_variance": np.var(visit_distribution)
        }
    
    def state_to_string(self, state: Tuple[int, ...]) -> str:
        """Convert discrete state to human-readable string"""
        lane_names = ['North', 'South', 'East', 'West']
        phase_names = ['North', 'South', 'East', 'West']
        
        state_str = "Traffic: "
        for i, lane_state in enumerate(state[:4]):
            state_str += f"{lane_names[i]}={lane_state}, "
        
        state_str += f"Green={phase_names[state[4]]}, Time={state[5]}"
        return state_str

class AdaptiveQLearningAgent:
    """
    Enhanced Q-Learning Agent with adaptive learning and better exploration
    """
    
    def __init__(self, 
                 action_size: int = 4,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.999,
                 use_replay_buffer: bool = False,
                 replay_buffer_size: int = 10000,
                 adaptive_learning: bool = True,
                 use_boltzmann_exploration: bool = False,
                 temperature: float = 1.0):
        """
        Initialize enhanced Q-Learning agent
        
        Args:
            action_size: Number of possible actions (4 for traffic signals)
            learning_rate: Initial learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate per episode
            use_replay_buffer: Whether to use experience replay
            replay_buffer_size: Size of replay buffer
            adaptive_learning: Whether to use adaptive learning rate
            use_boltzmann_exploration: Use Boltzmann exploration instead of epsilon-greedy
            temperature: Temperature for Boltzmann exploration
        """
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Enhanced exploration options
        self.use_boltzmann_exploration = use_boltzmann_exploration
        self.temperature = temperature
        self.adaptive_learning = adaptive_learning
        
        # Q-table with better initialization
        self.q_table = defaultdict(lambda: np.random.normal(0, 0.01, action_size))
        
        # Enhanced state discretizer
        self.discretizer = ImprovedStateDiscretizer()
        
        # Experience replay
        self.use_replay_buffer = use_replay_buffer
        if use_replay_buffer:
            self.replay_buffer = ExperienceReplayBuffer(replay_buffer_size)
        
        # Enhanced training statistics
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'states_visited': set(),
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_history': [],
            'learning_rate_history': [],
            'q_value_changes': [],
            'td_errors': [],
            'action_counts': [0] * action_size,
            'convergence_indicators': []
        }
        
        # Adaptive learning parameters
        self.state_visit_counts = defaultdict(int)
        self.state_learning_rates = defaultdict(lambda: self.learning_rate)
        
        print(f"🤖 Enhanced Q-Learning Agent initialized:")
        print(f"   Learning rate: {learning_rate} (adaptive: {adaptive_learning})")
        print(f"   Discount factor: {discount_factor}")
        print(f"   Exploration: {'Boltzmann' if use_boltzmann_exploration else 'Epsilon-greedy'}")
        print(f"   Experience replay: {use_replay_buffer}")
    
    def choose_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Enhanced action selection with multiple exploration strategies
        
        Args:
            observation: Current state observation
            training: Whether agent is in training mode
            
        Returns:
            Selected action (0-3)
        """
        # Discretize the state
        discrete_state = self.discretizer.discretize_state(observation)
        
        # Track visited states
        if training:
            self.training_stats['states_visited'].add(discrete_state)
            self.state_visit_counts[discrete_state] += 1
        
        # Get Q-values for current state
        q_values = self.q_table[discrete_state]
        
        if not training:
            # Exploitation only during evaluation
            action = np.argmax(q_values)
        elif self.use_boltzmann_exploration:
            # Boltzmann (softmax) exploration
            action = self._boltzmann_action_selection(q_values)
        else:
            # Enhanced epsilon-greedy with UCB-style bonus
            action = self._enhanced_epsilon_greedy(discrete_state, q_values, training)
        
        # Track action distribution
        if training:
            self.training_stats['action_counts'][action] += 1
        
        return action
    
    def _boltzmann_action_selection(self, q_values: np.ndarray) -> int:
        """Boltzmann (softmax) action selection"""
        if self.temperature <= 0:
            return np.argmax(q_values)
        
        # Softmax with temperature
        exp_values = np.exp(q_values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        
        # Sample from probability distribution
        return np.random.choice(self.action_size, p=probabilities)
    
    def _enhanced_epsilon_greedy(self, state: Tuple, q_values: np.ndarray, training: bool) -> int:
        """Enhanced epsilon-greedy with exploration bonus"""
        if training and np.random.random() < self.epsilon:
            # Exploration with UCB-style bias toward less-visited actions
            visit_count = self.state_visit_counts[state]
            if visit_count > 10:  # Only apply after some visits
                # Calculate exploration bonus (UCB-like)
                exploration_bonus = np.sqrt(2 * np.log(visit_count) / 
                                          (np.array(self.training_stats['action_counts']) + 1))
                adjusted_values = q_values + 0.1 * exploration_bonus
                return np.argmax(adjusted_values)
            else:
                # Pure random exploration for new states
                return np.random.randint(0, self.action_size)
        else:
            # Exploitation
            return np.argmax(q_values)
    
    def update_q_table(self, state: Tuple, action: int, reward: float, 
                      next_state: Tuple, done: bool):
        """
        Enhanced Q-table update with adaptive learning rate
        
        Args:
            state: Current discrete state
            action: Action taken
            reward: Reward received
            next_state: Next discrete state
            done: Whether episode is finished
        """
        # Get current Q-value
        current_q = self.q_table[state][action]
        
        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            next_q_values = self.q_table[next_state]
            target_q = reward + self.discount_factor * np.max(next_q_values)
        
        # Calculate TD error
        td_error = target_q - current_q
        
        # Adaptive learning rate based on state visitation
        if self.adaptive_learning:
            visit_count = self.state_visit_counts[state]
            # Decrease learning rate for frequently visited states
            adaptive_lr = self.initial_learning_rate / (1 + visit_count * 0.001)
            adaptive_lr = max(adaptive_lr, 0.01)  # Minimum learning rate
        else:
            adaptive_lr = self.learning_rate
        
        # Q-learning update with adaptive learning rate
        new_q = current_q + adaptive_lr * td_error
        
        # Update Q-table
        old_q = self.q_table[state][action]
        self.q_table[state][action] = new_q
        
        # Track learning metrics
        q_change = abs(new_q - old_q)
        self.training_stats['q_value_changes'].append(q_change)
        self.training_stats['td_errors'].append(abs(td_error))
        
        return td_error, adaptive_lr
    
    def learn(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Enhanced learning method with better experience handling
        
        Args:
            state: Previous state observation
            action: Action taken
            reward: Reward received
            next_state: New state observation
            done: Whether episode is finished
        """
        # Discretize states
        discrete_state = self.discretizer.discretize_state(state)
        discrete_next_state = self.discretizer.discretize_state(next_state)
        
        if self.use_replay_buffer:
            # Add to replay buffer
            self.replay_buffer.add(discrete_state, action, reward, 
                                 discrete_next_state, done)
            
            # Learn from replay buffer with multiple updates
            if self.replay_buffer.size() >= 64:  # Larger minimum batch
                batch_size = min(64, self.replay_buffer.size())
                batch = self.replay_buffer.sample(batch_size)
                
                total_td_error = 0
                for exp_state, exp_action, exp_reward, exp_next_state, exp_done in batch:
                    td_error, _ = self.update_q_table(exp_state, exp_action, exp_reward, 
                                                    exp_next_state, exp_done)
                    total_td_error += abs(td_error)
                
                # Track average TD error for convergence monitoring
                avg_td_error = total_td_error / len(batch)
                self.training_stats['convergence_indicators'].append(avg_td_error)
        else:
            # Direct Q-table update
            td_error, adaptive_lr = self.update_q_table(discrete_state, action, reward, 
                                                       discrete_next_state, done)
            self.training_stats['learning_rate_history'].append(adaptive_lr)
        
        # Update step counter
        self.training_stats['total_steps'] += 1
    
    def decay_epsilon(self):
        """Enhanced epsilon decay with minimum threshold"""
        if self.use_boltzmann_exploration:
            # Decay temperature for Boltzmann exploration
            self.temperature = max(0.1, self.temperature * 0.999)
        else:
            # Standard epsilon decay
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.training_stats['epsilon_history'].append(self.epsilon)
    
    def episode_finished(self, episode_reward: float, episode_length: int):
        """
        Enhanced episode completion handling
        
        Args:
            episode_reward: Total reward for the episode
            episode_length: Number of steps in episode
        """
        self.training_stats['episodes'] += 1
        self.training_stats['episode_rewards'].append(episode_reward)
        self.training_stats['episode_lengths'].append(episode_length)
        
        # Decay exploration parameter
        self.decay_epsilon()
        
        # Calculate learning progress indicators
        if len(self.training_stats['episode_rewards']) >= 10:
            recent_rewards = self.training_stats['episode_rewards'][-10:]
            reward_variance = np.var(recent_rewards)
            reward_trend = np.mean(recent_rewards[-5:]) - np.mean(recent_rewards[:5])
            
            self.training_stats['convergence_indicators'].extend([
                reward_variance, reward_trend
            ])
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        stats = self.training_stats.copy()
        
        # Basic statistics
        if len(stats['episode_rewards']) > 0:
            stats['avg_reward_last_100'] = np.mean(stats['episode_rewards'][-100:])
            stats['avg_length_last_100'] = np.mean(stats['episode_lengths'][-100:])
            stats['best_reward'] = np.max(stats['episode_rewards'])
            stats['worst_reward'] = np.min(stats['episode_rewards'])
            
            # Improvement metrics
            if len(stats['episode_rewards']) >= 200:
                early_avg = np.mean(stats['episode_rewards'][:100])
                recent_avg = np.mean(stats['episode_rewards'][-100:])
                stats['total_improvement'] = recent_avg - early_avg
        
        # Q-table statistics
        stats['q_table_size'] = len(self.q_table)
        stats['states_explored'] = len(stats['states_visited'])
        stats['exploration_coverage'] = len(stats['states_visited']) / self.discretizer.total_states
        stats['current_epsilon'] = self.epsilon
        
        # Learning stability metrics
        if len(stats['q_value_changes']) > 100:
            stats['recent_q_change'] = np.mean(stats['q_value_changes'][-100:])
            stats['early_q_change'] = np.mean(stats['q_value_changes'][:100])
            stats['q_change_ratio'] = stats['recent_q_change'] / max(stats['early_q_change'], 0.001)
        
        if len(stats['td_errors']) > 100:
            stats['avg_td_error'] = np.mean(stats['td_errors'][-100:])
            stats['td_error_trend'] = np.polyfit(range(len(stats['td_errors'][-100:])), 
                                               stats['td_errors'][-100:], 1)[0]
        
        # Action distribution analysis
        total_actions = sum(stats['action_counts'])
        if total_actions > 0:
            stats['action_distribution'] = [count/total_actions for count in stats['action_counts']]
            stats['action_entropy'] = -sum([p * np.log(p + 1e-10) for p in stats['action_distribution']])
        
        # State discretization insights
        stats['state_discretization_info'] = self.discretizer.get_state_density_info()
        
        return stats
    
    def analyze_learning_progress(self) -> Dict[str, str]:
        """Analyze learning progress and provide insights"""
        stats = self.get_enhanced_statistics()
        insights = {}
        
        # Convergence analysis
        if len(self.training_stats['convergence_indicators']) > 50:
            recent_indicators = self.training_stats['convergence_indicators'][-50:]
            if np.std(recent_indicators) < 0.1:
                insights['convergence'] = "Learning appears to be converging"
            elif np.mean(recent_indicators) > 1.0:
                insights['convergence'] = "Learning still unstable, may need more episodes"
            else:
                insights['convergence'] = "Learning progressing normally"
        
        # Exploration analysis
        if stats.get('action_entropy', 0) < 1.0:
            insights['exploration'] = "Low action diversity - may be over-exploiting"
        elif stats.get('action_entropy', 0) > 1.3:
            insights['exploration'] = "High action diversity - still exploring well"
        else:
            insights['exploration'] = "Balanced exploration-exploitation"
        
        # Performance trend
        if 'total_improvement' in stats and stats['total_improvement'] > 10:
            insights['performance'] = "Strong improvement trend"
        elif 'total_improvement' in stats and stats['total_improvement'] < -10:
            insights['performance'] = "Performance declining - check hyperparameters"
        else:
            insights['performance'] = "Stable performance"
        
        return insights
    
    def save_model(self, filepath: str):
        """Save Q-table and agent parameters"""
        # Convert defaultdict to regular dict for saving
        q_table_dict = dict(self.q_table)
        
        # Save Q-table
        with open(f"{filepath}_qtable.pkl", 'wb') as f:
            pickle.dump(q_table_dict, f)
        
        # Save agent parameters and statistics
        agent_data = {
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'training_stats': self.training_stats,
            'use_replay_buffer': self.use_replay_buffer
        }
        
        with open(f"{filepath}_params.json", 'w') as f:
            # Convert sets to lists for JSON serialization
            stats_copy = agent_data['training_stats'].copy()
            stats_copy['states_visited'] = list(stats_copy['states_visited'])
            agent_data['training_stats'] = stats_copy
            json.dump(agent_data, f, indent=2)
        
        print(f"💾 Model saved to {filepath}_qtable.pkl and {filepath}_params.json")

    

    def load_model(self, filepath: str):
        """Load a pre-trained Q-table and agent parameters"""
        q_table_path = f"{filepath}_qtable.pkl"
        if os.path.exists(q_table_path):
            with open(q_table_path, 'rb') as f:
                # Load the dictionary and convert it back to a defaultdict
                q_table_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.random.normal(0, 0.01, self.action_size), q_table_dict)
            print(f"✅ Q-table loaded successfully from {q_table_path}")
        else:
            print(f"⚠️ Warning: No Q-table found at {q_table_path}. Starting with a new, untrained agent.")
# Enhanced Experience Replay Buffer
class ExperienceReplayBuffer:
    """
    Enhanced experience replay buffer with prioritization
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        """
        Initialize enhanced replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent
        """
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.max_priority = 1.0
    
    def add(self, state: Tuple, action: int, reward: float, 
            next_state: Tuple, done: bool, td_error: float = None):
        """Add experience with priority"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
        # Set priority based on TD error or max priority for new experiences
        priority = self.max_priority if td_error is None else abs(td_error) + 1e-6
        self.priorities.append(priority ** self.alpha)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample batch with priority weighting"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / np.sum(priorities)
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, 
                                 replace=False, p=probabilities)
        
        return [self.buffer[i] for i in indices]
    
    def size(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)

def test_enhanced_agent():
    """Test function for enhanced Q-Learning agent"""
    print("🧪 Testing Enhanced Q-Learning Agent...\n")
    
    # Create enhanced agent
    agent = AdaptiveQLearningAgent(
        learning_rate=0.15,
        adaptive_learning=True,
        use_boltzmann_exploration=False,
        epsilon_decay=0.995
    )
    
    # Test enhanced state discretization
    print("1️⃣ Testing enhanced state discretization...")
    test_observation = np.array([0.25, 0.5, 0.8, 0.1, 0.33, 0.2])
    discrete_state = agent.discretizer.discretize_state(test_observation)
    print(f"   Observation: {test_observation}")
    print(f"   Discrete state: {discrete_state}")
    print(f"   State description: {agent.discretizer.state_to_string(discrete_state)}")
    
    # Test enhanced action selection
    print("\n2️⃣ Testing enhanced action selection...")
    for i in range(5):
        action = agent.choose_action(test_observation)
        print(f"   Action {i+1}: {action}")
    
    # Test enhanced learning
    print("\n3️⃣ Testing enhanced learning...")
    state = test_observation
    action = 2
    reward = -15.5
    next_state = np.array([0.2, 0.45, 0.75, 0.15, 0.5, 0.3])
    done = False
    
    agent.learn(state, action, reward, next_state, done)
    print(f"   Learned from experience: action={action}, reward={reward}")
    
    # Test enhanced statistics
    print("\n4️⃣ Testing enhanced statistics...")
    agent.episode_finished(-45.2, 150)
    stats = agent.get_enhanced_statistics()
    insights = agent.analyze_learning_progress()
    
    print(f"   Episodes: {stats['episodes']}")
    print(f"   States visited: {stats['states_explored']}")
    print(f"   Current epsilon: {stats['current_epsilon']:.4f}")
    print(f"   Learning insights: {insights}")
    
    print("\n✅ Enhanced Q-Learning Agent test completed!")

if __name__ == "__main__":
    test_enhanced_agent()