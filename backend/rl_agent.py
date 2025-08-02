import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import logging

logger = logging.getLogger(__name__)

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = config.get("learning_rate", 0.001)
        self.gamma = 0.99
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.batch_size = config.get("batch_size", 64)
        self.memory_size = config.get("memory_size", 10000)
        self.target_update = config.get("target_update", 10)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        self.memory = ReplayBuffer(self.memory_size)
        self.step_count = 0
        self.update_target_network()
        
        logger.info(f"DQN Agent initialized: state_size={state_size}, action_size={action_size}")
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.update_target_network()
    
    def is_trained(self):
        return len(self.memory) > 1000
    
    def get_optimal_policy(self):
        return {
            "model_state": self.q_network.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.step_count
        }
    
    def reset(self):
        self.epsilon = 1.0
        self.step_count = 0
        self.memory = ReplayBuffer(self.memory_size)

class AutomationEnvironment:
    def __init__(self, config):
        self.config = config
        self.state_size = 12  # System states: positions, velocities, temperatures, etc.
        self.action_size = 6  # Discrete actions: speed adjustments, process changes
        
        self.cycle_time_target = config.get("cycle_time_target", 30.0)
        self.throughput_target = config.get("throughput_target", 120.0)
        self.safety_threshold = config.get("safety_threshold", 0.95)
        self.energy_efficiency_target = config.get("energy_efficiency_target", 0.85)
        
        self.reset()
        logger.info("Automation Environment initialized")
    
    def reset(self):
        # Initialize system state: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, temp, pressure, vibration, load, quality, energy]
        self.state = np.random.normal(0, 0.1, self.state_size)
        self.state = np.clip(self.state, -1, 1)
        
        self.cycle_time = 0
        self.steps_in_episode = 0
        self.total_energy = 0
        self.safety_violations = 0
        self.production_count = 0
        
        return self.state.copy()
    
    def step(self, action):
        self.steps_in_episode += 1
        
        # Apply action effects
        action_effects = self._apply_action(action)
        
        # Update system dynamics
        self._update_dynamics(action_effects)
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = self._is_episode_done()
        
        return self.state.copy(), reward, done
    
    def _apply_action(self, action):
        # Map discrete actions to continuous control values
        action_map = {
            0: {"speed": 0.8, "pressure": 0.9, "temp": 0.85},  # Conservative
            1: {"speed": 1.0, "pressure": 1.0, "temp": 1.0},   # Normal
            2: {"speed": 1.2, "pressure": 1.1, "temp": 1.1},   # Aggressive
            3: {"speed": 0.6, "pressure": 0.8, "temp": 0.8},   # Energy saving
            4: {"speed": 1.1, "pressure": 1.05, "temp": 0.95}, # Quality focus
            5: {"speed": 0.9, "pressure": 0.95, "temp": 0.9}   # Balanced
        }
        return action_map.get(action, action_map[1])
    
    def _update_dynamics(self, action_effects):
        # Simulate industrial system dynamics
        speed_factor = action_effects["speed"]
        pressure_factor = action_effects["pressure"]
        temp_factor = action_effects["temp"]
        
        # Update positions (indices 0-2)
        self.state[0:3] += self.state[3:6] * 0.1 * speed_factor
        
        # Update velocities (indices 3-5) with some noise
        self.state[3:6] += np.random.normal(0, 0.02, 3) * speed_factor
        
        # Update temperature (index 6)
        self.state[6] = 0.9 * self.state[6] + 0.1 * temp_factor + np.random.normal(0, 0.01)
        
        # Update pressure (index 7)
        self.state[7] = 0.95 * self.state[7] + 0.05 * pressure_factor + np.random.normal(0, 0.01)
        
        # Update vibration (index 8)
        self.state[8] = np.random.normal(0, 0.1) * speed_factor
        
        # Update load (index 9)
        self.state[9] = 0.8 * speed_factor + np.random.normal(0, 0.05)
        
        # Update quality score (index 10)
        quality_impact = 1.0 - abs(speed_factor - 1.0) * 0.3
        self.state[10] = 0.95 * quality_impact + np.random.normal(0, 0.02)
        
        # Update energy consumption (index 11)
        energy = speed_factor * pressure_factor * temp_factor
        self.state[11] = energy + np.random.normal(0, 0.01)
        self.total_energy += energy
        
        # Clip states to valid ranges
        self.state = np.clip(self.state, -2, 2)
        
        # Update cycle metrics
        self.cycle_time += 1.0 / speed_factor
        if self.cycle_time >= self.cycle_time_target:
            self.production_count += 1
            self.cycle_time = 0
    
    def _calculate_reward(self, action):
        reward = 0
        
        # Production efficiency reward
        if self.production_count > 0:
            throughput_ratio = self.production_count / (self.steps_in_episode / 100.0)
            reward += (throughput_ratio / self.throughput_target) * 10
        
        # Quality reward
        quality_score = (self.state[10] + 1) / 2  # Normalize to 0-1
        reward += quality_score * 5
        
        # Energy efficiency reward
        if self.steps_in_episode > 0:
            energy_efficiency = 1.0 / (1.0 + self.total_energy / self.steps_in_episode)
            reward += energy_efficiency * 3
        
        # Safety penalty
        if abs(self.state[6]) > 1.5 or abs(self.state[7]) > 1.5:  # High temp/pressure
            reward -= 5
            self.safety_violations += 1
        
        # Vibration penalty
        if abs(self.state[8]) > 1.0:
            reward -= 2
        
        # Cycle time bonus
        if self.cycle_time <= self.cycle_time_target:
            reward += 2
        
        return reward
    
    def _is_episode_done(self):
        # Episode ends after 200 steps or if safety violations exceed threshold
        return self.steps_in_episode >= 200 or self.safety_violations > 5
    
    def is_done(self):
        return self._is_episode_done()
    
    def get_state_size(self):
        return self.state_size
    
    def get_action_size(self):
        return self.action_size
    
    def get_performance_metrics(self):
        safety_score = max(0, 1.0 - self.safety_violations / 10.0)
        energy_efficiency = 1.0 / (1.0 + self.total_energy / max(1, self.steps_in_episode))
        quality_score = (self.state[10] + 1) / 2
        throughput = self.production_count / max(1, self.steps_in_episode / 100.0)
        
        # Overall Equipment Effectiveness (OEE)
        availability = 1.0 - (self.safety_violations / max(1, self.steps_in_episode))
        performance = min(1.0, throughput / self.throughput_target)
        oee = availability * performance * quality_score
        
        return {
            "cycle_time": self.cycle_time,
            "throughput": throughput,
            "safety_score": safety_score,
            "energy_efficiency": energy_efficiency,
            "quality_score": quality_score,
            "oee": oee,
            "production_count": self.production_count,
            "safety_violations": self.safety_violations
        }
    
    def get_config(self):
        return self.config.copy()