#!/usr/bin/env python3
"""
WMG Automation Systems Group - Reinforcement Learning Backend
PyTorch-based RL implementation for automated PLC code generation
Compatible with VueOne engineering ecosystem
"""

import asyncio
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import websockets
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import threading
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data structures for automation environment
@dataclass
class AutomationState:
    """Represents the current state of the automation system"""
    conveyor_position: float
    station1_busy: bool
    station2_busy: bool
    workpiece_positions: List[float]
    cycle_time: float
    throughput: float
    safety_violations: int
    energy_consumption: float
    queue_length: int
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to PyTorch tensor for RL agent"""
        state_vector = [
            self.conveyor_position / 100.0,  # Normalize to [0,1]
            float(self.station1_busy),
            float(self.station2_busy),
            len(self.workpiece_positions) / 10.0,  # Normalize queue length
            self.cycle_time / 100.0,  # Normalize cycle time
            self.throughput / 200.0,  # Normalize throughput
            min(self.safety_violations / 10.0, 1.0),  # Cap safety violations
            self.energy_consumption / 1000.0,  # Normalize energy
            self.queue_length / 20.0  # Normalize queue length
        ]
        return torch.FloatTensor(state_vector)

@dataclass
class AutomationAction:
    """Represents actions the RL agent can take"""
    action_id: int
    station1_duration: float
    station2_duration: float
    conveyor_speed: float
    buffer_size: int
    
    @classmethod
    def from_action_id(cls, action_id: int) -> 'AutomationAction':
        """Convert discrete action ID to automation parameters"""
        # Map discrete actions to continuous parameters
        station1_durations = [2.0, 3.0, 4.0, 5.0]
        station2_durations = [1.5, 2.5, 3.5, 4.5]
        conveyor_speeds = [0.5, 1.0, 1.5, 2.0]
        buffer_sizes = [2, 4, 6, 8]
        
        s1_idx = action_id % 4
        s2_idx = (action_id // 4) % 4
        conv_idx = (action_id // 16) % 4
        buf_idx = (action_id // 64) % 4
        
        return cls(
            action_id=action_id,
            station1_duration=station1_durations[s1_idx],
            station2_duration=station2_durations[s2_idx],
            conveyor_speed=conveyor_speeds[conv_idx],
            buffer_size=buffer_sizes[buf_idx]
        )

class AutomationEnvironment:
    """
    Automation system environment for reinforcement learning
    Simulates a generic modular automation system with safety constraints
    """
    
    def __init__(self):
        self.state_size = 9
        self.action_size = 256  # 4^4 discrete actions
        self.reset()
        
    def reset(self) -> AutomationState:
        """Reset environment to initial state"""
        self.current_state = AutomationState(
            conveyor_position=0.0,
            station1_busy=False,
            station2_busy=False,
            workpiece_positions=[],
            cycle_time=45.0,
            throughput=80.0,
            safety_violations=0,
            energy_consumption=150.0,
            queue_length=0
        )
        self.step_count = 0
        self.total_reward = 0.0
        return self.current_state
    
    def step(self, action: AutomationAction) -> Tuple[AutomationState, float, bool, Dict]:
        """Execute action and return new state, reward, done, info"""
        self.step_count += 1
        
        # Simulate automation system dynamics
        self._update_system_state(action)
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self.total_reward += reward
        
        # Check termination conditions
        done = self.step_count >= 200 or self.current_state.safety_violations > 5
        
        # Additional info
        info = {
            'step': self.step_count,
            'total_reward': self.total_reward,
            'oee': self._calculate_oee(),
            'efficiency': self._calculate_efficiency()
        }
        
        return self.current_state, reward, done, info
    
    def _update_system_state(self, action: AutomationAction):
        """Update automation system state based on action"""
        # Simulate conveyor movement
        self.current_state.conveyor_position += action.conveyor_speed
        if self.current_state.conveyor_position > 100:
            self.current_state.conveyor_position = 0
            
        # Simulate workpiece processing
        base_cycle_time = action.station1_duration + action.station2_duration + 2.0
        efficiency_factor = 1.0 - (action.conveyor_speed - 1.0) ** 2 * 0.1
        self.current_state.cycle_time = base_cycle_time * efficiency_factor
        
        # Calculate throughput
        self.current_state.throughput = 3600.0 / max(self.current_state.cycle_time, 1.0)
        
        # Energy consumption model
        self.current_state.energy_consumption = (
            action.station1_duration * 50 +
            action.station2_duration * 40 +
            action.conveyor_speed * 30 +
            action.buffer_size * 10
        )
        
        # Safety violations (increase with extreme parameters)
        if action.conveyor_speed > 1.8 or action.station1_duration < 1.5:
            self.current_state.safety_violations += 1
            
        # Queue management
        self.current_state.queue_length = min(action.buffer_size, 15)
        
        # Station status simulation
        self.current_state.station1_busy = self.step_count % 20 < action.station1_duration * 4
        self.current_state.station2_busy = self.step_count % 25 < action.station2_duration * 4
    
    def _calculate_reward(self, action: AutomationAction) -> float:
        """Calculate reward based on multiple objectives"""
        # Throughput reward (maximize)
        throughput_reward = (self.current_state.throughput / 120.0) * 30
        
        # Cycle time reward (minimize)
        cycle_time_reward = max(0, (50 - self.current_state.cycle_time) / 50.0) * 25
        
        # Energy efficiency reward (minimize consumption)
        energy_reward = max(0, (200 - self.current_state.energy_consumption) / 200.0) * 20
        
        # Safety penalty (heavy penalty for violations)
        safety_penalty = -self.current_state.safety_violations * 50
        
        # Queue optimization (moderate queue length)
        queue_reward = max(0, (10 - abs(self.current_state.queue_length - 5)) / 10.0) * 15
        
        # Stability bonus (consistent performance)
        stability_bonus = 10 if 20 < self.current_state.cycle_time < 40 else 0
        
        total_reward = (
            throughput_reward + cycle_time_reward + energy_reward + 
            safety_penalty + queue_reward + stability_bonus
        )
        
        return total_reward
    
    def _calculate_oee(self) -> float:
        """Calculate Overall Equipment Effectiveness"""
        availability = max(0.8, 1.0 - self.current_state.safety_violations * 0.1)
        performance = min(1.0, self.current_state.throughput / 100.0)
        quality = max(0.9, 1.0 - self.current_state.safety_violations * 0.05)
        return availability * performance * quality * 100
    
    def _calculate_efficiency(self) -> float:
        """Calculate energy efficiency percentage"""
        optimal_energy = 120.0
        return max(0, (200 - self.current_state.energy_consumption) / 200.0) * 100

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for automation system optimization
    Architecture optimized for industrial control applications
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            
            # Hidden layers
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            
            # Output layer
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(state)

class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int = 10000):
        self.memory = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', 
                                   ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state: torch.Tensor, action: int, reward: float, 
             next_state: torch.Tensor, done: bool):
        """Store experience in replay buffer"""
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)
    
    def sample(self, batch_size: int) -> List:
        """Sample random batch of experiences"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        return len(self.memory)

class DQNAgent:
    """
    Deep Q-Network agent for automation system optimization
    Implements Double DQN with experience replay and target network
    """
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Training parameters
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_frequency = 100
        self.training_step = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.loss_history = []
        
        logger.info(f"DQN Agent initialized on device: {self.device}")
    
    def act(self, state: AutomationState) -> int:
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = state.to_tensor().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax().item()
    
    def remember(self, state: AutomationState, action: int, reward: float,
                 next_state: AutomationState, done: bool):
        """Store experience in replay buffer"""
        state_tensor = state.to_tensor()
        next_state_tensor = next_state.to_tensor()
        self.replay_buffer.push(state_tensor, action, reward, next_state_tensor, done)
    
    def train(self) -> float:
        """Train the DQN on a batch of experiences"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        states = torch.stack([e.state for e in experiences]).to(self.device)
        actions = torch.tensor([e.action for e in experiences]).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.stack([e.next_state for e in experiences]).to(self.device)
        dones = torch.tensor([e.done for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (Double DQN)
        next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions).detach()
        
        # Target Q values
        target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (1 - dones.unsqueeze(1)))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update training step
        self.training_step += 1
        
        # Update target network
        if self.training_step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Track loss
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'loss_history': self.loss_history
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint['episode_rewards']
        self.loss_history = checkpoint['loss_history']
        logger.info(f"Model loaded from {filepath}")

class PLCCodeGenerator:
    """
    Generates optimized PLC code based on learned RL policy
    Compatible with VueOne engineering ecosystem
    """
    
    def __init__(self, agent: DQNAgent):
        self.agent = agent
    
    def generate_code(self, optimal_action: AutomationAction, 
                     performance_metrics: Dict[str, float]) -> str:
        """Generate IEC 61499 compatible PLC code"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        code = f"""
// ========================================================================
// WMG AUTOMATION SYSTEMS GROUP - RL OPTIMIZED PLC CODE
// Generated: {timestamp}
// VueOne Engineering Ecosystem Compatible
// IEC 61499 Standard Compliant
// ========================================================================

// PERFORMANCE METRICS ACHIEVED:
// Cycle Time: {performance_metrics.get('cycle_time', 0):.1f}s
// Throughput: {performance_metrics.get('throughput', 0):.1f} parts/hour
// OEE: {performance_metrics.get('oee', 0):.1f}%
// Energy Efficiency: {performance_metrics.get('efficiency', 0):.1f}%
// Safety Score: {performance_metrics.get('safety_score', 0):.1f}%

FUNCTION_BLOCK RLOptimizedAutomationController
VAR_INPUT
    // System Inputs
    Start_System : BOOL;
    Emergency_Stop : BOOL;
    Reset_System : BOOL;
    
    // Station Status Inputs
    Station1_Ready : BOOL;
    Station2_Ready : BOOL;
    Conveyor_Ready : BOOL;
    
    // Sensor Inputs
    Part_Present_Station1 : BOOL;
    Part_Present_Station2 : BOOL;
    Conveyor_Position : INT;
    Queue_Length : INT;
END_VAR

VAR_OUTPUT
    // Control Outputs
    Conveyor_Run : BOOL;
    Conveyor_Speed : REAL;
    Station1_Activate : BOOL;
    Station2_Activate : BOOL;
    
    // Status Outputs
    System_Running : BOOL;
    Cycle_Complete : BOOL;
    Safety_OK : BOOL;
    Performance_OK : BOOL;
    
    // Performance Outputs
    Current_Cycle_Time : REAL;
    Current_Throughput : REAL;
    Energy_Consumption : REAL;
END_VAR

VAR
    // RL Optimized Parameters
    STATION1_DURATION : TIME := T#{optimal_action.station1_duration * 1000:.0f}ms;
    STATION2_DURATION : TIME := T#{optimal_action.station2_duration * 1000:.0f}ms;
    CONVEYOR_SPEED_SETPOINT : REAL := {optimal_action.conveyor_speed:.2f};
    BUFFER_SIZE_LIMIT : INT := {optimal_action.buffer_size};
    
    // State Machine Variables
    SystemState : INT := 0;
    PreviousState : INT := 0;
    StateTimer : TON;
    CycleTimer : TON;
    SafetyTimer : TON;
    
    // Performance Monitoring
    CycleCounter : INT := 0;
    TotalEnergy : REAL := 0.0;
    LastCycleTime : REAL := 0.0;
    
    // Safety Monitoring
    SafetyViolations : INT := 0;
    MaxSafetyViolations : INT := 3;
    
    // Internal Flags
    ProcessingStation1 : BOOL := FALSE;
    ProcessingStation2 : BOOL := FALSE;
    ConveyorMoving : BOOL := FALSE;
END_VAR

// ========================================================================
// MAIN CONTROL LOGIC - RL OPTIMIZED STATE MACHINE
// ========================================================================

// Safety Pre-Check
Safety_OK := NOT Emergency_Stop AND 
             Station1_Ready AND 
             Station2_Ready AND 
             Conveyor_Ready AND
             (SafetyViolations < MaxSafetyViolations);

// Main State Machine
IF Safety_OK AND Start_System THEN
    System_Running := TRUE;
    
    CASE SystemState OF
        0: // IDLE STATE
            IF Station1_Ready AND NOT ProcessingStation1 THEN
                SystemState := 1;
                CycleTimer(IN:=TRUE, PT:=T#0ms);
            END_IF;
        
        1: // MATERIAL_LOADING_STATION1
            Conveyor_Run := TRUE;
            Conveyor_Speed := CONVEYOR_SPEED_SETPOINT;
            ConveyorMoving := TRUE;
            
            IF Part_Present_Station1 AND Station1_Ready THEN
                SystemState := 2;
                ProcessingStation1 := TRUE;
                Station1_Activate := TRUE;
                StateTimer(IN:=TRUE, PT:=STATION1_DURATION);
            END_IF;
        
        2: // PROCESSING_STATION1
            StateTimer(IN:=TRUE, PT:=STATION1_DURATION);
            
            IF StateTimer.Q THEN
                ProcessingStation1 := FALSE;
                Station1_Activate := FALSE;
                StateTimer(IN:=FALSE);
                
                IF Station2_Ready THEN
                    SystemState := 3;
                ELSE
                    SystemState := 6; // Wait for Station2
                END_IF;
            END_IF;
        
        3: // TRANSFER_TO_STATION2
            Conveyor_Run := TRUE;
            Conveyor_Speed := CONVEYOR_SPEED_SETPOINT;
            
            IF Part_Present_Station2 AND Station2_Ready THEN
                SystemState := 4;
                ProcessingStation2 := TRUE;
                Station2_Activate := TRUE;
                StateTimer(IN:=TRUE, PT:=STATION2_DURATION);
            END_IF;
        
        4: // PROCESSING_STATION2
            StateTimer(IN:=TRUE, PT:=STATION2_DURATION);
            
            IF StateTimer.Q THEN
                ProcessingStation2 := FALSE;
                Station2_Activate := FALSE;
                StateTimer(IN:=FALSE);
                SystemState := 5;
            END_IF;
        
        5: // CYCLE_COMPLETE
            Cycle_Complete := TRUE;
            CycleCounter := CycleCounter + 1;
            
            // Calculate performance metrics
            LastCycleTime := TIME_TO_REAL(CycleTimer.ET) / 1000.0;
            Current_Cycle_Time := LastCycleTime;
            Current_Throughput := 3600.0 / LastCycleTime;
            
            // Energy consumption calculation
            TotalEnergy := TotalEnergy + 
                          (TIME_TO_REAL(STATION1_DURATION) * 0.05) +
                          (TIME_TO_REAL(STATION2_DURATION) * 0.04) +
                          (CONVEYOR_SPEED_SETPOINT * 0.03);
            Energy_Consumption := TotalEnergy;
            
            // Buffer management
            IF Queue_Length < BUFFER_SIZE_LIMIT THEN
                SystemState := 0; // Continue production
            ELSE
                SystemState := 7; // Buffer full, wait
            END_IF;
            
            CycleTimer(IN:=FALSE);
            Cycle_Complete := FALSE;
        
        6: // WAIT_FOR_STATION2
            Conveyor_Run := FALSE;
            ConveyorMoving := FALSE;
            
            IF Station2_Ready THEN
                SystemState := 3;
            END_IF;
            
            // Timeout protection
            SafetyTimer(IN:=TRUE, PT:=T#30s);
            IF SafetyTimer.Q THEN
                SafetyViolations := SafetyViolations + 1;
                SystemState := 0;
                SafetyTimer(IN:=FALSE);
            END_IF;
        
        7: // BUFFER_FULL_WAIT
            Conveyor_Run := FALSE;
            ConveyorMoving := FALSE;
            
            IF Queue_Length < (BUFFER_SIZE_LIMIT - 1) THEN
                SystemState := 0;
            END_IF;
    END_CASE;
    
ELSE
    // Emergency stop or safety violation
    System_Running := FALSE;
    Conveyor_Run := FALSE;
    Station1_Activate := FALSE;
    Station2_Activate := FALSE;
    ConveyorMoving := FALSE;
    ProcessingStation1 := FALSE;
    ProcessingStation2 := FALSE;
    SystemState := 0;
END_IF;

// Performance monitoring
Performance_OK := (Current_Cycle_Time < 50.0) AND 
                  (Current_Throughput > 60.0) AND
                  (SafetyViolations = 0);

// Reset handling
IF Reset_System THEN
    SystemState := 0;
    CycleCounter := 0;
    SafetyViolations := 0;
    TotalEnergy := 0.0;
    CycleTimer(IN:=FALSE);
    StateTimer(IN:=FALSE);
    SafetyTimer(IN:=FALSE);
END_IF;

END_FUNCTION_BLOCK

// ========================================================================
// COMPONENT-BASED FUNCTION BLOCKS (VueOne Compatible)
// ========================================================================

FUNCTION_BLOCK RLOptimizedConveyorControl
VAR_INPUT
    Run_Command : BOOL;
    Speed_Setpoint : REAL;
    Emergency_Stop : BOOL;
END_VAR
VAR_OUTPUT
    Motor_Running : BOOL;
    Current_Speed : REAL;
    Position_Feedback : INT;
    Fault_Status : BOOL;
END_VAR
VAR
    SpeedController : PID;
    MotorDrive : BOOL;
    CurrentPosition : INT := 0;
END_VAR

// RL-optimized conveyor control logic
IF Run_Command AND NOT Emergency_Stop THEN
    MotorDrive := TRUE;
    SpeedController(
        SetPoint := Speed_Setpoint,
        ProcessValue := Current_Speed,
        Output => Current_Speed
    );
    
    // Position tracking
    IF MotorDrive THEN
        CurrentPosition := CurrentPosition + INT(Current_Speed * 10);
        IF CurrentPosition > 1000 THEN
            CurrentPosition := 0;
        END_IF;
    END_IF;
    
    Position_Feedback := CurrentPosition;
    Motor_Running := MotorDrive;
    Fault_Status := FALSE;
ELSE
    MotorDrive := FALSE;
    Motor_Running := FALSE;
    Current_Speed := 0.0;
    Fault_Status := Emergency_Stop;
END_IF;

END_FUNCTION_BLOCK

FUNCTION_BLOCK RLOptimizedStationController
VAR_INPUT
    Activate : BOOL;
    Part_Present : BOOL;
    Processing_Duration : TIME;
    Emergency_Stop : BOOL;
END_VAR
VAR_OUTPUT
    Processing_Active : BOOL;
    Process_Complete : BOOL;
    Quality_OK : BOOL;
    Station_Ready : BOOL;
END_VAR
VAR
    ProcessTimer : TON;
    QualityCheck : BOOL := TRUE;
END_VAR

// RL-optimized station processing logic
Station_Ready := NOT Processing_Active AND NOT Emergency_Stop;

IF Activate AND Part_Present AND Station_Ready THEN
    Processing_Active := TRUE;
    ProcessTimer(IN:=TRUE, PT:=Processing_Duration);
    Process_Complete := FALSE;
    
ELSIF ProcessTimer.Q THEN
    Processing_Active := FALSE;
    Process_Complete := TRUE;
    Quality_OK := QualityCheck; // Assume good quality for this example
    ProcessTimer(IN:=FALSE);
    
ELSIF Emergency_Stop THEN
    Processing_Active := FALSE;
    Process_Complete := FALSE;
    ProcessTimer(IN:=FALSE);
END_IF;

END_FUNCTION_BLOCK

// ========================================================================
// HMI INTEGRATION (Auto-Generated for EAE/Siemens Compatibility)
// ========================================================================

PROGRAM HMI_RLOptimizedInterface
VAR
    MainController : RLOptimizedAutomationController;
    ConveyorCtrl : RLOptimizedConveyorControl;
    Station1Ctrl : RLOptimizedStationController;
    Station2Ctrl : RLOptimizedStationController;
    
    // HMI Variables
    HMI_StartButton : BOOL;
    HMI_StopButton : BOOL;
    HMI_ResetButton : BOOL;
    HMI_EmergencyStop : BOOL;
    
    // Status Display
    HMI_SystemStatus : STRING;
    HMI_PerformanceDisplay : STRING;
    HMI_CycleTimeDisplay : REAL;
    HMI_ThroughputDisplay : REAL;
END_VAR

// Main HMI control logic
MainController(
    Start_System := HMI_StartButton,
    Emergency_Stop := HMI_EmergencyStop,
    Reset_System := HMI_ResetButton,
    Station1_Ready := Station1Ctrl.Station_Ready,
    Station2_Ready := Station2Ctrl.Station_Ready,
    Conveyor_Ready := TRUE
);

// Status display logic
IF MainController.System_Running THEN
    HMI_SystemStatus := 'RUNNING - RL OPTIMIZED';
ELSIF MainController.Safety_OK THEN
    HMI_SystemStatus := 'READY';
ELSE
    HMI_SystemStatus := 'FAULT - CHECK SAFETY';
END_IF;

HMI_CycleTimeDisplay := MainController.Current_Cycle_Time;
HMI_ThroughputDisplay := MainController.Current_Throughput;

HMI_PerformanceDisplay := CONCAT('OEE: ', REAL_TO_STRING(
    {performance_metrics.get('oee', 0):.1f}), '%');

END_PROGRAM

// ========================================================================
// END OF RL-OPTIMIZED PLC CODE
// Compatible with VueOne Engineering Ecosystem
// WMG Automation Systems Group, University of Warwick
// ========================================================================
"""
        return code

class RLTrainingManager:
    """
    Manages RL training process and communicates with frontend
    Provides real-time training updates via WebSocket
    """
    
    def __init__(self):
        self.environment = AutomationEnvironment()
        self.agent = DQNAgent(self.environment.state_size, self.environment.action_size)
        self.code_generator = PLCCodeGenerator(self.agent)
        
        self.is_training = False
        self.training_thread = None
        self.websocket_clients = set()
        
        # Training metrics
        self.current_episode = 0
        self.max_episodes = 1000
        self.best_reward = float('-inf')
        self.training_metrics = {
            'episode': 0,
            'reward': 0.0,
            'loss': 0.0,
            'epsilon': 1.0,
            'cycle_time': 45.0,
            'throughput': 80.0,
            'oee': 85.0,
            'efficiency': 87.0,
            'safety_score': 98.0
        }
    
    def start_training(self):
        """Start RL training in separate thread"""
        if not self.is_training:
            self.is_training = True
            self.current_episode = 0
            self.training_thread = threading.Thread(target=self._training_loop)
            self.training_thread.start()
            logger.info("RL training started")
    
    def stop_training(self):
        """Stop RL training"""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join(timeout=5)
        logger.info("RL training stopped")
    
    def _training_loop(self):
        """Main training loop"""
        for episode in range(self.max_episodes):
            if not self.is_training:
                break
                
            self.current_episode = episode
            episode_reward = 0.0
            episode_loss = 0.0
            loss_count = 0
            
            # Reset environment
            state = self.environment.reset()
            
            while True:
                # Choose action
                action_id = self.agent.act(state)
                action = AutomationAction.from_action_id(action_id)
                
                # Execute action
                next_state, reward, done, info = self.environment.step(action)
                
                # Store experience
                self.agent.remember(state, action_id, reward, next_state, done)
                
                # Train agent
                loss = self.agent.train()
                if loss > 0:
                    episode_loss += loss
                    loss_count += 1
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update metrics
            self.training_metrics.update({
                'episode': episode,
                'reward': episode_reward,
                'loss': episode_loss / max(loss_count, 1),
                'epsilon': self.agent.epsilon,
                'cycle_time': state.cycle_time,
                'throughput': state.throughput,
                'oee': info.get('oee', 85.0),
                'efficiency': info.get('efficiency', 87.0),
                'safety_score': max(0, 100 - state.safety_violations * 10)
            })
            
            # Track best performance
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.agent.save_model(f'best_model_episode_{episode}.pth')
            
            # Send updates to frontend
            asyncio.run(self._broadcast_training_update())
            
            # Logging
            if episode % 50 == 0:
                logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                           f"Loss={episode_loss/max(loss_count, 1):.4f}, "
                           f"Epsilon={self.agent.epsilon:.3f}")
            
            time.sleep(0.1)  # Prevent overwhelming the frontend
        
        self.is_training = False
        logger.info("Training completed")
    
    async def _broadcast_training_update(self):
        """Send training updates to all connected WebSocket clients"""
        if self.websocket_clients:
            message = {
                'type': 'training_update',
                'data': self.training_metrics
            }
            
            # Create new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Send to all clients
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send_text(json.dumps(message))
                except:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
    
    def generate_optimized_code(self) -> str:
        """Generate PLC code based on best learned policy"""
        # Use best action from current policy
        best_state = self.environment.current_state
        best_action_id = self.agent.act(best_state)
        best_action = AutomationAction.from_action_id(best_action_id)
        
        return self.code_generator.generate_code(best_action, self.training_metrics)
    
    def add_websocket_client(self, websocket):
        """Add WebSocket client for real-time updates"""
        self.websocket_clients.add(websocket)
    
    def remove_websocket_client(self, websocket):
        """Remove WebSocket client"""
        self.websocket_clients.discard(websocket)

# Global training manager instance
training_manager = RLTrainingManager()

# FastAPI application
app = FastAPI(title="WMG RL Automation Backend")

# API Models
class TrainingRequest(BaseModel):
    episodes: int = 1000
    learning_rate: float = 0.001

class CodeGenerationRequest(BaseModel):
    optimize: bool = True
    format: str = "iec61499"

# REST API Endpoints
@app.post("/api/training/start")
async def start_training(request: TrainingRequest):
    """Start RL training"""
    training_manager.max_episodes = request.episodes
    training_manager.start_training()
    return {"status": "training_started", "episodes": request.episodes}

@app.post("/api/training/stop")
async def stop_training():
    """Stop RL training"""
    training_manager.stop_training()
    return {"status": "training_stopped"}

@app.get("/api/training/status")
async def get_training_status():
    """Get current training status"""
    return {
        "is_training": training_manager.is_training,
        "current_episode": training_manager.current_episode,
        "max_episodes": training_manager.max_episodes,
        "metrics": training_manager.training_metrics
    }

@app.post("/api/code/generate")
async def generate_code(request: CodeGenerationRequest):
    """Generate optimized PLC code"""
    code = training_manager.generate_optimized_code()
    return {
        "status": "code_generated",
        "code": code,
        "format": request.format,
        "timestamp": datetime.now().isoformat()
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    training_manager.add_websocket_client(websocket)
    
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'ping':
                await websocket.send_text(json.dumps({'type': 'pong'}))
    
    except WebSocketDisconnect:
        training_manager.remove_websocket_client(websocket)

@app.get("/")
async def read_root():
    return {"message": "WMG RL Automation Backend is running", "status": "active"}

def main():
    """Main entry point"""
    logger.info("Starting WMG RL Automation Backend")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Start FastAPI server - localhost only for security
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    main()