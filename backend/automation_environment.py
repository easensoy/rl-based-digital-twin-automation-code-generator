"""
WMG RL Digital Twin Platform - Automation Environment
University of Warwick - WMG Automation Systems Group

Industrial automation environment simulation for reinforcement learning training.
Provides realistic manufacturing process dynamics with safety constraints.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class ProcessState(Enum):
    """Manufacturing process states"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class SafetyLevel(Enum):
    """Safety alert levels"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class SensorReading:
    """Individual sensor reading with metadata"""
    value: float
    timestamp: datetime
    sensor_id: str
    unit: str
    min_value: float = 0.0
    max_value: float = 100.0
    warning_threshold: float = 80.0
    critical_threshold: float = 95.0
    
    @property
    def normalized_value(self) -> float:
        """Normalized value between 0 and 1"""
        return (self.value - self.min_value) / (self.max_value - self.min_value)
    
    @property
    def safety_level(self) -> SafetyLevel:
        """Current safety level based on thresholds"""
        if self.value >= self.critical_threshold:
            return SafetyLevel.CRITICAL
        elif self.value >= self.warning_threshold:
            return SafetyLevel.WARNING
        else:
            return SafetyLevel.NORMAL

@dataclass
class RobotState:
    """Robot kinematic and dynamic state"""
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(6))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(6))
    joint_torques: np.ndarray = field(default_factory=lambda: np.zeros(6))
    end_effector_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    is_moving: bool = False
    current_program: Optional[str] = None
    error_codes: List[str] = field(default_factory=list)

@dataclass
class ProductionMetrics:
    """Real-time production performance metrics"""
    cycle_time: float = 0.0
    throughput: float = 0.0
    quality_score: float = 1.0
    energy_consumption: float = 0.0
    oee: float = 0.0
    safety_score: float = 1.0
    scrap_rate: float = 0.0
    maintenance_score: float = 1.0

class AutomationEnvironment:
    """
    Comprehensive industrial automation environment for RL training.
    Simulates realistic manufacturing processes with safety constraints.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        
        # Environment state
        self.current_step = 0
        self.episode_length = self.config["max_episode_steps"]
        self.process_state = ProcessState.IDLE
        
        # Robot state
        self.robot_state = RobotState()
        self.robot_limits = self._initialize_robot_limits()
        
        # Sensor systems
        self.sensors = self._initialize_sensors()
        self.sensor_history = []
        
        # Production metrics
        self.production_metrics = ProductionMetrics()
        self.metrics_history = []
        
        # Process parameters
        self.process_parameters = self._initialize_process_parameters()
        
        # Safety monitoring
        self.safety_violations = []
        self.emergency_stop = False
        
        # Disturbance modeling
        self.disturbances = self._initialize_disturbances()
        
        # Performance targets
        self.performance_targets = self.config["performance_targets"]
        
        # State space definition
        self.state_size = 18  # Extended state representation
        self.action_size = 8  # Expanded action space
        
        # Quality system
        self.quality_parameters = self._initialize_quality_system()
        
        # Maintenance system
        self.maintenance_state = self._initialize_maintenance_system()
        
        logger.info("Automation Environment initialized with enhanced simulation")
        self.reset()
    
    def _get_default_config(self) -> Dict:
        """Default environment configuration"""
        return {
            "max_episode_steps": 2000,
            "simulation_timestep": 0.1,  # seconds
            "safety_enabled": True,
            "noise_level": 0.02,
            "disturbance_probability": 0.05,
            "performance_targets": {
                "cycle_time": 18.0,
                "throughput": 180.0,  # parts per hour
                "quality_score": 0.98,
                "energy_efficiency": 0.85,
                "safety_score": 1.0,
                "oee_target": 0.85
            },
            "process_limits": {
                "temperature_max": 85.0,
                "pressure_max": 8.0,
                "vibration_max": 2.0,
                "force_max": 500.0
            }
        }
    
    def _initialize_robot_limits(self) -> Dict:
        """Initialize robot joint limits and constraints"""
        return {
            "position_limits": [
                [-np.pi, np.pi],      # Base joint
                [-np.pi/2, np.pi/2],  # Shoulder
                [-2.35, 0.7],         # Elbow
                [-np.pi, np.pi],      # Wrist 1
                [-2.09, 2.09],        # Wrist 2
                [-2*np.pi, 2*np.pi]   # Wrist 3
            ],
            "velocity_limits": [2.0, 2.0, 3.0, 3.0, 3.0, 6.0],  # rad/s
            "acceleration_limits": [10.0, 10.0, 15.0, 15.0, 15.0, 20.0],  # rad/s²
            "torque_limits": [300.0, 300.0, 150.0, 54.0, 54.0, 54.0]  # Nm
        }
    
    def _initialize_sensors(self) -> Dict[str, SensorReading]:
        """Initialize industrial sensor systems"""
        current_time = datetime.now()
        
        return {
            "temperature_1": SensorReading(25.0, current_time, "TEMP_001", "°C", 0.0, 100.0, 70.0, 85.0),
            "temperature_2": SensorReading(24.0, current_time, "TEMP_002", "°C", 0.0, 100.0, 70.0, 85.0),
            "pressure_1": SensorReading(1.0, current_time, "PRES_001", "bar", 0.0, 10.0, 7.0, 8.5),
            "pressure_2": SensorReading(0.8, current_time, "PRES_002", "bar", 0.0, 10.0, 7.0, 8.5),
            "vibration_x": SensorReading(0.1, current_time, "VIB_X001", "mm/s", 0.0, 5.0, 2.0, 3.0),
            "vibration_y": SensorReading(0.1, current_time, "VIB_Y001", "mm/s", 0.0, 5.0, 2.0, 3.0),
            "vibration_z": SensorReading(0.1, current_time, "VIB_Z001", "mm/s", 0.0, 5.0, 2.0, 3.0),
            "force_x": SensorReading(0.0, current_time, "FORCE_X", "N", -1000.0, 1000.0, 400.0, 600.0),
            "force_y": SensorReading(0.0, current_time, "FORCE_Y", "N", -1000.0, 1000.0, 400.0, 600.0),
            "force_z": SensorReading(0.0, current_time, "FORCE_Z", "N", -1000.0, 1000.0, 400.0, 600.0),
            "power_consumption": SensorReading(5.2, current_time, "POWER_001", "kW", 0.0, 50.0, 35.0, 45.0),
            "flow_rate": SensorReading(15.0, current_time, "FLOW_001", "L/min", 0.0, 100.0, 80.0, 95.0)
        }
    
    def _initialize_process_parameters(self) -> Dict:
        """Initialize manufacturing process parameters"""
        return {
            "feed_rate": 100.0,           # mm/min
            "spindle_speed": 1000.0,      # RPM
            "cutting_depth": 2.0,         # mm
            "coolant_flow": 15.0,         # L/min
            "tool_wear": 0.0,             # 0-1 scale
            "material_hardness": 0.5,     # 0-1 scale
            "surface_finish": 0.95,       # 0-1 quality scale
            "dimensional_accuracy": 0.98   # 0-1 accuracy scale
        }
    
    def _initialize_disturbances(self) -> Dict:
        """Initialize system disturbance models"""
        return {
            "tool_wear_rate": 0.001,
            "thermal_drift": 0.002,
            "vibration_drift": 0.001,
            "power_fluctuation": 0.05,
            "material_variation": 0.03
        }
    
    def _initialize_quality_system(self) -> Dict:
        """Initialize quality control parameters"""
        return {
            "dimensional_tolerance": 0.02,    # mm
            "surface_roughness": 1.6,         # μm Ra
            "geometric_tolerance": 0.01,      # mm
            "material_properties": 0.95,      # compliance score
            "inspection_rate": 0.1,           # 10% of parts inspected
            "defect_rate": 0.02               # 2% baseline defect rate
        }
    
    def _initialize_maintenance_system(self) -> Dict:
        """Initialize predictive maintenance parameters"""
        return {
            "tool_life_remaining": 1.0,       # 0-1 scale
            "bearing_condition": 0.95,        # 0-1 health score
            "lubrication_level": 0.8,         # 0-1 level
            "calibration_drift": 0.0,         # calibration error
            "scheduled_maintenance": 100,      # hours until next service
            "unplanned_downtime_risk": 0.05    # probability
        }
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.process_state = ProcessState.IDLE
        self.emergency_stop = False
        self.safety_violations.clear()
        
        # Reset robot state
        self.robot_state = RobotState()
        
        # Reset sensors with baseline values
        self._reset_sensors()
        
        # Reset production metrics
        self.production_metrics = ProductionMetrics()
        
        # Reset process parameters
        self.process_parameters.update({
            "tool_wear": random.uniform(0.0, 0.1),
            "material_hardness": random.uniform(0.4, 0.6)
        })
        
        # Clear history
        self.sensor_history.clear()
        self.metrics_history.clear()
        
        logger.info("Environment reset completed")
        return self._get_state_vector()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one simulation step"""
        self.current_step += 1
        
        # Apply action to system
        action_effects = self._apply_action(action)
        
        # Update system dynamics
        self._update_system_dynamics(action_effects)
        
        # Update sensors
        self._update_sensors()
        
        # Update production metrics
        self._update_production_metrics()
        
        # Check safety constraints
        safety_violations = self._check_safety_constraints()
        
        # Calculate reward
        reward = self._calculate_reward(action, safety_violations)
        
        # Check termination conditions
        done = self._check_termination_conditions()
        
        # Compile info dictionary
        info = self._compile_step_info(action_effects, safety_violations)
        
        # Store history
        self._store_step_history()
        
        return self._get_state_vector(), reward, done, info
    
    def _apply_action(self, action: int) -> Dict:
        """Apply selected action to manufacturing system"""
        
        # Extended action space for comprehensive control
        action_mappings = {
            0: {  # Conservative operation
                "feed_rate_multiplier": 0.8,
                "spindle_speed_multiplier": 0.9,
                "coolant_flow_multiplier": 1.1,
                "power_target": 0.7,
                "process_intensity": 0.8
            },
            1: {  # Normal operation
                "feed_rate_multiplier": 1.0,
                "spindle_speed_multiplier": 1.0,
                "coolant_flow_multiplier": 1.0,
                "power_target": 1.0,
                "process_intensity": 1.0
            },
            2: {  # Aggressive operation
                "feed_rate_multiplier": 1.3,
                "spindle_speed_multiplier": 1.2,
                "coolant_flow_multiplier": 1.2,
                "power_target": 1.4,
                "process_intensity": 1.3
            },
            3: {  # Energy optimization
                "feed_rate_multiplier": 0.9,
                "spindle_speed_multiplier": 0.85,
                "coolant_flow_multiplier": 0.9,
                "power_target": 0.6,
                "process_intensity": 0.9
            },
            4: {  # Quality focus
                "feed_rate_multiplier": 0.7,
                "spindle_speed_multiplier": 0.8,
                "coolant_flow_multiplier": 1.3,
                "power_target": 0.9,
                "process_intensity": 0.8
            },
            5: {  # Throughput optimization
                "feed_rate_multiplier": 1.4,
                "spindle_speed_multiplier": 1.3,
                "coolant_flow_multiplier": 1.1,
                "power_target": 1.5,
                "process_intensity": 1.4
            },
            6: {  # Maintenance mode
                "feed_rate_multiplier": 0.5,
                "spindle_speed_multiplier": 0.6,
                "coolant_flow_multiplier": 1.5,
                "power_target": 0.4,
                "process_intensity": 0.5
            },
            7: {  # Emergency reduction
                "feed_rate_multiplier": 0.3,
                "spindle_speed_multiplier": 0.4,
                "coolant_flow_multiplier": 2.0,
                "power_target": 0.2,
                "process_intensity": 0.3
            }
        }
        
        effects = action_mappings.get(action, action_mappings[1])
        
        # Apply effects to process parameters
        self.process_parameters["feed_rate"] *= effects["feed_rate_multiplier"]
        self.process_parameters["spindle_speed"] *= effects["spindle_speed_multiplier"]
        self.process_parameters["coolant_flow"] *= effects["coolant_flow_multiplier"]
        
        # Constrain parameters to realistic limits
        self.process_parameters["feed_rate"] = np.clip(self.process_parameters["feed_rate"], 20.0, 500.0)
        self.process_parameters["spindle_speed"] = np.clip(self.process_parameters["spindle_speed"], 200.0, 5000.0)
        self.process_parameters["coolant_flow"] = np.clip(self.process_parameters["coolant_flow"], 5.0, 50.0)
        
        return effects
    
    def _update_system_dynamics(self, action_effects: Dict):
        """Update manufacturing system dynamics"""
        
        # Tool wear progression
        wear_rate = self.disturbances["tool_wear_rate"] * action_effects["process_intensity"]
        self.process_parameters["tool_wear"] += wear_rate
        self.process_parameters["tool_wear"] = np.clip(self.process_parameters["tool_wear"], 0.0, 1.0)
        
        # Thermal effects
        thermal_load = action_effects["power_target"] * action_effects["process_intensity"]
        self.sensors["temperature_1"].value += (thermal_load - 1.0) * 2.0 + np.random.normal(0, 0.5)
        self.sensors["temperature_2"].value += (thermal_load - 1.0) * 1.8 + np.random.normal(0, 0.4)
        
        # Pressure dynamics
        pressure_demand = action_effects["coolant_flow_multiplier"]
        self.sensors["pressure_1"].value = 1.0 + (pressure_demand - 1.0) * 2.0 + np.random.normal(0, 0.1)
        
        # Vibration based on operating conditions
        vibration_base = action_effects["process_intensity"] * 0.2
        self.sensors["vibration_x"].value = vibration_base + np.random.normal(0, 0.05)
        self.sensors["vibration_y"].value = vibration_base + np.random.normal(0, 0.05)
        self.sensors["vibration_z"].value = vibration_base + np.random.normal(0, 0.05)
        
        # Power consumption
        base_power = 5.0
        load_power = action_effects["power_target"] * 15.0
        efficiency_factor = 0.8 + (1.0 - action_effects["process_intensity"]) * 0.15
        self.sensors["power_consumption"].value = (base_power + load_power) / efficiency_factor
        
        # Add process noise
        self._add_process_noise()
        
        # Apply disturbances
        self._apply_disturbances()
    
    def _update_sensors(self):
        """Update all sensor readings with realistic dynamics"""
        current_time = datetime.now()
        
        for sensor_name, sensor in self.sensors.items():
            # Update timestamp
            sensor.timestamp = current_time
            
            # Apply sensor noise
            noise_level = self.config["noise_level"]
            sensor.value += np.random.normal(0, noise_level * sensor.max_value * 0.01)
            
            # Ensure values stay within physical limits
            sensor.value = np.clip(sensor.value, sensor.min_value, sensor.max_value)
    
    def _update_production_metrics(self):
        """Update real-time production performance metrics"""
        
        # Cycle time calculation
        base_cycle_time = self.performance_targets["cycle_time"]
        efficiency_factor = self.process_parameters["feed_rate"] / 100.0
        tool_factor = 1.0 + self.process_parameters["tool_wear"] * 0.3
        self.production_metrics.cycle_time = base_cycle_time / efficiency_factor * tool_factor
        
        # Throughput calculation
        if self.production_metrics.cycle_time > 0:
            self.production_metrics.throughput = 3600.0 / self.production_metrics.cycle_time
        
        # Quality score
        tool_impact = 1.0 - self.process_parameters["tool_wear"] * 0.4
        process_stability = 1.0 - max(0, (self.sensors["vibration_x"].value - 0.5)) * 0.2
        temperature_impact = 1.0 - max(0, (self.sensors["temperature_1"].value - 50.0)) * 0.01
        self.production_metrics.quality_score = tool_impact * process_stability * temperature_impact
        
        # Energy consumption (kWh per part)
        power_kw = self.sensors["power_consumption"].value
        cycle_hours = self.production_metrics.cycle_time / 3600.0
        self.production_metrics.energy_consumption = power_kw * cycle_hours
        
        # Overall Equipment Effectiveness (OEE)
        availability = 0.95  # Assume 95% availability
        performance = min(1.0, self.production_metrics.throughput / self.performance_targets["throughput"])
        quality = self.production_metrics.quality_score
        self.production_metrics.oee = availability * performance * quality
        
        # Safety score
        safety_violations = len([s for s in self.sensors.values() 
                               if s.safety_level in [SafetyLevel.WARNING, SafetyLevel.CRITICAL]])
        self.production_metrics.safety_score = max(0.0, 1.0 - safety_violations * 0.1)
        
        # Scrap rate
        base_scrap = self.quality_parameters["defect_rate"]
        quality_impact = (1.0 - self.production_metrics.quality_score) * 0.5
        self.production_metrics.scrap_rate = base_scrap + quality_impact
        
        # Maintenance score
        tool_health = 1.0 - self.process_parameters["tool_wear"]
        vibration_health = max(0.0, 1.0 - self.sensors["vibration_x"].value / 2.0)
        self.production_metrics.maintenance_score = (tool_health + vibration_health) / 2.0
    
    def _check_safety_constraints(self) -> List[str]:
        """Check safety constraints and return violations"""
        violations = []
        
        if not self.config["safety_enabled"]:
            return violations
        
        # Temperature limits
        for temp_sensor in ["temperature_1", "temperature_2"]:
            if self.sensors[temp_sensor].safety_level == SafetyLevel.CRITICAL:
                violations.append(f"Critical temperature: {temp_sensor}")
                
        # Pressure limits
        if self.sensors["pressure_1"].safety_level == SafetyLevel.CRITICAL:
            violations.append("Critical pressure detected")
        
        # Vibration limits
        for vib_sensor in ["vibration_x", "vibration_y", "vibration_z"]:
            if self.sensors[vib_sensor].safety_level == SafetyLevel.CRITICAL:
                violations.append(f"Excessive vibration: {vib_sensor}")
        
        # Power consumption limits
        if self.sensors["power_consumption"].safety_level == SafetyLevel.CRITICAL:
            violations.append("Power consumption exceeded")
        
        # Robot joint limits
        for i, (pos, limits) in enumerate(zip(self.robot_state.joint_positions, 
                                            self.robot_limits["position_limits"])):
            if pos < limits[0] or pos > limits[1]:
                violations.append(f"Joint {i} position limit exceeded")
        
        # Emergency stop conditions
        if len(violations) >= 3:
            self.emergency_stop = True
            violations.append("Emergency stop activated")
        
        self.safety_violations.extend(violations)
        return violations
    
    def _calculate_reward(self, action: int, safety_violations: List[str]) -> float:
        """Calculate comprehensive reward function"""
        
        # Safety penalty (highest priority)
        safety_penalty = 0.0
        if safety_violations:
            safety_penalty = -100.0 * len(safety_violations)
            if self.emergency_stop:
                safety_penalty -= 500.0
        
        # Production efficiency rewards
        throughput_reward = min(1.0, self.production_metrics.throughput / 
                              self.performance_targets["throughput"]) * 50.0
        
        cycle_time_reward = max(0.0, 1.0 - abs(self.production_metrics.cycle_time - 
                                              self.performance_targets["cycle_time"]) / 
                               self.performance_targets["cycle_time"]) * 30.0
        
        # Quality rewards
        quality_reward = self.production_metrics.quality_score * 40.0
        
        # Energy efficiency
        target_energy = 0.1  # kWh per part
        energy_efficiency = max(0.0, 1.0 - abs(self.production_metrics.energy_consumption - 
                                              target_energy) / target_energy)
        energy_reward = energy_efficiency * 25.0
        
        # OEE reward
        oee_reward = self.production_metrics.oee * 45.0
        
        # Maintenance considerations
        maintenance_reward = self.production_metrics.maintenance_score * 20.0
        
        # Process stability bonus
        vibration_stability = max(0.0, 1.0 - self.sensors["vibration_x"].value / 2.0)
        temperature_stability = max(0.0, 1.0 - abs(self.sensors["temperature_1"].value - 35.0) / 50.0)
        stability_bonus = (vibration_stability + temperature_stability) * 15.0
        
        # Tool wear penalty
        tool_penalty = -self.process_parameters["tool_wear"] * 30.0
        
        # Composite reward
        total_reward = (
            safety_penalty +
            throughput_reward +
            cycle_time_reward +
            quality_reward +
            energy_reward +
            oee_reward +
            maintenance_reward +
            stability_bonus +
            tool_penalty
        )
        
        return total_reward
    
    def _check_termination_conditions(self) -> bool:
        """Check if episode should terminate"""
        
        # Maximum steps reached
        if self.current_step >= self.episode_length:
            return True
        
        # Emergency stop
        if self.emergency_stop:
            return True
        
        # Critical system failure
        critical_sensors = [s for s in self.sensors.values() 
                          if s.safety_level == SafetyLevel.CRITICAL]
        if len(critical_sensors) >= 2:
            return True
        
        # Tool completely worn
        if self.process_parameters["tool_wear"] >= 0.95:
            return True
        
        return False
    
    def _compile_step_info(self, action_effects: Dict, safety_violations: List[str]) -> Dict:
        """Compile comprehensive step information"""
        return {
            "step": self.current_step,
            "process_state": self.process_state.value,
            "action_effects": action_effects,
            "safety_violations": safety_violations,
            "emergency_stop": self.emergency_stop,
            "production_metrics": {
                "cycle_time": self.production_metrics.cycle_time,
                "throughput": self.production_metrics.throughput,
                "quality_score": self.production_metrics.quality_score,
                "energy_consumption": self.production_metrics.energy_consumption,
                "oee": self.production_metrics.oee,
                "safety_score": self.production_metrics.safety_score,
                "scrap_rate": self.production_metrics.scrap_rate,
                "maintenance_score": self.production_metrics.maintenance_score
            },
            "sensor_readings": {name: sensor.value for name, sensor in self.sensors.items()},
            "process_parameters": self.process_parameters.copy(),
            "robot_state": {
                "joint_positions": self.robot_state.joint_positions.tolist(),
                "joint_velocities": self.robot_state.joint_velocities.tolist(),
                "end_effector_pose": self.robot_state.end_effector_pose.tolist(),
                "is_moving": self.robot_state.is_moving
            }
        }
    
    def _get_state_vector(self) -> np.ndarray:
        """Get current state as normalized vector for RL agent"""
        state = np.zeros(self.state_size)
        
        # Robot joint positions (normalized)
        for i, pos in enumerate(self.robot_state.joint_positions[:6]):
            limits = self.robot_limits["position_limits"][i]
            state[i] = (pos - limits[0]) / (limits[1] - limits[0])
        
        # Key sensor readings (normalized)
        state[6] = self.sensors["temperature_1"].normalized_value
        state[7] = self.sensors["pressure_1"].normalized_value
        state[8] = self.sensors["vibration_x"].normalized_value
        state[9] = self.sensors["power_consumption"].normalized_value
        
        # Production metrics (normalized)
        state[10] = min(1.0, self.production_metrics.throughput / 300.0)
        state[11] = self.production_metrics.quality_score
        state[12] = min(1.0, self.production_metrics.energy_consumption / 1.0)
        state[13] = self.production_metrics.oee
        state[14] = self.production_metrics.safety_score
        
        # Process parameters (normalized)
        state[15] = self.process_parameters["tool_wear"]
        state[16] = self.process_parameters["feed_rate"] / 500.0
        state[17] = self.process_parameters["spindle_speed"] / 5000.0
        
        return state
    
    def _reset_sensors(self):
        """Reset sensors to baseline values"""
        baseline_values = {
            "temperature_1": 25.0 + np.random.normal(0, 2),
            "temperature_2": 24.0 + np.random.normal(0, 2),
            "pressure_1": 1.0 + np.random.normal(0, 0.1),
            "pressure_2": 0.8 + np.random.normal(0, 0.1),
            "vibration_x": 0.1 + np.random.normal(0, 0.02),
            "vibration_y": 0.1 + np.random.normal(0, 0.02),
            "vibration_z": 0.1 + np.random.normal(0, 0.02),
            "force_x": 0.0,
            "force_y": 0.0,
            "force_z": 0.0,
            "power_consumption": 5.2 + np.random.normal(0, 0.5),
            "flow_rate": 15.0 + np.random.normal(0, 1)
        }
        
        for sensor_name, value in baseline_values.items():
            self.sensors[sensor_name].value = value
    
    def _add_process_noise(self):
        """Add realistic process noise to sensors"""
        noise_level = self.config["noise_level"]
        
        for sensor in self.sensors.values():
            noise = np.random.normal(0, noise_level * (sensor.max_value - sensor.min_value) * 0.01)
            sensor.value += noise
    
    def _apply_disturbances(self):
        """Apply random disturbances to system"""
        if np.random.random() < self.config["disturbance_probability"]:
            disturbance_type = np.random.choice([
                "thermal_spike", "pressure_drop", "vibration_burst", 
                "power_fluctuation", "material_change"
            ])
            
            if disturbance_type == "thermal_spike":
                self.sensors["temperature_1"].value += np.random.uniform(5, 15)
            elif disturbance_type == "pressure_drop":
                self.sensors["pressure_1"].value *= np.random.uniform(0.7, 0.9)
            elif disturbance_type == "vibration_burst":
                for vib_sensor in ["vibration_x", "vibration_y", "vibration_z"]:
                    self.sensors[vib_sensor].value += np.random.uniform(0.5, 1.5)
            elif disturbance_type == "power_fluctuation":
                self.sensors["power_consumption"].value *= np.random.uniform(1.1, 1.3)
            elif disturbance_type == "material_change":
                self.process_parameters["material_hardness"] *= np.random.uniform(0.8, 1.2)
    
    def _store_step_history(self):
        """Store current step data in history"""
        step_data = {
            "step": self.current_step,
            "sensors": {name: sensor.value for name, sensor in self.sensors.items()},
            "metrics": self.production_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        self.sensor_history.append(step_data)
        self.metrics_history.append(self.production_metrics)
        
        # Keep history manageable
        if len(self.sensor_history) > 1000:
            self.sensor_history = self.sensor_history[-800:]
            self.metrics_history = self.metrics_history[-800:]
    
    def update_robot_angles(self, joint_angles: Dict):
        """Update robot joint angles from external control"""
        for i, (joint_name, angle) in enumerate(joint_angles.items()):
            if i < len(self.robot_state.joint_positions):
                self.robot_state.joint_positions[i] = float(angle)
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics for external access"""
        return {
            "cycle_time": self.production_metrics.cycle_time,
            "throughput": self.production_metrics.throughput,
            "quality_score": self.production_metrics.quality_score,
            "energy_efficiency": 1.0 / max(0.1, self.production_metrics.energy_consumption),
            "oee": self.production_metrics.oee,
            "safety_score": self.production_metrics.safety_score,
            "scrap_rate": self.production_metrics.scrap_rate,
            "maintenance_score": self.production_metrics.maintenance_score
        }
    
    def get_state_size(self) -> int:
        """Get size of state vector"""
        return self.state_size
    
    def get_action_size(self) -> int:
        """Get size of action space"""
        return self.action_size
    
    def get_sensor_data(self) -> Dict:
        """Get current sensor readings for monitoring"""
        return {
            name: {
                "value": sensor.value,
                "unit": sensor.unit,
                "safety_level": sensor.safety_level.value,
                "normalized": sensor.normalized_value
            }
            for name, sensor in self.sensors.items()
        }
    
    def is_done(self) -> bool:
        """Check if current episode is complete"""
        return self._check_termination_conditions()
    
    def get_config(self) -> Dict:
        """Get environment configuration"""
        return self.config.copy()