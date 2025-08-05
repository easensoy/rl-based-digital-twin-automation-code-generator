"""
WMG RL Digital Twin Platform - Automation Environment (Refactored)
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

# Constants
DEFAULT_STATE_SIZE = 18
DEFAULT_ACTION_SIZE = 8
MAX_HISTORY_SIZE = 1000
HISTORY_TRIM_SIZE = 800

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
class SensorConfig:
    """Sensor configuration template"""
    sensor_id: str
    unit: str
    min_value: float
    max_value: float
    warning_threshold: float
    critical_threshold: float
    initial_value: float = 0.0

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
    
    # Class constants for sensor configurations
    SENSOR_CONFIGS = {
        "temperature_1": SensorConfig("TEMP_001", "°C", 0.0, 100.0, 70.0, 85.0, 25.0),
        "temperature_2": SensorConfig("TEMP_002", "°C", 0.0, 100.0, 70.0, 85.0, 24.0),
        "pressure_1": SensorConfig("PRES_001", "bar", 0.0, 10.0, 7.0, 8.5, 1.0),
        "pressure_2": SensorConfig("PRES_002", "bar", 0.0, 10.0, 7.0, 8.5, 0.8),
        "vibration_x": SensorConfig("VIB_X001", "mm/s", 0.0, 5.0, 2.0, 3.0, 0.1),
        "vibration_y": SensorConfig("VIB_Y001", "mm/s", 0.0, 5.0, 2.0, 3.0, 0.1),
        "vibration_z": SensorConfig("VIB_Z001", "mm/s", 0.0, 5.0, 2.0, 3.0, 0.1),
        "force_x": SensorConfig("FORCE_X", "N", -1000.0, 1000.0, 400.0, 600.0, 0.0),
        "force_y": SensorConfig("FORCE_Y", "N", -1000.0, 1000.0, 400.0, 600.0, 0.0),
        "force_z": SensorConfig("FORCE_Z", "N", -1000.0, 1000.0, 400.0, 600.0, 0.0),
        "power_consumption": SensorConfig("POWER_001", "kW", 0.0, 50.0, 35.0, 45.0, 5.2),
        "flow_rate": SensorConfig("FLOW_001", "L/min", 0.0, 100.0, 80.0, 95.0, 15.0)
    }

    # Action configurations
    ACTION_CONFIGS = {
        0: ("Conservative", {"feed_rate": 0.8, "spindle_speed": 0.9, "coolant_flow": 1.1, "power_target": 0.7, "process_intensity": 0.8}),
        1: ("Normal", {"feed_rate": 1.0, "spindle_speed": 1.0, "coolant_flow": 1.0, "power_target": 1.0, "process_intensity": 1.0}),
        2: ("Aggressive", {"feed_rate": 1.3, "spindle_speed": 1.2, "coolant_flow": 1.2, "power_target": 1.4, "process_intensity": 1.3}),
        3: ("Energy", {"feed_rate": 0.9, "spindle_speed": 0.85, "coolant_flow": 0.9, "power_target": 0.6, "process_intensity": 0.9}),
        4: ("Quality", {"feed_rate": 0.7, "spindle_speed": 0.8, "coolant_flow": 1.3, "power_target": 0.9, "process_intensity": 0.8}),
        5: ("Throughput", {"feed_rate": 1.4, "spindle_speed": 1.3, "coolant_flow": 1.1, "power_target": 1.5, "process_intensity": 1.4}),
        6: ("Maintenance", {"feed_rate": 0.5, "spindle_speed": 0.6, "coolant_flow": 1.5, "power_target": 0.4, "process_intensity": 0.5}),
        7: ("Emergency", {"feed_rate": 0.3, "spindle_speed": 0.4, "coolant_flow": 2.0, "power_target": 0.2, "process_intensity": 0.3})
    }

    # Robot joint limits
    ROBOT_LIMITS = {
        "position_limits": [[-np.pi, np.pi], [-np.pi/2, np.pi/2], [-2.35, 0.7], [-np.pi, np.pi], [-2.09, 2.09], [-2*np.pi, 2*np.pi]],
        "velocity_limits": [2.0, 2.0, 3.0, 3.0, 3.0, 6.0],
        "acceleration_limits": [10.0, 10.0, 15.0, 15.0, 15.0, 20.0],
        "torque_limits": [300.0, 300.0, 150.0, 54.0, 54.0, 54.0]
    }

    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        
        # Environment state
        self.current_step = 0
        self.episode_length = self.config["max_episode_steps"]
        self.process_state = ProcessState.IDLE
        
        # Initialize components
        self.robot_state = RobotState()
        self.sensors = self._create_sensors()
        self.production_metrics = ProductionMetrics()
        self.process_parameters = self._get_initial_process_parameters()
        
        # Monitoring and control
        self.sensor_history = []
        self.metrics_history = []
        self.safety_violations = []
        self.emergency_stop = False
        
        # System parameters
        self.disturbances = self._get_disturbance_rates()
        self.performance_targets = self.config["performance_targets"]
        self.state_size = DEFAULT_STATE_SIZE
        self.action_size = DEFAULT_ACTION_SIZE
        
        logger.info("Automation Environment initialized with enhanced simulation")
        self.reset()
    
    def _get_default_config(self) -> Dict:
        """Default environment configuration"""
        return {
            "max_episode_steps": 2000,
            "simulation_timestep": 0.1,
            "safety_enabled": True,
            "noise_level": 0.02,
            "disturbance_probability": 0.05,
            "performance_targets": {
                "cycle_time": 18.0,
                "throughput": 180.0,
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

    def _create_sensors(self) -> Dict[str, SensorReading]:
        """Create sensors from configuration templates"""
        current_time = datetime.now()
        sensors = {}
        
        for name, config in self.SENSOR_CONFIGS.items():
            sensors[name] = SensorReading(
                value=config.initial_value,
                timestamp=current_time,
                sensor_id=config.sensor_id,
                unit=config.unit,
                min_value=config.min_value,
                max_value=config.max_value,
                warning_threshold=config.warning_threshold,
                critical_threshold=config.critical_threshold
            )
        
        return sensors

    def _get_initial_process_parameters(self) -> Dict:
        """Initialize manufacturing process parameters"""
        return {
            "feed_rate": 100.0,
            "spindle_speed": 1000.0,
            "cutting_depth": 2.0,
            "coolant_flow": 15.0,
            "tool_wear": 0.0,
            "material_hardness": 0.5,
            "surface_finish": 0.95,
            "dimensional_accuracy": 0.98
        }

    def _get_disturbance_rates(self) -> Dict:
        """System disturbance model parameters"""
        return {
            "tool_wear_rate": 0.001,
            "thermal_drift": 0.002,
            "vibration_drift": 0.001,
            "power_fluctuation": 0.05,
            "material_variation": 0.03
        }

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.process_state = ProcessState.IDLE
        self.emergency_stop = False
        
        # Reset all components
        self._reset_all_components()
        
        logger.info("Environment reset completed")
        return self._get_state_vector()

    def _reset_all_components(self):
        """Reset all environment components"""
        self.safety_violations.clear()
        self.robot_state = RobotState()
        self.production_metrics = ProductionMetrics()
        self.sensor_history.clear()
        self.metrics_history.clear()
        
        # Reset process parameters with randomization
        self.process_parameters.update({
            "tool_wear": random.uniform(0.0, 0.1),
            "material_hardness": random.uniform(0.4, 0.6)
        })
        
        # Reset sensor values with noise
        self._reset_sensor_values()

    def _reset_sensor_values(self):
        """Reset sensor values to baseline with noise"""
        noise_factors = {
            "temperature_1": 2.0, "temperature_2": 2.0, "pressure_1": 0.1, "pressure_2": 0.1,
            "vibration_x": 0.02, "vibration_y": 0.02, "vibration_z": 0.02,
            "power_consumption": 0.5, "flow_rate": 1.0
        }
        
        for name, sensor in self.sensors.items():
            config = self.SENSOR_CONFIGS[name]
            noise = noise_factors.get(name, 0.0)
            sensor.value = config.initial_value + np.random.normal(0, noise)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one simulation step"""
        self.current_step += 1
        
        # Execute action and update system
        action_effects = self._apply_action(action)
        self._update_system_dynamics(action_effects)
        self._update_sensors()
        self._update_production_metrics()
        
        # Check safety and calculate reward
        safety_violations = self._check_safety_constraints()
        reward = self._calculate_reward(action, safety_violations)
        done = self._check_termination_conditions()
        
        # Compile step information
        info = self._compile_step_info(action_effects, safety_violations)
        self._store_step_history()
        
        return self._get_state_vector(), reward, done, info

    def _apply_action(self, action: int) -> Dict:
        """Apply selected action to manufacturing system"""
        _, effects = self.ACTION_CONFIGS.get(action, self.ACTION_CONFIGS[1])
        
        # Apply multipliers to process parameters
        parameter_updates = {
            "feed_rate": effects["feed_rate"],
            "spindle_speed": effects["spindle_speed"],
            "coolant_flow": effects["coolant_flow"]
        }
        
        for param, multiplier in parameter_updates.items():
            self.process_parameters[param] *= multiplier
        
        # Apply constraints
        self._constrain_process_parameters()
        
        return effects

    def _constrain_process_parameters(self):
        """Apply realistic limits to process parameters"""
        constraints = {
            "feed_rate": (20.0, 500.0),
            "spindle_speed": (200.0, 5000.0),
            "coolant_flow": (5.0, 50.0)
        }
        
        for param, (min_val, max_val) in constraints.items():
            self.process_parameters[param] = np.clip(self.process_parameters[param], min_val, max_val)

    def _update_system_dynamics(self, action_effects: Dict):
        """Update manufacturing system dynamics"""
        # Update tool wear
        wear_rate = self.disturbances["tool_wear_rate"] * action_effects["process_intensity"]
        self.process_parameters["tool_wear"] = np.clip(
            self.process_parameters["tool_wear"] + wear_rate, 0.0, 1.0
        )
        
        # Update thermal system
        thermal_load = action_effects["power_target"] * action_effects["process_intensity"]
        self._update_thermal_sensors(thermal_load)
        
        # Update pressure system
        self._update_pressure_sensors(action_effects["coolant_flow"])
        
        # Update vibration system
        self._update_vibration_sensors(action_effects["process_intensity"])
        
        # Update power system
        self._update_power_consumption(action_effects)
        
        # Apply noise and disturbances
        self._add_process_noise()
        self._apply_disturbances()

    def _update_thermal_sensors(self, thermal_load: float):
        """Update temperature sensors based on thermal load"""
        self.sensors["temperature_1"].value += (thermal_load - 1.0) * 2.0 + np.random.normal(0, 0.5)
        self.sensors["temperature_2"].value += (thermal_load - 1.0) * 1.8 + np.random.normal(0, 0.4)

    def _update_pressure_sensors(self, coolant_multiplier: float):
        """Update pressure sensors based on coolant flow"""
        self.sensors["pressure_1"].value = 1.0 + (coolant_multiplier - 1.0) * 2.0 + np.random.normal(0, 0.1)

    def _update_vibration_sensors(self, intensity: float):
        """Update vibration sensors based on process intensity"""
        vibration_base = intensity * 0.2
        for axis in ["vibration_x", "vibration_y", "vibration_z"]:
            self.sensors[axis].value = vibration_base + np.random.normal(0, 0.05)

    def _update_power_consumption(self, action_effects: Dict):
        """Update power consumption based on process parameters"""
        base_power = 5.0
        load_power = action_effects["power_target"] * 15.0
        efficiency_factor = 0.8 + (1.0 - action_effects["process_intensity"]) * 0.15
        self.sensors["power_consumption"].value = (base_power + load_power) / efficiency_factor

    def _update_sensors(self):
        """Update all sensor readings with realistic dynamics"""
        current_time = datetime.now()
        noise_level = self.config["noise_level"]
        
        for sensor in self.sensors.values():
            sensor.timestamp = current_time
            noise = np.random.normal(0, noise_level * sensor.max_value * 0.01)
            sensor.value = np.clip(sensor.value + noise, sensor.min_value, sensor.max_value)

    def _update_production_metrics(self):
        """Update real-time production performance metrics"""
        # Calculate cycle time
        base_cycle_time = self.performance_targets["cycle_time"]
        efficiency_factor = self.process_parameters["feed_rate"] / 100.0
        tool_factor = 1.0 + self.process_parameters["tool_wear"] * 0.3
        self.production_metrics.cycle_time = base_cycle_time / efficiency_factor * tool_factor
        
        # Calculate throughput
        if self.production_metrics.cycle_time > 0:
            self.production_metrics.throughput = 3600.0 / self.production_metrics.cycle_time
        
        # Calculate quality metrics
        self._calculate_quality_metrics()
        
        # Calculate energy and efficiency metrics
        self._calculate_efficiency_metrics()

    def _calculate_quality_metrics(self):
        """Calculate quality-related production metrics"""
        tool_impact = 1.0 - self.process_parameters["tool_wear"] * 0.4
        process_stability = 1.0 - max(0, (self.sensors["vibration_x"].value - 0.5)) * 0.2
        temperature_impact = 1.0 - max(0, (self.sensors["temperature_1"].value - 50.0)) * 0.01
        self.production_metrics.quality_score = tool_impact * process_stability * temperature_impact

    def _calculate_efficiency_metrics(self):
        """Calculate efficiency and performance metrics"""
        # Energy consumption
        power_kw = self.sensors["power_consumption"].value
        cycle_hours = self.production_metrics.cycle_time / 3600.0
        self.production_metrics.energy_consumption = power_kw * cycle_hours
        
        # Overall Equipment Effectiveness
        availability = 0.95
        performance = min(1.0, self.production_metrics.throughput / self.performance_targets["throughput"])
        quality = self.production_metrics.quality_score
        self.production_metrics.oee = availability * performance * quality
        
        # Safety and maintenance scores
        self._calculate_safety_maintenance_scores()

    def _calculate_safety_maintenance_scores(self):
        """Calculate safety and maintenance-related scores"""
        # Safety score
        critical_sensors = sum(1 for s in self.sensors.values() if s.safety_level in [SafetyLevel.WARNING, SafetyLevel.CRITICAL])
        self.production_metrics.safety_score = max(0.0, 1.0 - critical_sensors * 0.1)
        
        # Maintenance score
        tool_health = 1.0 - self.process_parameters["tool_wear"]
        vibration_health = max(0.0, 1.0 - self.sensors["vibration_x"].value / 2.0)
        self.production_metrics.maintenance_score = (tool_health + vibration_health) / 2.0
        
        # Scrap rate
        base_scrap = 0.02  # 2% baseline
        quality_impact = (1.0 - self.production_metrics.quality_score) * 0.5
        self.production_metrics.scrap_rate = base_scrap + quality_impact

    def _check_safety_constraints(self) -> List[str]:
        """Check safety constraints and return violations"""
        if not self.config["safety_enabled"]:
            return []
        
        violations = []
        
        # Check critical sensor levels
        critical_sensors = {
            "temperature": ["temperature_1", "temperature_2"],
            "pressure": ["pressure_1"],
            "vibration": ["vibration_x", "vibration_y", "vibration_z"],
            "power": ["power_consumption"]
        }
        
        for category, sensor_names in critical_sensors.items():
            for sensor_name in sensor_names:
                if self.sensors[sensor_name].safety_level == SafetyLevel.CRITICAL:
                    violations.append(f"Critical {category}: {sensor_name}")
        
        # Check robot joint limits
        for i, (pos, limits) in enumerate(zip(self.robot_state.joint_positions, self.ROBOT_LIMITS["position_limits"])):
            if pos < limits[0] or pos > limits[1]:
                violations.append(f"Joint {i} position limit exceeded")
        
        # Check for emergency stop condition
        if len(violations) >= 3:
            self.emergency_stop = True
            violations.append("Emergency stop activated")
        
        self.safety_violations.extend(violations)
        return violations

    def _calculate_reward(self, action: int, safety_violations: List[str]) -> float:
        """Calculate comprehensive reward function"""
        reward_components = {}
        
        # Safety penalty (highest priority)
        reward_components["safety"] = self._calculate_safety_reward(safety_violations)
        
        # Performance rewards
        reward_components["throughput"] = min(1.0, self.production_metrics.throughput / self.performance_targets["throughput"]) * 50.0
        reward_components["cycle_time"] = self._calculate_cycle_time_reward() * 30.0
        reward_components["quality"] = self.production_metrics.quality_score * 40.0
        reward_components["energy"] = self._calculate_energy_reward() * 25.0
        reward_components["oee"] = self.production_metrics.oee * 45.0
        reward_components["maintenance"] = self.production_metrics.maintenance_score * 20.0
        reward_components["stability"] = self._calculate_stability_reward() * 15.0
        reward_components["tool_wear"] = -self.process_parameters["tool_wear"] * 30.0
        
        return sum(reward_components.values())

    def _calculate_safety_reward(self, safety_violations: List[str]) -> float:
        """Calculate safety-related reward component"""
        if not safety_violations:
            return 0.0
        
        penalty = -100.0 * len(safety_violations)
        if self.emergency_stop:
            penalty -= 500.0
        return penalty

    def _calculate_cycle_time_reward(self) -> float:
        """Calculate cycle time reward component"""
        target_cycle = self.performance_targets["cycle_time"]
        return max(0.0, 1.0 - abs(self.production_metrics.cycle_time - target_cycle) / target_cycle)

    def _calculate_energy_reward(self) -> float:
        """Calculate energy efficiency reward component"""
        target_energy = 0.1  # kWh per part
        return max(0.0, 1.0 - abs(self.production_metrics.energy_consumption - target_energy) / target_energy)

    def _calculate_stability_reward(self) -> float:
        """Calculate process stability reward component"""
        vibration_stability = max(0.0, 1.0 - self.sensors["vibration_x"].value / 2.0)
        temperature_stability = max(0.0, 1.0 - abs(self.sensors["temperature_1"].value - 35.0) / 50.0)
        return (vibration_stability + temperature_stability) / 2.0

    def _check_termination_conditions(self) -> bool:
        """Check if episode should terminate"""
        termination_conditions = [
            self.current_step >= self.episode_length,
            self.emergency_stop,
            len([s for s in self.sensors.values() if s.safety_level == SafetyLevel.CRITICAL]) >= 2,
            self.process_parameters["tool_wear"] >= 0.95
        ]
        
        return any(termination_conditions)

    def _get_state_vector(self) -> np.ndarray:
        """Get current state as normalized vector for RL agent"""
        state = np.zeros(self.state_size)
        
        # Robot joint positions (normalized)
        for i, pos in enumerate(self.robot_state.joint_positions[:6]):
            limits = self.ROBOT_LIMITS["position_limits"][i]
            state[i] = (pos - limits[0]) / (limits[1] - limits[0])
        
        # Key sensor readings (normalized)
        sensor_indices = {
            "temperature_1": 6, "pressure_1": 7, "vibration_x": 8, "power_consumption": 9
        }
        for sensor_name, index in sensor_indices.items():
            state[index] = self.sensors[sensor_name].normalized_value
        
        # Production metrics (normalized)
        metrics_data = [
            min(1.0, self.production_metrics.throughput / 300.0),
            self.production_metrics.quality_score,
            min(1.0, self.production_metrics.energy_consumption / 1.0),
            self.production_metrics.oee,
            self.production_metrics.safety_score
        ]
        state[10:15] = metrics_data
        
        # Process parameters (normalized)
        state[15] = self.process_parameters["tool_wear"]
        state[16] = self.process_parameters["feed_rate"] / 500.0
        state[17] = self.process_parameters["spindle_speed"] / 5000.0
        
        return state

    def _add_process_noise(self):
        """Add realistic process noise to sensors"""
        noise_level = self.config["noise_level"]
        for sensor in self.sensors.values():
            noise = np.random.normal(0, noise_level * (sensor.max_value - sensor.min_value) * 0.01)
            sensor.value += noise

    def _apply_disturbances(self):
        """Apply random disturbances to system"""
        if np.random.random() < self.config["disturbance_probability"]:
            disturbance_functions = {
                "thermal_spike": lambda: self._apply_thermal_disturbance(),
                "pressure_drop": lambda: self._apply_pressure_disturbance(),
                "vibration_burst": lambda: self._apply_vibration_disturbance(),
                "power_fluctuation": lambda: self._apply_power_disturbance(),
                "material_change": lambda: self._apply_material_disturbance()
            }
            
            disturbance_type = np.random.choice(list(disturbance_functions.keys()))
            disturbance_functions[disturbance_type]()

    def _apply_thermal_disturbance(self):
        """Apply thermal disturbance"""
        self.sensors["temperature_1"].value += np.random.uniform(5, 15)

    def _apply_pressure_disturbance(self):
        """Apply pressure disturbance"""
        self.sensors["pressure_1"].value *= np.random.uniform(0.7, 0.9)

    def _apply_vibration_disturbance(self):
        """Apply vibration disturbance"""
        for vib_sensor in ["vibration_x", "vibration_y", "vibration_z"]:
            self.sensors[vib_sensor].value += np.random.uniform(0.5, 1.5)

    def _apply_power_disturbance(self):
        """Apply power disturbance"""
        self.sensors["power_consumption"].value *= np.random.uniform(1.1, 1.3)

    def _apply_material_disturbance(self):
        """Apply material property disturbance"""
        self.process_parameters["material_hardness"] *= np.random.uniform(0.8, 1.2)

    def _compile_step_info(self, action_effects: Dict, safety_violations: List[str]) -> Dict:
        """Compile comprehensive step information"""
        return {
            "step": self.current_step,
            "process_state": self.process_state.value,
            "action_effects": action_effects,
            "safety_violations": safety_violations,
            "emergency_stop": self.emergency_stop,
            "production_metrics": self._get_metrics_dict(),
            "sensor_readings": {name: sensor.value for name, sensor in self.sensors.items()},
            "process_parameters": self.process_parameters.copy(),
            "robot_state": self._get_robot_state_dict()
        }

    def _get_metrics_dict(self) -> Dict:
        """Get production metrics as dictionary"""
        return {
            "cycle_time": self.production_metrics.cycle_time,
            "throughput": self.production_metrics.throughput,
            "quality_score": self.production_metrics.quality_score,
            "energy_consumption": self.production_metrics.energy_consumption,
            "oee": self.production_metrics.oee,
            "safety_score": self.production_metrics.safety_score,
            "scrap_rate": self.production_metrics.scrap_rate,
            "maintenance_score": self.production_metrics.maintenance_score
        }

    def _get_robot_state_dict(self) -> Dict:
        """Get robot state as dictionary"""
        return {
            "joint_positions": self.robot_state.joint_positions.tolist(),
            "joint_velocities": self.robot_state.joint_velocities.tolist(),
            "end_effector_pose": self.robot_state.end_effector_pose.tolist(),
            "is_moving": self.robot_state.is_moving
        }

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
        
        # Maintain history size
        if len(self.sensor_history) > MAX_HISTORY_SIZE:
            self.sensor_history = self.sensor_history[-HISTORY_TRIM_SIZE:]
            self.metrics_history = self.metrics_history[-HISTORY_TRIM_SIZE:]

    # Public interface methods
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

    def get_state_size(self) -> int:
        """Get size of state vector"""
        return self.state_size

    def get_action_size(self) -> int:
        """Get size of action space"""
        return self.action_size

    def is_done(self) -> bool:
        """Check if current episode is complete"""
        return self._check_termination_conditions()

    def get_config(self) -> Dict:
        """Get environment configuration"""
        return self.config.copy()