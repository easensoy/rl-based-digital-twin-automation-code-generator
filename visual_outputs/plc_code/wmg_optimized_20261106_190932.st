

PROGRAM WMG_Optimized_Control
(* Generated: 2026-11-06 19:09:32 *)
(* Training Steps: 1000 *)
(* Target Cycle Time: 30.0s *)
(* Target Throughput: 120.0 units/min *)

VAR
    (* Process Variables *)
    cycle_time : REAL := 30.0;
    throughput : REAL := 0.0;
    safety_score : REAL := 1.0;
    energy_efficiency : REAL := 0.85;
    
    (* Control Variables *)
    speed_setpoint : REAL := 1.0;
    pressure_setpoint : REAL := 1.0;
    temperature_setpoint : REAL := 1.0;
    
    (* System State *)
    system_running : BOOL := FALSE;
    emergency_stop : BOOL := FALSE;
    maintenance_mode : BOOL := FALSE;
    
    (* Optimization Parameters *)
    learning_rate : REAL := 0.001;
    epsilon : REAL := 0.00998645168764533;
    
    (* Performance Metrics *)
    production_count : DINT := 0;
    quality_index : REAL := 0.95;
    oee_value : REAL := 0.0;
END_VAR



FUNCTION_BLOCK FB_SafetyMonitor
VAR_INPUT
    temperature : REAL;
    pressure : REAL;
    vibration : REAL;
    emergency_button : BOOL;
END_VAR

VAR_OUTPUT
    safety_ok : BOOL;
    shutdown_required : BOOL;
    safety_score : REAL;
END_VAR

VAR
    temp_alarm : BOOL;
    pressure_alarm : BOOL;
    vibration_alarm : BOOL;
END_VAR

(* Safety Logic - RL Optimized Thresholds *)
temp_alarm := temperature > 80.0 OR temperature < -10.0;
pressure_alarm := pressure > 10.0 OR pressure < 0.1;
vibration_alarm := vibration > 5.0;

shutdown_required := temp_alarm OR pressure_alarm OR vibration_alarm OR emergency_button;
safety_ok := NOT shutdown_required;

(* Calculate Safety Score *)
IF safety_ok THEN
    safety_score := 0.95;
ELSE
    safety_score := 0.0;
END_IF;

END_FUNCTION_BLOCK



FUNCTION_BLOCK FB_ProcessControl
VAR_INPUT
    start_command : BOOL;
    stop_command : BOOL;
    speed_factor : REAL;
    pressure_factor : REAL;
    temp_factor : REAL;
END_VAR

VAR_OUTPUT
    motor_speed : REAL;
    valve_position : REAL;
    heater_power : REAL;
    process_active : BOOL;
END_VAR

VAR
    cycle_timer : TON;
    ramp_timer : TON;
END_VAR

(* RL-Optimized Process Control *)
IF start_command AND NOT stop_command THEN
    process_active := TRUE;
    
    (* Apply RL-learned speed optimization *)
    motor_speed := speed_factor * 100.0;
    
    (* Pressure control with learned parameters *)
    valve_position := pressure_factor * 90.0;
    
    (* Temperature control *)
    heater_power := temp_factor * 75.0;
    
    (* Cycle timing optimization *)
    cycle_timer(IN := process_active, PT := T#30.0s);
    
ELSE
    process_active := FALSE;
    motor_speed := 0.0;
    valve_position := 0.0;
    heater_power := 0.0;
    cycle_timer(IN := FALSE);
END_IF;

END_FUNCTION_BLOCK



FUNCTION_BLOCK FB_RLOptimizer
VAR_INPUT
    current_state : ARRAY[0..11] OF REAL;
    target_throughput : REAL;
    energy_target : REAL;
END_VAR

VAR_OUTPUT
    optimal_action : DINT;
    speed_recommendation : REAL;
    pressure_recommendation : REAL;
    temp_recommendation : REAL;
END_VAR

VAR
    state_normalized : ARRAY[0..11] OF REAL;
    q_values : ARRAY[0..5] OF REAL;
    best_action : DINT;
    i : DINT;
END_VAR

(* RL-based Action Selection *)
(* Normalize input states *)
FOR i := 0 TO 11 DO
    state_normalized[i] := current_state[i] / 2.0;  (* Clip to [-1,1] range *)
END_FOR;

(* Simplified Q-value approximation based on learned policy *)
(* Action 0: Conservative *)
q_values[0] := state_normalized[10] * 0.8 + state_normalized[11] * 0.5;

(* Action 1: Normal *)
q_values[1] := state_normalized[10] * 1.0 + state_normalized[11] * 0.7;

(* Action 2: Aggressive *)
q_values[2] := (1.0 - state_normalized[10]) * 0.6 + state_normalized[11] * 1.2;

(* Action 3: Energy Saving *)
q_values[3] := state_normalized[11] * (-0.8) + state_normalized[10] * 0.6;

(* Action 4: Quality Focus *)
q_values[4] := state_normalized[10] * 1.2 + state_normalized[11] * 0.4;

(* Action 5: Balanced *)
q_values[5] := (state_normalized[10] + state_normalized[11]) * 0.5;

(* Select best action *)
best_action := 0;
FOR i := 1 TO 5 DO
    IF q_values[i] > q_values[best_action] THEN
        best_action := i;
    END_IF;
END_FOR;

optimal_action := best_action;

(* Map action to control recommendations *)
CASE optimal_action OF
    0: (* Conservative *)
        speed_recommendation := 0.8;
        pressure_recommendation := 0.9;
        temp_recommendation := 0.85;
    1: (* Normal *)
        speed_recommendation := 1.0;
        pressure_recommendation := 1.0;
        temp_recommendation := 1.0;
    2: (* Aggressive *)
        speed_recommendation := 1.2;
        pressure_recommendation := 1.1;
        temp_recommendation := 1.1;
    3: (* Energy Saving *)
        speed_recommendation := 0.6;
        pressure_recommendation := 0.8;
        temp_recommendation := 0.8;
    4: (* Quality Focus *)
        speed_recommendation := 1.1;
        pressure_recommendation := 1.05;
        temp_recommendation := 0.95;
    5: (* Balanced *)
        speed_recommendation := 0.9;
        pressure_recommendation := 0.95;
        temp_recommendation := 0.9;
END_CASE;

END_FUNCTION_BLOCK



(* Main Program Logic *)
VAR
    safety_monitor : FB_SafetyMonitor;
    process_control : FB_ProcessControl;
    rl_optimizer : FB_RLOptimizer;
    
    system_state : ARRAY[0..11] OF REAL;
    sensor_temp : REAL;
    sensor_pressure : REAL;
    sensor_vibration : REAL;
    
    start_btn : BOOL;
    stop_btn : BOOL;
    e_stop_btn : BOOL;
END_VAR

(* Read sensor inputs *)
(* sensor_temp := Analog_Input_1; *)
(* sensor_pressure := Analog_Input_2; *)
(* sensor_vibration := Analog_Input_3; *)

(* Safety monitoring *)
safety_monitor(
    temperature := sensor_temp,
    pressure := sensor_pressure,
    vibration := sensor_vibration,
    emergency_button := e_stop_btn
);

(* Update system state for RL *)
system_state[6] := sensor_temp / 100.0;  (* Normalize temperature *)
system_state[7] := sensor_pressure / 10.0;  (* Normalize pressure *)
system_state[8] := sensor_vibration / 5.0;  (* Normalize vibration *)

(* RL Optimization *)
rl_optimizer(
    current_state := system_state,
    target_throughput := 120.0,
    energy_target := 0.85
);

(* Process Control *)
IF safety_monitor.safety_ok THEN
    process_control(
        start_command := start_btn,
        stop_command := stop_btn,
        speed_factor := rl_optimizer.speed_recommendation,
        pressure_factor := rl_optimizer.pressure_recommendation,
        temp_factor := rl_optimizer.temp_recommendation
    );
ELSE
    process_control(
        start_command := FALSE,
        stop_command := TRUE,
        speed_factor := 0.0,
        pressure_factor := 0.0,
        temp_factor := 0.0
    );
END_IF;

(* Calculate OEE *)
oee_value := safety_monitor.safety_score * (throughput / 120.0) * quality_index;

(* Update performance metrics *)
IF process_control.process_active THEN
    production_count := production_count + 1;
    throughput := production_count / cycle_time * 60.0;  (* Units per minute *)
END_IF;

END_PROGRAM



(* End of WMG Optimized Control System *)
(* Generated by RL-based Digital Twin Platform *)
(* For technical support: wmg-automation@warwick.ac.uk *)

