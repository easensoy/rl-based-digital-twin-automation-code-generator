# Generated PLC code: `wmg_optimized_20261106_190932.st`

**How it was produced.** `POST /api/code/generate` on the running backend, after the 1,000-episode
training run (seed 42). Call chain: `WMGDigitalTwinServer.generate_optimized_code()` →
`DQNAgent.get_optimal_policy()` → `PLCCodeGenerator.generate_code()`. The backend writes the file to
`data/generated_code/`, and a copy is kept here. The file is 290 lines.

**Language and target.** IEC 61131-3 **Structured Text**: `PROGRAM`, `FUNCTION_BLOCK`, `VAR_INPUT/VAR_OUTPUT`,
`TON`, `CASE`, `FOR`. There is no vendor target, no I/O mapping and no `CONFIGURATION/RESOURCE/TASK`.
Docstrings and the UI call the output "IEC 61499 compliant", but it contains **no IEC 61499 construct**:
no event inputs/outputs, no `WITH` associations, no ECC, no `.fbt` XML, and no device or resource model.

See `../07_plc_code_structure.png` for the block diagram.

## Structure (line numbers refer to the generated file)

| Lines | Block | Content |
|---|---|---|
| 3–34 | `PROGRAM WMG_Optimized_Control` header and first `VAR` block | Process variables, setpoints and flags. **L5** `(* Training Steps: 1000 *)` and **L28** `epsilon : REAL := 0.00998645…` are the **only two values that come from the training run**. `learning_rate := 0.001` is a template literal. |
| 38–73 | `FUNCTION_BLOCK FB_SafetyMonitor` | Fixed thresholds: temp > 80 or < −10, pressure > 10 or < 0.1, vibration > 5, plus the e-stop. Outputs `safety_ok` and `shutdown_required`. `safety_score := 0.95` (the config `safety_threshold`) when OK. |
| 77–122 | `FUNCTION_BLOCK FB_ProcessControl` | `motor_speed := speed_factor*100`, `valve_position := pressure_factor*90`, `heater_power := temp_factor*75`. `cycle_timer : TON` runs with `PT := T#30.0s`, but its `Q` is never read. `ramp_timer` is declared and never used. |
| 126–210 | `FUNCTION_BLOCK FB_RLOptimizer` | Divides the 12 inputs by 2. Computes six "Q-values" as **fixed linear formulas** in `state[10]` (quality) and `state[11]` (energy) (L155–170), then takes the argmax. A `CASE` maps the chosen action to the same speed/pressure/temperature multipliers as `AutomationEnvironment._apply_action()`. |
| 214–283 | Second `VAR` block and main body (same `PROGRAM`) | Calls `safety_monitor`, writes `system_state[6..8]` only (L244–246), calls `rl_optimizer`, then calls `process_control` (forced stop when `NOT safety_ok`). Computes OEE and increments `production_count` on every scan (L279–280). |
| 287–289 | Footer | Comments. |

## What the trained policy contributes

`get_optimal_policy()` returns the Q-network `state_dict` (35,462 parameters), ε and the step count.
`generate_code()` reads **only ε and the step count**. The network weights are never used, so a run that
learned something different produces a file that differs in only those two values.

## What the code actually does (line-for-line emulation, `scripts/evaluate_policies.py`)

* With the main-program wiring, `system_state[10]` and `[11]` are never written, so they stay 0.0. The
  Q-values become `[0, 0, 0.6, 0, 0, 0]`, and **the program always selects action 2 "Aggressive"**,
  whatever the sensor values.
* Fed the full 12-value twin state instead, it still locks onto action 2.
* Agreement with the trained DQN, which always picks "Quality focus": **0.0%** over 20,000 visited
  states.
* On the backend's twin, "Aggressive" scores **1,622** reward per episode, the lowest of all six actions.
  The best fixed action ("Energy saving") scores 1,827.
* The sensor reads are commented out (L231–233), so `sensor_pressure` stays 0.0. `FB_SafetyMonitor` then
  raises `pressure_alarm` (`pressure < 0.1`), and **the program as emitted holds the process in a
  permanent safety stop** until real I/O is mapped.

## IEC 61131-3 conformance (why it does not compile as one unit)

1. **Nested POUs.** `FUNCTION_BLOCK` declarations (L38, L77, L126) sit between `PROGRAM` (L3) and
   `END_PROGRAM` (L283). POUs cannot be nested.
2. **Split declaration part.** The `PROGRAM` has two `VAR` blocks (L9 and L215) separated by other POUs.
3. **Type mixing.** L280 divides `DINT` by `REAL`. CODESYS accepts this with a warning; a strict IEC check,
   or TIA Portal with "IEC check" enabled, needs `DINT_TO_REAL`.
4. **Semantics.** L279 counts PLC scans as parts produced.
5. **No execution model.** There is no task, no cycle time and no I/O mapping.

## Portability

| Target | Usable as-is? | Work needed |
|---|---|---|
| **CODESYS / TwinCAT 3** (IEC 61131-3) | No; closest fit | Split into 3 FBs + 1 PROGRAM, merge the PROGRAM's two `VAR` blocks, map `sensor_*` and `*_btn` to I/O, assign the PROGRAM to a task. The FB bodies are then standard ST. |
| **TIA Portal** (S7-1200/1500 SCL) | No | TIA has no `PROGRAM` POU: move the main logic into an FB called from OB1 (or into OB1 directly). Wrap each block in SCL source syntax (quoted names, `BEGIN`), use TON multi-instances and an instance DB, add explicit conversions. |
| **EAE** (EcoStruxure Automation Expert, IEC 61499) | No | Re-model each FB as a Basic FB with event inputs/outputs (e.g. `INIT`/`REQ` to `INITO`/`CNF`), `WITH` associations and an ECC whose state runs the ST body as an algorithm. Replace the `PROGRAM` sequencing with an event-connected FB network (or a CAT), and bind I/O through the EAE hardware configuration. |

In every case the result would still not contain the learned policy. That needs the weights exported,
for example as constant arrays plus a matrix-vector loop in an ST algorithm. For a 12-128-128-128-6 MLP
this is about 35k multiply-accumulates per decision.
