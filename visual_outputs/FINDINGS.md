# Findings: `rl-based-digital-twin-automation-code-generator` @ `898c112`

Detailed companion to `REPORT.md` (the one-page summary). Repo location:
`C:\rl-based-digital-twin-automation-code-generator`, branch `main`, HEAD `898c112` (2025-08-05).
Remote: `https://github.com/easensoy/rl-based-digital-twin-automation-code-generator.git`.
No source file under `backend/`, `frontend/`, `config/`, `tests/` or the HTML was edited. Everything
added is listed in the [Files created by this review](#files-created-by-this-review) section.

---

## 1. Code map

| Folder / file | What it is | State |
|---|---|---|
| `backend/main.py` (535 lines) | `WMGDigitalTwinServer`: FastAPI app, static mounts, REST + WebSocket routes, training loop, code generation | Runs. Must be started from the repo root as `python backend/main.py`, because it uses flat imports (L26) and CWD-relative paths. |
| `backend/rl_agent.py` (292) | `DQNNetwork`, `ReplayBuffer`, `DQNAgent` **and the `AutomationEnvironment` actually used** (12 states, 6 actions) | Runs |
| `backend/automation_environment.py` (708) | A second, richer `AutomationEnvironment`: 18 states, 8 actions, sensor configs with units/thresholds, disturbances, tool wear, robot joint limits | **Dead code.** Never imported. Its interface is incompatible (`step()` returns a 4-tuple, and the constructor expects other config keys). The last commit (`898c112`) changed only this file. |
| `backend/plc_generator.py` (357) | `PLCCodeGenerator`: f-string templates that emit Structured Text | Runs. See `plc_code/PLC_CODE_EXPLAINED.md`. |
| `backend/websocket_handler.py` (77) | `WebSocketConnectionManager`: connect / disconnect / broadcast | Runs. Its three `send_*` helpers are unused. |
| `config/config.json` | Server host/port, DQN hyperparameters, automation targets | Loaded with a shallow `dict.update` |
| `data/` (untracked) | `training_logs/wmg_platform.log` (runtime log), `generated_code/` (output `.st`) | **Required at import time:** main.py L35 opens the log file when the module loads, so a fresh clone without `data/` crashes. |
| `frontend/main.js` (841) | `WMGDigitalTwin`: UI, Three.js scene, simulation fallback | Aborts during initialisation (see below) |
| `frontend/js/core/` | `websocket-manager.js` (ES module: reconnect, heartbeat, queue), `event-dispatcher.js` (priority event bus), `state-manager.js` (Redux-style store, not loaded by the HTML) | The WebSocket manager fails to load |
| `frontend/js/robotics/` | `inverse-kinematics.js` (DH forward/inverse kinematics, Jacobian), `robot-visualizer.js` (Three.js robot cells), `robot-controller.js` (trajectory planner, safety monitor, program executor) | The first two fail to load. None of the three is used by `main.js`. |
| `frontend/js/ui/` | `code-display.js`, `performance-dashboard.js`, `training-controls.js` | **All three are 0 bytes** |
| `frontend/css/` | Theme and component styles | The HTML links only `wmg-theme.css` and `dashboard.css` |
| `tests/` | `test_rl_agent.py`, `test_kinematics.js` | **Both are 0 bytes** |
| `wmg-digital-twin.html` | The single page, served at `/` | See next section |
| `README.md` | n/a | **0 bytes** (empty since `c28ad51`) |

### What `wmg-digital-twin.html` shows

A three-column dashboard:
* left: SYSTEM, TRAINING (Start/Stop/Reset) and CODE (Generate/Download) panels
* centre: a "Robot Digital Twin" Three.js canvas
* right: METRICS and six joint sliders
* bottom: SYSTEM LOG

The displayed numbers are **hard-coded in the markup**: Episode 525, ε 0.072, OEE 91.8%, Reward 1628.22,
and the log lines "Episode 519…525".

Verified at runtime in headless Edge (`screenshots/01_frontend_console_log.txt`):

1. **Module loading fails.** `websocket-manager.js`, `inverse-kinematics.js` and `robot-visualizer.js`
   use `export`, but `wmg-digital-twin.html` L296–298 loads them as classic scripts. Result: 3×
   `SyntaxError: Unexpected token 'export'`, `WebSocketManager` stays undefined, and the page falls back
   to "Simulation".
2. **3D initialisation aborts.** `initialize3D()` calls `this.createSimpleRobot()` (`main.js` L160), but
   the class defines `createAdvancedRobot()`. The catch block then calls `createFallbackCanvas()` (L176),
   which is also undefined. `initialize()` therefore aborts before `setupUIHandlers()` and
   `startUpdateLoops()` run. The buttons and sliders get no handlers, the canvas stays black, and the
   metrics never change. This was introduced by `c01f6b6` (2025-08-04).
3. **Wrong WebSocket address.** Even if the manager loaded, `main.js` L70 hard-codes
   `ws://localhost:8000/ws`, and there is no handler for the backend's `plc_code` message, so Generate
   would stay on "Generating…".

`screenshots/03_DIAGNOSTIC_…png` shows what `createAdvancedRobot()` would draw if it were reached. It is
not the app's behaviour: the method was called by hand from the DevTools protocol. Its parts are
independent meshes, not a kinematic chain, and they are not driven by the backend.

### One full run, traced

| Step | Classes / functions (file) |
|---|---|
| Start | `main()` → `WMGDigitalTwinServer.__init__` → `load_configuration`, `setup_middleware`, `setup_static_files`, `setup_routes`, `initialize_rl_components` → `AutomationEnvironment(config["automation"])`, `DQNAgent(12, 6, config["rl_agent"])` (`main.py`, `rl_agent.py`) |
| Trigger | `POST /api/training/start` or WS `{"action":"start_training"}` → `start_rl_training` → `asyncio.create_task(training_loop)` |
| Loop | `training_loop`: while active and `current_episode < 1000` (hard-coded, L65) → `run_training_episode` → `broadcast_training_progress`, `broadcast_performance_update` → `sleep(0.1)` (L359) |
| Process state | `AutomationEnvironment.reset()`: 12-vector ~ N(0, 0.1) [pos×3, vel×3, temp, pressure, vibration, load, quality, energy] |
| Observe and act | `DQNAgent.act(state)`: ε-greedy over `DQNNetwork` (12→128→128→128→6 MLP, ReLU, dropout 0.2, 35,462 parameters) |
| Environment step | `step()` → `_apply_action` (6 speed/pressure/temperature multiplier sets) → `_update_dynamics` (first-order lags, random vibration, quality = 0.95·(1−0.3·\|speed−1\|), energy = speed·pressure·temp, cycle counter) → `_calculate_reward` → `_is_episode_done` (200 steps) |
| Reward | quality·5 + energy-efficiency·3 + throughput/120·10 + 2 (cycle bonus, always paid, L252) − 5 (temp/pressure above 1.5; unreachable, L243) − 2 (vibration above 1; practically unreachable) |
| Learn | `remember` → `ReplayBuffer.push`. After the episode, **one** `replay()` call (L403): sample 64, MSE between Q(s,a) and r + 0.99·max Q_target(s′), Adam step, ε ← ε·0.995, target sync every 10 replays |
| Metrics | `get_performance_metrics()`: cycle_time (current cycle counter), throughput (parts per 100 steps), safety, energy efficiency, OEE |
| Generate code | `POST /api/code/generate` → `is_trained()` (replay buffer above 1,000 transitions, L113) → `get_optimal_policy()` → `PLCCodeGenerator.generate_code()` → `data/generated_code/wmg_optimized_<ts>.st` |

**PLC language and target.** The generator emits IEC 61131-3 Structured Text with no vendor target. It is
not IEC 61499. It is not usable as-is in EAE or TIA Portal, and not compilable as one unit even in
CODESYS. The detail and porting steps are in `plc_code/PLC_CODE_EXPLAINED.md`.

---

## 2. Claims check

`README.md` is empty, so the claims are taken from the docstrings, UI strings, commit messages and
`requirements.txt`.

| Claim | Status | Evidence |
|---|---|---|
| **Real-time digital twin** | **Partial** (backend) / **missing** (frontend) | The backend broadcasts once per **episode**, not per process step: 1,000 `training_progress` and 1,000 `performance_update` messages were captured (`run/ws_messages.jsonl`). The browser never connects and shows hard-coded values. The "twin" is a synthetic 12-variable stochastic model with no link to a plant, PLC, OPC UA or the 3D robot, and robot joints are not part of the RL state. |
| **Automated PLC code generation from the learned policy** | Generation **implemented**; "from the policy" **missing** | A 290-line `.st` is produced. From the policy only ε and the step count are used; the Q-network weights are ignored. `FB_RLOptimizer` coefficients are hard-coded. Agreement with the trained DQN is **0.0%**. The code always picks "Aggressive". |
| **IEC 61499 compliant** | **Missing** | The output is IEC 61131-3 ST with no events, ECC, `WITH` or `.fbt`. It is also not valid 61131-3 as a single unit (nested POUs). |
| **PyTorch DQN** | **Implemented (basic)** | Q-network, target network, replay buffer and ε-greedy are present. Deviations: one gradient step per 200 environment steps, dropout without `eval()`, no checkpointing, no loss logging, "trained" means the buffer holds more than 1,000 transitions. Training does not converge: median loss goes from 71.5 to 7,847. |
| **GPU support** | **Partial** | The device selects CUDA if available (`rl_agent.py` L58). The pinned `torch==2.1.0` from PyPI is a CPU-only build on Windows, so CUDA is unavailable on this PC's RTX 1000 Ada. |
| **Industrial connectivity** (OPC UA, Modbus) | **Missing** | `opcua` and `pymodbus` are pinned but never imported. There is no PLC or fieldbus link. |
| **EAE / TIA Portal compatibility** | **Missing** | Implied by the "IEC 61499" wording. See the portability table in `plc_code/PLC_CODE_EXPLAINED.md`. |
| **Safety constraints** | **Partial** | The environment's safety penalty is unreachable (temperature and pressure settle at 1.1 or below, threshold 1.5): 0 violations in 200,000 steps. The generated safety FB uses fixed literals. |
| **Configuration-driven sensors and actions** (last commit) | Implemented **in dead code** | `SENSOR_CONFIGS` and `ACTION_CONFIGS` live in the unused `automation_environment.py`. |
| **Robot IK / trajectory control** | Present, **unreachable** | The ES modules fail to load, and `main.js` does not use them. |
| Auth, task queue, metrics (`python-jose`, `passlib`, `celery`, `redis`, `prometheus-client`, `structlog`) | **Missing** | None is imported. `passlib` 1.7.4 with the resolved `bcrypt` 5.0.0 also fails at runtime. |

### The PDFs

* **The two "GPU" PDFs are the same article.** Schmidt, Schellroth, Fischer, Allimant and Riedel (ISW,
  University of Stuttgart), *Reinforcement learning methods based on GPU accelerated industrial control
  hardware*, Neural Computing and Applications 33 (2021) 12191–12207, doi:10.1007/s00521-021-05848-4,
  CC BY 4.0. `Reinforcement_learning_methods_based_on_GPU_accele.pdf` is a ResearchGate copy with the
  Springer Nature terms page appended.
* **What it proposes.**
  * Three requirements for RL in production: few experiments, nonlinear processes, and real-time
    execution on control hardware (1–100 ms cycles).
  * An evaluation on a physical ball-on-rods testbed with a 4 ms PLC cycle:
    * NEAT needs 6,750 agents and no prior knowledge.
    * Bayesian optimisation with Gaussian processes needs about 500 experiments, plus a prior control
      structure.
    * CNN agents are too heavy for real time on a standard PLC.
  * A taxonomy of four PLC programming architectures: pure IEC 61131; integrated C/C++; a separate
    real-time process on the same machine; and an external system over fieldbus.
  * The proposal itself is architecture 3: a CODESYS Control for Linux SL PLC that exchanges data through
    shared-memory lock-free queues with a PREEMPT_RT process running TensorRT on a GPU.
  * Measured results: about 4.5 µs PLC↔process round trip with about 10 µs jitter, and about 3 ms PLC
    cycles including TinyYOLOv2 inference.
* **How much the repo reflects it.** Only the motivation (RL to create and tune control programs) and a
  `cuda if available` line. None of the following is present: real-time constraints, PLC runtime
  integration in any of the four architectures, deployment of the learned agent to a controller, a
  physical or physics-based testbed, sample-efficient methods, or latency and jitter measurement.
* `Reinforcement Learning Intro Stanford.pdf` is the Sutton & Barto textbook, 2nd edition draft
  (2014/15), as background reading.

### The 9 uncommitted changes (as found at the start)

| # | Path | What it is |
|---|---|---|
| 1 | `M backend/__pycache__/rl_agent.cpython-311.pyc` | Bytecode. The committed copy (177 B) records source size 0, so it was compiled from the empty placeholder in `c28ad51`. The working copy (18,388 B) matches the current `rl_agent.py` (mtime 2025-08-02 13:18:23, 11,441 B). This is a build artefact and should not be tracked. |
| 2 | `M requirements.txt` | Adds `aiofiles 23.2.1`, `jinja2 3.1.2`, `python-dotenv 1.0.0`, `redis 5.0.1` and `celery 5.3.4` under "Additional Dependencies for Full Functionality", and drops the final newline. None is imported. All five install cleanly. |
| 3 | `?? Reinforcement Learning Intro Stanford.pdf` | Sutton & Barto draft textbook (4.2 MB) |
| 4 | `?? Reinforcement learning methods … hardware.pdf` | Schmidt et al. 2021 (0.99 MB) |
| 5 | `?? Reinforcement_learning_methods_based_on_GPU_accele.pdf` | Duplicate of #4 with a terms page (0.90 MB) |
| 6 | `?? backend/__pycache__/plc_generator.cpython-311.pyc` | Bytecode matching the current source |
| 7 | `?? backend/__pycache__/websocket_handler.cpython-311.pyc` | Bytecode matching the current source |
| 8 | `?? data/` | Runtime output (log and generated code). Needed at import time (see above). |
| 9 | `?? wmg_env/` | Old venv (248 MB). Its base interpreter (`…\Python311\python.exe`, 3.11.9) no longer exists, so it is broken. |

The repo has **no `.gitignore`**, which is why items 1, 6, 7 and 9 show up at all.

---

## 3. Setup and tests

* **Python on this PC:** 3.13.5 (default, `py`), uv-managed CPython 3.11.14 and 3.15.2, and MSYS2 3.12.7.
  The 3.11.9 interpreter used by `wmg_env` is gone.
* **`.venv`:** created with `…\uv\python\cpython-3.11-windows-x86_64-none\python.exe -m venv .venv`.
  This uses the plain stdlib venv; no uv command was run and nothing global was changed. The pins require
  Python 3.11 or older (`torch 2.1.0`, `numpy 1.24.3` and `pandas 2.0.3` have no cp313 wheels).
  * **All 34 pinned requirements installed. `pip check` is clean.**
  * One latent conflict: `passlib 1.7.4` with the unpinned transitive `bcrypt 5.0.0` fails at runtime
    (`ValueError: password cannot be longer than 72 bytes`). It is unused, but `bcrypt<4.1` should be
    pinned if auth is ever wired up.
  * Note: `.venv` points at uv's 3.11 junction, so `uv python uninstall 3.11` would break it, in the same
    way the old `wmg_env` broke.
* **GPU:**
  * Hardware: NVIDIA RTX 1000 Ada Generation Laptop GPU (6 GB, compute capability 8.9, driver 596.71,
    CUDA 13.2).
  * `torch 2.1.0+cpu`: **`torch.cuda.is_available()` is False.** Using the GPU needs a CUDA build such as
    `torch==2.1.0+cu121` from `download.pytorch.org`. It was not installed, to keep the pins (about
    2.5 GB).
  * For this workload a GPU would not help. The model is a 35k-parameter MLP with one batch-64 update per
    episode, and 100 s of the 189 s run is the server's own per-episode sleep.
* **Frontend:** nothing to install (static files served by FastAPI; three.js r128 comes from cdnjs).
* **Tests:** **none exist.** `pytest tests` collected 0 items (exit code 5). `node tests/test_kinematics.js`
  ran an empty file (exit code 0). See `run/test_results.txt`.

---

## 4. Run

* Port 8000 was already in use by another local process (a uv Python 3.15 app, PID 55884, left
  untouched), so the backend ran on **127.0.0.1:8001**. It was started through
  `scripts/run_instrumented_server.py`: the unmodified backend plus read-only observers for loss, actions
  and traces, with seed 42. It bound 127.0.0.1 rather than 0.0.0.0, because the API has no auth and
  allows CORS `*`.
* `POST /api/code/generate` before training returned **400 "Model not trained"**.
* `POST /api/training/start` ran 1,000 episodes (200,000 steps, 1,000 gradient steps) in **188.8 s**.
* `POST /api/code/generate` then returned 200 and wrote
  `data/generated_code/wmg_optimized_20261106_190932.st`, copied to `plc_code/`. The Q-network snapshot
  used for that generation is `run/policy_snapshot_20261106_190932.pt`.

### Results (100 evaluation episodes per policy, identical seeds; `run/evaluation.json`)

| Policy | Reward / episode | Energy eff. | Quality | Throughput (/100 steps) |
|---|---|---|---|---|
| Uniform random | 1,716.0 | 0.530 | 0.951 | 3.50 |
| **Before:** untrained DQN, greedy | 1,713.7 | 0.530 | 0.961 | 3.12 |
| **After:** trained DQN, greedy (always "Quality focus") | **1,682.6** | 0.477 | 0.961 | 3.00 |
| Generated ST (always "Aggressive") | 1,622.0 | 0.408 | 0.947 | 2.50 |
| Best fixed action ("Energy saving") | **1,827.4** | 0.723 | 0.918 | 5.50 |

Training changed greedy reward by **−31.1 (−1.8%)**. The learned policy is a single fixed action. Across
6 seeds (42 plus `run/multi_seed_check.json`) the chosen action varies with the seed, and no seed finds
"Energy saving". The cause can be read off the Q-values: the gap between the best and second-best action
values (about 9 on average) is far below the TD error (RMSE about 88), so the greedy choice is set by
noise.

### Other defects found (file:line)

* `backend/main.py:217`: `await` on the synchronous `disconnect()` raises `TypeError` in the generic
  WebSocket error path.
* `backend/main.py:85/121`: binds `0.0.0.0` with CORS `*` and no auth on endpoints that start training
  and write files.
* `backend/plc_generator.py:318`: reads the key `energy_efficiency_target`, but the targets dict
  (`main.py:504`) uses `energy_efficiency`, so the value always falls back to 0.85.
* `backend/rl_agent.py:278`: OEE compares throughput in parts per 100 steps against a 120 "units/min"
  target, so the backend reports OEE ≈ 2.6% while the UI hard-codes 91.8%.
* `backend/rl_agent.py:122`: `reset()` keeps the network weights, and nothing is saved to disk.

---

## Files created by this review

| Path | What it is |
|---|---|
| `.venv/` | The new environment |
| `.claude/launch.json` | Preview server config |
| `visual_outputs/` | Everything in this folder |
| `data/generated_code/wmg_optimized_20261106_190932.st` | Written by the backend itself |
| `data/training_logs/wmg_platform.log` | Appended by the backend |

No file under version control was modified.
