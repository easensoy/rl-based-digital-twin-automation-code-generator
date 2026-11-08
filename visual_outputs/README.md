# visual_outputs: index

Start with these two files:

* **`REPORT.md`** / **`REPORT.html`**: the one-page WMG report. The HTML prints to one A4 page.
* **`FINDINGS.md`**: the detailed notes. It covers the code map, the run trace, the claims check, the
  PDFs, the 9 uncommitted changes, setup, tests and the defect list.

## Charts (training run: seed 42, 1,000 episodes, default `config/config.json`)

| File | Shows |
|---|---|
| `01_reward_curve.png` | Episode reward with evaluation reference lines |
| `02_loss_curve.png` | TD loss per gradient step (log scale) |
| `03_epsilon_decay.png` | ε schedule vs the measured share of random actions |
| `04_action_distribution.png` | Greedy action share over training; final policy vs generated ST |
| `05_before_after_process_metrics.png` | Reward, energy efficiency, quality and throughput per policy |
| `06_digital_twin_process_state.png` | Process-state traces, episode 0 vs episode 999 |
| `07_plc_code_structure.png` | Block diagram of the generated `.st` file |
| `data/*.csv` | The data behind each chart |

## Other folders

| Folder | Contents |
|---|---|
| `screenshots/` | Frontend served by the backend (01), the HTML opened as a file (02), a labelled diagnostic of the unreachable 3D model (03), and the console logs |
| `plc_code/` | Generated `.st` file and `PLC_CODE_EXPLAINED.md` |
| `run/` | Raw telemetry (`episodes.jsonl`, `traces.jsonl`, `ws_messages.jsonl`), API responses, `evaluation.json`, `multi_seed_check.json`, the policy snapshot `.pt` and the test output |
| `scripts/` | Everything needed to reproduce the outputs (below) |

## Reproduce (from the repo root, Windows)

```bat
.venv\Scripts\python.exe visual_outputs\scripts\run_instrumented_server.py --seed 42 --port 8001
REM in a second terminal:
curl -X POST http://127.0.0.1:8001/api/training/start
REM wait ~190 s (GET /api/system/status until training_active is false), then:
curl -X POST http://127.0.0.1:8001/api/code/generate
copy data\generated_code\wmg_optimized_*.st visual_outputs\plc_code\
.venv\Scripts\python.exe visual_outputs\scripts\evaluate_policies.py --episodes 100
.venv\Scripts\python.exe visual_outputs\scripts\multi_seed_check.py --seeds 1 2 3 4 5
.venv\Scripts\python.exe visual_outputs\scripts\make_plots.py
```

`run_instrumented_server.py` runs the unmodified backend and only observes it: it records loss, actions,
traces and a weight snapshot. Use port 8000 if it is free; on this PC another local app already uses it.
