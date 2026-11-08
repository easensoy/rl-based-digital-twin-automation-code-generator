"""
Instrumented launcher for the WMG RL Digital Twin backend.

Runs the UNMODIFIED backend (backend/main.py -> WMGDigitalTwinServer) in this
process and records telemetry that the backend computes but never exposes:

  * DQN training loss: the tensor returned by torch.nn.functional.mse_loss
    inside DQNAgent.replay()
  * every action returned by DQNAgent.act(), tagged explore / exploit
    (detected by observing calls to random.randrange, which act() only
    uses on its exploration branch; the RNG stream is not altered)
  * the per-episode result dict returned by run_training_episode()
  * the process-state trace of selected episodes (AutomationEnvironment.step)
  * a snapshot of the Q-network weights each time PLC code is generated

Nothing under backend/ is edited. Each wrapper calls the original function and
only observes its inputs and outputs. The only behavioural difference from
`python backend/main.py` is the bind address (default 127.0.0.1 instead of the
0.0.0.0 in config/config.json) and fixed RNG seeds for reproducibility.

Usage (from the repo root):
    .venv\\Scripts\\python.exe visual_outputs\\scripts\\run_instrumented_server.py --seed 42
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)                                # backend uses CWD-relative paths
sys.path.insert(0, str(REPO / "backend"))     # backend uses flat imports

import numpy as np            # noqa: E402
import torch                  # noqa: E402
import torch.nn.functional as F  # noqa: E402

OUT = REPO / "visual_outputs" / "run"
TRACE_EPISODES = {0, 1, 50, 100, 250, 500, 750, 900, 998, 999}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "traces.jsonl"):
        (OUT / name).unlink(missing_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs("data/training_logs", exist_ok=True)   # main.py's FileHandler needs it at import
    os.makedirs("data/generated_code", exist_ok=True)

    import rl_agent                  # original backend module
    import main as backend_main      # original backend module

    tel = {"losses": [], "actions": [], "explore": [], "trace": [], "explore_flag": False}

    # 1) loss observer
    _orig_mse = F.mse_loss

    def _mse_observer(*a, **k):
        loss = _orig_mse(*a, **k)
        tel["losses"].append(float(loss.detach().cpu()))
        return loss

    rl_agent.F.mse_loss = _mse_observer

    # 2) explore/exploit observer: proxy for the `random` module inside rl_agent
    class _RandomProxy:
        def __getattr__(self, name):
            return getattr(random, name)

        def randrange(self, *a, **k):
            tel["explore_flag"] = True
            return random.randrange(*a, **k)

    rl_agent.random = _RandomProxy()

    _orig_act = rl_agent.DQNAgent.act

    def _act_observer(self, state):
        tel["explore_flag"] = False
        action = _orig_act(self, state)
        tel["actions"].append(int(action))
        tel["explore"].append(bool(tel["explore_flag"]))
        return action

    rl_agent.DQNAgent.act = _act_observer

    # 3) process-state observer
    _orig_step = rl_agent.AutomationEnvironment.step

    def _step_observer(self, action):
        next_state, reward, done = _orig_step(self, action)
        tel["trace"].append({
            "action": int(action),
            "reward": float(reward),
            "state": [round(float(x), 5) for x in next_state],
            "production_count": int(self.production_count),
            "total_energy": float(self.total_energy),
            "safety_violations": int(self.safety_violations),
        })
        return next_state, reward, done

    rl_agent.AutomationEnvironment.step = _step_observer

    Server = backend_main.WMGDigitalTwinServer

    # 4) per-episode observer
    _orig_episode = Server.run_training_episode

    async def _episode_observer(self):
        for key in ("losses", "actions", "explore", "trace"):
            tel[key].clear()
        result = await _orig_episode(self)
        rec = dict(result)
        rec["losses"] = list(tel["losses"])
        rec["action_counts"] = np.bincount(tel["actions"], minlength=6).tolist()
        rec["explore_steps"] = int(sum(tel["explore"]))
        greedy = [a for a, e in zip(tel["actions"], tel["explore"]) if not e]
        rec["greedy_action_counts"] = np.bincount(greedy, minlength=6).tolist() if greedy else [0] * 6
        rec["replay_buffer_len"] = len(self.rl_agent.memory)
        rec["gradient_steps"] = self.rl_agent.step_count
        rec["production_count"] = int(self.automation_environment.production_count)
        rec["total_energy"] = float(self.automation_environment.total_energy)
        rec["wall_time"] = time.time()
        with open(OUT / "episodes.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if result["episode"] in TRACE_EPISODES:
            with open(OUT / "traces.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"episode": result["episode"],
                                     "epsilon_after": result["epsilon"],
                                     "steps": tel["trace"]}) + "\n")
        return result

    Server.run_training_episode = _episode_observer

    # 5) policy snapshot at code-generation time
    _orig_gen = Server.generate_optimized_code

    async def _gen_observer(self):
        code = await _orig_gen(self)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        torch.save({
            "state_dict": self.rl_agent.q_network.state_dict(),
            "epsilon": self.rl_agent.epsilon,
            "gradient_steps": self.rl_agent.step_count,
            "episodes_completed": self.current_episode,
            "replay_buffer_len": len(self.rl_agent.memory),
            "config": self.config,
            "seed": args.seed,
        }, OUT / f"policy_snapshot_{stamp}.pt")
        return code

    Server.generate_optimized_code = _gen_observer

    server = Server()
    cfg = server.config["server"]
    port = args.port or int(os.environ.get("PORT", 0)) or cfg["port"]
    with open(OUT / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"seed": args.seed, "host": args.host, "port": port,
                   "config": server.config, "torch": torch.__version__,
                   "numpy": np.__version__, "python": sys.version,
                   "device": str(server.rl_agent.device),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)

    import uvicorn
    uvicorn.run(server.app, host=args.host, port=port, reload=cfg["reload"], log_level="info")


if __name__ == "__main__":
    main()
