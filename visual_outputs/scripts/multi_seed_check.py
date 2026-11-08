"""
Robustness check: re-run the default training scenario on several seeds using
the backend's own WMGDigitalTwinServer.run_training_episode() (no HTTP, no
0.1 s sleep, no broadcasts; none of these affect learning), then evaluate the
greedy policy with the same harness as evaluate_policies.py.

Usage (from the repo root):
    .venv\\Scripts\\python.exe visual_outputs\\scripts\\multi_seed_check.py --seeds 1 2 3 4 5
"""
import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_policies as ev    # noqa: E402  (sets CWD + sys.path for backend)

import numpy as np                # noqa: E402
import torch                      # noqa: E402


async def train(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    import main as backend_main
    server = backend_main.WMGDigitalTwinServer()
    rewards = []
    while server.current_episode < server.max_episodes:
        r = await server.run_training_episode()      # same call training_loop() makes
        server.current_episode += 1
        rewards.append(r["total_reward"])
    return server, rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--eval-episodes", type=int, default=30)
    a = ap.parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    cfg = ev.load_config()
    env = ev.AutomationEnvironment(config=cfg["automation"])
    gen = torch.Generator().manual_seed(777)
    seeds_eval = [10_000 + i for i in range(a.eval_episodes)]
    out = []
    for seed in a.seeds:
        t0 = time.time()
        server, rewards = asyncio.run(train(seed))
        net = server.rl_agent.q_network
        eps = [ev.run_episode(env, lambda s: ev.greedy(net, s, gen, False), sd) for sd in seeds_eval]
        counts = np.sum([e["action_counts"] for e in eps], axis=0)
        rec = {"seed": seed,
               "train_seconds": round(time.time() - t0, 1),
               "final_epsilon": server.rl_agent.epsilon,
               "gradient_steps": server.rl_agent.step_count,
               "train_reward_last100_mean": float(np.mean(rewards[-100:])),
               "greedy_eval_reward_mean": float(np.mean([e["total_reward"] for e in eps])),
               "greedy_action_share": (counts / counts.sum()).round(4).tolist()}
        out.append(rec)
        print(json.dumps(rec))
    (ev.OUT / "multi_seed_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", ev.OUT / "multi_seed_check.json")


if __name__ == "__main__":
    main()
