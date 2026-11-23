"""
Phase 1 audit of the existing environment (benchmark v0 = backend.rl_agent.AutomationEnvironment).

Development evidence only: it uses environment seeds 10000-10099, the seeds of the historical evaluation,
which have already informed earlier work. No training.

1. Reward decomposition per fixed action (both production models), recomputed from the environment's own
   counters after every step and checked against the reward the environment returns.
2. Constraint reachability: safety events (|temp| or |pressure| > 1.5) and vibration events (|vib| > 1)
   under uniform random actions and every fixed action.
3. Value of non-stationary control: every two-segment schedule (action a for steps < k, then action b).
4. Reward-outcome ledger: the training reward against physically consistent production (sum of speed / 30
   per step, the speed-proportional definition), energy (sum of speed * pressure * temp), mean quality and
   constraint events, for every fixed action, uniform random and the 13 trained networks.

Writes experiments/v1/results/v0_audit.json and experiments/v1/results/v0_ledger.csv.
Run from the repository root:  .venv\\Scripts\\python.exe experiments\\v1\\audit_v0.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
warnings.simplefilter("ignore", UserWarning)
from backend.rl_agent import AutomationEnvironment, DQNNetwork  # noqa: E402

OUT = REPO / "experiments" / "v1" / "results"
SEEDS = [10_000 + i for i in range(100)]
SEARCH_SEEDS = SEEDS[:20]
MODELS = ("legacy", "speed_proportional")
NAMES = ["Conservative", "Normal", "Aggressive", "Energy saving", "Quality focus", "Balanced"]
CFG = {**json.loads((REPO / "config" / "config.json").read_text(encoding="utf-8")).get("automation", {})}


def components(env, action):
    """The terms of AutomationEnvironment._calculate_reward, recomputed after env.step()."""
    st, n = env.state, env.steps_in_episode
    prod = (env.production_count / (n / 100.0)) / env.throughput_target * 10 if env.production_count > 0 else 0.0
    qual = (st[10] + 1) / 2 * 5
    eff = 1.0 / (1.0 + env.total_energy / n) * 3
    safety = -5.0 if (abs(st[6]) > 1.5 or abs(st[7]) > 1.5) else 0.0
    vib = -2.0 if abs(st[8]) > 1.0 else 0.0
    bonus = 2.0 if env.cycle_time <= env.cycle_time_target else 0.0
    return np.array([prod, qual, eff, safety, vib, bonus])


def episode(model, policy, seed, record=False):
    env = AutomationEnvironment({**CFG, "production_model": model})
    np.random.seed(seed)
    random.seed(seed)
    s = env.reset()
    total, comp, phys_prod, energy, qual, events, max_abs = 0.0, np.zeros(6), 0.0, 0.0, [], 0, np.zeros(3)
    for t in range(200):
        a = policy(s, t)
        s, r, done = env.step(a)
        c = components(env, a)
        assert abs(c.sum() - r) < 1e-9, (model, seed, t, c.sum(), r)
        sp = env._apply_action(a)
        total += r
        comp += c
        phys_prod += sp["speed"] / env.cycle_time_target
        energy += sp["speed"] * sp["pressure"] * sp["temp"]
        qual.append(s[10])
        events += int(c[3] < 0) + int(c[4] < 0)
        max_abs = np.maximum(max_abs, np.abs([s[6], s[7], s[8]]))
        if done:
            break
    return dict(reward=total, comp=comp, counted_parts=env.production_count, physical_parts=phys_prod, energy=energy,
                quality=float(np.mean(qual)), constraint_events=events, max_abs_temp_pressure_vib=max_abs.tolist(),
                steps=t + 1)


def fixed(a):
    return lambda s, t: a


def uniform():
    return lambda s, t: int(np.random.randint(6)) if False else random.randrange(6)


def schedule(a, b, k):
    return lambda s, t: a if t < k else b


def network(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    net = DQNNetwork(12, 6)
    net.load_state_dict(sd)
    net.eval()

    def act(s, t):
        with torch.no_grad():
            return int(net(torch.from_numpy(np.asarray(s, dtype=np.float32))[None]).argmax())
    return act


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"seeds": [SEEDS[0], SEEDS[-1]], "models": {}}
    ledger_rows = []
    policies = {f"Always {i}: {n}": fixed(i) for i, n in enumerate(NAMES)}
    policies["Uniform random"] = uniform()
    nets = {p.stem: p for p in sorted((REPO / "visual_outputs/run/eval_v2/policies").glob("*.pt"))}
    for model in MODELS:
        m = {"policies": {}}
        pols = dict(policies)
        for stem, path in nets.items():
            if stem.startswith(model + "_seed"):
                pols[f"DQN {stem}"] = network(path)
        if model == "legacy":
            pols["DQN server snapshot"] = network(REPO / "visual_outputs/run/policy_snapshot_20261106_190932.pt")
        for name, pol in pols.items():
            eps = [episode(model, pol, sd) for sd in SEEDS]
            comp = np.mean([e["comp"] for e in eps], axis=0) / 200
            m["policies"][name] = {
                "reward_mean": float(np.mean([e["reward"] for e in eps])),
                "per_step_components": dict(zip(["production", "quality", "energy_efficiency", "safety", "vibration",
                                                 "cycle_bonus"], comp.round(4).tolist())),
                "counted_parts": float(np.mean([e["counted_parts"] for e in eps])),
                "physical_parts": float(np.mean([e["physical_parts"] for e in eps])),
                "energy": float(np.mean([e["energy"] for e in eps])),
                "quality": float(np.mean([e["quality"] for e in eps])),
                "constraint_events_total": int(sum(e["constraint_events"] for e in eps)),
                "max_abs_temp_pressure_vib": np.max([e["max_abs_temp_pressure_vib"] for e in eps], axis=0).round(3).tolist(),
            }
            for sd, e in zip(SEEDS, eps):
                ledger_rows.append([model, name, sd, round(e["reward"], 6), e["counted_parts"], round(e["physical_parts"], 4),
                                    round(e["energy"], 4), round(e["quality"], 6), e["constraint_events"]])
            print(f"{model:18s} {name:32s} reward {m['policies'][name]['reward_mean']:8.2f}  physical parts "
                  f"{m['policies'][name]['physical_parts']:6.2f}  events {m['policies'][name]['constraint_events_total']}",
                  flush=True)
        # two-segment schedules on 20 seeds, then the best one against the best fixed action on all 100
        best_fixed = max(range(6), key=lambda i: m["policies"][f"Always {i}: {NAMES[i]}"]["reward_mean"])
        search = []
        for a in range(6):
            for b in range(6):
                for k in range(25, 200, 25):
                    if a != b:
                        search.append((float(np.mean([episode(model, schedule(a, b, k), sd)["reward"]
                                                      for sd in SEARCH_SEEDS])), a, b, k))
        search.sort(reverse=True)
        _, a, b, k = search[0]
        best_sched = [episode(model, schedule(a, b, k), sd)["reward"] for sd in SEEDS]
        fixed_r = [episode(model, fixed(best_fixed), sd)["reward"] for sd in SEEDS]
        diff = np.array(best_sched) - np.array(fixed_r)
        m["schedule_search"] = {
            "schedules": len(search), "search_seeds": len(SEARCH_SEEDS), "best": {"a": NAMES[a], "b": NAMES[b], "k": k},
            "best_fixed": NAMES[best_fixed], "paired_diff_mean": float(diff.mean()),
            "paired_diff_ci95": float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff))),
            "relative_gain": float(diff.mean() / np.mean(fixed_r))}
        print(model, "schedule search:", m["schedule_search"], flush=True)
        result["models"][model] = m
    result["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / "v0_audit.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    with open(OUT / "v0_ledger.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["model", "policy", "env_seed", "training_reward", "counted_parts", "physical_parts", "energy",
                    "quality_mean", "constraint_events"])
        w.writerows(ledger_rows)
    print(f"wrote v0_audit.json and v0_ledger.csv in {result['wall_seconds']} s")


if __name__ == "__main__":
    main()
