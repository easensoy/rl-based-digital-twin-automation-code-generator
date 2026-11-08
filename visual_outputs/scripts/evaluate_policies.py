"""
Before/after evaluation of the trained DQN policy on the backend's own
AutomationEnvironment (backend/rl_agent.py), plus a fidelity check of the
generated PLC code (FB_RLOptimizer in backend/plc_generator.py) against the
trained network.

All policies are run on the same environment seeds (common random numbers):
the environment only draws from numpy's global RNG, and every policy here
draws its own randomness from separate generators, so the process noise
sequence is identical across policies.

Usage (from the repo root):
    .venv\\Scripts\\python.exe visual_outputs\\scripts\\evaluate_policies.py [--episodes 100]
"""
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "backend"))

import numpy as np   # noqa: E402
import torch         # noqa: E402

from rl_agent import AutomationEnvironment, DQNAgent  # noqa: E402

OUT = REPO / "visual_outputs" / "run"
ACTION_NAMES = ["Conservative", "Normal", "Aggressive", "Energy saving", "Quality focus", "Balanced"]


def load_config() -> dict:
    """Same merge logic as WMGDigitalTwinServer.load_configuration()."""
    cfg = {
        "rl_agent": {"learning_rate": 0.001, "batch_size": 64, "memory_size": 10000,
                     "epsilon_start": 1.0, "epsilon_end": 0.01, "epsilon_decay": 0.995,
                     "target_update": 10},
        "automation": {"cycle_time_target": 30.0, "throughput_target": 120.0,
                       "safety_threshold": 0.95, "energy_efficiency_target": 0.85},
    }
    with open("config/config.json", encoding="utf-8") as fh:
        cfg.update(json.load(fh))
    return cfg


# --------------------------------------------------------------------------
# Emulation of the generated IEC 61131-3 ST (plc_generator.py), line for line
# --------------------------------------------------------------------------
def st_fb_rloptimizer(current_state: np.ndarray) -> int:
    """FB_RLOptimizer body: normalise, fixed linear 'Q-values', argmax (FOR loop, strict >)."""
    sn = [current_state[i] / 2.0 for i in range(12)]
    q = [
        sn[10] * 0.8 + sn[11] * 0.5,           # 0 Conservative
        sn[10] * 1.0 + sn[11] * 0.7,           # 1 Normal
        (1.0 - sn[10]) * 0.6 + sn[11] * 1.2,   # 2 Aggressive
        sn[11] * (-0.8) + sn[10] * 0.6,        # 3 Energy saving
        sn[10] * 1.2 + sn[11] * 0.4,           # 4 Quality focus
        (sn[10] + sn[11]) * 0.5,               # 5 Balanced
    ]
    best = 0
    for i in range(1, 6):
        if q[i] > q[best]:
            best = i
    return best


def st_main_program_wiring(twin_state: np.ndarray) -> np.ndarray:
    """Main program: only system_state[6..8] are written (temp/100, pressure/10,
    vibration/5); every other element keeps its initial value 0.0. The twin's
    temperature/pressure/vibration are dimensionless, so they are passed through
    unchanged; this does not affect the result because indices 10 and 11 (the
    only ones FB_RLOptimizer reads) are never written."""
    plc = np.zeros(12)
    plc[6] = twin_state[6]
    plc[7] = twin_state[7]
    plc[8] = twin_state[8]
    return plc


# --------------------------------------------------------------------------
def make_agent(cfg: dict, seed: int, state_dict=None) -> DQNAgent:
    torch.manual_seed(seed)
    agent = DQNAgent(state_size=12, action_size=6, config=cfg["rl_agent"])
    if state_dict is not None:
        agent.q_network.load_state_dict(state_dict)
    return agent


def greedy(net: torch.nn.Module, state: np.ndarray, gen: torch.Generator, train_mode: bool) -> int:
    net.train(train_mode)
    with torch.no_grad():
        x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        if train_mode:
            # reproduce dropout without touching torch's global RNG
            h = torch.relu(net.fc1(x)); h = h * (torch.rand(h.shape, generator=gen) > 0.2) / 0.8
            h = torch.relu(net.fc2(h)); h = h * (torch.rand(h.shape, generator=gen) > 0.2) / 0.8
            h = torch.relu(net.fc3(h))
            q = net.fc4(h)
        else:
            q = net(x)
    return int(torch.argmax(q, dim=1).item())


def run_episode(env: AutomationEnvironment, policy, seed: int) -> dict:
    np.random.seed(seed)
    random.seed(seed)
    state = env.reset()
    total, steps, actions, quality, energy_step = 0.0, 0, [], [], []
    states = []
    while not env.is_done() and steps < 200:          # same loop guard as main.py
        a = policy(state)
        states.append(state.copy())
        state, r, done = env.step(a)
        total += r
        steps += 1
        actions.append(a)
        quality.append((state[10] + 1) / 2)
        energy_step.append(state[11])
        if done:
            break
    m = env.get_performance_metrics()
    return {
        "total_reward": total, "steps": steps,
        "production_count": m["production_count"], "throughput": m["throughput"],
        "energy_efficiency": m["energy_efficiency"], "quality_final": m["quality_score"],
        "quality_mean": float(np.mean(quality)), "energy_per_step": float(np.mean(energy_step)),
        "oee": m["oee"], "safety_score": m["safety_score"],
        "safety_violations": m["safety_violations"],
        "action_counts": np.bincount(actions, minlength=6).tolist(),
        "_states": states, "_actions": actions,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--snapshot", default=None)
    args = ap.parse_args()

    cfg = load_config()
    meta = json.loads((OUT / "run_meta.json").read_text(encoding="utf-8"))
    seed = int(meta["seed"])
    snap_path = Path(args.snapshot) if args.snapshot else sorted(OUT.glob("policy_snapshot_*.pt"))[-1]
    snap = torch.load(snap_path, map_location="cpu")

    untrained = make_agent(cfg, seed)                  # identical init to the training run
    trained = make_agent(cfg, seed, snap["state_dict"])
    env = AutomationEnvironment(config=cfg["automation"])

    rng = np.random.default_rng(12345)
    gen = torch.Generator().manual_seed(777)

    policies = {
        "Uniform random": lambda s: int(rng.integers(6)),
        "Untrained DQN (greedy)": lambda s: greedy(untrained.q_network, s, gen, True),
        "Trained DQN (greedy, as backend runs it)": lambda s: greedy(trained.q_network, s, gen, True),
        "Trained DQN (greedy, eval mode)": lambda s: greedy(trained.q_network, s, gen, False),
        "Generated ST, as wired in main program": lambda s: st_fb_rloptimizer(st_main_program_wiring(s)),
        "Generated ST, fed full twin state": lambda s: st_fb_rloptimizer(np.asarray(s)),
    }
    for k, name in enumerate(ACTION_NAMES):
        policies[f"Always {k}: {name}"] = (lambda kk: (lambda s: kk))(k)

    seeds = [10_000 + i for i in range(args.episodes)]
    results, rows = {}, []
    trained_states = []
    for pname, pol in policies.items():
        eps = [run_episode(env, pol, sd) for sd in seeds]
        if pname == "Trained DQN (greedy, eval mode)":
            for e in eps:
                trained_states.extend(e["_states"])
        for sd, e in zip(seeds, eps):
            rows.append({"policy": pname, "seed": sd,
                         **{k: v for k, v in e.items() if not k.startswith("_") and k != "action_counts"}})
        keys = ["total_reward", "production_count", "throughput", "energy_efficiency",
                "quality_mean", "energy_per_step", "oee", "safety_score", "steps"]
        summ = {k: {"mean": float(np.mean([e[k] for e in eps])),
                    "std": float(np.std([e[k] for e in eps], ddof=1)),
                    "ci95": float(1.96 * np.std([e[k] for e in eps], ddof=1) / np.sqrt(len(eps)))}
                for k in keys}
        counts = np.sum([e["action_counts"] for e in eps], axis=0)
        summ["action_share"] = (counts / counts.sum()).tolist()
        results[pname] = summ
        print(f"{pname:45s} reward {summ['total_reward']['mean']:9.2f} ± {summ['total_reward']['ci95']:.2f}"
              f" | energy_eff {summ['energy_efficiency']['mean']:.4f} | quality {summ['quality_mean']['mean']:.4f}"
              f" | throughput {summ['throughput']['mean']:.3f} | oee {summ['oee']['mean']:.4f}")

    # Fidelity of the generated PLC code to the trained network, on states the
    # trained policy actually visits.
    S = np.array(trained_states)
    dqn_eval = [greedy(trained.q_network, s, gen, False) for s in S]
    dqn_train = [greedy(trained.q_network, s, gen, True) for s in S]
    st_wired = [st_fb_rloptimizer(st_main_program_wiring(s)) for s in S]
    st_full = [st_fb_rloptimizer(s) for s in S]
    with torch.no_grad():
        trained.q_network.eval()
        q = trained.q_network(torch.as_tensor(S, dtype=torch.float32)).numpy()
    fidelity = {
        "n_states": int(len(S)),
        "agreement_st_as_wired_vs_dqn_eval": float(np.mean(np.array(st_wired) == np.array(dqn_eval))),
        "agreement_st_full_state_vs_dqn_eval": float(np.mean(np.array(st_full) == np.array(dqn_eval))),
        "agreement_dqn_train_mode_vs_eval_mode": float(np.mean(np.array(dqn_train) == np.array(dqn_eval))),
        "st_as_wired_action_share": (np.bincount(st_wired, minlength=6) / len(S)).tolist(),
        "st_full_state_action_share": (np.bincount(st_full, minlength=6) / len(S)).tolist(),
        "dqn_eval_action_share": (np.bincount(dqn_eval, minlength=6) / len(S)).tolist(),
        "q_value_mean_per_action": q.mean(axis=0).tolist(),
        "q_value_std_per_action": q.std(axis=0).tolist(),
        "q_gap_best_minus_second_mean": float(np.mean(np.sort(q, axis=1)[:, -1] - np.sort(q, axis=1)[:, -2])),
    }
    print(json.dumps(fidelity, indent=2))

    out = {"snapshot": snap_path.name, "snapshot_meta": {k: v for k, v in snap.items() if k != "state_dict"},
           "episodes_per_policy": args.episodes, "env_seeds": [seeds[0], seeds[-1]],
           "action_names": ACTION_NAMES, "policies": results, "plc_fidelity": fidelity}
    (OUT / "evaluation.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    with open(OUT / "evaluation_episodes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT / "evaluation.json")


if __name__ == "__main__":
    main()
