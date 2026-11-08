"""
Per-seed, per-episode evaluation of the default DQN training procedure, with seed-to-seed
spread reported separately from episode noise.

For one simulator production model ("legacy" = commit 898c112, "speed_proportional" =
corrected; see PRODUCTION_MODELS in backend/rl_agent.py) and each training seed:

  * train exactly as WMGDigitalTwinServer.training_loop / run_training_episode does
    (backend/main.py): 1,000 episodes of at most 200 steps, act -> step -> remember,
    one replay() per episode once the buffer exceeds the batch size. The loop is
    re-implemented here so this script does not import main.py; its equivalence is
    checked by retraining seed 42 on "legacy" and comparing the weights bit for bit
    with the snapshot saved from the real server run.
  * evaluate the greedy policy (eval mode) on 100 fixed environment seeds.

Baselines (uniform random, each fixed action) run once per model on the same environment
seeds, so every comparison is paired episode by episode.

Outputs, under visual_outputs/run/eval_v2/:
  episodes_<model>.csv          one row per (policy, train_seed, env_seed)
  training_rewards_<model>.csv  training reward per episode per seed
  summary_<model>.json          per-seed results, seed spread, paired differences
  policies/<model>_seed<k>.pt   trained Q-network weights
  repro_check.json              legacy seed 42 vs policy_snapshot_20261106_190932.pt

Usage (from the repo root):
    .venv\\Scripts\\python.exe visual_outputs\\scripts\\multi_seed_eval.py --model legacy
    .venv\\Scripts\\python.exe visual_outputs\\scripts\\multi_seed_eval.py --model speed_proportional
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_policies as ev    # noqa: E402  (sets CWD and sys.path for backend/)

import numpy as np                # noqa: E402
import torch                      # noqa: E402
from scipy import stats           # noqa: E402

from rl_agent import PRODUCTION_MODELS, AutomationEnvironment, DQNAgent  # noqa: E402

OUT = ev.OUT / "eval_v2"
SNAPSHOT_42 = ev.OUT / "policy_snapshot_20261106_190932.pt"
METRICS = ["total_reward", "energy_efficiency", "quality_mean", "throughput", "production_count", "oee"]


def train(seed: int, model: str, cfg: dict, episodes: int):
    """Same RNG order and update schedule as the backend server."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = AutomationEnvironment(config={**cfg["automation"], "production_model": model})
    agent = DQNAgent(state_size=env.get_state_size(), action_size=env.get_action_size(),
                     config=cfg["rl_agent"])
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        total, steps = 0.0, 0
        while not env.is_done() and steps < 200:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total += reward
            steps += 1
            if done:
                break
        if len(agent.memory) > cfg["rl_agent"]["batch_size"]:
            agent.replay()
        rewards.append(total)
    return agent, rewards


def ci95(x):
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def episode_rows(policy, train_seed, eps, env_seeds):
    rows = []
    for sd, e in zip(env_seeds, eps):
        rows.append({"policy": policy, "train_seed": train_seed, "env_seed": sd,
                     **{m: e[m] for m in METRICS},
                     "dominant_action": int(np.argmax(e["action_counts"]))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=PRODUCTION_MODELS, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4, 5])
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--train-episodes", type=int, default=1000)
    a = ap.parse_args()

    (OUT / "policies").mkdir(parents=True, exist_ok=True)
    cfg = ev.load_config()
    env_seeds = [10_000 + i for i in range(a.eval_episodes)]
    eval_env = AutomationEnvironment(config={**cfg["automation"], "production_model": a.model})
    names = ev.ACTION_NAMES
    t0 = time.time()

    # ---- baselines (independent of training seed) ----------------------------------
    rng = np.random.default_rng(12345)
    baselines = {"Uniform random": lambda s: int(rng.integers(6))}
    for k, n in enumerate(names):
        baselines[f"Always {k}: {n}"] = (lambda kk: (lambda s: kk))(k)
    rows, base_eps = [], {}
    for pname, pol in baselines.items():
        eps = [ev.run_episode(eval_env, pol, sd) for sd in env_seeds]
        base_eps[pname] = eps
        rows += episode_rows(pname, "", eps, env_seeds)
    base_summary = {p: {m: {"mean": float(np.mean([e[m] for e in eps])), "ci95": ci95([e[m] for e in eps])}
                        for m in METRICS} for p, eps in base_eps.items()}
    best_fixed = max((p for p in baselines if p.startswith("Always")),
                     key=lambda p: base_summary[p]["total_reward"]["mean"])
    r_random = np.array([e["total_reward"] for e in base_eps["Uniform random"]])
    r_best = np.array([e["total_reward"] for e in base_eps[best_fixed]])
    print(f"[{a.model}] baselines done; best fixed action = {best_fixed} "
          f"({base_summary[best_fixed]['total_reward']['mean']:.1f})", flush=True)

    # ---- train + evaluate each seed -------------------------------------------------
    per_seed, train_rows, repro = [], [], None
    gen = torch.Generator().manual_seed(777)
    for seed in a.seeds:
        ts = time.time()
        agent, train_rewards = train(seed, a.model, cfg, a.train_episodes)
        net = agent.q_network
        torch.save({"state_dict": net.state_dict(), "seed": seed, "production_model": a.model,
                    "epsilon": agent.epsilon, "gradient_steps": agent.step_count},
                   OUT / "policies" / f"{a.model}_seed{seed}.pt")
        if a.model == "legacy" and seed == 42 and SNAPSHOT_42.exists():
            ref = torch.load(SNAPSHOT_42, map_location="cpu")["state_dict"]
            cur = net.state_dict()
            diffs = {k: float((cur[k] - ref[k]).abs().max()) for k in ref}
            repro = {"compared_with": SNAPSHOT_42.name,
                     "bit_identical": all(torch.equal(cur[k], ref[k]) for k in ref),
                     "max_abs_diff": max(diffs.values()), "per_tensor_max_abs_diff": diffs}
            (OUT / "repro_check.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")
            print(f"[{a.model}] seed 42 reproduces the server snapshot bit for bit: {repro['bit_identical']}",
                  flush=True)
        eps = [ev.run_episode(eval_env, lambda s: ev.greedy(net, s, gen, False), sd) for sd in env_seeds]
        rows += episode_rows("Trained DQN (greedy, eval mode)", seed, eps, env_seeds)
        train_rows += [{"train_seed": seed, "episode": i, "total_reward": r} for i, r in enumerate(train_rewards)]
        r = np.array([e["total_reward"] for e in eps])
        counts = np.sum([e["action_counts"] for e in eps], axis=0)
        share = counts / counts.sum()
        rec = {"train_seed": seed,
               "train_seconds": round(time.time() - ts, 1),
               "final_epsilon": agent.epsilon,
               "gradient_steps": agent.step_count,
               "train_reward_last100_mean": float(np.mean(train_rewards[-100:])),
               "eval_reward_mean": float(r.mean()), "eval_reward_ci95_episodes": ci95(r),
               "paired_diff_vs_random": float((r - r_random).mean()),
               "paired_diff_vs_random_ci95": ci95(r - r_random),
               "paired_diff_vs_best_fixed": float((r - r_best).mean()),
               "paired_diff_vs_best_fixed_ci95": ci95(r - r_best),
               "action_share": share.round(4).tolist(),
               "dominant_action": names[int(np.argmax(share))],
               "dominant_share": float(share.max()),
               **{f"eval_{m}_mean": float(np.mean([e[m] for e in eps])) for m in METRICS if m != "total_reward"}}
        per_seed.append(rec)
        print(f"[{a.model}] seed {seed}: {rec['eval_reward_mean']:.1f} reward, "
              f"{rec['dominant_action']} {rec['dominant_share']:.0%}, "
              f"vs random {rec['paired_diff_vs_random']:+.1f}, vs best fixed {rec['paired_diff_vs_best_fixed']:+.1f} "
              f"({rec['train_seconds']} s)", flush=True)

    # ---- across seeds -----------------------------------------------------------------
    means = np.array([p["eval_reward_mean"] for p in per_seed])
    n = len(means)
    t = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")

    def across(key):
        v = np.array([p[key] for p in per_seed])
        return {"mean": float(v.mean()), "sd_across_seeds": float(v.std(ddof=1)) if n > 1 else float("nan"),
                "ci95_across_seeds": float(t * v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
                "min": float(v.min()), "max": float(v.max())}

    summary = {
        "production_model": a.model,
        "training_seeds": a.seeds, "eval_episodes_per_policy": a.eval_episodes,
        "env_seeds": [env_seeds[0], env_seeds[-1]], "train_episodes": a.train_episodes,
        "baselines": base_summary, "best_fixed_action": best_fixed,
        "per_seed": per_seed,
        "across_seeds": {
            "eval_reward": across("eval_reward_mean"),
            "paired_diff_vs_random": across("paired_diff_vs_random"),
            "paired_diff_vs_best_fixed": across("paired_diff_vs_best_fixed"),
            "seeds_with_single_action_policy": int(sum(p["dominant_share"] >= 0.99 for p in per_seed)),
            "seeds_choosing_best_fixed_action": int(sum(p["dominant_action"] == best_fixed.split(": ")[1]
                                                        and p["dominant_share"] >= 0.99 for p in per_seed)),
            "seeds_beating_random": int(sum(p["paired_diff_vs_random"] > 0 for p in per_seed)),
            "note": "ci95_across_seeds uses Student t over training seeds (training reliability); "
                    "per-seed ci95 fields cover episode noise only.",
        },
        "reproducibility": repro,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (OUT / f"summary_{a.model}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, data in ((f"episodes_{a.model}.csv", rows), (f"training_rewards_{a.model}.csv", train_rows)):
        with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    acr = summary["across_seeds"]
    print(f"[{a.model}] across {n} seeds: reward {acr['eval_reward']['mean']:.1f} "
          f"± {acr['eval_reward']['ci95_across_seeds']:.1f} (t-CI over seeds); "
          f"vs best fixed {acr['paired_diff_vs_best_fixed']['mean']:+.1f}; "
          f"single-action policies {acr['seeds_with_single_action_policy']}/{n}; "
          f"beat random {acr['seeds_beating_random']}/{n}; wall {summary['wall_seconds']} s", flush=True)


if __name__ == "__main__":
    main()
