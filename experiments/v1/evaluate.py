"""Run controllers on benchmark v1 scenario sets and score them with the outcome ledger."""

from __future__ import annotations

import numpy as np

from ledger import outcomes
from process import P0, VecProcess

# Deterministic numeric graders, one verdict per episode (True = pass), and the controller-level feasibility
# criteria: the minimum pass rate over a scenario set. Frozen in docs/experiments/PROTOCOL_v1.md.
GRADERS = {
    "safety": lambda o: o["excursion_steps"] == 0,                 # never above T_safe
    "delivery": lambda o: o["shortfall"] <= 0.01 * o["demand"],    # at least 99 % of the order delivered
    "quality": lambda o: o["scrap_fraction"] <= 0.05,              # first-pass yield at least 95 %
}
FEASIBLE = {"safety": 0.99, "delivery": 0.95, "quality": 0.95}


def run(controller, sc, keep_traj: bool = False) -> dict:
    env = VecProcess(sc, reward="intended", log=True)
    obs = env.reset()
    total = np.zeros(len(sc))
    for _ in range(P0.steps):
        obs, r, _ = env.step(controller(obs))
        total += r
    traj = env.trajectory()
    out = outcomes(traj, sc.demand)
    out["demand"] = np.asarray(sc.demand, dtype=float)
    out["reward_intended"] = total
    out["reward_throughput"] = (P0.part_value * (out["good_parts"] + out["scrap_parts"]) - out["energy_cost"]
                                - P0.safety_cost * out["excursion_steps"])
    for g, f in GRADERS.items():
        out["pass_" + g] = f(out)
    if keep_traj:
        out["traj"] = traj
    return out


def summary(out: dict) -> dict:
    keys = ("margin", "good_parts", "scrap_fraction", "energy_kwh", "peak_energy_kwh", "shortfall", "excursion_steps",
            "reward_intended", "reward_throughput")
    s = {k: float(np.mean(out[k])) for k in keys}
    s["margin_sd"] = float(np.std(out["margin"], ddof=1))
    s["demand_met_rate"] = float(np.mean(out["demand_met"]))
    for g in GRADERS:
        s["pass_rate_" + g] = float(np.mean(out["pass_" + g]))
    s["feasible"] = bool(all(s["pass_rate_" + g] >= FEASIBLE[g] for g in GRADERS))
    s["n"] = int(len(out["margin"]))
    return s


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a) - np.asarray(b)
    return {"mean": float(d.mean()), "ci95": float(1.96 * d.std(ddof=1) / np.sqrt(len(d))), "n": int(len(d))}
