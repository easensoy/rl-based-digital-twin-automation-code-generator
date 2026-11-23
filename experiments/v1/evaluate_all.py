"""
Confirmatory evaluation of benchmark v1 (docs/experiments/PROTOCOL_v1.md, sections 7 and 8), run once after
the confirmatory training.

Controllers: uniform random, the six fixed presets, the locked rule, and the final checkpoint of every
confirmatory run (7 conditions x 6 seeds). Scenario sets: dev nominal, held-out nominal, the 25-cell
generalisation map (50 episodes per cell) and the six held-out fault families (100 episodes each). Every
controller sees the same episodes.

Writes
  experiments/v1/results/episodes.csv.gz        one row per controller x episode (raw ledger outcomes)
  experiments/v1/results/confirmatory.json      summaries, per-seed means, contrasts and the decisions
  experiments/v1/results/trajectories/*.csv     output D traces
Run:  .venv\\Scripts\\python.exe experiments\\v1\\evaluate_all.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO)]

import controllers as C  # noqa: E402
import scenarios as S  # noqa: E402
from evaluate import FEASIBLE, GRADERS, run  # noqa: E402
from process import NAMES, P0, PC, TH, V, Scenario  # noqa: E402

RES = HERE / "results"
CONDS = ("I0", "I1", "I2", "I3", "I4", "I5", "R1")
SEEDS = (1, 2, 3, 4, 5, 6)
T975_5 = 2.570581835636314           # Student t, 0.975 quantile, 5 degrees of freedom
CAL = json.loads((RES / "calibration.json").read_text(encoding="utf-8"))
LOCKED, BEST_FIXED = CAL["locked"]["rule"], CAL["locked"]["best_fixed"]
OUTCOME_KEYS = ("margin", "good_parts", "scrap_parts", "scrap_fraction", "energy_kwh", "peak_energy_kwh",
                "energy_cost", "shortfall", "excursion_steps", "excess_parts", "reward_intended",
                "reward_throughput")


def concat(scs) -> Scenario:
    return Scenario(**{k: np.concatenate([getattr(s, k) for s in scs]) for k in scs[0].__dict__})


def scenario_sets():
    cells = [(i, j) for i in range(len(S.MAP_DT)) for j in range(len(S.MAP_LAG))]
    sets = {"dev": (S.dev(), ["nominal"] * len(S.DEV)),
            "heldout": (S.heldout(), ["nominal"] * len(S.HELDOUT)),
            "map": (concat([S.map_cell(i, j) for i, j in cells]),
                    [f"{i},{j}" for i, j in cells for _ in range(S.MAP_EPISODES)]),
            "faults": (concat([S.fault_family(f, "heldout") for f in S.FAULTS]),
                       [f for f in S.FAULTS for _ in range(S.FAULT_EPISODES)])}
    return sets


def controllers():
    out = {"uniform": ("baseline", None, lambda: C.uniform(0))}
    for a in range(6):
        out[f"fixed:{NAMES[a]}"] = ("baseline", None, (lambda a=a: C.fixed(a)))
    out["locked rule"] = ("baseline", None, lambda: C.rule(LOCKED))
    for c in CONDS:
        for s in SEEDS:
            p = REPO / "experiments" / "v1" / "checkpoints" / "confirmatory" / f"{c}_seed{s}.pt"
            if p.exists():
                sd = torch.load(p, map_location="cpu", weights_only=True)
                out[f"{c}_seed{s}"] = (c, s, (lambda sd=sd: C.dqn(sd)))
    return out


def rates(o: dict, mask=None) -> dict:
    m = slice(None) if mask is None else mask
    return {g: float(np.mean(o["pass_" + g][m])) for g in GRADERS}


def summarise(o: dict, mask=None) -> dict:
    m = slice(None) if mask is None else mask
    s = {k: float(np.mean(o[k][m])) for k in OUTCOME_KEYS}
    s["n"] = int(np.size(o["margin"][m]))
    s["pass_rate"] = rates(o, mask)
    s["fail_any_rate"] = float(np.mean(~np.all([o["pass_" + g][m] for g in GRADERS], axis=0)))
    s["feasible"] = all(s["pass_rate"][g] >= FEASIBLE[g] for g in GRADERS)
    return s


def contrast(perseed_a: np.ndarray, b) -> dict:
    """b: per-seed array (paired by seed) or a scalar baseline mean."""
    d = perseed_a - b
    sd = float(np.std(d, ddof=1))
    hw = T975_5 * sd / np.sqrt(len(d))
    return dict(mean=float(d.mean()), lower=float(d.mean() - hw), upper=float(d.mean() + hw), sd_between_seeds=sd,
                per_seed=np.round(d, 3).tolist(), n_seeds=int(len(d)))


def main():
    t0 = time.perf_counter()
    sets, ctrls = scenario_sets(), controllers()
    raw, results = {}, {}
    for name, (cond, seed, make) in ctrls.items():
        raw[name] = {}
        for set_name, (sc, _) in sets.items():
            o = run(make(), sc, keep_traj=False)
            o["reward_incremental"] = o["reward_intended"] + P0.shortfall_cost * o["demand"]
            raw[name][set_name] = o
        print(f"{name:26s} heldout margin {raw[name]['heldout']['margin'].mean():8.2f}", flush=True)
    # raw per-episode ledger
    with gzip.open(RES / "episodes.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["controller", "condition", "train_seed", "set", "group", "scenario_seed", *OUTCOME_KEYS,
                    "reward_incremental", "demand", *("pass_" + g for g in GRADERS)])
        for name, (cond, seed, _) in ctrls.items():
            for set_name, (sc, groups) in sets.items():
                o = raw[name][set_name]
                for e in range(len(sc)):
                    w.writerow([name, cond, seed if seed is not None else "", set_name, groups[e], int(sc.seed[e]),
                                *(round(float(o[k][e]), 4) for k in OUTCOME_KEYS),
                                round(float(o["reward_incremental"][e]), 4), int(o["demand"][e]),
                                *(int(o["pass_" + g][e]) for g in GRADERS)])
    # summaries per controller, set and group
    for name in ctrls:
        results[name] = {}
        for set_name, (sc, groups) in sets.items():
            g = np.array(groups)
            results[name][set_name] = {"all": summarise(raw[name][set_name])}
            if set_name in ("map", "faults"):
                for grp in dict.fromkeys(groups):
                    results[name][set_name][grp] = summarise(raw[name][set_name], g == grp)

    def perseed(cond, set_name="heldout", grp="all"):
        return np.array([results[f"{cond}_seed{s}"][set_name][grp]["margin"] for s in SEEDS])

    def pooled(cond, set_name="heldout"):
        o = [raw[f"{cond}_seed{s}"][set_name] for s in SEEDS]
        return {g: float(np.mean(np.concatenate([x["pass_" + g] for x in o]))) for g in GRADERS}

    base_mean = {n: results[n]["heldout"]["all"]["margin"] for n in ctrls if ctrls[n][0] == "baseline"}
    base_pass = {n: results[n]["heldout"]["all"]["pass_rate"] for n in base_mean}

    def decide(a, b):
        is_cond = b in CONDS
        c = contrast(perseed(a), perseed(b) if is_cond else base_mean[b])
        pa, pb = pooled(a), (pooled(b) if is_cond else base_pass[b])
        noninf = {g: (1 - pa[g]) <= (1 - pb[g]) + 0.02 for g in GRADERS}
        feasible = all(pa[g] >= FEASIBLE[g] for g in GRADERS)
        c.update(pass_rate_a=pa, pass_rate_b=pb, non_inferior=noninf, a_feasible=feasible,
                 improved=bool(c["lower"] > 0 and all(noninf.values()) and feasible))
        # episode-level noise: SD over episodes of the paired difference, averaged over seeds
        eps_sd = [np.std(raw[f"{a}_seed{s}"]["heldout"]["margin"]
                         - (raw[f"{b}_seed{s}"]["heldout"]["margin"] if is_cond else raw[b]["heldout"]["margin"]),
                         ddof=1) for s in SEEDS]
        c["sd_within_seed_episodes"] = float(np.mean(eps_sd))
        return c

    bf = f"fixed:{NAMES[BEST_FIXED]}"
    complete = all(f"{c}_seed{s}" in ctrls for c in CONDS for s in SEEDS)
    contrasts = {}
    if complete:
        plan = {"H1: I5 vs I0": ("I5", "I0"), "H2: I5 vs locked rule": ("I5", "locked rule"),
                "H3: I5 vs best fixed": ("I5", bf), "I1 vs I0": ("I1", "I0"), "I2 vs I0": ("I2", "I0"),
                "I3 vs I1": ("I3", "I1"), "I3 vs I2": ("I3", "I2"), "I4 vs I3": ("I4", "I3"),
                "I5 vs I4": ("I5", "I4"), "R1 vs I3": ("R1", "I3"), "I0 vs best fixed": ("I0", bf),
                "I0 vs locked rule": ("I0", "locked rule"), "I4 vs locked rule": ("I4", "locked rule"),
                "I3 vs locked rule": ("I3", "locked rule")}
        contrasts = {k: decide(a, b) for k, (a, b) in plan.items()}
        inter = (perseed("I3") - perseed("I2")) - (perseed("I1") - perseed("I0"))
        contrasts["interaction (I3-I2)-(I1-I0)"] = contrast(inter, 0.0)
        # scenario variation: I5 minus locked rule across fault families and map cells
        var = {}
        for set_name in ("faults", "map"):
            groups = [g for g in results["locked rule"][set_name] if g != "all"]
            diffs = {g: contrast(perseed("I5", set_name, g), results["locked rule"][set_name][g]["margin"])
                     for g in groups}
            var[set_name] = dict(per_group=diffs, sd_across_groups=float(np.std([d["mean"] for d in diffs.values()],
                                                                                ddof=1)))
        contrasts["scenario_variation_I5_vs_locked_rule"] = var
    # output D: trajectory selected by the protocol rule
    fam = S.fault_family("cold_saturation", "heldout")
    onset = fam.cold_on
    k = int(np.lexsort((fam.seed, np.abs(onset - np.median(onset))))[0])
    traj_dir = RES / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    ep = fam.subset(np.array([k]))
    shown = {"locked rule": C.rule(LOCKED), bf: C.fixed(BEST_FIXED)}
    for c in ("I0", "I5"):
        if f"{c}_seed1" in ctrls:
            shown[f"{c}_seed1"] = ctrls[f"{c}_seed1"][2]()
    trace_rows = []
    for name, ctrl in shown.items():
        o = run(ctrl, ep, keep_traj=True)
        tr = o["traj"]
        a = tr["action"][0].astype(int)
        v = V[a]
        parts = P0.rate * v * P0.dt_min
        lo = P0.T_req0 + P0.T_req1 * v
        good = np.where((tr["T"][0] >= lo) & (tr["T"][0] <= lo + P0.window), parts, 0.0)
        kwh = (tr["P_h"][0] + P0.P_motor * v + P0.P_comp * PC[a]) * P0.dt_min / 60.0
        t = np.arange(P0.steps)
        price = np.where((t >= P0.peak_start) & (t < P0.peak_end), P0.price_peak, P0.price_off)
        for i in range(P0.steps):
            trace_rows.append([name, i, round(i * P0.dt_min / 60.0, 4), round(float(tr["T"][0, i]), 4),
                               round(float(tr["T_meas"][0, i]), 4), int(a[i]), NAMES[a[i]], float(V[a[i]]),
                               round(float(TH[a[i]] * P0.T_nom), 2), round(float(lo[i]), 2),
                               round(float(lo[i] + P0.window), 2), P0.T_safe, round(float(tr["P_h"][0, i]), 3),
                               round(float(tr["P_max"][0, i]), 3), round(float(tr["T_amb"][0, i]), 3),
                               int(i >= ep.cold_on[0] and i < ep.cold_on[0] + ep.cold_len[0]),
                               round(float(good[:i + 1].sum()), 2), int(ep.demand[0]),
                               round(float((kwh * price)[:i + 1].sum()), 4),
                               round(float(np.cumsum(tr["reward"][0])[i]), 4)])
    with open(traj_dir / f"cold_saturation_seed{int(ep.seed[0])}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["controller", "step", "hour", "T_true_degC", "T_measured_degC", "preset", "preset_name",
                    "line_speed", "heater_setpoint_degC", "cure_window_low_degC", "cure_window_high_degC",
                    "safety_limit_degC", "heater_kW", "heater_limit_kW", "ambient_degC", "fault_active",
                    "cumulative_good_parts", "order_parts", "cumulative_energy_cost_EUR", "cumulative_margin_EUR"])
        w.writerows(trace_rows)
    out = dict(protocol_sha256=hashlib.sha256((REPO / "docs/experiments/PROTOCOL_v1.md").read_bytes()).hexdigest(),
               complete=complete, controllers={n: dict(condition=c, seed=s) for n, (c, s, _) in ctrls.items()},
               summaries=results, contrasts=contrasts,
               trajectory=dict(family="cold_saturation", index=k, scenario_seed=int(ep.seed[0]),
                               onset_step=int(ep.cold_on[0]), median_onset=float(np.median(onset)),
                               rule="held-out cold_saturation episode whose onset is closest to the median onset; "
                                    "ties: lowest seed", controllers=list(shown)),
               wall_seconds=round(time.perf_counter() - t0, 1))
    (RES / "confirmatory.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote confirmatory.json, episodes.csv.gz and the output D trace in {out['wall_seconds']} s")
    for k_, c in contrasts.items():
        if "mean" in c:
            print(f"{k_:30s} {c['mean']:8.2f} [{c['lower']:8.2f}, {c['upper']:8.2f}]",
                  "improved" if c.get("improved") else "")


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    main()
