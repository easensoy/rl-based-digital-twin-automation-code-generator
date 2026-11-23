"""
Development-stage calibration of benchmark v1. Uses development scenarios only (seeds 1,000,000-1,000,099 and
the development versions of the fault families); no held-out scenario is evaluated here.

1. Fixed presets and uniform random on the development set.
2. The rule's 56 configurations (controllers.RULE_GRID); the feasible one (evaluate.FEASIBLE) with the highest development mean
   margin is locked.
3. The feasible fixed preset with the highest development mean margin is locked.
4. Evaluator validation: references expected to pass (locked rule, best fixed preset) and deliberately
   defective controllers (constant Aggressive, the locked rule blind to the temperature sensor, the locked
   rule blind to the good-part counter, the throughput maximiser) on the development nominal set and the
   development fault families, scored by the deterministic graders.

Writes experiments/v1/results/calibration.json. Run:  .venv\Scripts\python.exe experiments\v1\calibrate.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import controllers as C  # noqa: E402
import scenarios as S  # noqa: E402
from evaluate import GRADERS, paired, run, summary  # noqa: E402
from process import NAMES  # noqa: E402

OUT = HERE / "results" / "calibration.json"


def main():
    t0 = time.perf_counter()
    dev = S.dev()
    res = {"dev_seeds": [S.DEV[0], S.DEV[-1]], "fixed": {}, "rule_grid": []}
    runs = {}
    for a in range(6):
        runs[f"fixed_{a}"] = run(C.fixed(a), dev)
        res["fixed"][NAMES[a]] = summary(runs[f"fixed_{a}"])
    runs["uniform"] = run(C.uniform(0), dev)
    res["uniform"] = summary(runs["uniform"])
    for cfg in C.RULE_GRID:
        o = run(C.rule(cfg), dev)
        res["rule_grid"].append({"cfg": cfg, **summary(o)})
    feasible_rules = [r for r in res["rule_grid"] if r["feasible"]]
    best_rule = max(feasible_rules, key=lambda r: r["margin"])
    best_fixed = max((a for a in range(6) if res["fixed"][NAMES[a]]["feasible"]),
                     key=lambda a: res["fixed"][NAMES[a]]["margin"])
    res["locked"] = {"rule": best_rule["cfg"], "best_fixed": best_fixed, "best_fixed_name": NAMES[best_fixed],
                     "feasible_rule_configs": len(feasible_rules), "rule_configs": len(res["rule_grid"])}
    rule_run = run(C.rule(best_rule["cfg"]), dev)
    res["rule_vs_best_fixed_dev"] = paired(rule_run["margin"], runs[f"fixed_{best_fixed}"]["margin"])
    print("fixed:", {k: (round(v["margin"], 1), v["feasible"]) for k, v in res["fixed"].items()},
          "| uniform", round(res["uniform"]["margin"], 1))
    print("locked rule:", best_rule["cfg"], round(best_rule["margin"], 2), f"({len(feasible_rules)} of {len(C.RULE_GRID)} feasible)",
          "| best fixed:", NAMES[best_fixed], "| rule - best fixed:", res["rule_vs_best_fixed_dev"])

    # evaluator validation on development scenarios
    cfg = best_rule["cfg"]
    ctrls = {"reference: locked rule": C.rule(cfg),
             f"reference: best fixed ({NAMES[best_fixed]})": C.fixed(best_fixed),
             "defect: constant Aggressive": C.fixed(2),
             "defect: rule blind to temperature": C.rule(cfg, sensor_blind=True),
             "defect: rule blind to part counter": C.rule(cfg, counter_blind=True),
             "defect: throughput maximiser": C.throughput()}
    families = {"nominal": dev, **{f: S.fault_family(f, "dev") for f in S.FAULTS}}
    res["evaluator_validation"] = {}
    for fam, sc in families.items():
        outs = {name: run(c, sc) for name, c in ctrls.items()}
        ref = outs["reference: locked rule"]
        row = {}
        for name, o in outs.items():
            sm = summary(o)
            sm["margin_vs_locked_rule"] = paired(o["margin"], ref["margin"])
            sm["identical_actions_to_locked_rule"] = None
            row[name] = sm
        res["evaluator_validation"][fam] = row
        print(f"{fam:17s}", " | ".join(
            f"{n.split(': ')[1][:12]} {v['margin']:6.1f} " + "".join("P" if v["pass_rate_" + g] >= 0.5 else "-"
                                                                    for g in GRADERS)
            + ("*" if v["feasible"] else " ") for n, v in row.items()))
    # which defects change the actions at all (a defect whose code path is never reached is harmless)
    for fam, sc in families.items():
        base = run(C.rule(cfg), sc, keep_traj=True)["traj"]["action"]
        for name in ("defect: rule blind to temperature", "defect: rule blind to part counter"):
            c = C.rule(cfg, sensor_blind="temperature" in name, counter_blind="counter" in name)
            act = run(c, sc, keep_traj=True)["traj"]["action"]
            res["evaluator_validation"][fam][name]["identical_actions_to_locked_rule"] = bool(np.array_equal(act, base))
            res["evaluator_validation"][fam][name]["changed_action_share"] = float(np.mean(act != base))
    res["wall_seconds"] = round(time.perf_counter() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(HERE.parents[1])} in {res['wall_seconds']} s")


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    main()
