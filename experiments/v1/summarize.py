"""
Markdown tables of the benchmark-v1 results, generated from the stored JSON files (no number is typed by hand).
Writes experiments/v1/results/summary.md. Run:  .venv\\Scripts\\python.exe experiments\\v1\\summarize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scenarios as S  # noqa: E402

RES = HERE / "results"
CONDS = ("I0", "I1", "I2", "I3", "I4", "I5", "R1")
SEEDS = (1, 2, 3, 4, 5, 6)
T975_5 = 2.570581835636314
G = ("safety", "delivery", "quality")


def j(path):
    return json.loads((RES / path).read_text(encoding="utf-8"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    conf, cal = j("confirmatory.json"), j("calibration.json")
    summ, con = conf["summaries"], conf["contrasts"]
    runs = {(c, s): j(f"runs/confirmatory/{c}_seed{s}.json") for c in CONDS for s in SEEDS}
    out = []
    w = out.append

    w("### Held-out nominal, 100 paired episodes (conditions: mean over 6 seeds ± 95 % t half-width)\n")
    w("| Controller | Margin (EUR/shift) | Safety | Delivery | Quality | Feasible | Dev margin, final checkpoint |")
    w("|---|---|---|---|---|---|---|")
    for name in ["uniform"] + [n for n in summ if n.startswith("fixed:")] + ["locked rule"]:
        s = summ[name]["heldout"]["all"]
        w(f"| {name.replace('fixed:', 'fixed: ')} | {s['margin']:.1f} | " + " | ".join(
            f"{100 * s['pass_rate'][g]:.0f} %" for g in G) + f" | {'yes' if s['feasible'] else 'no'} | |")
    for c in CONDS:
        m = np.array([summ[f"{c}_seed{s}"]["heldout"]["all"]["margin"] for s in SEEDS])
        hw = T975_5 * m.std(ddof=1) / np.sqrt(len(m))
        pr = {g: np.mean([summ[f"{c}_seed{s}"]["heldout"]["all"]["pass_rate"][g] for s in SEEDS]) for g in G}
        feas = sum(summ[f"{c}_seed{s}"]["heldout"]["all"]["feasible"] for s in SEEDS)
        dev = np.mean([runs[(c, s)]["curve"][-1]["margin"] for s in SEEDS])
        w(f"| {c} | {m.mean():.1f} ± {hw:.1f} | " + " | ".join(f"{100 * pr[g]:.0f} %" for g in G)
          + f" | {feas}/6 seeds | {dev:.1f} |")

    w("\n### Preregistered contrasts on held-out nominal (paired by seed; 95 % t-interval, 5 df)\n")
    w("| Contrast | Mean difference (EUR/shift) | 95 % CI | SD between seeds | SD over episodes, within seed "
      "| Constraint non-inferiority (S/D/Q) | A feasible | Decision |")
    w("|---|---|---|---|---|---|---|---|")
    for k, c in con.items():
        if "mean" not in c:
            continue
        if "non_inferior" in c:
            ni = "/".join("yes" if c["non_inferior"][g] else "no" for g in G)
            dec = "improved" if c["improved"] else "no improvement shown"
            w(f"| {k} | {c['mean']:+.1f} | [{c['lower']:+.1f}, {c['upper']:+.1f}] | {c['sd_between_seeds']:.1f} | "
              f"{c['sd_within_seed_episodes']:.1f} | {ni} | {'yes' if c['a_feasible'] else 'no'} | {dec} |")
        else:
            w(f"| {k} | {c['mean']:+.1f} | [{c['lower']:+.1f}, {c['upper']:+.1f}] | {c['sd_between_seeds']:.1f} | "
              "| | | |")

    w("\n### Held-out fault families (100 episodes each): margin and share of episodes failing any grader\n")
    w("| Family | Locked rule | I0 − rule | I5 − rule [95 % CI] | Fail any: rule | I0 | I5 |")
    w("|---|---|---|---|---|---|---|")
    var = con.get("scenario_variation_I5_vs_locked_rule", {})
    for f in S.FAULTS:
        r = summ["locked rule"]["faults"][f]
        d0 = np.mean([summ[f"I0_seed{s}"]["faults"][f]["margin"] for s in SEEDS]) - r["margin"]
        d5 = var["faults"]["per_group"][f]
        f0 = np.mean([summ[f"I0_seed{s}"]["faults"][f]["fail_any_rate"] for s in SEEDS])
        f5 = np.mean([summ[f"I5_seed{s}"]["faults"][f]["fail_any_rate"] for s in SEEDS])
        w(f"| {f} | {r['margin']:.1f} | {d0:+.1f} | {d5['mean']:+.1f} [{d5['lower']:+.1f}, {d5['upper']:+.1f}] | "
          f"{100 * r['fail_any_rate']:.0f} % | {100 * f0:.0f} % | {100 * f5:.0f} % |")
    mp = var["map"]["per_group"]
    inside = [mp[k]["mean"] for k in mp if S.in_dev_region(S.MAP_DT[int(k[0])], S.MAP_LAG[int(k[2])])]
    outside = [mp[k]["mean"] for k in mp if not S.in_dev_region(S.MAP_DT[int(k[0])], S.MAP_LAG[int(k[2])])]
    worst = min(mp, key=lambda k: mp[k]["mean"])
    best = max(mp, key=lambda k: mp[k]["mean"])
    w(f"\nMap, I5 − locked rule: mean over the {len(inside)} cells inside the development range "
      f"{np.mean(inside):+.1f} EUR, over the {len(outside)} cells outside {np.mean(outside):+.1f} EUR; "
      f"best cell (cold front {S.MAP_DT[int(best[0])]:g} K, lag {S.MAP_LAG[int(best[2])]:g}) {mp[best]['mean']:+.1f}, "
      f"worst cell ({S.MAP_DT[int(worst[0])]:g} K, lag {S.MAP_LAG[int(worst[2])]:g}) {mp[worst]['mean']:+.1f}. "
      f"SD of the I5 − rule difference across fault families {var['faults']['sd_across_groups']:.1f} EUR, "
      f"across map cells {var['map']['sd_across_groups']:.1f} EUR.")

    w("\n### Compute per run (mean over 6 seeds)\n")
    w("| Condition | Interactions | Gradient updates | Training wall time (s) | CPU time (s) |")
    w("|---|---|---|---|---|")
    for c in CONDS:
        r = [runs[(c, s)] for s in SEEDS]
        w(f"| {c} | {r[0]['interactions']:,} | {np.mean([x['updates'] for x in r]):,.0f} | "
          f"{np.mean([x['train_wall_s'] for x in r]):.0f} | {np.mean([x['cpu_s'] for x in r]):.0f} |")
    man = {t: j(f"manifest_{t}.json") for t in ("pilot", "confirmatory") if (RES / f"manifest_{t}.json").exists()}
    cpu_total = sum(x["cpu_s"] for x in runs.values())
    w(f"\nConfirmatory training: {len(runs)} runs, {sum(len(m['failed']) for m in man.values())} failed; wall "
      f"{man['confirmatory']['wall_s'] / 60:.1f} min on {man['confirmatory']['workers']} worker processes; summed "
      f"process CPU time {cpu_total / 3600:.2f} h. Pilot: wall {man['pilot']['wall_s'] / 60:.1f} min.")
    stages = {"v0 audit": j("v0_audit.json")["wall_seconds"], "calibration": cal["wall_seconds"],
              "boundary audit": j("boundary_audit.json")["wall_seconds"],
              "pilot": man["pilot"]["wall_s"], "confirmatory training": man["confirmatory"]["wall_s"],
              "held-out evaluation": conf["wall_seconds"]}
    w("Wall time by stage (s): " + ", ".join(f"{k} {v:.0f}" for k, v in stages.items())
      + f"; total {sum(stages.values()) / 60:.1f} min.")
    text = "\n".join(out) + "\n"
    (RES / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
