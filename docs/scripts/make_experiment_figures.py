"""
Figures A-F of the benchmark-v1 experiment (docs/experiments/REPORT_v1.md), light and dark, SVG and PNG,
written to docs/figures/v1/. The data behind each figure is written to experiments/v1/results/figure_data/.
Theme, palette and save settings are those of docs/scripts/make_figures.py.

Run after experiments/v1/evaluate_all.py and experiments/v1/boundary_audit.py:
    .venv\\Scripts\\python.exe docs\\scripts\\make_experiment_figures.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("mf", REPO / "docs" / "scripts" / "make_figures.py")
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)

RES = REPO / "experiments" / "v1" / "results"
FIG = REPO / "docs" / "figures" / "v1"
DATA = RES / "figure_data"
CONDS = ("I0", "I1", "I2", "I3", "I4", "I5", "R1")
SEEDS = (1, 2, 3, 4, 5, 6)
LABEL = {"I0": "I0 original", "I1": "I1 eval-mode fix", "I2": "I2 update ratio", "I3": "I3 = I1 + I2",
         "I4": "I4 incremental accounting", "I5": "I5 Double DQN (proposed)", "R1": "R1 throughput reward"}
SHORT = {"I0": "I0 original", "I1": "I1 eval-mode fix", "I2": "I2 update ratio", "I3": "I3 = I1 + I2",
         "I4": "I4 accounting", "I5": "I5 Double DQN", "R1": "R1 throughput reward"}
SLOTS = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"],
         "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]}
T975_5 = 2.570581835636314
BIN_TICKS = ["<1e-6", "1e-6\nto 1e-5", "1e-5\nto 1e-4", "1e-4\nto 1e-3", "1e-3\nto 1e-2", "≥1e-2"]
NL = "\n"
NAMES = ("Conservative", "Normal", "Aggressive", "Energy saving", "Quality focus", "Balanced")


def pm(v):
    """+1.0 / −1.0 with a true minus sign."""
    return f"{v:+.1f}".replace("-", "−")


def color(mode, cond):
    return SLOTS[mode][CONDS.index(cond)]


def load():
    runs = {(c, s): json.loads((RES / "runs" / "confirmatory" / f"{c}_seed{s}.json").read_text(encoding="utf-8"))
            for c in CONDS for s in SEEDS}
    cal = json.loads((RES / "calibration.json").read_text(encoding="utf-8"))
    conf = json.loads((RES / "confirmatory.json").read_text(encoding="utf-8"))
    ba = json.loads((RES / "boundary_audit.json").read_text(encoding="utf-8"))
    rule_dev = next(r["margin"] for r in cal["rule_grid"] if r["cfg"] == cal["locked"]["rule"])
    refs_dev = {"locked rule": rule_dev, "best fixed": cal["fixed"][cal["locked"]["best_fixed_name"]]["margin"],
                "uniform": cal["uniform"]["margin"]}
    return runs, cal, conf, ba, refs_dev


def write(name, header, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def save(fig, name, mode):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(FIG / f"{name}-{mode}.{ext}", transparent=True, bbox_inches="tight", pad_inches=0.12,
                    metadata={"Date": None} if ext == "svg" else None, **kw)
    plt.close(fig)


def titles(fig, t, title, subtitle, y=1.0):
    """Conclusion-led title above a wrapped subtitle, both left-aligned above the axes (wrapped to fit 8 in)."""
    H = fig.get_size_inches()[1]
    sub = NL.join(textwrap.wrap(" ".join(subtitle.split()), 120))
    ttl = NL.join(textwrap.wrap(title, 82))
    y_sub = y + 0.10 / H
    fig.text(0.0, y_sub, sub, fontsize=9, color=t["ink2"], ha="left", va="bottom", linespacing=1.4)
    y_ttl = y_sub + ((sub.count(NL) + 1) * 9 * 1.4 / 72 + 0.10) / H
    fig.text(0.0, y_ttl, ttl, fontsize=12, fontweight="bold", color=t["ink"], ha="left", va="bottom",
             linespacing=1.25)


def ref_lines(ax, t, refs, label=False, xmax=None):
    styles = {"locked rule": ("-", t["ink"]), "best fixed": ("--", t["ink2"]), "uniform": (":", t["muted"])}
    for k, v in refs.items():
        ls, c = styles[k]
        ax.axhline(v, ls=ls, lw=1.0, color=c, zorder=1)
        if label:
            ax.text(xmax, v, f" {k}", color=t["ink2"], fontsize=7.5, va="center", ha="left")


# --------------------------------------------------------------------------------------------- A
def fig_a(t, mode, runs, refs, conf):
    fig, axes = plt.subplots(2, 4, figsize=(mf.WIDTH, 4.9), sharey=True)
    rows = []
    finals = {}
    for k, cond in enumerate(CONDS):
        ax = axes.flat[k]
        x = np.array([p["interactions"] for p in runs[(cond, 1)]["curve"]]) / 1000.0
        Y = np.array([[p["margin"] for p in runs[(cond, s)]["curve"]] for s in SEEDS])
        for s, y in zip(SEEDS, Y):
            ax.plot(x, y, color=color(mode, cond), lw=0.7, alpha=0.45, zorder=2)
        m, sd = Y.mean(0), Y.std(0, ddof=1)
        hw = T975_5 * sd / np.sqrt(len(SEEDS))
        ax.fill_between(x, m - hw, m + hw, color=color(mode, cond), alpha=0.18, lw=0, zorder=2)
        ax.plot(x, m, color=color(mode, cond), lw=2.0, zorder=3)
        ref_lines(ax, t, refs, label=(k == 3), xmax=x.max())
        upd = runs[(cond, 1)]["updates"]
        cpu = np.mean([runs[(cond, s)]["cpu_s"] for s in SEEDS])
        ax.set_title(f"{SHORT[cond]}\n{upd / 1000:.0f}k updates, {cpu / 60:.0f} CPU-min", fontsize=7.5,
                     color=t["ink"], loc="left")
        ax.grid(axis="y")
        ax.set_ylim(-260, 160)
        ax.set_xticks([0, 120, 240])
        finals[cond] = Y[:, -1]
        for s, y in zip(SEEDS, Y):
            for xi, yi, p in zip(x, y, runs[(cond, s)]["curve"]):
                rows.append([cond, s, p["episode"], p["interactions"], p["updates"], round(yi, 3)])
    for ax in axes[:, 0]:
        ax.set_ylabel("dev margin (EUR per shift)")
    axes[1, 0].set_xlabel("interactions (thousands)")
    # matched updates: seed means against gradient updates (I0 against I3, the update-ratio change)
    ax = axes.flat[7]
    for cond in ("I0", "I3"):
        u = np.array([p["updates"] for p in runs[(cond, 1)]["curve"]])
        Y = np.array([[p["margin"] for p in runs[(cond, s)]["curve"]] for s in SEEDS])
        ok = u > 0
        ax.plot(u[ok], Y.mean(0)[ok], color=color(mode, cond), lw=1.8, marker="o", ms=3, label=cond)
    ax.axvline(1000, color=t["ink2"], lw=0.8, ls=":")
    ax.text(1100, -250, "I0's total", fontsize=7, color=t["ink2"], va="bottom")
    ref_lines(ax, t, refs)
    ax.set_xscale("log")
    ax.set_xlabel("gradient updates (log)")
    ax.legend(loc="lower right", fontsize=7.5, handletextpad=0.3)
    ax.set_title("Matched updates\nI0 against I3, seed means", fontsize=7.5, color=t["ink"], loc="left")
    ax.grid(axis="y")
    fig.tight_layout(h_pad=1.6, w_pad=0.6)
    h1 = conf["contrasts"]["H1: I5 vs I0"]
    titles(fig, t, A_TITLE,
           "Development-set margin of the greedy policy against environment interactions; thin lines are the six "
           "training seeds, the band is the seed mean ± t-interval (95 %, 5 df). Reference lines: locked rule (solid), "
           "best fixed preset (dashed), uniform random (dotted). Last panel: the same seed means against gradient "
           "updates. "
           f"Held-out, I5 − I0 = {pm(h1['mean'])} EUR [{pm(h1['lower'])}, {pm(h1['upper'])}].")
    save(fig, "expA_learning", mode)
    write("expA_learning", ["condition", "seed", "episode", "interactions", "updates", "dev_margin_EUR"], rows)


# --------------------------------------------------------------------------------------------- B
def diverging(t, mode):
    neg, pos = ("#eb6834", "#2a78d6") if mode == "light" else ("#d95926", "#3987e5")
    mid = "#f0efec" if mode == "light" else "#383835"
    return LinearSegmentedColormap.from_list("div", [neg, mid, pos])


def sequential(t, mode):
    lo = "#fcfcfb" if mode == "light" else "#1a1a19"
    hi = "#e34948" if mode == "light" else "#e66767"
    return LinearSegmentedColormap.from_list("seq", [lo, hi])


def fig_b(t, mode, conf):
    import sys
    sys.path.insert(0, str(REPO / "experiments" / "v1"))
    import scenarios as S
    summ = conf["summaries"]
    nI, nJ = len(S.MAP_DT), len(S.MAP_LAG)
    D = np.zeros((nI, nJ))
    F = np.zeros((nI, nJ))
    Fr = np.zeros((nI, nJ))
    rows = []
    for i in range(nI):
        for j in range(nJ):
            key = f"{i},{j}"
            d = [summ[f"I5_seed{s}"]["map"][key]["margin"] - summ["locked rule"]["map"][key]["margin"] for s in SEEDS]
            D[i, j] = np.mean(d)
            F[i, j] = np.mean([summ[f"I5_seed{s}"]["map"][key]["fail_any_rate"] for s in SEEDS])
            Fr[i, j] = summ["locked rule"]["map"][key]["fail_any_rate"]
            rows.append([S.MAP_DT[i], S.MAP_LAG[j], int(S.in_dev_region(S.MAP_DT[i], S.MAP_LAG[j])), round(D[i, j], 3),
                         round(float(np.std(d, ddof=1)), 3), round(F[i, j], 4), round(Fr[i, j], 4),
                         6 * S.MAP_EPISODES])
    fig = plt.figure(figsize=(mf.WIDTH, 6.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.78], hspace=0.62, wspace=0.42)
    lim = max(abs(D.min()), abs(D.max()), 1.0)
    panels = [(fig.add_subplot(gs[0, 0]), D, diverging(t, mode), TwoSlopeNorm(0.0, -lim, lim),
               "I5 − locked rule, margin (EUR per shift)", "{:+.0f}"),
              (fig.add_subplot(gs[0, 1]), 100 * F, sequential(t, mode), plt.Normalize(0, 100),
               "I5 episodes failing any grader (%)\nsmall: locked rule", "{:.0f}")]
    for ax, Z, cmap, norm, ttl, fmt in panels:
        ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto", origin="upper")
        for i in range(nI):
            for j in range(nJ):
                val = Z[i, j]
                ax.text(j, i - (0.12 if Z is not D else 0), fmt.format(val), ha="center", va="center", fontsize=8,
                        color=t["ink"])
                if Z is not D:
                    ax.text(j, i + 0.28, f"{100 * Fr[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                            color=t["ink2"])
        ax.set_xticks(range(nJ), [f"{v:g}" for v in S.MAP_LAG])
        ax.set_yticks(range(nI), [f"{v:g}" for v in S.MAP_DT])
        ax.set_xlabel("thermal-lag factor")
        ax.set_ylabel("cold-front magnitude (K)")
        ax.set_title(ttl, fontsize=8.5, color=t["ink"], loc="left")
        # development region: magnitude 0 or -5 K, lag 0.8-1.25
        ax.add_patch(plt.Rectangle((0.5, -0.5), 3, 2, fill=False, lw=2.0, ec=t["ink"], ls="-"))
        for sp in ax.spines.values():
            sp.set_visible(False)
    # fault families
    ax = fig.add_subplot(gs[1, :])
    fams = [g for g in summ["locked rule"]["faults"] if g != "all"]
    y = np.arange(len(fams))
    frows = []
    for off, cond in ((-0.15, "I0"), (0.15, "I5")):
        d = np.array([[summ[f"{cond}_seed{s}"]["faults"][f]["margin"] - summ["locked rule"]["faults"][f]["margin"]
                       for s in SEEDS] for f in fams])
        m = d.mean(1)
        hw = T975_5 * d.std(1, ddof=1) / np.sqrt(len(SEEDS))
        ax.errorbar(m, y + off, xerr=hw, fmt="o", ms=5, color=color(mode, cond), ecolor=color(mode, cond), lw=1.4,
                    capsize=0, label=LABEL[cond])
        for f, mm, hh in zip(fams, m, hw):
            fail = np.mean([summ[f"{cond}_seed{s}"]["faults"][f]["fail_any_rate"] for s in SEEDS])
            frows.append([cond, f, round(mm, 3), round(hh, 3), round(fail, 4)])
    ticks = []
    for f in fams:
        fr = summ["locked rule"]["faults"][f]["fail_any_rate"]
        f5 = np.mean([summ[f"I5_seed{s}"]["faults"][f]["fail_any_rate"] for s in SEEDS])
        ticks.append(f"{f.replace('_', ' ')}{NL}fail any: rule {100 * fr:.0f} %, I5 {100 * f5:.0f} %")
    ax.axvline(0, color=t["axis"], lw=1)
    ax.set_yticks(y, ticks, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("margin difference against the locked rule (EUR per shift), seed mean ± 95 % t-interval")
    ax.grid(axis="x")
    ax.legend(loc="lower left", fontsize=7.5, ncol=2, bbox_to_anchor=(0.0, 1.0), handletextpad=0.3)
    ax.set_title("Held-out fault families (100 episodes each, onset in steps 30-150)", fontsize=8.5, color=t["ink"],
                 loc="left", pad=22)
    titles(fig, t, B_TITLE,
           "Top: 25 cells of cold-front magnitude × thermal-lag factor, 50 episodes per cell for each of six seeds "
           "(n = 300 per cell);\nthe outlined block is the training and development range. Bottom: held-out fault "
           "families, single and composed.", y=0.95)
    save(fig, "expB_generalisation", mode)
    write("expB_map", ["cold_front_K", "lag_factor", "inside_dev_region", "I5_minus_rule_EUR", "sd_between_seeds",
                       "I5_fail_any_rate", "rule_fail_any_rate", "episodes"], rows)
    write("expB_faults", ["condition", "family", "minus_rule_EUR", "t_halfwidth", "fail_any_rate"], frows)


# --------------------------------------------------------------------------------------------- C
def fig_c(t, mode, conf):
    summ = conf["summaries"]
    pts = []
    for name, meta in conf["controllers"].items():
        s = summ[name]["heldout"]["all"]
        pts.append((name, meta["condition"], s))
    fig, axes = plt.subplots(2, 2, figsize=(mf.WIDTH, 5.6))
    spec_ = [("margin", "ledger margin (EUR per shift)", 1.0), ("excess_parts", "good parts beyond the order", 1.0),
             ("energy_kwh", "energy (kWh per shift)", 1.0), ("scrap_fraction", "scrap (% of parts made)", 100.0)]
    rows = []
    for k, (ax, (key, lab, sc)) in enumerate(zip(axes.flat, spec_)):
        for name, cond, s in pts:
            x, yv = s["reward_throughput"], s[key] * sc
            if cond == "baseline":
                if name == "uniform":
                    continue
                ax.plot(x, yv, marker="s", ms=5, color=t["ink"], ls="none", zorder=4)
                short = name.replace("fixed:", "").replace("locked rule", "rule")
                if k == 0 or name == "locked rule":
                    ax.annotate(short, (x, yv), xytext=(4, 3), textcoords="offset points", fontsize=6.5,
                                color=t["ink2"])
            else:
                hi = cond in ("I3", "R1")
                ax.plot(x, yv, marker="o", ms=5 if hi else 3.5, ls="none", zorder=3 if hi else 2,
                        color=color(mode, cond) if hi else t["muted"], alpha=1.0 if hi else 0.6,
                        mec=mf.THEMES[mode]["page"], mew=0.6)
            if key == "margin":
                rows.append([name, cond, round(s["reward_throughput"], 3), round(s["margin"], 3),
                             round(s["excess_parts"], 2), round(s["energy_kwh"], 2), round(s["scrap_fraction"], 5),
                             round(s["shortfall"], 2), round(s["fail_any_rate"], 4)])
        ax.set_ylabel(lab)
        ax.grid(True)
    for ax in axes[1]:
        ax.set_xlabel("throughput reward (EUR per shift), held-out nominal")
    handles = [Line2D([], [], marker="o", ls="none", color=color(mode, "R1"),
                      label="R1, trained on throughput reward"),
               Line2D([], [], marker="o", ls="none", color=color(mode, "I3"),
                      label="I3, same algorithm, intended reward"),
               Line2D([], [], marker="o", ls="none", color=t["muted"], label="other DQN runs"),
               Line2D([], [], marker="s", ls="none", color=t["ink"], label="non-learning baselines")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=7.5,
               handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    titles(fig, t, C_TITLE,
           "Each point is one controller or training run on the 100 held-out nominal episodes. The x-axis is the "
           "misspecified throughput\nreward (every part paid, no order cap, no shortfall cost); the panels are "
           "ledger outcomes computed from the trajectories. Baselines are named in the first panel. The intended "
           "reward equals the ledger margin by construction (tests/test_benchmark_v1.py); constraint pass rates are "
           "in Figure E.")
    save(fig, "expC_reward_outcome", mode)
    write("expC_reward_outcome", ["controller", "condition", "throughput_reward_EUR", "margin_EUR", "excess_parts",
                                  "energy_kWh", "scrap_fraction", "shortfall_parts", "fail_any_rate"], rows)


# --------------------------------------------------------------------------------------------- D
def fig_d(t, mode, conf):
    tr = conf["trajectory"]
    path = RES / "trajectories" / f"cold_saturation_seed{tr['scenario_seed']}.csv"
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ctrls = list(dict.fromkeys(r["controller"] for r in rows))
    sty = {}
    for c in ctrls:
        if c.startswith("I0"):
            sty[c] = dict(color=color(mode, "I0"), ls="-", lw=1.5, label="I0 seed 1 (original)")
        elif c.startswith("I5"):
            sty[c] = dict(color=color(mode, "I5"), ls="-", lw=1.5, label="I5 seed 1 (proposed)")
        elif c == "locked rule":
            sty[c] = dict(color=t["ink"], ls="-", lw=1.5, label="locked rule")
        else:
            sty[c] = dict(color=t["ink2"], ls="--", lw=1.3, label="best fixed (" + c.split(":")[1] + ")")
    fig, axes = plt.subplots(5, 1, figsize=(mf.WIDTH, 8.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.5, 0.95, 0.9, 0.8, 0.9]))
    on = [float(r["hour"]) for r in rows if r["controller"] == ctrls[0] and r["fault_active"] == "1"]
    for ax in axes:
        ax.axvspan(min(on), max(on) + 2 / 60, color=t["grid"], alpha=0.9, lw=0, zorder=0)
        ax.axvspan(90 * 2 / 60, 180 * 2 / 60, color=t["grid"], alpha=0.0, lw=0)
        ax.grid(axis="y")
    for c in ctrls:
        r = [x for x in rows if x["controller"] == c]
        h = np.array([float(x["hour"]) for x in r]) + 2 / 60
        axes[0].plot(h, [float(x["T_true_degC"]) for x in r], **sty[c])
        axes[0].step(h, [float(x["cure_window_low_degC"]) for x in r], where="pre", color=sty[c]["color"], lw=0.8,
                     ls=":", alpha=0.8)
        axes[1].step(h, [float(x["heater_setpoint_degC"]) for x in r], where="pre", color=sty[c]["color"],
                     ls=sty[c]["ls"], lw=1.2)
        good = np.array([float(x["cumulative_good_parts"]) for x in r])
        axes[2].plot(h, good, color=sty[c]["color"], ls=sty[c]["ls"], lw=1.4)
        axes[3].plot(h, [float(x["cumulative_energy_cost_EUR"]) for x in r], color=sty[c]["color"], ls=sty[c]["ls"],
                     lw=1.4)
        axes[4].plot(h, [float(x["cumulative_margin_EUR"]) for x in r], color=sty[c]["color"], ls=sty[c]["ls"], lw=1.4)
        end = float(r[-1]["cumulative_margin_EUR"])
        axes[4].text(8.02, end, f" {end:.0f}", color=t["ink2"], fontsize=7.5, va="center")
    axes[0].axhline(205, color=mf.STATUS["critical"], lw=1.0, ls=(0, (4, 2)))
    axes[0].text(0.02, 206, "safety limit 205 °C", color=t["ink2"], fontsize=7.5, va="bottom")
    order = int(rows[0]["order_parts"])
    axes[2].axhline(order, color=t["ink2"], lw=1.0, ls=":")
    axes[2].text(0.02, order, f"order {order:,} parts", color=t["ink2"], fontsize=7.5, va="bottom")
    for ax in axes:
        ax.axvline(3, color=t["axis"], lw=0.8)
        ax.axvline(6, color=t["axis"], lw=0.8)
    axes[0].text(3.02, axes[0].get_ylim()[0], " peak tariff from 3 h", color=t["ink2"], fontsize=7, va="bottom")
    axes[0].text(min(on) + 0.02, axes[0].get_ylim()[1], " cold front −15 K + heater at 60 %", color=t["ink2"],
                 fontsize=7.5, va="top")
    axes[0].set_ylabel("oven temperature\n(°C, true)")
    sp = sorted({(round(th * 180.0), nm) for th, nm in zip((0.85, 1.0, 1.1, 0.8, 0.95, 0.9), NAMES)})
    axes[1].set_yticks([v for v, _ in sp], [f"{nm} {v}" for v, nm in sp], fontsize=6.5)
    axes[1].set_ylabel("preset\n(heater setpoint, °C)")
    axes[2].set_ylabel("cumulative\ngood parts")
    axes[3].set_ylabel("cumulative energy\ncost (EUR)")
    axes[4].set_ylabel("cumulative\nmargin (EUR)")
    axes[4].set_xlabel("shift time (h)")
    axes[4].set_xlim(0, 8)
    handles = [Line2D([], [], **sty[c]) for c in ctrls]
    handles.append(Line2D([], [], color=t["ink2"], lw=0.8, ls=":", label="lower edge of each controller's cure window"))
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.32), fontsize=7.5, ncol=3)
    fig.tight_layout(h_pad=0.6)
    titles(fig, t, D_TITLE,
           f"Held-out composed-fault episode {tr['scenario_seed']} (selection rule: onset closest to the family's "
           f"median onset, step {tr['onset_step']}).\nAll four controllers see the same episode. Shaded: fault active. "
           "Traces: experiments/v1/results/trajectories/.")
    save(fig, "expD_trajectory", mode)


# --------------------------------------------------------------------------------------------- E
def fig_e(t, mode, conf):
    summ = conf["summaries"]
    fig, (ax, az) = plt.subplots(1, 2, figsize=(mf.WIDTH, 4.4), gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    rows = []
    zoom = dict(x=(-1.5, 22.0), y=(60.0, 135.0))
    for name, meta in conf["controllers"].items():
        s = summ[name]["heldout"]["all"]
        x, y, feas = 100 * s["fail_any_rate"], s["margin"], s["feasible"]
        cond = meta["condition"]
        for a in (ax, az):
            inside = zoom["x"][0] <= x <= zoom["x"][1] and zoom["y"][0] <= y <= zoom["y"][1]
            if a is az and not inside:
                continue
            if cond == "baseline":
                a.plot(x, y, marker="s", ms=6, ls="none", color=t["ink"], mfc=t["ink"] if feas else "none", zorder=4)
                if a is az or not inside:
                    a.annotate(name.replace("fixed:", ""), (x, y), xytext=(5, -2), textcoords="offset points",
                               fontsize=7, color=t["ink2"])
            else:
                c = color(mode, cond)
                a.plot(x, y, marker="o", ms=5.5, ls="none", color=c, mfc=c if feas else "none", mew=1.3, zorder=3)
        rows.append([name, cond, meta["seed"] if meta["seed"] is not None else "", round(y, 3), round(x, 3),
                     int(feas), *(round(s["pass_rate"][g], 4) for g in ("safety", "delivery", "quality"))])
    ax.add_patch(plt.Rectangle((zoom["x"][0], zoom["y"][0]), zoom["x"][1] - zoom["x"][0], zoom["y"][1] - zoom["y"][0],
                               fill=False, ec=t["ink2"], lw=0.8, ls="--"))
    az.set_xlim(*zoom["x"])
    az.set_ylim(*zoom["y"])
    az.set_title("zoom on the dashed box", fontsize=8, color=t["ink"], loc="left")
    for a in (ax, az):
        a.set_xlabel("held-out episodes failing at least one grader (%)")
        a.grid(True)
    ax.set_ylabel("held-out margin (EUR per shift)")
    handles = [Line2D([], [], marker="o", ls="none", color=color(mode, c), label=c) for c in CONDS]
    handles += [Line2D([], [], marker="s", ls="none", color=t["ink"], label="baseline"),
                Line2D([], [], marker="o", ls="none", color=t["ink2"], mfc="none", label="infeasible (hollow)")]
    az.set_xlabel("failing a grader (%)")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=9, fontsize=7.5, handletextpad=0.2,
               columnspacing=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.0)
    titles(fig, t, E_TITLE,
           "Measured points only, no fitted frontier: one point per baseline and per training run (seed). Filled = "
           "feasible: safety pass rate ≥ 99 %, delivery (≥ 99 % of the order) ≥ 95 %, quality (scrap ≤ 5 %) ≥ 95 % "
           "of the 100 held-out nominal episodes. Right: the dashed box enlarged.")
    save(fig, "expE_frontier", mode)
    write("expE_frontier", ["controller", "condition", "seed", "margin_EUR", "fail_any_pct", "feasible",
                            "pass_safety", "pass_delivery", "pass_quality"], rows)


# --------------------------------------------------------------------------------------------- F
def fig_f(t, mode, ba):
    agg, bins = ba["by_source_and_margin"], ba["margin_bins"]
    fig, axes = plt.subplots(1, 2, figsize=(mf.WIDTH, 3.6), gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    ax = axes[0]
    x = np.arange(len(bins))
    srcs = [("rollout", "corpus A, rollout", SLOTS[mode][0]), ("synthetic", "corpus A, synthetic", SLOTS[mode][1]),
            ("visited", "visited closed-loop", SLOTS[mode][2])]
    rows = []
    for k, (src, lab, c) in enumerate(srcs):
        n = np.array([agg[src][b]["states"] for b in bins], dtype=float)
        d = np.array([agg[src][b]["st_vs_torch"] for b in bins])
        ax.bar(x + (k - 1) * 0.27, np.where(n > 0, n, np.nan), width=0.25, color=c, label=lab, zorder=2)
        for xi, ni, di in zip(x + (k - 1) * 0.27, n, d):
            if ni > 0 and di > 0:
                ax.annotate(f"{di} flip{'s' if di > 1 else ''} ({src})", (xi, ni), xytext=(xi - 0.35, ni * 12),
                            ha="center", fontsize=7, color=t["ink"], fontweight="bold",
                            arrowprops=dict(arrowstyle="-", color=t["ink2"], lw=0.8))
        for b, ni, di in zip(bins, n, d):
            rows.append([src, b, int(ni), int(di), agg[src][b]["torch_vs_f64"], agg[src][b]["max_abs_dq_st_torch"],
                         agg[src][b]["max_abs_dq_torch_f64"]])
    ax.set_yscale("log")
    ax.set_xticks(x, BIN_TICKS, fontsize=7.5)
    ax.set_xlabel("top-two Q gap (float64)")
    ax.set_ylabel("states (18 networks)")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(axis="y")
    ax.set_ylim(0.7, 1e6)
    ax = axes[1]
    for k, (src, lab, c) in enumerate(srcs):
        e = [agg[src][b]["max_abs_dq_st_torch"] if agg[src][b]["states"] else np.nan for b in bins]
        ax.plot(x + (k - 1) * 0.18, e, marker="o", ms=5, color=c, ls="none", label=lab)
    ax.axhline(1e-3, color=t["ink2"], ls="--", lw=1)
    ax.text(len(bins) - 1, 1.15e-3, "gate tolerance 1e-3", fontsize=7, color=t["ink2"], ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(x, BIN_TICKS, fontsize=7.5)
    ax.set_xlabel("top-two Q gap (float64)")
    ax.set_ylabel("max |Q(.st) − Q(PyTorch)|")
    ax.legend(loc="center right", fontsize=7, handletextpad=0.1)
    ax.grid(axis="y")
    fig.tight_layout(w_pad=1.5)
    titles(fig, t, F_TITLE,
           "18 networks: 20,000 corpus-A states each (10,000 rollout, 10,000 synthetic) and the 20,000 states each "
           "visits under its own greedy policy\n(100 dev episodes). A flip is a .st action differing from PyTorch "
           "float32; tie_tolerance = 0. The .fbt outputs are bit-identical to the .st.")
    save(fig, "expF_decision_boundary", mode)
    write("expF_decision_boundary", ["source", "margin_bin", "states", "st_vs_torch_flips", "torch_vs_f64_flips",
                                     "max_abs_dq_st_torch", "max_abs_dq_torch_f64"], rows)


A_TITLE = B_TITLE = C_TITLE = D_TITLE = E_TITLE = F_TITLE = ""


def main():
    global A_TITLE, B_TITLE, C_TITLE, D_TITLE, E_TITLE, F_TITLE
    runs, cal, conf, ba, refs = load()
    T = json.loads((RES / "figure_titles.json").read_text(encoding="utf-8"))
    A_TITLE, B_TITLE, C_TITLE, D_TITLE, E_TITLE, F_TITLE = (T[k] for k in "ABCDEF")
    for mode, t in mf.THEMES.items():
        mf.setup(t)
        plt.rcParams["svg.hashsalt"] = "make_experiment_figures"
        fig_a(t, mode, runs, refs, conf)
        fig_b(t, mode, conf)
        fig_c(t, mode, conf)
        fig_d(t, mode, conf)
        fig_e(t, mode, conf)
        fig_f(t, mode, ba)
    print("wrote", sorted(p.name for p in FIG.glob("exp*")))


if __name__ == "__main__":
    main()
