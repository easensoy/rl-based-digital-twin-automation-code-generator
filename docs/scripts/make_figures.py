"""
README figures, drawn from stored evidence only (no training, no server, no network calls).

  fig1_export_verification     hero: frozen policy -> generated code -> checks, with the measured outcomes
  fig2_policy_evaluation       trained DQN against untrained, uniform random and best fixed action (historical)
  fig3_near_tie_rejection      the export the strict gate rejected: Q-values, actions and setpoints at 2 states
  fig4_verification_coverage   what the checks exercised, and what was not run or not verified
  supp_td_error_vs_margin      supporting diagnostic for docs/TRAINING_DIAGNOSTIC.md

Inputs (read-only):
  docs/data/export_verification.json                     measure_export_verification.py (2026-11-08)
  tests/fixtures/near_tie_speed_proportional_seed5.json  extract_near_tie_fixture.py (2026-11-08)
  visual_outputs/run/eval_v2/summary_*.json              multi-seed evaluation (2026-11-07, historical)
  visual_outputs/run/evaluation.json                     single server training run, seed 42 (2026-11-06, historical)
  visual_outputs/data/training_loss.csv                  the same server run

Writes docs/figures/<name>-{light,dark}.{svg,png} and the plotted values as docs/data/<name>.csv
(figure 3 plots docs/data/near_tie_states.csv, written by the extractor, and checks it against the fixture).
Run from the repository root:
    .venv\\Scripts\\python.exe docs\\scripts\\make_figures.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIG_DIR = REPO / "docs" / "figures"
DATA_DIR = REPO / "docs" / "data"
EVAL_V2 = REPO / "visual_outputs" / "run" / "eval_v2"
EVAL_HIST = REPO / "visual_outputs" / "run" / "evaluation.json"
LOSS_CSV = REPO / "visual_outputs" / "data" / "training_loss.csv"
MEASURED = DATA_DIR / "export_verification.json"
NEAR_TIE = REPO / "tests" / "fixtures" / "near_tie_speed_proportional_seed5.json"
NEAR_TIE_CSV = DATA_DIR / "near_tie_states.csv"

# Reference palette of the data-viz method, validated on GitHub's page surfaces (#ffffff, #0d1117).
# Status colours are fixed across themes and always come with an icon and a label.
STATUS = dict(good="#0ca30c", warning="#fab219", critical="#d03b3b")
THEMES = {
    "light": dict(ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", card="#f6f8fa", edge="#d0d7de",
                  page="#ffffff", **STATUS),
    "dark": dict(ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 blue="#3987e5", orange="#d95926", aqua="#199e70", card="#161b22", edge="#30363d",
                 page="#0d1117", **STATUS),
}
WIDTH = 8.0  # inches; shown at ~830 px in a GitHub README column, so 8 pt is ~11.5 px
thousands = FuncFormatter(lambda v, _: f"{v:,.0f}")


def setup(t):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "svg.fonttype": "path",
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"],
        "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "axes.facecolor": "none", "figure.facecolor": "none", "savefig.facecolor": "none",
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.linewidth": 0.8, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 5, "ytick.major.pad": 8,
        "legend.frameon": False,
    })


def save(fig, name, mode):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(FIG_DIR / f"{name}-{mode}.{ext}", transparent=True, bbox_inches="tight", pad_inches=0.12,
                    metadata={"Date": None} if ext == "svg" else None, **kw)
    plt.close(fig)


def write_csv(name, header, rows):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def sci(v, digits=1):
    """1.2e-04 -> '1.2×10⁻⁴' for labels."""
    m, e = f"{v:.{digits}e}".split("e")
    sup = str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m}×10{sup}"


# ---------------------------------------------------------------------------
# Drawing helpers: a full-figure canvas in inches, cards, status badges
# ---------------------------------------------------------------------------
def canvas(fig, w, h):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return ax


def head(ax, t, h, title, subtitle):
    ax.text(0, h, title, fontsize=12, fontweight="bold", va="top", color=t["ink"])
    ax.text(0, h - 0.3, subtitle, fontsize=9, va="top", color=t["ink2"], linespacing=1.4)


def card(ax, t, x, y, w, h, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", linewidth=0.9,
                                edgecolor=t["muted"] if dashed else t["edge"],
                                facecolor="none" if dashed else t["card"], linestyle=(0, (3, 2)) if dashed else "-"))


def badge(ax, t, x, y, kind, r=0.075):
    """Status icon, always drawn next to a text label (colour never carries meaning alone)."""
    if kind in ("pass", "reject", "warn"):
        fill = {"pass": t["good"], "reject": t["critical"], "warn": t["warning"]}[kind]
        ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor="none", zorder=5))
        ax.text(x, y - 0.004, {"pass": "✓", "reject": "✗", "warn": "!"}[kind], ha="center", va="center",
                fontsize=r * 105, fontweight="bold", color="#0b0b0b" if kind == "warn" else "#ffffff", zorder=6)
    elif kind == "notverified":
        ax.add_patch(Circle((x, y), r * 0.92, facecolor="none", edgecolor=t["muted"], linewidth=1.2,
                            linestyle=(0, (1.6, 1.3)), zorder=5))
    else:  # not run
        ax.text(x, y, "–", ha="center", va="center", fontsize=10, color=t["muted"], zorder=6)


def two_tone(ax, t, x, y, bold, rest, size):
    """A bold lead-in followed by regular text on the same baseline."""
    a = ax.text(x, y, bold, fontsize=size, fontweight="bold", va="baseline", color=t["ink"])
    bb = a.get_window_extent(renderer=ax.figure.canvas.get_renderer()).transformed(ax.transData.inverted())
    ax.text(bb.x1, y, rest, fontsize=size, va="baseline", color=t["ink2"])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def measured_facts(m):
    nets = m["networks"]
    rejected = [n for n in nets if not n["ok"]]
    eae = m.get("eae") or {}
    accepted = list((eae.get("accepted") or {}).values())
    negative = eae.get("negative_controls") or []
    st = m.get("st_check") or {}
    return dict(
        st_runs=1 if st else 0,
        st_bit_identical=bool(st.get("q_bit_identical_to_fbt")) and bool(st.get("outputs_identical_to_fbt")),
        n=len(nets), n_trained=sum(n["group"].startswith("trained") for n in nets),
        n_synthetic=sum(n["group"].startswith("synthetic") for n in nets),
        states=m["protocol"]["states"]["rollout"]["n"] + m["protocol"]["states"]["box"]["n"],
        n_rollout=m["protocol"]["states"]["rollout"]["n"], n_box=m["protocol"]["states"]["box"]["n"],
        q_tol=m["protocol"]["q_tolerance"], tie_tol=m["protocol"]["tie_tolerance"],
        n_q_ok=sum(n["checks"]["q_values"] for n in nets), worst_dq=max(n["max_abs_q_diff"] for n in nets),
        n_ok=sum(n["ok"] for n in nets), rejected=rejected,
        mutants=len(m["mutations"]), mutants_rejected=sum(not x["ok"] for x in m["mutations"]),
        eae_version=eae.get("version"), eae_accepted=sum(a["ok"] for a in accepted), eae_runs=len(accepted),
        eae_errors=sum(a["errors"] for a in accepted), eae_warnings=sum(a["warnings"] for a in accepted),
        eae_known=sorted({a["known_standalone_messages"] for a in accepted}),
        neg_rejected=sum(not x["ok"] for x in negative), neg_runs=len(negative),
        st_identical=(m.get("st_check") or {}).get("outputs_identical_to_fbt"),
    )


def load_eval():
    out = {}
    for model in ("legacy", "speed_proportional"):
        s = load_json(EVAL_V2 / f"summary_{model}.json")
        best = s["best_fixed_action"]
        out[model] = dict(
            best_name=best.split(": ", 1)[1], best=s["baselines"][best]["total_reward"],
            random=s["baselines"]["Uniform random"]["total_reward"],
            seeds=[(p["train_seed"], p["eval_reward_mean"], p["eval_reward_ci95_episodes"], p["dominant_action"],
                    p["dominant_share"]) for p in s["per_seed"]],
            across=s["across_seeds"]["eval_reward"], episodes=s["eval_episodes_per_policy"],
            env_seeds=s["env_seeds"])
    hist = load_json(EVAL_HIST)
    # comparable only if the 2026-11-06 run used the same evaluation episodes: check it, do not assume it
    assert hist["env_seeds"] == out["legacy"]["env_seeds"] and hist["episodes_per_policy"] == out["legacy"]["episodes"]
    assert math.isclose(hist["policies"]["Uniform random"]["total_reward"]["mean"], out["legacy"]["random"]["mean"])
    out["untrained_legacy"] = hist["policies"]["Untrained DQN (greedy)"]["total_reward"]
    out["hist_q_gap"] = hist["plc_fidelity"]["q_gap_best_minus_second_mean"]
    out["hist_q_gap_states"] = hist["plc_fidelity"]["n_states"]
    out["episode_ci_max"] = max([out["untrained_legacy"]["ci95"]] + [out[k][b]["ci95"] for k in ("legacy", "speed_proportional")
                                                                     for b in ("best", "random")]
                                + [ci for k in ("legacy", "speed_proportional") for _, _, ci, _, _ in out[k]["seeds"]])
    return out


def load_near_tie():
    case = load_json(NEAR_TIE)
    with open(NEAR_TIE_CSV, encoding="utf-8") as fh:
        rows = {int(r["state_index"]): r for r in csv.DictReader(fh)}
    for s in case["states"]:  # the linked table and the fixture must say the same
        a, b = s["competing_actions"]
        r = rows[s["index"]]
        assert float(r["reference_q_first"]) == s["reference_q"][a] and float(r["generated_q_second"]) == s["generated_q"][b]
    return case


# ---------------------------------------------------------------------------
# Figure 1 (hero): the export, the checks, and what they found
# ---------------------------------------------------------------------------
def fig1(t, mode, f):
    W, H = WIDTH, 5.3
    fig = plt.figure(figsize=(W, H))
    ax = canvas(fig, W, H)
    head(ax, t, H, "The gate caught an action mismatch that Q-value checks alone would accept",
         "A frozen PyTorch DQN exported as IEC 61499 (.fbt) and IEC 61131-3 (.st) code, checked against the network.")
    y0, bh, bw, gap = 4.0, 0.66, 2.42, 0.37
    boxes = [("Frozen PyTorch policy", ["DQN, 12-128-128-128-6 MLP", "35,462 weights, eval mode"]),
             ("Exporter", ["iec61499_exporter.py", "weights → REAL arrays"]),
             ("Generated function block", [".fbt  IEC 61499, EAE format", ".st   IEC 61131-3"])]
    for i, (hd, lines) in enumerate(boxes):
        x = i * (bw + gap)
        card(ax, t, x, y0, bw, bh)
        ax.text(x + 0.12, y0 + bh - 0.1, hd, fontsize=9.2, fontweight="bold", va="top")
        for j, line in enumerate(lines):
            ax.text(x + 0.12, y0 + bh - 0.33 - 0.15 * j, line, fontsize=8.2, color=t["ink2"], va="top")
        if i < 2:
            ax.add_patch(FancyArrowPatch((x + bw + 0.05, y0 + bh / 2), (x + bw + gap - 0.05, y0 + bh / 2),
                                         arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color=t["ink2"]))
    xa = 2 * (bw + gap) + bw / 2
    ax.add_patch(FancyArrowPatch((xa, y0 - 0.03), (xa, 3.72), arrowstyle="-|>", mutation_scale=10, linewidth=1.1,
                                 color=t["ink2"]))

    two_tone(ax, t, 0, 3.57, "The .fbt, run against PyTorch", f"  in a float32 interpreter: {f['n']} networks × "
             f"{f['states']:,} sampled states, strict gate", 9.2)
    dis = sum(r["argmax_disagreements"] for r in f["rejected"])
    tiles = [
        ("pass", "passed", f"{f['n_q_ok']} / {f['n']}", ["networks with every Q-value", "within 10⁻³",
                                                         f"(worst {sci(f['worst_dq'])})"]),
        ("warn" if f["n_ok"] < f["n"] else "pass", f"{f['n'] - f['n_ok']} rejected" if f["n_ok"] < f["n"] else "passed",
         f"{f['n_ok']} / {f['n']}", ["networks with the same", "action and setpoints on", "every sampled state"]),
        ("reject", "caught by the gate", f"{dis} / {f['states']:,}",
         ["states where one network's FB", "picks another action while", "its Q-values pass"]),
        ("pass", "rejected as intended", f"{f['mutants_rejected']} / {f['mutants']}",
         ["injected defects rejected", "(5 kinds × 2 networks)"]),
    ]
    tw, ty, th = 1.9, 1.98, 1.42
    tg = (W - 4 * tw) / 3
    for i, (kind, label, big, lines) in enumerate(tiles):
        x = i * (tw + tg)
        card(ax, t, x, ty, tw, th)
        badge(ax, t, x + 0.2, ty + th - 0.2, kind)
        ax.text(x + 0.34, ty + th - 0.2, label, fontsize=8.3, fontweight="bold", va="center", color=t["ink"])
        ax.text(x + 0.12, ty + th - 0.6, big, fontsize=19, fontweight="bold", va="center", color=t["ink"])
        for j, line in enumerate(lines):
            ax.text(x + 0.12, ty + th - 0.93 - 0.15 * j, line, fontsize=8.1, va="center", color=t["ink2"])

    # the other artefact, the vendor compiler and the controller, each in its own tile
    by, bh2 = 0.6, 1.28
    bw3 = (W - 2 * tg) / 3
    lower = [
        ("pass" if f["st_bit_identical"] else "reject", "passed: .st (IEC 61131-3)", f"{f['st_runs']} / {f['n']}",
         ["network run (the server run):", "Q-values bit-identical to its",
          f".fbt; the other {f['n'] - f['st_runs']}: not run"], False),
        ("pass" if f["eae_accepted"] == f["eae_runs"] else "reject", "accepted: EAE type compiler",
         f"{f['eae_accepted']} / {f['eae_runs']}",
         [f".fbt exports (EAE {f['eae_version']}): no", "unfiltered errors or warnings;",
          f"{f['neg_rejected']} / {f['neg_runs']} broken variants rejected"], False),
        ("notverified", "on a controller", "not verified",
         ["no import into an EAE solution,", "no deployment, no cycle time"], True),
    ]
    for i, (kind, label, big, lines, dashed) in enumerate(lower):
        x = i * (bw3 + tg)
        card(ax, t, x, by, bw3, bh2, dashed=dashed)
        badge(ax, t, x + 0.2, by + bh2 - 0.2, kind)
        ax.text(x + 0.34, by + bh2 - 0.2, label, fontsize=8.3, fontweight="bold", va="center")
        ax.text(x + 0.12, by + bh2 - 0.56, big, fontsize=17, fontweight="bold", va="center",
                color=t["muted"] if dashed else t["ink"])
        for j, line in enumerate(lines):
            ax.text(x + 0.12, by + bh2 - 0.84 - 0.14 * j, line, fontsize=8.0, va="center", color=t["ink2"])
    ax.text(0, 0.42, f"Sampled states: {f['n_rollout']:,} from simulator rollouts + {f['n_box']:,} uniform in [−2, 2]¹². "
            "Agreement on sampled states is not proof for every input,\nand acceptance by the type compiler is not "
            "evidence of correct execution on a controller. Measured 2026-11-08 on frozen weights.",
            fontsize=8.0, va="top", color=t["ink2"], linespacing=1.4)
    rows = [("networks with every Q-value within q_tolerance", f"{f['n_q_ok']} of {f['n']}", "networks[].checks.q_values"),
            ("largest |dQ| over all networks and states", f"{f['worst_dq']:.4e}", "networks[].max_abs_q_diff"),
            ("networks passing the strict gate (tie tolerance 0)", f"{f['n_ok']} of {f['n']}", "networks[].ok"),
            ("states where the rejected network's action differs", f"{dis} of {f['states']}",
             "networks[speed_proportional_seed5].argmax_disagreements"),
            ("injected defects rejected", f"{f['mutants_rejected']} of {f['mutants']}", "mutations[].ok"),
            ("networks whose .st was run in the interpreter", f"{f['st_runs']} of {f['n']}", "st_check.network"),
            (".st Q-values and outputs identical to the .fbt", str(f["st_bit_identical"]),
             "st_check.q_bit_identical_to_fbt, .outputs_identical_to_fbt"),
            ("exports accepted by the EAE type compiler", f"{f['eae_accepted']} of {f['eae_runs']}", "eae.accepted[].ok"),
            ("unfiltered compiler errors / warnings", f"{f['eae_errors']} / {f['eae_warnings']}",
             "eae.accepted[].errors, .warnings"),
            ("known standalone-compile messages per export (kept raw)", "/".join(map(str, f["eae_known"])),
             "eae.accepted[].known_standalone_messages"),
            ("broken variants rejected by the compiler check", f"{f['neg_rejected']} of {f['neg_runs']}",
             "eae.negative_controls[].ok"),
            ("execution on a controller, cycle time", "not verified", "")]
    write_csv("fig1_export_verification", ["quantity", "value", "source in docs/data/export_verification.json"], rows)
    save(fig, "fig1_export_verification", mode)


# ---------------------------------------------------------------------------
# Figure 2: policy evaluation (historical results)
# ---------------------------------------------------------------------------
def fig2(t, mode, ev):
    fig = plt.figure(figsize=(WIDTH, 5.4))
    fig.text(0.0, 0.985, "No trained DQN beat the best fixed action; 3 of 12 beat uniform random", ha="left", va="top",
             fontsize=12, fontweight="bold", color=t["ink"])
    fig.text(0.0, 0.928, "Historical multi-seed evaluation (2026-11-07): mean reward per 200-step episode over "
             f"{ev['legacy']['episodes']} paired episodes\n(environment seeds {ev['legacy']['env_seeds'][0]}–"
             f"{ev['legacy']['env_seeds'][1]}). One dot per training seed; the bar is the 95% t-interval across the 6 "
             f"seeds.\nEpisode-level 95% CIs are ≤ {ev['episode_ci_max']:.1f} reward units, narrower than the markers.",
             ha="left", va="top", fontsize=9, color=t["ink2"], linespacing=1.4)
    panels = [("legacy", "Default simulator (legacy production model, as published)", 0.455),
              ("speed_proportional", "Corrected production model (a part takes target ÷ speed steps)", 0.05)]
    rows = ["Trained DQN\n(6 training seeds)", "Untrained network\n(one init., 2026-11-06)", "Uniform random\nactions",
            "Best fixed action\n(always Energy saving)"]
    out_rows = []
    for model, label, bottom in panels:
        d = ev[model]
        ax = fig.add_axes([0.235, bottom, 0.735, 0.285])
        ax.set_xlim(1590, 1850)
        ax.set_ylim(-1.05, 3.55)
        ax.set_yticks(range(4))
        ax.set_yticklabels(rows, fontsize=8.8, linespacing=1.15)
        ax.xaxis.set_major_formatter(thousands)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.text(1590, 3.5, label, fontsize=9.3, fontweight="bold", va="bottom", color=t["ink"])
        best, rnd = d["best"]["mean"], d["random"]["mean"]
        ax.axvline(best, color=t["orange"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.axvline(rnd, color=t["muted"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        backing = dict(boxstyle="square,pad=0.12", facecolor=t["page"], edgecolor="none", alpha=0.9)
        ax.plot([best], [3], marker="D", markersize=8, color=t["orange"], zorder=3)
        ax.text(best + 6, 3, f"{best:,.1f}", ha="left", va="center", fontsize=8.5, color=t["ink2"])
        ax.plot([rnd], [2], marker="D", markersize=8, color=t["muted"], zorder=3)
        ax.text(rnd + 5, 2.28, f"{rnd:,.1f}", ha="left", fontsize=8.5, color=t["ink2"])
        out_rows += [(model, "best fixed action (Energy saving)", "", f"{best:.2f}", f"{d['best']['ci95']:.3f}", "episodes"),
                     (model, "uniform random", "", f"{rnd:.2f}", f"{d['random']['ci95']:.3f}", "episodes")]
        if model == "legacy":
            u = ev["untrained_legacy"]
            ax.plot([u["mean"]], [1], marker="D", markersize=8, markerfacecolor="none", markeredgecolor=t["ink2"],
                    markeredgewidth=1.4, zorder=3)
            ax.text(u["mean"] - 5, 1.28, f"{u['mean']:,.1f}", ha="right", fontsize=8.5, color=t["ink2"])
            out_rows.append((model, "untrained network (seed-42 initialisation, 2026-11-06 run, same 100 episodes)", "",
                             f"{u['mean']:.2f}", f"{u['ci95']:.3f}", "episodes"))
        else:
            ax.text(1597, 1, "not evaluated", fontsize=8.5, color=t["ink2"], va="center", fontstyle="italic")
        clusters = {}
        for seed, mean, ci, action, share in d["seeds"]:
            clusters.setdefault((round(mean, 1), action), []).append(seed)
            out_rows.append((model, f"trained DQN, seed {seed}", action, f"{mean:.2f}", f"{ci:.3f}", "episodes"))
        last_right = -1e9  # right edge of the last label above the dots, in data units
        for (mean, action), seeds in sorted(clusters.items()):
            offs = [0.0] if len(seeds) == 1 else [(j - (len(seeds) - 1) / 2) * 0.17 for j in range(len(seeds))]
            for o in offs:
                ax.plot([mean], [o], marker="o", markersize=7.5, color=t["blue"], markeredgecolor="none", zorder=4)
            text = f"{action} ×{len(seeds)}"
            half = len(text) * 1.5  # ~ half the label width in reward units at 8.3 pt on this axis
            if mean - half > last_right + 3:
                last_right = mean + half
                ax.text(mean, 0.36, text, ha="center", va="bottom", fontsize=8.3, color=t["ink2"], bbox=backing, zorder=5)
            else:
                ax.text(mean + 6, 0, text, ha="left", va="center", fontsize=8.3, color=t["ink2"], bbox=backing, zorder=5)
        a = d["across"]
        lo, hi = a["mean"] - a["ci95_across_seeds"], a["mean"] + a["ci95_across_seeds"]
        ax.plot([lo, hi], [-0.45, -0.45], color=t["blue"], linewidth=2.0, solid_capstyle="butt", zorder=3)
        ax.plot([lo, lo], [-0.53, -0.37], color=t["blue"], linewidth=1.4)
        ax.plot([hi, hi], [-0.53, -0.37], color=t["blue"], linewidth=1.4)
        ax.plot([a["mean"]], [-0.45], marker="|", markersize=11, markeredgewidth=2.0, color=t["blue"])
        ax.text(a["mean"], -0.62, f"mean of 6 seeds {a['mean']:,.1f} ± {a['ci95_across_seeds']:.1f} (95% t-interval)",
                ha="center", va="top", fontsize=8.0, color=t["ink2"], bbox=backing, zorder=5)
        out_rows.append((model, "trained DQN, mean of 6 seeds", "", f"{a['mean']:.2f}", f"{a['ci95_across_seeds']:.3f}",
                         "training seeds (Student t, n=6)"))
        if model == "speed_proportional":
            ax.set_xlabel("Mean reward per episode (higher is better)", fontsize=9)
    write_csv("fig2_policy_evaluation", ["production_model", "policy", "chosen_action", "mean_reward", "ci95",
                                         "ci95_over"], out_rows)
    save(fig, "fig2_policy_evaluation", mode)


# ---------------------------------------------------------------------------
# Figure 3: the near-tie the strict gate rejected
# ---------------------------------------------------------------------------
def fig3(t, mode, case, f):
    W, H = WIDTH, 6.15
    fig = plt.figure(figsize=(W, H))
    ax0 = canvas(fig, W, H)
    names = case["action_names"]
    states = case["states"]
    first, second = states[0]["competing_actions"]  # PyTorch's pick, the generated code's pick
    assert all(s["competing_actions"] == [first, second] for s in states)
    gap = states[0]["reference_top2_gap"]
    ulps = [round(s["reference_top2_gap"] / s["float32_ulp_at_reference_max"]) for s in states]
    head(ax0, t, H, f"A {sci(gap)} near-tie changed the action and setpoints; the Q-values passed",
         f"Network {case['checkpoint']['path'].rsplit('/', 1)[-1][:-3]} at the {len(states)} of "
         f"{case['state_set']['n']:,} sampled states where the strict gate failed it; both are uniform\ndraws "
         "from [−2, 2]¹², not simulator rollouts. PyTorch's two best Q-values differ there by "
         f"{min(ulps)}–{max(ulps)} float32 ulps;\nin the generated code they round to an exact tie.")
    blue, orange = t["blue"], t["orange"]

    # (a) Q-values of the two competing actions
    L, B, w, h = 1.45, 2.95, 3.95, 1.9
    ax = fig.add_axes([L / W, B / H, w / W, h / H])
    ax.set_xlim(-0.42, 1.72)
    ax.set_ylim(-0.35, 4.05)
    ax.set_yticks([])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.set_xticklabels(["0", "+0.5", "+1.0", "+1.5"], fontsize=8.3)
    ax.set_xlabel("Q − PyTorch's Q(Aggressive) at that state, in units of 10⁻⁴ (the Q tolerance 10⁻³ is 10 units)",
                  fontsize=8.3)
    ax0.text(0, B + h + 0.3, "a   Q-values of the two competing actions", fontsize=9.6, fontweight="bold", va="bottom")
    to_y = lambda yd: B + (yd + 0.35) / 4.4 * h  # noqa: E731  data y -> canvas inches
    plotted = []
    for k, s in enumerate(states):
        y_ref, y_gen = 3.35 - 2.0 * k, 2.55 - 2.0 * k
        base = s["reference_q"][second]
        off = lambda q: (q - base) / 1e-4  # noqa: E731
        qf_r, ag_r = s["reference_q"][first], s["reference_q"][second]
        qf_g, ag_g = s["generated_q"][first], s["generated_q"][second]
        assert qf_g == ag_g  # the tie
        ax.plot([off(qf_r)], [y_ref], marker="o", markersize=8.5, color=blue, markeredgecolor="none", zorder=4)
        ax.plot([off(ag_r)], [y_ref], marker="D", markersize=7.5, color=orange, markeredgecolor="none", zorder=4)
        ax.plot([off(qf_g)], [y_gen], marker="o", markersize=12.5, markerfacecolor="none", markeredgecolor=blue,
                markeredgewidth=1.6, zorder=4)
        ax.plot([off(ag_g)], [y_gen], marker="D", markersize=6.5, color=orange, markeredgecolor="none", zorder=5)
        ax.text(off(qf_r), y_ref + 0.3, f"{qf_r:.6f}", ha="center", va="bottom", fontsize=8.0, color=t["ink2"])
        ax.text(off(ag_r), y_ref + 0.3, f"{ag_r:.6f}", ha="center", va="bottom", fontsize=8.0, color=t["ink2"])
        ax.text(off(qf_g) + 0.1, y_gen, f"both {qf_g:.6f}", ha="left", va="center", fontsize=8.0, color=t["ink2"])
        ax0.text(L - 0.1, to_y(y_ref), "PyTorch", ha="right", va="center", fontsize=8.5)
        ax0.text(L - 0.1, to_y(y_gen), "Generated FB", ha="right", va="center", fontsize=8.5)
        ax0.text(0, to_y(y_ref) + 0.21, f"State {s['index']:,}".replace(",", ""), fontsize=8.6, fontweight="bold",
                 va="center")
        xr = L + w + 0.14
        ax0.plot([xr], [to_y(y_ref)], marker="o", markersize=7, color=blue, markeredgecolor="none")
        ax0.text(xr + 0.12, to_y(y_ref), f"picks {names[first]}: higher by\n{sci(s['reference_top2_gap'], 2)} "
                 f"({ulps[k]} ulps of {sci(s['float32_ulp_at_reference_max'], 2)})", fontsize=8.0, va="center",
                 color=t["ink"], linespacing=1.3)
        ax0.plot([xr], [to_y(y_gen)], marker="D", markersize=6, color=orange, markeredgecolor="none")
        ax0.text(xr + 0.12, to_y(y_gen), f"picks {names[second]}: exact tie,\nthe first maximum wins", fontsize=8.0,
                 va="center", color=t["ink"], linespacing=1.3)
        plotted.append((s["index"], qf_r, ag_r, qf_g, ag_g))
    ax0.plot([0.06], [B + h + 0.12], marker="o", markersize=7, color=blue, markeredgecolor="none")
    ax0.text(0.16, B + h + 0.12, f"{names[first]} (action {first})", fontsize=8.2, va="center", color=t["ink2"])
    ax0.plot([1.75], [B + h + 0.12], marker="D", markersize=6, color=orange, markeredgecolor="none")
    ax0.text(1.85, B + h + 0.12, f"{names[second]} (action {second})", fontsize=8.2, va="center", color=t["ink2"])

    # (b) the setpoints that follow from the action
    ax0.text(0, 2.0, "b   Setpoints sent with the action", fontsize=9.6, fontweight="bold", va="bottom")
    ref_sp, gen_sp = states[0]["reference_setpoints"], states[0]["generated_setpoints"]
    assert all(s["reference_setpoints"] == ref_sp and s["generated_setpoints"] == gen_sp for s in states)
    L2, B2, w2, h2 = 1.45, 0.82, 2.65, 1.08
    bx = fig.add_axes([L2 / W, B2 / H, w2 / W, h2 / H])
    bx.set_xlim(0.88, 1.29)
    bx.set_ylim(-0.6, 2.6)
    bx.set_yticks([])
    bx.grid(axis="x")
    bx.set_axisbelow(True)
    bx.set_xticks([0.9, 1.0, 1.1, 1.2])
    bx.set_xticklabels(["0.9", "1.0", "1.1", "1.2"], fontsize=8.3)
    bx.set_xlabel("setpoint factor (× nominal), identical at both states", fontsize=8.3)
    for i, label in enumerate(("Speed", "Pressure", "Temperature")):
        y = 2 - i
        r, g = ref_sp[i], gen_sp[i]
        bx.plot([r, g], [y, y], color=t["ink2"], linewidth=1.1, zorder=2)
        bx.plot([r], [y], marker="o", markersize=8, color=blue, markeredgecolor="none", zorder=3)
        bx.plot([g], [y], marker="D", markersize=7, color=orange, markeredgecolor="none", zorder=3)
        bx.text(r - 0.012, y, f"{r:.2f}", ha="right", va="center", fontsize=8.0, color=t["ink2"])
        bx.text(g + 0.012, y, f"{g:.2f} ({(g / r - 1) * 100:+.0f}%)", ha="left", va="center", fontsize=8.0,
                color=t["ink2"])
        ax0.text(L2 - 0.1, B2 + (y + 0.6) / 3.2 * h2, label, ha="right", va="center", fontsize=8.5)

    # (c) the two verdicts, kept apart from the diagnostic
    x0, cw = 4.55, 3.45
    q_only = case["q_only_check"]
    strict = case["strict_gate"]["on_20000_states"]
    diag = case["diagnostic_only_tie_tolerance_recheck"]["on_20000_states"]
    ax0.text(x0, 2.0, "c   Verdicts", fontsize=9.6, fontweight="bold", va="bottom")
    for j, (kind, title, detail) in enumerate([
            ("pass", "Q-only check: would accept",
             f"max |ΔQ| {sci(q_only['max_abs_q_diff_over_20000'])} < 10⁻³ over all {strict['states']:,} states"),
            ("reject", "Strict gate (tie tolerance 0): rejected",
             f"action and setpoints differ on {strict['argmax_disagreements']} of {strict['states']:,} states")]):
        y = 1.42 - 0.62 * j
        card(ax0, t, x0, y, cw, 0.5)
        badge(ax0, t, x0 + 0.2, y + 0.33, kind)
        ax0.text(x0 + 0.34, y + 0.33, title, fontsize=8.5, fontweight="bold", va="center")
        ax0.text(x0 + 0.34, y + 0.13, detail, fontsize=8.0, va="center", color=t["ink2"])
    ax0.text(x0, 0.68, f"Diagnostic only, not a pass: with tie tolerance {sci(diag['tie_tolerance'], 0)}\nthe "
             f"{diag['near_tie_disagreements']} states are counted as near-ties; the generated\naction and setpoints "
             "still differ there.", fontsize=7.9, va="top", color=t["ink2"], linespacing=1.35)
    moves = max(r["max_abs_diff_vs_gate_reference"] for s in states for r in s["reference_by_batch_size"])
    ax0.text(0, 0.22, f"PyTorch's own Q-values for these states move by up to {sci(moves, 2)} with batch size (1, 2 or "
             "20,000 rows), and it picks Quality focus in all three.\nInputs, all six Q-values and both setpoint sets: "
             "docs/data/near_tie_states.csv and tests/fixtures/near_tie_speed_proportional_seed5.json.",
             fontsize=7.9, va="top", color=t["ink2"], linespacing=1.35)
    save(fig, "fig3_near_tie_rejection", mode)
    return plotted


# ---------------------------------------------------------------------------
# Figure 4: verification coverage
# ---------------------------------------------------------------------------
GROUPS4 = [("trained_snapshot", "Trained · server run (2026-11-06)"),
           ("legacy_seed", "Trained · default simulator (2026-11-07)"),
           ("speed_proportional_seed", "Trained · corrected model (2026-11-07)"),
           ("synthetic_seed", "Untrained · random initialisation")]
DEFECTS = [("no_latch", "one WITH input removed"), ("relu", "ReLU missing in a block"),
           ("row", "one weight row skipped"), ("setpoint", "wrong setpoint literal"), ("argmin", "argmax → argmin")]


def seed_key(nid):
    tail = nid.rsplit("seed", 1)[-1] if "seed" in nid else ""
    return (0, 0) if not tail else (0 if tail == "42" else 1, int(tail))


def fig4(t, mode, m, f):
    W, H = WIDTH, 6.45
    fig = plt.figure(figsize=(W, H))
    ax = canvas(fig, W, H)
    head(ax, t, H, f"Checks covered {f['n']} networks and all six actions; none ran on a controller",
         f"Measured 2026-11-08 on frozen weights. Each network: {f['states']:,} sampled states\n({f['n_rollout']:,} from "
         f"simulator rollouts + {f['n_box']:,} uniform in [−2, 2]¹²).")
    nets = {n["id"]: n for n in m["networks"]}
    st_net = (m.get("st_check") or {}).get("network")
    compiled = set((m.get("eae") or {}).get("accepted", {}))
    cols = dict(q=2.66, act=3.1, sp=3.54, st=3.98, eae=4.42)
    strip0, pitch, sq = 1.0, 0.13, 0.095
    top = H - 1.0
    ax.text(0, top + 0.08, "a   Exported networks and the checks run on each", fontsize=9.2, fontweight="bold", va="bottom")
    hy = top - 0.2
    for a in range(6):
        ax.text(strip0 + a * pitch + sq / 2, hy, str(a), ha="center", va="center", fontsize=7.6, color=t["ink2"])
    ax.text(strip0 + 3 * pitch, hy + 0.17, "actions chosen", ha="center", va="center", fontsize=7.8, color=t["ink2"])
    ax.text(2.33, hy + 0.08, "max |ΔQ|", ha="right", va="center", fontsize=7.8, color=t["ink2"])
    for key, lab in (("q", "Q-values"), ("act", "action"), ("sp", "setpoints"), ("st", ".st run"), ("eae", "EAE\ncompile")):
        ax.text(cols[key], hy + 0.08, lab, ha="center", va="center", fontsize=7.8, color=t["ink2"], linespacing=1.1)
    y = hy - 0.24
    rows_out = []
    for prefix, label in GROUPS4:
        members = sorted((n for n in nets if n == prefix or (prefix.endswith("seed") and n.startswith(prefix))),
                         key=seed_key)
        ax.text(0, y, label, fontsize=8.3, fontweight="bold", va="center")
        y -= 0.19
        for nid in members:
            n = nets[nid]
            synth = nid.startswith("synthetic")
            color = t["aqua"] if synth else t["blue"]
            ax.text(0.12, y, "server run" if nid == "trained_snapshot" else f"seed {nid.rsplit('seed', 1)[-1]}",
                    fontsize=8.1, va="center", color=t["ink2"])
            for a, share in enumerate(n["network_action_share"]):
                ax.add_patch(Rectangle((strip0 + a * pitch, y - sq / 2), sq, sq, linewidth=0.8,
                                       facecolor=color if share > 0 else "none",
                                       edgecolor=color if share > 0 else t["axis"]))
            ax.text(2.33, y, sci(n["max_abs_q_diff"]), ha="right", va="center", fontsize=7.9, color=t["ink2"])
            marks = {"q": "pass" if n["checks"]["q_values"] else "reject",
                     "act": "pass" if n["checks"]["argmax"] else "reject",
                     "sp": "pass" if n["checks"]["setpoints"] else "reject",
                     "st": "pass" if nid == st_net and f["st_identical"] else "notrun",
                     "eae": ("pass" if m["eae"]["accepted"][nid]["ok"] else "reject") if nid in compiled else "notrun"}
            for key, kind in marks.items():
                badge(ax, t, cols[key], y, kind, r=0.06)
            rows_out.append(("network", nid, label, " ".join(str(a) for a, s in enumerate(n["network_action_share"]) if s > 0),
                             f"{n['max_abs_q_diff']:.4e}", *marks.values()))
            y -= 0.163
        y -= 0.05
    used = sorted({a for n in nets.values() for a, s in enumerate(n["network_action_share"]) if s > 0})
    ax.plot([0, 4.62], [y + 0.1, y + 0.1], color=t["edge"], linewidth=0.8)
    ax.text(0.12, y - 0.04, f"all {f['n']}", fontsize=8.1, fontweight="bold", va="center")
    ax.text(strip0 + 3 * pitch, y - 0.04, f"{len(used)} of 6 used", ha="center", va="center", fontsize=8.0,
            fontweight="bold")
    n_act = sum(n["checks"]["argmax"] for n in nets.values())
    n_sp = sum(n["checks"]["setpoints"] for n in nets.values())
    for key, txt in (("q", f"{f['n_q_ok']}/{f['n']}"), ("act", f"{n_act}/{f['n']}"), ("sp", f"{n_sp}/{f['n']}"),
                     ("st", f"{int(bool(st_net))} run"), ("eae", f"{len(compiled)} run")):
        ax.text(cols[key], y - 0.04, txt, ha="center", va="center", fontsize=8.0, fontweight="bold")
    ax.text(0, y - 0.3, "Actions: 0 Conservative · 1 Normal · 2 Aggressive · 3 Energy saving · 4 Quality focus · 5 Balanced",
            fontsize=7.8, va="center", color=t["ink2"])

    # (b) injected defects
    x0 = 5.0
    every = f["mutants_rejected"] == f["mutants"]
    verdict = f"all {f['mutants']} rejected" if every else f"{f['mutants_rejected']} of {f['mutants']} rejected"
    ax.text(x0, top + 0.08, f"b   Injected defects: {verdict}", fontsize=9.2, fontweight="bold", va="bottom")
    targets = [("trained_snapshot", "trained", [6.6, 6.84, 7.08], "o", t["blue"]),
               ("synthetic_seed0", "untrained", [7.36, 7.6, 7.84], "s", t["aqua"])]
    for _, lab, xs, _, _ in targets:
        ax.text(xs[1], hy + 0.1, lab, ha="center", va="center", fontsize=7.8, color=t["ink2"])
        for xx, c in zip(xs, ("Q", "A", "S")):
            ax.text(xx, hy - 0.06, c, ha="center", va="center", fontsize=7.8, color=t["ink2"])
    for ri, (mid, lab) in enumerate(DEFECTS):
        yy = hy - 0.3 - 0.2 * ri
        ax.text(6.42, yy, lab, ha="right", va="center", fontsize=8.1)
        for nid, _, xs, marker, color in targets:
            res = next(x for x in m["mutations"] if x["id"] == mid and x["network"] == nid)
            for xx, ck in zip(xs, ("q_values", "argmax", "setpoints")):
                failed = not res["checks"][ck]
                ax.plot([xx], [yy], marker=marker, markersize=7, markerfacecolor=color if failed else "none",
                        markeredgecolor=color if failed else t["muted"], markeredgewidth=1.1)
            rows_out.append(("defect", f"{nid}:{mid}", lab, "", f"{res.get('max_abs_q_diff', float('nan')):.4e}",
                             *("fails" if not res["checks"][c] else "passes" for c in ("q_values", "argmax", "setpoints")),
                             "rejected" if not res["ok"] else "ACCEPTED"))
    yl = hy - 0.3 - 0.2 * len(DEFECTS) - 0.04
    ax.text(x0, yl, "Trained: server run. Untrained: seed 0.\nQ, A, S: Q-values, action, setpoints.\nFilled: the check "
            "fails on the defect;\nhollow: the defect is invisible to that check.", fontsize=7.6, va="top",
            color=t["ink2"], linespacing=1.3)

    # (c) compiler negative controls
    yc = yl - 0.8
    ax.text(x0, yc, f"c   Compiler faults: {f['neg_rejected']} of {f['neg_runs']} rejected", fontsize=9.2,
            fontweight="bold", va="bottom")
    for k, ctl in enumerate(m["eae"]["negative_controls"]):
        yy = yc - 0.17 - 0.33 * k
        code = (ctl["error_codes"] or ctl["warning_codes"])[0]
        how = "" if ctl["error_codes"] else ": a warning fails it"
        ax.text(x0, yy, ctl["defect"], fontsize=8.1, va="center")
        ax.text(x0, yy - 0.14, f"{code}, exit {ctl['exit_code']}{how}", fontsize=7.7, va="center", color=t["ink2"])
        rows_out.append(("compiler_negative_control", ctl["defect"], code, "", "", "", "", "", "",
                         "rejected" if not ctl["ok"] else "ACCEPTED"))

    # (d) not verified
    yd = yc - 0.17 - 0.33 * len(m["eae"]["negative_controls"]) - 0.12
    card(ax, t, x0, yd - 0.7, W - x0, 0.78, dashed=True)
    badge(ax, t, x0 + 0.17, yd - 0.06, "notverified")
    ax.text(x0 + 0.31, yd - 0.06, "Not verified", fontsize=8.6, fontweight="bold", va="center")
    for k, line in enumerate(["import into an EAE solution, build, deployment", "execution and cycle time on a controller",
                              "target float arithmetic (binary32 assumed)"]):
        ax.text(x0 + 0.12, yd - 0.26 - 0.15 * k, line, fontsize=7.9, va="center", color=t["ink2"])

    # legend: shape + label for every status
    ly = 0.16
    items = [("pass", "passed"), ("reject", "failed: export rejected"), ("notrun", "not run")]
    x = 0
    for kind, lab in items:
        badge(ax, t, x + 0.07, ly, kind, r=0.06)
        ax.text(x + 0.18, ly, lab, fontsize=7.9, va="center", color=t["ink2"])
        x += 0.32 + len(lab) * 0.058
    ax.plot([x + 0.07], [ly], marker="o", markersize=7, color=t["blue"], markeredgecolor="none")
    ax.text(x + 0.18, ly, "check fails on the defect", fontsize=7.9, va="center", color=t["ink2"])
    x += 0.32 + 25 * 0.058
    ax.plot([x + 0.07], [ly], marker="o", markersize=7, markerfacecolor="none", markeredgecolor=t["muted"])
    ax.text(x + 0.18, ly, "check passes it", fontsize=7.9, va="center", color=t["ink2"])
    write_csv("fig4_verification_coverage", ["kind", "id", "group_or_description", "actions_chosen", "max_abs_q_diff",
                                             "q_values", "action", "setpoints", "st_run", "eae_compile_or_verdict"],
              rows_out)
    save(fig, "fig4_verification_coverage", mode)


# ---------------------------------------------------------------------------
# Supporting figure: why the learned ranking is arbitrary
# ---------------------------------------------------------------------------
def supp_td(t, mode, ev):
    with open(LOSS_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    steps = [int(r["gradient_steps"]) for r in rows]
    rmse = [math.sqrt(float(r["loss"])) for r in rows]
    win = 51
    med = []
    for i in range(len(rmse)):
        w = sorted(rmse[max(0, i - win // 2): i + win // 2 + 1])
        med.append(w[len(w) // 2])
    gap = ev["hist_q_gap"]
    first = sorted(rmse[:50])[25]
    last = sorted(rmse[-50:])[25]
    fig = plt.figure(figsize=(WIDTH, 4.1))
    fig.text(0.0, 0.985, f"By the end of training the TD error is {last / gap:.0f}× the margin that decides the action",
             ha="left", va="top", fontsize=12, fontweight="bold", color=t["ink"])
    fig.text(0.0, 0.905, "Historical: one training run of the published server (2026-11-06, seed 42, default simulator), "
             "1,000 gradient steps, one per\n200-step episode. TD error = √ of the logged mean-squared TD loss of each "
             "batch of 64. Margin = best minus second-best\nQ-value of the final network, mean over the "
             f"{ev['hist_q_gap_states']:,} states its greedy policy visited in evaluation.",
             ha="left", va="top", fontsize=9, color=t["ink2"], linespacing=1.4)
    ax = fig.add_axes([0.075, 0.12, 0.9, 0.6])
    ax.plot(steps, rmse, color=t["muted"], linewidth=0.6, alpha=0.55, zorder=2)
    ax.plot(steps, med, color=t["blue"], linewidth=2.0, zorder=3)
    ax.axhline(gap, color=t["orange"], linewidth=1.3, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 180)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Gradient step (one per training episode)", fontsize=9)
    ax.set_ylabel("Q-value units", fontsize=9)
    backing = dict(boxstyle="square,pad=0.15", facecolor=t["page"], edgecolor="none", alpha=0.9)
    ax.text(985, 116, f"TD error (rolling median of 51): {last:.0f} over the last 50 steps", ha="right", va="bottom",
            fontsize=8.6, color=t["ink"], bbox=backing, zorder=5)
    ax.annotate(f"TD error at the start: {first:.1f}", xy=(8, first + 1), xytext=(40, 62), fontsize=8.6,
                color=t["ink2"], ha="left", va="bottom", bbox=backing, zorder=5,
                arrowprops=dict(arrowstyle="-", color=t["ink2"], linewidth=0.8, shrinkA=2, shrinkB=2))
    ax.text(985, gap + 3, f"margin between the two best actions: {gap:.1f}", ha="right", va="bottom", fontsize=8.6,
            color=t["ink"], bbox=backing, zorder=5)
    ax.text(330, 152, "grey: value of each batch", fontsize=8.3, color=t["ink2"], ha="left", bbox=backing, zorder=5)
    write_csv("supp_td_error_vs_margin", ["gradient_step", "td_loss_mse", "td_error_rmse", "rmse_rolling_median_51"],
              [(s, f"{float(r['loss']):.4f}", f"{x:.4f}", f"{mm:.4f}") for s, r, x, mm in zip(steps, rows, rmse, med)])
    save(fig, "supp_td_error_vs_margin", mode)


def main():
    measured = load_json(MEASURED)
    facts = measured_facts(measured)
    ev = load_eval()
    case = load_near_tie()
    for mode, t in THEMES.items():
        setup(t)
        fig1(t, mode, facts)
        fig2(t, mode, ev)
        fig3(t, mode, case, facts)
        fig4(t, mode, measured, facts)
        supp_td(t, mode, ev)
    print(json.dumps({k: v for k, v in facts.items() if k != "rejected"}, indent=1, default=str))
    print("rejected:", [(r["id"], r["argmax_disagreements"]) for r in facts["rejected"]])
    print("episode-level CI max:", round(ev["episode_ci_max"], 3))


if __name__ == "__main__":
    main()
