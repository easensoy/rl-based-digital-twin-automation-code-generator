"""
Builds the visual outputs from the instrumented training run and the policy
evaluation:

  visual_outputs/01_reward_curve.png
  visual_outputs/02_loss_curve.png
  visual_outputs/03_epsilon_decay.png
  visual_outputs/04_action_distribution.png
  visual_outputs/05_before_after_process_metrics.png
  visual_outputs/06_digital_twin_process_state.png
  visual_outputs/07_plc_code_structure.png
  visual_outputs/data/*.csv   (table view of every chart)

Usage (from the repo root):
    .venv\\Scripts\\python.exe visual_outputs\\scripts\\make_plots.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import FancyBboxPatch        # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

VO = Path(__file__).resolve().parents[1]
RUN = VO / "run"
DATA = VO / "data"
DATA.mkdir(exist_ok=True)

# ---- palette: dataviz reference instance (validated, light mode) ----------
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ACTIONS = ["0 Conservative", "1 Normal", "2 Aggressive", "3 Energy saving", "4 Quality focus", "5 Balanced"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.6,
    "figure.dpi": 100, "savefig.dpi": 150,
})


def suptitle(fig, title, subtitle=None):
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=INK)
    if subtitle:
        fig.text(0.012, 0.935, subtitle, ha="left", va="top", fontsize=9, color=INK2)


def footer(fig, text):
    fig.text(0.012, 0.012, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def save(fig, name):
    fig.savefig(VO / name, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
def load():
    eps = pd.read_json(RUN / "episodes.jsonl", lines=True)
    eps["loss"] = eps["losses"].apply(lambda l: l[0] if len(l) else np.nan)
    eps["eps_during"] = eps["epsilon"].shift(1).fillna(1.0)      # result['epsilon'] is post-replay
    eps["explore_share"] = eps["explore_steps"] / eps["steps"]
    traces = {}
    with open(RUN / "traces.jsonl", encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            traces[t["episode"]] = t
    ev = json.loads((RUN / "evaluation.json").read_text(encoding="utf-8"))
    meta = json.loads((RUN / "run_meta.json").read_text(encoding="utf-8"))
    return eps, traces, ev, meta


def fig_reward(eps, ev, meta):
    P = ev["policies"]
    fig, ax = plt.subplots(figsize=(10, 4.9))
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.07, right=0.80)
    ma = eps["total_reward"].rolling(50, min_periods=10).mean()
    ax.plot(eps["episode"], eps["total_reward"], color=CAT[0], alpha=0.28, lw=0.8)
    ax.plot(eps["episode"], ma, color=CAT[0], lw=2.0)
    xr = eps["episode"].max()
    refs = [("Uniform random", "Uniform random policy"),
            ("Trained DQN (greedy, eval mode)", "Trained DQN, greedy"),
            (max((k for k in P if k.startswith("Always")), key=lambda k: P[k]["total_reward"]["mean"]),
             "Best fixed action")]
    used = []
    for key, label in refs:
        v = P[key]["total_reward"]["mean"]
        ax.axhline(v, color=MUTED, lw=0.9)
        lab = f"{label}: {v:,.0f}" + (f"\n({key.split(': ')[1]})" if key.startswith("Always") else "")
        y = v
        for u in used:                                   # nudge labels that would collide
            if abs(y - u) < 6:
                y = u + 7 if y >= u else u - 7
        used.append(y)
        ax.text(xr + 8, y, lab, color=INK2, fontsize=8, va="center")
    ax.set_xlim(0, xr)
    ax.set_xlabel("Training episode (200 steps each)")
    ax.set_ylabel("Total reward per episode")
    suptitle(fig, "Episode reward during DQN training on the default scenario",
             f"1,000 episodes, ε-greedy behaviour policy, seed {meta['seed']}. Light line: each episode; dark line: "
             f"50-episode mean. Grey lines: mean of {ev['episodes_per_policy']} evaluation episodes per policy.\n"
             "Flat plateaus in the light line are stretches where the greedy policy sits on one fixed action "
             "(1,622 Aggressive, 1,683 Quality focus, 1,716 Normal, 1,744 Balanced, 1,772 Conservative).")
    footer(fig, "Source: backend WMGDigitalTwinServer.run_training_episode via POST /api/training/start; "
                "evaluation: visual_outputs/scripts/evaluate_policies.py")
    eps[["episode", "total_reward", "steps", "epsilon", "throughput", "energy_efficiency", "oee",
         "production_count"]].to_csv(DATA / "training_episodes.csv", index=False)
    save(fig, "01_reward_curve.png")


def fig_loss(eps, meta):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.08, right=0.97)
    m = eps["loss"].notna()
    ax.plot(eps.loc[m, "gradient_steps"], eps.loc[m, "loss"], color=CAT[1], alpha=0.30, lw=0.8)
    ax.plot(eps.loc[m, "gradient_steps"], eps.loc[m, "loss"].rolling(25, min_periods=5).median(),
            color=CAT[1], lw=2.0)
    ax.set_yscale("log")
    ax.set_xlim(0, eps["gradient_steps"].max())
    ax.set_xlabel("Gradient step (DQNAgent.replay() is called once per episode)")
    ax.set_ylabel("TD loss, MSE (log scale)")
    last = eps.loc[m, "loss"].rolling(25, min_periods=5).median().iloc[-1]
    ax.text(eps["gradient_steps"].max(), last, f"25-step median\n{last:,.1f}  ", color=INK2,
            fontsize=8.5, ha="right", va="bottom")
    suptitle(fig, "DQN temporal-difference loss",
             f"{int(m.sum())} gradient steps in total (batch 64, Adam lr 1e-3, target network synced every 10 steps). "
             "Loss is computed by the backend but never logged or exposed; captured here by observation.")
    footer(fig, "Source: value returned by torch.nn.functional.mse_loss inside DQNAgent.replay(), recorded by "
                "visual_outputs/scripts/run_instrumented_server.py")
    eps.loc[m, ["episode", "gradient_steps", "loss"]].to_csv(DATA / "training_loss.csv", index=False)
    save(fig, "02_loss_curve.png")


def fig_epsilon(eps):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.07, right=0.97)
    ax.scatter(eps["episode"], eps["explore_share"], s=7, color=CAT[1], alpha=0.45, lw=0,
               label="Measured share of random actions in the episode")
    ax.plot(eps["episode"], eps["eps_during"], color=CAT[0], lw=2.0, label="ε in force during the episode")
    floor_ep = int(eps.loc[eps["eps_during"] <= 0.0101, "episode"].min())
    ax.axvline(floor_ep, color=MUTED, lw=0.9)
    ax.text(floor_ep - 10, 0.30, f"ε reaches its floor (0.01)\nat episode {floor_ep}", color=INK2,
            fontsize=8.5, ha="right")
    overall = eps["explore_steps"].sum() / eps["steps"].sum()
    ax.text(0.99, 0.95, f"Random actions over the whole run: {overall:.1%}", transform=ax.transAxes,
            ha="right", va="top", color=INK, fontsize=9.5)
    ax.set_xlim(0, eps["episode"].max())
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Probability / share")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.89))
    suptitle(fig, "Exploration schedule: ε decay",
             "ε starts at 1.0 and is multiplied by 0.995 once per gradient step. Because replay() runs once per "
             "episode, ε decays per episode, not per environment step.")
    footer(fig, "Source: result['epsilon'] from run_training_episode; random-action share observed on DQNAgent.act()")
    eps[["episode", "eps_during", "explore_steps", "steps", "explore_share"]].to_csv(
        DATA / "epsilon_schedule.csv", index=False)
    save(fig, "03_epsilon_decay.png")


def _stack(ax, x, shares, title):
    ax.stackplot(x, shares.T, colors=CAT[:6], edgecolor=SURF, linewidth=0.6)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title)
    ax.grid(False)


def fig_actions(eps, ev):
    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.55, wspace=0.35,
                          top=0.86, bottom=0.08, left=0.07, right=0.98)
    ax1 = fig.add_subplot(gs[0, :2])
    win = 50
    allc = np.vstack(eps["action_counts"].to_list()).astype(float)
    grc = np.vstack(eps["greedy_action_counts"].to_list()).astype(float)
    roll_all = pd.DataFrame(allc).rolling(win, min_periods=1).sum().to_numpy()
    roll_gr = pd.DataFrame(grc).rolling(win, min_periods=1).sum().to_numpy()
    g_tot = roll_gr.sum(axis=1, keepdims=True)
    share_gr = np.divide(roll_gr, g_tot, out=np.full_like(roll_gr, np.nan), where=g_tot > 0)
    x = eps["episode"].to_numpy()
    ok = g_tot[:, 0] >= 200          # only show once >= 200 greedy decisions are in the window
    _stack(ax1, x[ok], share_gr[ok], f"Greedy (exploit) decisions only, rolling {win}-episode window "
                                     f"(shown from episode {x[ok].min()}, once it holds 200 decisions)")
    ax1.set_xlabel("Training episode")
    ax1.set_ylabel("Share of greedy actions")
    ax1.set_xlim(0, x.max())
    # legend in the free cell, in slot order
    axl = fig.add_subplot(gs[0, 2])
    axl.axis("off")
    for i, name in enumerate(ACTIONS):
        axl.add_patch(FancyBboxPatch((0.02, 0.86 - i * 0.14), 0.07, 0.08, boxstyle="round,pad=0,rounding_size=0.015",
                                     color=CAT[i], transform=axl.transAxes))
        axl.text(0.13, 0.90 - i * 0.14, name, transform=axl.transAxes, va="center", color=INK, fontsize=9)
    axl.text(0.02, 1.0, "Action (AutomationEnvironment._apply_action)", transform=axl.transAxes,
             color=INK2, fontsize=8.5, va="bottom")

    fid = ev["plc_fidelity"]
    P = ev["policies"]
    panels = [("Trained DQN, greedy", np.array(fid["dqn_eval_action_share"])),
              ("Untrained DQN, greedy", np.array(P["Untrained DQN (greedy)"]["action_share"])),
              ("Generated ST (FB_RLOptimizer as wired)", np.array(fid["st_as_wired_action_share"]))]
    rows = []
    for j, (title, share) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j])
        y = np.arange(6)[::-1]
        ax.barh(y, share, color=CAT[:6], height=0.62)
        for yy, s in zip(y, share):
            ax.text(s + 0.02, yy, f"{s:.0%}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(ACTIONS if j == 0 else [""] * 6)
        ax.set_xlim(0, 1.15)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False)
        for a, s in enumerate(share):
            rows.append({"policy": title, "action": ACTIONS[a], "share": s})
    suptitle(fig, "Action distribution: what the agent learned, and what the generated PLC code does",
             f"Bottom row: share of actions over {fid['n_states']:,} states visited by the trained greedy policy "
             f"(evaluation). Generated ST agrees with the trained network on "
             f"{fid['agreement_st_as_wired_vs_dqn_eval']:.1%} of these states.")
    footer(fig, "Source: DQNAgent.act() observations during training; evaluate_policies.py (line-for-line "
                "emulation of the generated FB_RLOptimizer and main-program wiring)")
    pd.DataFrame(rows).to_csv(DATA / "action_distribution_final.csv", index=False)
    pd.DataFrame(share_gr, columns=ACTIONS).assign(episode=x).to_csv(DATA / "greedy_action_share_rolling.csv",
                                                                     index=False)
    save(fig, "04_action_distribution.png")


def fig_before_after(ev):
    P = ev["policies"]
    best_fixed = max((k for k in P if k.startswith("Always")), key=lambda k: P[k]["total_reward"]["mean"])
    order = [("Uniform random", "Uniform random"),
             ("Untrained DQN (greedy)", "Before: untrained DQN"),
             ("Trained DQN (greedy, as backend runs it)", "After: trained DQN (dropout on)"),
             ("Trained DQN (greedy, eval mode)", "After: trained DQN (eval mode)"),
             ("Generated ST, as wired in main program", "Generated ST code"),
             (best_fixed, f"Best fixed action ({best_fixed.split(': ')[1]})")]
    metrics = [("total_reward", "Total reward per episode\n(the optimised quantity)", "{:,.0f}"),
               ("energy_efficiency", "Energy efficiency\n1 / (1 + mean energy per step)", "{:.3f}"),
               ("quality_mean", "Mean quality score\n(state[10] + 1) / 2", "{:.4f}"),
               ("throughput", "Throughput\n(parts per 100 steps)", "{:.2f}")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.9), sharey=True)
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.21, right=0.985, wspace=0.12)
    y = np.arange(len(order))[::-1]
    rows = []
    for ax, (key, label, fmt) in zip(axes, metrics):
        for yy, (pk, pl) in zip(y, order):
            m, ci = P[pk][key]["mean"], P[pk][key]["ci95"]
            color = CAT[0] if pl.startswith("After") else (CAT[1] if pl.startswith("Before") else MUTED)
            ax.errorbar(m, yy, xerr=ci, fmt="o", ms=6.5, color=color, ecolor=color, elinewidth=1.4, capsize=0)
            rows.append({"policy": pl, "metric": key, "mean": m, "ci95": ci})
        vals = [P[pk][key]["mean"] for pk, _ in order]
        span = max(vals) - min(vals) or 1
        ax.set_xlim(min(vals) - 0.35 * span, max(vals) + 0.35 * span)
        for yy, (pk, _) in zip(y, order):
            ax.text(P[pk][key]["mean"], yy + 0.28, fmt.format(P[pk][key]["mean"]), ha="center",
                    fontsize=7.5, color=INK2)
        ax.set_title(label, fontsize=9.2)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(order) - 0.4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([pl for _, pl in order])
    tr, ut = P["Trained DQN (greedy, eval mode)"]["total_reward"], P["Untrained DQN (greedy)"]["total_reward"]
    suptitle(fig, "Before / after: process metrics under each policy on the backend's digital twin",
             f"Mean ± 95% CI over {ev['episodes_per_policy']} evaluation episodes per policy on identical "
             f"environment seeds (most CIs are narrower than the dot). Reward change, untrained → trained: "
             f"{tr['mean'] - ut['mean']:+,.1f} ({(tr['mean'] - ut['mean']) / ut['mean']:+.2%}).")
    footer(fig, "Source: visual_outputs/scripts/evaluate_policies.py; full per-episode table in "
                "run/evaluation_episodes.csv. Blue = after training, orange = before, grey = references.")
    pd.DataFrame(rows).to_csv(DATA / "before_after_metrics.csv", index=False)
    save(fig, "05_before_after_process_metrics.png")


def fig_twin(traces):
    early, late = min(traces), max(traces)
    te, tl = traces[early], traces[late]
    def arr(t, f):
        return np.array([f(s) for s in t["steps"]])
    panels = [("Temperature, state[6]", lambda s: s["state"][6]),
              ("Pressure, state[7]", lambda s: s["state"][7]),
              ("Vibration, state[8]", lambda s: s["state"][8]),
              ("Load, state[9]", lambda s: s["state"][9]),
              ("Quality, state[10]", lambda s: s["state"][10]),
              ("Energy per step, state[11]", lambda s: s["state"][11]),
              ("Cumulative parts produced", lambda s: s["production_count"]),
              ("Reward per step", lambda s: s["reward"]),
              ("Action chosen", lambda s: s["action"])]
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.4), sharex=True)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.06, right=0.98, hspace=0.45, wspace=0.22)
    lab_e = f"Episode {early} (ε = {1.0 if early == 0 else traces[early]['epsilon_after']:.2f}, random)"
    lab_l = f"Episode {late} (ε = {tl['epsilon_after']:.3f}, near-greedy)"
    rows = []
    for ax, (title, f) in zip(axes.ravel(), panels):
        xe, xl = np.arange(1, len(te["steps"]) + 1), np.arange(1, len(tl["steps"]) + 1)
        ye, yl = arr(te, f), arr(tl, f)
        if title == "Action chosen":
            ax.scatter(xe, ye + 0.12, s=5, color=CAT[1], lw=0)
            ax.scatter(xl, yl - 0.12, s=5, color=CAT[0], lw=0)
            ax.set_yticks(range(6))
            ax.set_yticklabels([a.split(" ", 1)[1] for a in ACTIONS], fontsize=7.5)
        else:
            ax.plot(xe, ye, color=CAT[1], lw=1.2)
            ax.plot(xl, yl, color=CAT[0], lw=1.2)
        ax.set_title(title, fontsize=9.2)
        if title.startswith("Cumulative"):
            ax.text(xe[-1], ye[-1], f"{int(ye[-1])}", color=INK2, fontsize=8, ha="right", va="bottom")
            ax.text(xl[-1], yl[-1], f"{int(yl[-1])}", color=INK2, fontsize=8, ha="right", va="top")
    for ax in axes[-1]:
        ax.set_xlabel("Environment step")
    for k, t in ((early, te), (late, tl)):
        for i, s in enumerate(t["steps"]):
            rows.append({"episode": k, "step": i + 1, "action": s["action"], "reward": s["reward"],
                         "production_count": s["production_count"],
                         **{f"state_{j}": v for j, v in enumerate(s["state"])}})
    handles = [plt.Line2D([], [], color=CAT[1], lw=2), plt.Line2D([], [], color=CAT[0], lw=2)]
    fig.legend(handles, [lab_e, lab_l], loc="upper left", ncol=2, bbox_to_anchor=(0.004, 0.925))
    suptitle(fig, "Digital twin view during a run: AutomationEnvironment process state",
             "12-variable synthetic process in backend/rl_agent.py. state[0:6] (positions, velocities) omitted: "
             "a random walk that never enters the reward.")
    footer(fig, "Source: AutomationEnvironment.step() observed during training (run/traces.jsonl)")
    pd.DataFrame(rows).to_csv(DATA / "digital_twin_traces.csv", index=False)
    save(fig, "06_digital_twin_process_state.png")


def fig_plc_structure(code_path: Path):
    """Block diagram of the generated .st file (structure as emitted)."""
    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, title, lines, fc="#ffffff", ec=BASE, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + 1.0, y + h - 1.4, title, fontsize=9.5, fontweight="semibold", color=tc, va="top")
        for i, ln in enumerate(lines):
            ax.text(x + 1.0, y + h - 4.2 - i * 2.35, ln, fontsize=7.8, color=INK2, va="top", family="Consolas")

    # outer PROGRAM frame
    ax.add_patch(FancyBboxPatch((1, 1), 98, 53, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc="#f4f3ef", ec=BASE, lw=1.2))
    ax.text(2.2, 52.6, "PROGRAM WMG_Optimized_Control   (single .st file, emitted by PLCCodeGenerator.generate_code)",
            fontsize=10, fontweight="semibold", color=INK, va="top")
    box(3, 38.5, 28, 11.5, "① Header + VAR … END_VAR",
        ["(* Training Steps: n *)  ← run", "epsilon : REAL := ε    ← run",
         "+ 14 fixed variables/setpoints"], fc="#eef4fc", ec=CAT[0])
    box(3, 21.5, 28, 14.5, "② FUNCTION_BLOCK FB_SafetyMonitor",
        ["IN : temperature, pressure,", "     vibration, emergency_button", "temp > 80.0, pressure > 10.0,",
         "vibration > 5.0  (fixed literals)", "OUT: safety_ok, shutdown_required"])
    box(35, 21.5, 28, 28.5, "③ FUNCTION_BLOCK FB_RLOptimizer",
        ["IN : current_state : ARRAY[0..11]", "state_normalized[i] := x[i] / 2.0", "q_values[k] := fixed linear",
         "   terms of state[10], state[11]", "   (hard-coded coefficients,", "    no network weights)",
         "best_action := argmax (FOR loop)", "CASE → speed / pressure / temp", "   recommendations",
         "OUT: optimal_action, *_recommendation"], fc="#fdf0ea", ec=CAT[1])
    box(67, 21.5, 29.5, 28.5, "④ FUNCTION_BLOCK FB_ProcessControl",
        ["IN : start/stop_command,", "     speed/pressure/temp_factor", "motor_speed  := speed_factor * 100",
         "valve_position := pressure_f * 90", "heater_power := temp_factor * 75",
         "cycle_timer(IN, PT := T#30.0s)", "  (instance declared, Q unused)", "OUT: motor_speed, valve_position,",
         "     heater_power, process_active"])
    box(3, 3, 93.5, 15.5, "⑤ Second VAR block + main body (same PROGRAM)",
        ["safety_monitor(...);  system_state[6..8] := temp/100, pressure/10, vibration/5;  "
         "(indices 10, 11 never written)",
         "rl_optimizer(current_state := system_state, ...);   IF safety_monitor.safety_ok THEN process_control(...) "
         "ELSE stop;",
         "oee_value := ...;   IF process_active THEN production_count := production_count + 1;  (counts PLC scans)",
         "Sensor reads are commented out: (* sensor_temp := Analog_Input_1; *)  → no I/O mapping, no TASK/CONFIGURATION",
         "END_PROGRAM    (header ① opens the PROGRAM; ②③④ are declared inside it, which IEC 61131-3 does not allow)"])
    arrow = dict(arrowstyle="-|>", color=INK2, lw=1.1, mutation_scale=12)
    ax.annotate("", xy=(67, 35.5), xytext=(63, 35.5), arrowprops=arrow)
    ax.text(63.2, 36.3, "setpoints", fontsize=7.5, color=MUTED)
    ax.annotate("", xy=(17, 21.3), xytext=(17, 18.7), arrowprops=arrow)
    ax.annotate("", xy=(49, 21.3), xytext=(49, 18.7), arrowprops=arrow)
    ax.annotate("", xy=(81, 21.3), xytext=(81, 18.7), arrowprops=arrow)
    ax.text(18, 19.2, "called from main body", fontsize=7.5, color=MUTED)
    ax.text(82, 19.2, "called every scan; forced stop if NOT safety_ok", fontsize=7.5, color=MUTED)
    ax.text(0.01, 1.035, "Generated PLC code: structure of " + code_path.name, transform=ax.transAxes,
            fontsize=12.5, fontweight="semibold", color=INK)
    ax.text(0.01, 1.005, "IEC 61131-3 Structured Text (not IEC 61499). Blue: the only part the training run "
            "changes (two header values). Orange: the block named after RL, which holds fixed coefficients "
            "and no network weights.", transform=ax.transAxes, fontsize=9, color=INK2)
    save(fig, "07_plc_code_structure.png")


def main():
    eps, traces, ev, meta = load()
    fig_reward(eps, ev, meta)
    fig_loss(eps, meta)
    fig_epsilon(eps)
    fig_actions(eps, ev)
    fig_before_after(ev)
    fig_twin(traces)
    codes = sorted((VO / "plc_code").glob("*.st"))
    if codes:
        fig_plc_structure(codes[-1])


if __name__ == "__main__":
    main()
