"""
Output F: numerical decision-boundary audit, extending the near-tie analysis of docs/EVIDENCE.md.

Networks and states: the 18 networks and the 20,000 corpus-A states of docs/scripts/measure_export_verification.py
(10,000 rollout states of the legacy simulator under uniform random actions; 10,000 synthetic states drawn
uniformly in [-2, 2]^12), each network exported with the pinned date (the .fbt hash must equal the one in
docs/data/export_verification.json). Per network and state:
  q64   float64 forward pass of the weights (tests/st_corpus.reference_q); the margin is its gap between the
        largest and the second-largest Q-value
  q32   PyTorch DQNNetwork in eval mode, float32, all states in one batch (the reference of the strict gate)
  st    the .st FUNCTION_BLOCK in backend/st_interpreter.py (binary32)
  fbt   the .fbt basic FB in the same interpreter (INIT, then REQ)
A disagreement is a different first-maximum action. Results are stratified by margin decade and by source.

Reachability: each network's greedy PyTorch policy is rolled out on the 100 development episodes of the v0 audit
(seeds 10000-10099; its own production model, legacy for the snapshot and the untrained networks) and the .st
is evaluated on every visited state. If the .st and PyTorch choose the same action at every visited state, the
closed loop with the exported block reproduces the PyTorch trajectories exactly (same seed, same action at
every step), so no closed-loop simulation is needed; otherwise the affected episodes are re-run with the .st
in the loop and their rewards compared.

tie_tolerance = 0 throughout; there is no tolerant re-check. Writes experiments/v1/results/boundary_audit.json
and experiments/v1/results/boundary_audit_disagreements.csv.
Run:  .venv\\Scripts\\python.exe experiments\\v1\\boundary_audit.py
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import random
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tests")]
warnings.simplefilter("ignore", UserWarning)

import st_corpus  # noqa: E402
from backend import iec61499_exporter as ex  # noqa: E402
from backend import st_interpreter as sti  # noqa: E402
from backend.rl_agent import AutomationEnvironment  # noqa: E402

spec = importlib.util.spec_from_file_location("mev", REPO / "docs" / "scripts" / "measure_export_verification.py")
mev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mev)

OUT = HERE / "results" / "boundary_audit.json"
CSV = HERE / "results" / "boundary_audit_disagreements.csv"
EDGES = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, np.inf]
LABELS = ["<1e-6", "1e-6-1e-5", "1e-5-1e-4", "1e-4-1e-3", "1e-3-1e-2", ">=1e-2"]
SEEDS = [10_000 + i for i in range(100)]
CFG = json.loads((REPO / "config" / "config.json").read_text(encoding="utf-8")).get("automation", {})
TABLE = np.array([row[1:] for row in ex.ACTION_SETPOINTS], dtype=np.float32)


def first_max(q):
    return np.argmax(q, axis=1)


def gap(q64):
    s = np.sort(q64, axis=1)
    return s[:, -1] - s[:, -2]


def run_st(path, x):
    fb = sti.STFunctionBlock(Path(path), n=len(x))
    out = fb(**{nm: x[:, j] for j, nm in enumerate(ex.STATE_NAMES)})
    return fb.get("Q").T.astype(np.float32), np.asarray(sti.lanes(out["ACTION"], len(x))).astype(np.int64)


def run_fbt(path, x):
    fb = sti.BasicFB(Path(path), n=len(x))
    fb.fire("INIT")
    out = fb.fire("REQ", {nm: x[:, j] for j, nm in enumerate(ex.STATE_NAMES)})[0][1]
    return fb.get("Q").T.astype(np.float32), np.asarray(sti.lanes(out["ACTION"], len(x))).astype(np.int64)


def model_of(nid: str) -> str:
    return "speed_proportional" if nid.startswith("speed_proportional") else "legacy"


def rollout(policy_fn, model, seeds, record_actions=False):
    """The v0 audit's episode loop (experiments/v1/audit_v0.py): visited states, actions and rewards."""
    states, actions, rewards = [], [], []
    for sd in seeds:
        env = AutomationEnvironment({**CFG, "production_model": model})
        np.random.seed(sd)
        random.seed(sd)
        s, total, acts = env.reset(), 0.0, []
        for _ in range(200):
            a = policy_fn(np.asarray(s, dtype=np.float32))
            states.append(np.asarray(s, dtype=np.float32))
            acts.append(a)
            s, r, done = env.step(a)
            total += r
            if done:
                break
        actions.append(acts)
        rewards.append(total)
    return np.stack(states), actions, np.array(rewards)


def main():
    t0 = time.perf_counter()
    stored = json.loads((REPO / "docs" / "data" / "export_verification.json").read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in stored["networks"]}
    x = mev.states()
    assert hashlib.sha256(x.tobytes()).hexdigest() == stored["protocol"]["states"]["float32_sha256"]
    source = np.array(["rollout"] * mev.N_ROLLOUT + ["synthetic"] * mev.N_BOX)
    rows, disagreements = [], []
    agg = {src: {lab: dict(states=0, st_vs_torch=0, torch_vs_f64=0, st_vs_f64=0, fbt_vs_st=0,
                           max_abs_dq_st_torch=0.0, max_abs_dq_torch_f64=0.0) for lab in LABELS}
           for src in ("rollout", "synthetic", "visited")}

    def account(src, bins, a64, a32, ast, afb, q32, qst, q64):
        for k, lab in enumerate(LABELS):
            m = bins == k
            if not m.any():
                continue
            c = agg[src][lab]
            c["states"] += int(m.sum())
            c["st_vs_torch"] += int((ast[m] != a32[m]).sum())
            c["torch_vs_f64"] += int((a32[m] != a64[m]).sum())
            c["st_vs_f64"] += int((ast[m] != a64[m]).sum())
            if afb is not None:
                c["fbt_vs_st"] += int((afb[m] != ast[m]).sum())
            c["max_abs_dq_st_torch"] = max(c["max_abs_dq_st_torch"], float(np.abs(qst[m] - q32[m]).max()))
            c["max_abs_dq_torch_f64"] = max(c["max_abs_dq_torch_f64"], float(np.abs(q32[m] - q64[m]).max()))

    with tempfile.TemporaryDirectory() as tmp:
        for nid, group, src_path, _sha, policy in mev.networks():
            t1 = time.perf_counter()
            res = ex.export(policy, Path(tmp) / nid, ex.ExportOptions(date=mev.EXPORT_DATE), st=True)
            assert mev.sha256(res.fbt_path) == by_id[nid]["fbt_sha256"], f"{nid}: .fbt differs from the measured export"
            net = ex.reference_network(policy)
            sd = {k: v.detach() for k, v in net.state_dict().items()}
            with torch.no_grad():
                q32 = net(torch.from_numpy(x)).numpy()
            q64 = st_corpus.reference_q(sd, x)
            qst, ast = run_st(res.st_path, x)
            qfb, afb = run_fbt(res.fbt_path, x)
            a64, a32 = first_max(q64), first_max(q32)
            g = gap(q64)
            bins = np.digitize(g, EDGES[1:-1], right=False)
            for src in ("rollout", "synthetic"):
                m = source == src
                account(src, bins[m], a64[m], a32[m], ast[m], afb[m], q32[m], qst[m], q64[m])
            for i in np.flatnonzero((ast != a32) | (a32 != a64) | (ast != a64) | (afb != ast)):
                disagreements.append(dict(network=nid, corpus="A", index=int(i), source=source[i],
                                          margin_f64=float(g[i]), action_f64=int(a64[i]), action_torch=int(a32[i]),
                                          action_st=int(ast[i]), action_fbt=int(afb[i]),
                                          q_gap_torch=float(q32[i, a32[i]] - q32[i, ast[i]]),
                                          q_gap_st=float(qst[i, ast[i]] - qst[i, a32[i]]),
                                          setpoint_delta=(TABLE[ast[i]] - TABLE[a32[i]]).round(4).tolist()))
            # reachability: the network's own greedy policy on the v0 audit's dev episodes

            def greedy(s):
                with torch.no_grad():
                    return int(np.argmax(net(torch.from_numpy(s)[None]).numpy()[0]))
            vis, acts, rew = rollout(greedy, model_of(nid), SEEDS)
            with torch.no_grad():
                v32 = net(torch.from_numpy(vis)).numpy()
            v64 = st_corpus.reference_q(sd, vis)
            vst, vast = run_st(res.st_path, vis)
            va64, va32, vg = first_max(v64), first_max(v32), gap(v64)
            account("visited", np.digitize(vg, EDGES[1:-1]), va64, va32, vast, None, v32, vst, v64)
            n_dis = int((vast != va32).sum())
            closed = None
            if n_dis:   # re-run the affected episodes with the exported block in the loop
                bounds = np.cumsum([0] + [len(a) for a in acts])
                eps = sorted({int(np.searchsorted(bounds, i, side="right") - 1) for i in np.flatnonzero(vast != va32)})

                def st_policy(s):
                    return int(run_st(res.st_path, s[None])[1][0])
                _, acts_st, rew_st = rollout(st_policy, model_of(nid), [SEEDS[e] for e in eps])
                closed = dict(episodes=[SEEDS[e] for e in eps], reward_torch=rew[eps].round(4).tolist(),
                              reward_st=rew_st.round(4).tolist(),
                              actions_changed=[int(sum(p != q for p, q in zip(acts[e], acts_st[j])))
                                               for j, e in enumerate(eps)])
            share = np.bincount(np.concatenate([np.asarray(a) for a in acts]), minlength=6) / len(vis)
            rows.append(dict(id=nid, group=group, production_model=model_of(nid),
                             corpus_a=dict(st_vs_torch=int((ast != a32).sum()), torch_vs_f64=int((a32 != a64).sum()),
                                           st_vs_f64=int((ast != a64).sum()), fbt_vs_st=int((afb != ast).sum()),
                                           fbt_q_bit_identical_to_st=bool(np.array_equal(qfb.view(np.uint32),
                                                                                         qst.view(np.uint32))),
                                           min_margin_f64=float(g.min()),
                                           states_margin_below_1e_3=int((g < 1e-3).sum())),
                             visited=dict(states=int(len(vis)), episodes=len(SEEDS), st_vs_torch=n_dis,
                                          torch_vs_f64=int((va32 != va64).sum()), min_margin_f64=float(vg.min()),
                                          max_abs_dq_st_torch=float(np.abs(vst - v32).max()),
                                          action_share=share.round(4).tolist(), mean_reward=float(rew.mean()),
                                          closed_loop=closed),
                             seconds=round(time.perf_counter() - t1, 1)))
            print(f"{nid:28s} A: st/torch {rows[-1]['corpus_a']['st_vs_torch']}, torch/f64 "
                  f"{rows[-1]['corpus_a']['torch_vs_f64']}, min gap {g.min():.2e} | visited {len(vis)}: st/torch "
                  f"{n_dis}, min gap {vg.min():.2e}, actions {share.round(2).tolist()} ({rows[-1]['seconds']} s)",
                  flush=True)
    res = dict(script="experiments/v1/boundary_audit.py", tie_tolerance=0.0, margin_bins=LABELS,
               states_float32_sha256=stored["protocol"]["states"]["float32_sha256"], by_source_and_margin=agg,
               networks=rows, disagreements=len(disagreements), wall_seconds=round(time.perf_counter() - t0, 1),
               versions=dict(torch=torch.__version__, numpy=np.__version__))
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    with open(CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(disagreements[0]) if disagreements else ["network"])
        w.writeheader()
        w.writerows(disagreements)
    print(f"wrote {OUT.relative_to(REPO)} and {CSV.relative_to(REPO)} ({len(disagreements)} rows) in "
          f"{res['wall_seconds']} s")


if __name__ == "__main__":
    main()
