"""
The two states on which the strict gate rejected speed_proportional_seed5 in the
2026-11-08 export check, extracted as a deterministic regression fixture.
Frozen weights only: no training, no server.

Steps, each checked against docs/data/export_verification.json:
  1. rebuild the 20,000 states of measure_export_verification.py (same hash);
  2. load the checkpoint (same file and weight hashes) and re-export it with the
     pinned export date (same .fbt hash);
  3. rerun the strict gate on all 20,000 states and take the states where the
     generated action differs from PyTorch's (the same two states);
  4. record, for each, the full input vector, PyTorch's and the generated Q-values,
     both actions and both setpoint sets, plus the strict verdict and, separately,
     the tie-tolerance diagnostic.

Writes
  tests/fixtures/near_tie_speed_proportional_seed5.json   read by tests/test_iec61499_exporter.py
  docs/data/near_tie_states.csv                            the compact table linked from the README

Run from the repository root:
    .venv\\Scripts\\python.exe docs\\scripts\\extract_near_tie_fixture.py
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("measure", REPO / "docs" / "scripts" / "measure_export_verification.py")
measure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measure)  # defines states(), sha256(), EXPORT_DATE; main() is not run
ex, sti = measure.ex, measure.sti

NETWORK_ID = "speed_proportional_seed5"
MEASURED = REPO / "docs" / "data" / "export_verification.json"
FIXTURE = REPO / "tests" / "fixtures" / f"near_tie_{NETWORK_ID}.json"
TABLE = REPO / "docs" / "data" / "near_tie_states.csv"
DIAGNOSTIC_TIE = 2e-3  # 2 x q_tolerance, as in the stored re-check; a diagnostic, never the verdict
ACTION_NAMES = [row[0] for row in ex.ACTION_SETPOINTS]


def f32(values):
    """float32 values as Python floats: exact, and np.float32(v) gives back the same bits."""
    return [float(v) for v in np.asarray(values, dtype=np.float32).ravel()]


def hex32(values):
    return [f"0x{int(b):08x}" for b in np.ascontiguousarray(np.asarray(values, dtype=np.float32)).view(np.uint32).ravel()]


def gate(rep: dict) -> dict:
    keep = ("ok", "checks", "max_abs_q_diff", "argmax_agreement", "argmax_disagreements", "near_tie_disagreements",
            "setpoint_mismatches", "tie_tolerance", "states")
    return {k: rep[k] for k in keep}


def run_fb(fbt: Path, x: np.ndarray):
    fb = sti.BasicFB(fbt, n=len(x))
    fb.fire("INIT")
    cnf = fb.fire("REQ", {nm: x[:, j] for j, nm in enumerate(ex.STATE_NAMES)})
    assert [e for e, _ in cnf] == ["CNF"]
    out = cnf[0][1]
    q = fb.get("Q").T.astype(np.float32)
    action = sti.lanes(out["ACTION"], len(x)).astype(np.int64)
    sp = np.stack([sti.lanes(out[k], len(x)) for k in ex.SETPOINT_OUTPUTS], axis=1).astype(np.float32)
    return q, action, sp


def main() -> None:
    stored = json.loads(MEASURED.read_text(encoding="utf-8"))
    entry = next(n for n in stored["networks"] if n["id"] == NETWORK_ID)
    x = measure.states()
    x_sha = hashlib.sha256(x.tobytes()).hexdigest()
    assert x_sha == stored["protocol"]["states"]["float32_sha256"], "the state set differs from the measured one"
    ckpt = REPO / entry["source"]
    assert measure.sha256(ckpt) == entry["source_sha256"], "checkpoint file differs"
    policy = ex.load_policy(ckpt)
    assert policy.sha256 == entry["weights_sha256"], "checkpoint weights differ"
    net = ex.reference_network(policy)
    table = np.array([row[1:] for row in ex.ACTION_SETPOINTS], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        res = ex.export(policy, Path(tmp), ex.ExportOptions(date=measure.EXPORT_DATE), st=True)
        assert measure.sha256(res.fbt_path) == entry["fbt_sha256"], "the export differs from the measured one"
        full = ex.verify_fbt(res.fbt_path, policy, x)  # strict, exactly as measured
        idx = [e["state"] for e in full["examples"] if e["network_action"] != e["fb_action"]]
        assert full["argmax_disagreements"] == len(idx) == 2, full
        assert idx == [e["state"] for e in entry["examples"]], (idx, entry["examples"])

        with torch.no_grad():
            q_ref_batch = net(torch.from_numpy(x)).numpy()  # the reference the gate compared against
        q_gen_full, a_gen_full, sp_gen_full = run_fb(res.fbt_path, x)
        xs = x[idx]
        with torch.no_grad():  # PyTorch's own result for the same rows depends on the batch it is computed in
            q_ref_by_batch = {"20000 (as the gate runs it)": q_ref_batch[idx],
                              "2 (these two states)": net(torch.from_numpy(xs)).numpy(),
                              "1 (each state alone)": np.concatenate([net(torch.from_numpy(xs[k:k + 1])).numpy()
                                                                      for k in range(len(xs))])}
        q_gen, a_gen, sp_gen = run_fb(res.fbt_path, xs)
        assert np.array_equal(q_gen, q_gen_full[idx]) and np.array_equal(a_gen, a_gen_full[idx]), \
            "interpreter lanes are not independent"
        stfb = sti.STFunctionBlock(res.st_path, n=len(xs))
        st_out = stfb(**{nm: xs[:, j] for j, nm in enumerate(ex.STATE_NAMES)})
        q_st = stfb.get("Q").T.astype(np.float32)
        strict_pair = ex.verify_fbt(res.fbt_path, policy, xs, net=lambda _: torch.from_numpy(q_ref_batch[idx]))
        tolerant_pair = ex.verify_fbt(res.fbt_path, policy, xs, net=lambda _: torch.from_numpy(q_ref_batch[idx]),
                                      tie_tolerance=DIAGNOSTIC_TIE)
        tolerant_full = ex.verify_fbt(res.fbt_path, policy, x, tie_tolerance=DIAGNOSTIC_TIE)
        fbt_sha, st_sha = measure.sha256(res.fbt_path), measure.sha256(res.st_path)
        st_bits_equal = bool(np.array_equal(q_st, q_gen))
        st_action_equal = bool(np.array_equal(sti.lanes(st_out["ACTION"], len(xs)).astype(np.int64), a_gen))

    q_ref = q_ref_batch[idx]
    a_ref = q_ref.astype(np.float64).argmax(axis=1)
    states = []
    for k, i in enumerate(idx):
        top = np.argsort(-q_ref[k].astype(np.float64), kind="stable")[:2]
        e = next(e for e in entry["examples"] if e["state"] == i)
        gap = float(np.float64(q_ref[k, top[0]]) - np.float64(q_ref[k, top[1]]))
        dq = float(np.abs(q_gen[k].astype(np.float64) - q_ref[k].astype(np.float64)).max())
        assert (int(a_ref[k]), int(a_gen[k])) == (e["network_action"], e["fb_action"]), (i, e)
        assert gap == e["network_top2_gap"] and dq == e["max_abs_q_diff"], (i, gap, dq, e)
        states.append({
            "index": int(i),
            "sampled_from": f"uniform box, draw {int(i) - measure.N_ROLLOUT} of {measure.N_BOX}"
                            if i >= measure.N_ROLLOUT else f"simulator rollout, step {int(i)}",
            "input": f32(xs[k]),
            "reference_q": f32(q_ref[k]), "reference_q_bits": hex32(q_ref[k]),
            "generated_q": f32(q_gen[k]), "generated_q_bits": hex32(q_gen[k]),
            "reference_action": int(a_ref[k]), "generated_action": int(a_gen[k]),
            "competing_actions": [int(a) for a in top],
            "reference_top2_gap": gap,
            "float32_ulp_at_reference_max": float(np.spacing(np.abs(q_ref[k, top[0]]))),
            "generated_gap_same_pair": float(np.float64(q_gen[k, top[0]]) - np.float64(q_gen[k, top[1]])),
            "max_abs_q_diff": dq,
            "reference_setpoints": f32(table[a_ref[k]]), "generated_setpoints": f32(sp_gen[k]),
            "reference_by_batch_size": [
                {"batch": label, "q_competing": f32(q[k, top]),
                 "gap": float(np.float64(q[k, top[0]]) - np.float64(q[k, top[1]])),
                 "argmax": ACTION_NAMES[int(q[k].astype(np.float64).argmax())],
                 "max_abs_diff_vs_gate_reference": float(np.abs(q[k].astype(np.float64) - q_ref[k]).max())}
                for label, q in q_ref_by_batch.items()],
        })

    fixture = {
        "what": "The two states on which the strict export gate rejected speed_proportional_seed5 on 2026-11-08. "
                "PyTorch's best two actions (Quality focus, Aggressive) differ by 1.22e-4, a few float32 ulps. The "
                "generated function block sums in another order, its two Q-values round to an exact tie, and its "
                "argmax keeps the first maximum: Aggressive, with Aggressive's setpoints. Every Q-value stays within "
                "the 1e-3 tolerance, so a Q-only check would accept this export; the strict gate rejects it.",
        "generated_by": "docs/scripts/extract_near_tie_fixture.py (deterministic: no timestamp, so the file hash is stable)",
        "measurement": f"docs/data/export_verification.json, generated {stored['generated_at']}",
        "values": "float32 values written as exact decimals; *_bits are the IEEE 754 binary32 patterns",
        "checkpoint": {"path": entry["source"], "sha256": entry["source_sha256"], "weights_sha256": entry["weights_sha256"],
                       "origin": "multi-seed evaluation of 2026-11-07: training seed 5, corrected "
                                 "(speed_proportional) production model, 1,000 training episodes"},
        "export": {"date": measure.EXPORT_DATE, "fbt_sha256": fbt_sha, "st_sha256": st_sha},
        "state_set": {"construction": "measure_export_verification.states(): 10,000 rollout states of the default "
                                      "simulator under uniform random actions, then 10,000 draws uniform in [-2, 2]^12, "
                                      "both seeded 20261108, as float32",
                      "seed": measure.STATE_SEED, "n_rollout": measure.N_ROLLOUT, "n_box": measure.N_BOX,
                      "n": int(len(x)), "float32_sha256": x_sha, "rejected_indices": idx},
        "state_names": list(ex.STATE_NAMES), "action_names": ACTION_NAMES,
        "setpoint_outputs": list(ex.SETPOINT_OUTPUTS),
        "reference": "rl_agent.DQNNetwork in eval mode, float32 CPU, evaluated on the full 20,000-state batch as the "
                     "gate does; reference_by_batch_size shows how PyTorch's own values move with batch size",
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "torch_threads": torch.get_num_threads()},
        "st_on_these_states": {"q_bit_identical_to_fbt": st_bits_equal, "action_identical_to_fbt": st_action_equal},
        "states": states,
        "q_only_check": {"q_tolerance": stored["protocol"]["q_tolerance"], "max_abs_q_diff_over_20000": entry["max_abs_q_diff"],
                         "would_accept": bool(entry["max_abs_q_diff"] < stored["protocol"]["q_tolerance"])},
        "strict_gate": {"tie_tolerance": 0.0, "on_20000_states": gate(full), "on_these_two_states": gate(strict_pair)},
        "diagnostic_only_tie_tolerance_recheck": {
            "note": "Counts the two disagreements as near-ties. The generated action and setpoints still differ from "
                    "PyTorch's on both states, so this is not behavioural equivalence and not a strict pass.",
            "on_20000_states": gate(tolerant_full), "on_these_two_states": gate(tolerant_pair)},
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")

    header = (["state_index", "sampled_from"] + list(ex.STATE_NAMES)
              + ["competing_actions", "reference_q_first", "reference_q_second", "reference_gap",
                 "generated_q_first", "generated_q_second", "generated_gap", "max_abs_q_diff",
                 "reference_action", "generated_action", "reference_setpoints", "generated_setpoints",
                 "q_only_check", "strict_gate"])
    rows = []
    for s in states:
        a, b = s["competing_actions"]
        rows.append([s["index"], s["sampled_from"]] + [repr(v) for v in s["input"]]
                    + [f"{ACTION_NAMES[a]} / {ACTION_NAMES[b]}", repr(s["reference_q"][a]), repr(s["reference_q"][b]),
                       repr(s["reference_top2_gap"]), repr(s["generated_q"][a]), repr(s["generated_q"][b]),
                       repr(s["generated_gap_same_pair"]), repr(s["max_abs_q_diff"]),
                       ACTION_NAMES[s["reference_action"]], ACTION_NAMES[s["generated_action"]],
                       " ".join(f"{v:.2f}" for v in s["reference_setpoints"]),
                       " ".join(f"{v:.2f}" for v in s["generated_setpoints"]),
                       "accept", "reject"])
    with open(TABLE, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)

    print(json.dumps({k: fixture[k] for k in ("st_on_these_states",
                                              "q_only_check", "strict_gate", "diagnostic_only_tie_tolerance_recheck")},
                     indent=1))
    for s in states:
        a, b = s["competing_actions"]
        print(f"state {s['index']} ({s['sampled_from']}): reference {ACTION_NAMES[a]} {s['reference_q'][a]!r} vs "
              f"{ACTION_NAMES[b]} {s['reference_q'][b]!r} (gap {s['reference_top2_gap']:.4e}, ulp "
              f"{s['float32_ulp_at_reference_max']:.4e}); generated {s['generated_q'][a]!r} vs {s['generated_q'][b]!r} "
              f"(gap {s['generated_gap_same_pair']:+.4e}); actions {ACTION_NAMES[s['reference_action']]} -> "
              f"{ACTION_NAMES[s['generated_action']]}; setpoints {s['reference_setpoints']} -> {s['generated_setpoints']}")
        for r in s["reference_by_batch_size"]:
            print(f"    PyTorch, batch {r['batch']:28s} gap {r['gap']:+.4e}  argmax {r['argmax']:14s} "
                  f"max |diff| vs gate reference {r['max_abs_diff_vs_gate_reference']:.3e}")
    print(f"wrote {FIXTURE.relative_to(REPO)} and {TABLE.relative_to(REPO)}")


if __name__ == "__main__":
    main()
