"""
Standalone IEC 61131-3 (.st) verification on frozen weights (no training, no server).

Corpus A: the 18 networks and 20,000 states of docs/data/export_verification.json. Each network is
re-exported with the pinned date (its .fbt hash must equal the stored one). Its .st runs in
st_interpreter and iec61499_exporter.verify_st compares it with PyTorch, evaluated on all 20,000 states
in one batch as the gate does (tie tolerance 0). The .fbt verdict is recomputed with verify_fbt and
compared with the stored measurement, and the .st and .fbt outputs are compared bit for bit.

Corpus B: the synthetic corpus of tests/st_corpus.py (14 networks x 11,128 states; reference = float64
forward pass): coverage of all six actions, exact ties, boundary inputs.

Mutation sensitivity on corpus B: argmax -> argmin, first -> last maximum, and the SPEED_SP literal of each
action's branch, each applied to every corpus-B network. A mutant counts as reachable on a network when
the mutated path changes an output there: argmin always, last_max where the generated code ties the top
two Q-values exactly, setpoint a where the reference selects action a on some state.

Writes docs/data/st_verification.json. Run from the repository root:
    .venv\\Scripts\\python.exe docs\\scripts\\measure_st_verification.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


measure = _load("measure", REPO / "docs" / "scripts" / "measure_export_verification.py")  # states(), networks()
st_corpus = _load("st_corpus", REPO / "tests" / "st_corpus.py")
ex, sti = measure.ex, measure.sti

MEASURED = REPO / "docs" / "data" / "export_verification.json"
OUT = REPO / "docs" / "data" / "st_verification.json"
CORPUS_B_DATE = "2026-11-09"  # export date pinned in tests/test_st_standalone.py
TABLE = np.array([row[1:] for row in ex.ACTION_SETPOINTS], dtype=np.float32)
KEEP = ("ok", "checks", "max_abs_q_diff", "argmax_agreement", "argmax_disagreements", "near_tie_disagreements",
        "setpoint_mismatches", "action_histogram", "states", "tie_tolerance")


def gate(rep):
    return {k: rep[k] for k in KEEP if k in rep}


def run_st(path, x):
    fb = sti.STFunctionBlock(Path(path), n=len(x))
    out = fb(**{nm: x[:, j] for j, nm in enumerate(ex.STATE_NAMES)})
    return (fb.get("Q").T.astype(np.float32), np.asarray(sti.lanes(out["ACTION"], len(x))).astype(np.int64),
            np.stack([sti.lanes(out[nm], len(x)) for nm in ex.SETPOINT_OUTPUTS], axis=1).astype(np.float32))


def run_fbt(path, x):
    fb = sti.BasicFB(Path(path), n=len(x))
    fb.fire("INIT")
    out = fb.fire("REQ", {nm: x[:, j] for j, nm in enumerate(ex.STATE_NAMES)})[0][1]
    return (fb.get("Q").T.astype(np.float32), np.asarray(sti.lanes(out["ACTION"], len(x))).astype(np.int64),
            np.stack([sti.lanes(out[nm], len(x)) for nm in ex.SETPOINT_OUTPUTS], axis=1).astype(np.float32))


def same_bits(a, b):
    return bool(np.array_equal(np.ascontiguousarray(a).view(np.uint32), np.ascontiguousarray(b).view(np.uint32)))


def corpus_a(tmp: Path, stored: dict) -> dict:
    x = measure.states()
    x_sha = hashlib.sha256(x.tobytes()).hexdigest()
    assert x_sha == stored["protocol"]["states"]["float32_sha256"], "corpus A states differ from the stored measurement"
    by_id = {n["id"]: n for n in stored["networks"]}
    rows = []
    for nid, group, source, source_sha, policy in measure.networks():
        t0 = time.perf_counter()
        res = ex.export(policy, tmp / f"A_{nid}", ex.ExportOptions(date=measure.EXPORT_DATE), st=True)
        fbt_sha = measure.sha256(res.fbt_path)
        assert fbt_sha == by_id[nid]["fbt_sha256"], f"{nid}: .fbt differs from the measured export"
        with torch.no_grad():
            q_ref = ex.reference_network(policy)(torch.from_numpy(x)).numpy()
        ref = st_corpus.recorded(q_ref)
        st_rep = ex.verify_st(res.st_path, policy, x, net=ref)
        fbt_rep = ex.verify_fbt(res.fbt_path, policy, x, net=ref)
        s = by_id[nid]
        compared = ("ok", "checks", "max_abs_q_diff", "argmax_disagreements", "near_tie_disagreements",
                    "setpoint_mismatches")
        replicated = all(fbt_rep[k] == s[k] for k in compared)
        q_s, a_s, sp_s = run_st(res.st_path, x)
        q_f, a_f, sp_f = run_fbt(res.fbt_path, x)
        identical = same_bits(q_s, q_f) and bool(np.array_equal(a_s, a_f)) and same_bits(sp_s, sp_f)
        rows.append({"id": nid, "group": group, "st_sha256": measure.sha256(res.st_path),
                     "st": gate(st_rep), "fbt_recomputed": gate(fbt_rep), "fbt_replicates_stored": replicated,
                     "st_outputs_bit_identical_to_fbt": identical, "seconds": round(time.perf_counter() - t0, 1)})
        print(f"A {nid:26s} .st {ex.verification_summary(st_rep)} | .fbt replicated {replicated} | "
              f"bit-identical {rows[-1]['st_outputs_bit_identical_to_fbt']}", flush=True)
    return {"states": int(len(x)), "states_float32_sha256": x_sha, "reference": "PyTorch DQNNetwork, eval mode, float32 "
            "CPU, all 20,000 states in one batch (as verify_fbt and the stored measurement)", "networks": rows}


def corpus_b(tmp: Path) -> tuple:
    x = st_corpus.states()
    nets = {}
    rows = []
    for nid, kind, sd in st_corpus.networks():
        policy = ex.policy_from_state_dict(sd)
        res = ex.export(policy, tmp / f"B_{nid}", ex.ExportOptions(date=CORPUS_B_DATE), st=True)
        q_ref = st_corpus.reference_q(sd, x)
        rep = ex.verify_st(res.st_path, policy, x, net=st_corpus.recorded(q_ref))
        q_s, a_s, sp_s = run_st(res.st_path, x)
        ref_actions = q_ref.argmax(axis=1)
        top2 = np.sort(q_s, axis=1)[:, -2:]
        row = {"id": nid, "kind": kind, "weights_sha256": policy.sha256, "st_sha256": measure.sha256(res.st_path),
               "st": gate(rep), "reference_action_histogram": np.bincount(ref_actions, minlength=6).tolist(),
               "generated_exact_top2_ties": int(np.sum(top2[:, 0] == top2[:, 1]))}
        if kind == "tie":
            i, j = (int(v) for v in nid.split("_")[1:])
            row["tie"] = {"pair": [i, j], "generated_q_equal_everywhere": bool(np.array_equal(q_s[:, i], q_s[:, j])),
                          "reference_q_equal_everywhere": bool(np.array_equal(q_ref[:, i], q_ref[:, j])),
                          "first_maximum_selected_everywhere": bool((a_s == i).all() and (ref_actions == i).all())}
        if kind == "dominant":
            a = int(nid[-1])
            row["setpoint_correspondence"] = {"action": a, "selected_everywhere": bool((a_s == a).all()),
                                              "setpoints_equal_table_row": same_bits(sp_s, np.broadcast_to(TABLE[a],
                                                                                                           sp_s.shape))}
        rows.append(row)
        nets[nid] = dict(policy=policy, st=res.st_path, q_ref=q_ref, ref_actions=ref_actions,
                         ties=row["generated_exact_top2_ties"])
        print(f"B {nid:18s} {ex.verification_summary(rep)}", flush=True)
    seg = st_corpus.segments()
    counts = np.sum([r["reference_action_histogram"] for r in rows], axis=0)
    info = {"states": int(len(x)), "states_float32_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "segments": {k: int(b - a) for k, (a, b) in seg.items()}, "seed": st_corpus.STATE_SEED,
            "offset": st_corpus.OFFSET, "tie_pairs": [list(p) for p in st_corpus.TIE_PAIRS],
            "reference": "float64 forward pass of the same weights (tests/st_corpus.reference_q)",
            "reference_action_counts": counts.tolist(),
            "untrained_networks_select": sorted({int(a) for r in rows if r["kind"] == "untrained"
                                                 for a, c in enumerate(r["reference_action_histogram"]) if c}),
            "networks": rows}
    return x, nets, info


def mutations(x, nets: dict, tmp: Path) -> dict:
    kinds = [("argmin", 0), ("last_max", 0)] + [("setpoint", a) for a in range(6)]
    matrix = []
    for kind, action in kinds:
        label = kind if kind != "setpoint" else f"setpoint_{action}"
        for nid, net in nets.items():
            path = tmp / f"M_{nid}_{label}.st"
            path.write_text(st_corpus.mutate(net["st"].read_text(encoding="utf-8"), kind, action), encoding="utf-8")
            rep = ex.verify_st(path, net["policy"], x, net=st_corpus.recorded(net["q_ref"]))
            reachable = (kind == "argmin" or (kind == "last_max" and net["ties"] > 0)
                         or (kind == "setpoint" and bool(np.any(net["ref_actions"] == action))))
            matrix.append({"mutant": label, "network": nid, "reachable": reachable, "detected": not rep["ok"],
                           "failed_checks": [k for k, v in rep["checks"].items() if not v]})
        print(f"M {label:11s} detected on {sum(m['detected'] for m in matrix if m['mutant'] == label)} of "
              f"{len(nets)} networks, reachable on {sum(m['reachable'] for m in matrix if m['mutant'] == label)}",
              flush=True)
    return {"mutants": [k if k != "setpoint" else f"setpoint_{a}" for k, a in kinds],
            "pairs": len(matrix), "reachable_pairs": sum(m["reachable"] for m in matrix),
            "detected_reachable": sum(m["reachable"] and m["detected"] for m in matrix),
            "detected_unreachable": sum((not m["reachable"]) and m["detected"] for m in matrix),
            "matrix": matrix}


def main() -> None:
    t_start = time.perf_counter()
    stored = json.loads(MEASURED.read_text(encoding="utf-8"))
    result = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script": "docs/scripts/measure_st_verification.py",
        "predicate": "iec61499_exporter.verify_st (the predicate of verify_fbt): max_x max_k |Q_st - Q_ref| < 1e-3; "
                     "ACTION = first maximum of Q_ref on every state; SPEED_SP/PRESSURE_SP/TEMP_SP equal to the "
                     "ACTION_SETPOINTS row of that action as binary32; tie_tolerance 0",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "torch_threads": torch.get_num_threads()},
        "code": {"git_head": measure.git("rev-parse", "HEAD"),
                 "working_tree_changes": [ln[3:] for ln in measure.git("status", "--porcelain").splitlines()],
                 "sha256": {p: measure.sha256(REPO / p) for p in (
                     "backend/iec61499_exporter.py", "backend/st_interpreter.py", "backend/rl_agent.py",
                     "tests/st_corpus.py")}},
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        result["corpus_a"] = corpus_a(tmp, stored)
        x, nets, info = corpus_b(tmp)
        result["corpus_b"] = info
        result["mutations"] = mutations(x, nets, tmp)
    result["wall_seconds"] = round(time.perf_counter() - t_start, 1)
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} in {result['wall_seconds']} s")


if __name__ == "__main__":
    main()
