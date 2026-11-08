"""
Fresh export-verification measurements on frozen weights (no training, no server).

For every network below, export it with backend/iec61499_exporter.py, run the
generated .fbt in backend/st_interpreter.py (IEC REAL = float32, IEC 61499 event
semantics) on the same 20,000 states, and compare with rl_agent.DQNNetwork in eval
mode using iec61499_exporter.verify_fbt (the gate the server applies):

  * trained, server run   visual_outputs/run/policy_snapshot_20261106_190932.pt
  * trained, multi-seed   visual_outputs/run/eval_v2/policies/*.pt (6 seeds x 2 models)
  * synthetic             DQNNetwork(12, 6) at torch seeds 0-4 (untrained, all 6 actions)

Then: the .st of the server-run snapshot is run on the same states; five deliberate
defects are injected into the .fbt of the snapshot and of synthetic network 0; and
EAE's FBTypeCompiler.exe (if installed) compiles two exports plus three broken
variants. Writes docs/data/export_verification.json.

Run from the repository root:
    .venv\\Scripts\\python.exe docs\\scripts\\measure_export_verification.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
warnings.simplefilter("ignore", UserWarning)  # torch 2.1 TypedStorage notice

from backend import eae_fbtype_compiler as eae  # noqa: E402
from backend import iec61499_exporter as ex  # noqa: E402
from backend import st_interpreter as sti  # noqa: E402
from backend.rl_agent import DQNNetwork  # noqa: E402

OUT = REPO / "docs" / "data" / "export_verification.json"
SNAPSHOT = REPO / "visual_outputs" / "run" / "policy_snapshot_20261106_190932.pt"
EVAL_V2 = REPO / "visual_outputs" / "run" / "eval_v2" / "policies"
STATE_SEED = 20261108
N_ROLLOUT, N_BOX = 10_000, 10_000
EXPORT_DATE = "2026-11-08"  # fixed, so the generated files hash the same on every run
SYNTHETIC_SEEDS = range(5)

MUTATIONS = [  # (id, description, old, new); each replaces the first occurrence only
    ("no_latch", "REQ does not latch the ENERGY input (one WITH removed)",
     '        <With Var="ENERGY" />\n      </Event>\n    </EventInputs>', "      </Event>\n    </EventInputs>"),
    ("relu", "ReLU missing in one block of layer 2",
     "MAX(ACC + B2[I], 0.0)", "ACC + B2[I]"),
    ("row", "last row of a layer-2 weight block skipped",
     "FOR I := 16 TO 31 DO", "FOR I := 16 TO 30 DO"),
    ("setpoint", "wrong TEMP_SP literal in the Quality focus row",
     "        TEMP_SP := 0.95;", "        TEMP_SP := 0.9;"),
    ("argmin", "argmax replaced by argmin",
     "IF Q[A] > Q[ACTION] THEN", "IF Q[A] < Q[ACTION] THEN"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def states() -> np.ndarray:
    roll = ex.rollout_states(N_ROLLOUT, seed=STATE_SEED)  # legacy simulator, uniform random actions
    box = np.random.default_rng(STATE_SEED).uniform(-2.0, 2.0, size=(N_BOX, 12))  # the simulator clips to [-2, 2]
    return np.concatenate([roll, box]).astype(np.float32)


def networks():
    yield ("trained_snapshot", "trained, server run (2026-11-06)", SNAPSHOT.relative_to(REPO).as_posix(),
           sha256(SNAPSHOT), ex.load_policy(SNAPSHOT))
    for path in sorted(EVAL_V2.glob("*.pt")):
        yield (path.stem, "trained, multi-seed (2026-11-07)", path.relative_to(REPO).as_posix(),
               sha256(path), ex.load_policy(path))
    for seed in SYNTHETIC_SEEDS:
        saved = torch.get_rng_state()
        try:
            torch.manual_seed(seed)
            net = DQNNetwork(12, 6)
        finally:
            torch.set_rng_state(saved)
        yield (f"synthetic_seed{seed}", "synthetic (untrained, torch seed)", f"DQNNetwork(12, 6), torch.manual_seed({seed})",
               None, ex.policy_from_state_dict(net.state_dict()))


def gate_summary(rep: dict) -> dict:
    keep = ("ok", "checks", "max_abs_q_diff", "argmax_agreement", "argmax_disagreements",
            "near_tie_disagreements", "setpoint_mismatches", "q_range", "action_histogram", "examples", "error")
    return {k: rep[k] for k in keep if k in rep}


def compile_report(fbt: Path, out: Path, exe: Path) -> dict:
    t0 = time.perf_counter()
    rep = eae.compile_fbt(fbt, out, exe)

    def codes(lines):
        return [m.group(1) for m in (re.search(r"\b([A-Z][A-Z0-9_]+):", line) for line in lines) if m]

    return {
        "ok": rep.ok, "exit_code": rep.returncode, "bin_bytes": rep.bin_bytes, "seconds": round(time.perf_counter() - t0, 1),
        "errors": len(rep.errors), "warnings": len(rep.warnings), "known_standalone_messages": len(rep.context_messages),
        "error_codes": codes(rep.errors), "warning_codes": codes(rep.warnings),
        "known_message_codes": sorted(set(codes(rep.context_messages))),
        "raw_message_lines": len(rep.errors) + len(rep.warnings) + len(rep.context_messages),
        "lines_truncated_to_160_chars": [ln[:160] + (f" [+{len(ln) - 160} chars]" if len(ln) > 160 else "")
                                         for ln in rep.errors + rep.warnings + rep.context_messages],
    }


def main() -> None:
    t_start = time.perf_counter()
    x = states()
    result = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script": "docs/scripts/measure_export_verification.py",
        "protocol": {
            "states": {"rollout": {"n": N_ROLLOUT, "seed": STATE_SEED, "source": "iec61499_exporter.rollout_states: "
                                   "AutomationEnvironment (default 'legacy' production model), uniform random actions"},
                       "box": {"n": N_BOX, "seed": STATE_SEED, "source": "numpy default_rng uniform in [-2, 2]^12"},
                       "float32_sha256": hashlib.sha256(x.tobytes()).hexdigest()},
            "gate": "iec61499_exporter.verify_fbt: ok = max|dQ| < q_tolerance AND argmax on every state AND "
                    "SPEED_SP/PRESSURE_SP/TEMP_SP equal to the action-table row of the network's action",
            "q_tolerance": 1e-3, "tie_tolerance": 0.0, "export_date": EXPORT_DATE,
            "reference": "rl_agent.DQNNetwork in eval mode (dropout off), float32, CPU",
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                        "platform": platform.platform()},
        "code": {"git_head": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
                 "uncommitted_paths": [line[3:] for line in git("status", "--porcelain").splitlines()
                                       if "__pycache__" not in line and not line[3:].startswith((".venv", "wmg_env"))],
                 "sha256": {p: sha256(REPO / p) for p in ("backend/iec61499_exporter.py", "backend/st_interpreter.py",
                                                          "backend/eae_fbtype_compiler.py", "backend/rl_agent.py")}},
        "networks": [], "st_check": None, "mutations": [], "eae": None,
    }
    exe = eae.find_fbtype_compiler()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        exports = {}
        for nid, group, source, source_sha, policy in networks():
            t0 = time.perf_counter()
            opts = ex.ExportOptions(date=EXPORT_DATE)
            res = ex.export(policy, tmp / nid, opts, st=(nid == "trained_snapshot"))
            rep = ex.verify_fbt(res.fbt_path, policy, x)
            with torch.no_grad():
                ref_actions = ex.reference_network(policy)(torch.from_numpy(x)).argmax(1).numpy()
            share = (np.bincount(ref_actions, minlength=6) / len(x)).round(5).tolist()
            exports[nid] = (res, policy)
            entry = {
                "id": nid, "group": group, "source": source, "source_sha256": source_sha,
                "weights_sha256": policy.sha256, "fbt_sha256": sha256(res.fbt_path), "fbt_bytes": res.fbt_path.stat().st_size,
                "network_action_share": share, "actions_covered": int(np.count_nonzero(share)),
                "seconds": round(time.perf_counter() - t0, 1), **gate_summary(rep)}
            print(f"{nid:28s} {ex.verification_summary(rep)}  actions {share}", flush=True)
            if not rep["ok"]:
                # Diagnostic only, reported separately: the strict result above stays the verdict.
                # 2 x q_tolerance bounds what float32 rounding alone can do to the order of two actions.
                tie = 2 * rep["q_tolerance"]
                again = ex.verify_fbt(res.fbt_path, policy, x, tie_tolerance=tie)
                entry["tie_tolerance_recheck"] = {"tie_tolerance": tie, **gate_summary(again)}
                print(f"{'':28s} re-check, tie tolerance {tie:g}: {ex.verification_summary(again)}", flush=True)
            result["networks"].append(entry)

        # the IEC 61131-3 .st of the snapshot, run on the same states
        res, policy = exports["trained_snapshot"]
        fb = sti.STFunctionBlock(res.st_path, n=len(x))
        out = {k: sti.lanes(v, len(x)) for k, v in fb(**{nm: x[:, j] for j, nm in enumerate(ex.STATE_NAMES)}).items()}
        net = ex.reference_network(policy)
        with torch.no_grad():
            q_ref = net(torch.from_numpy(x)).numpy().astype(np.float64)
        q_st = fb.get("Q").T.astype(np.float64)
        fbt_fb = sti.BasicFB(res.fbt_path, n=len(x))
        fbt_fb.fire("INIT")
        fbt_out = fbt_fb.fire("REQ", {nm: x[:, j] for j, nm in enumerate(ex.STATE_NAMES)})[0][1]
        result["st_check"] = {
            "network": "trained_snapshot", "st_sha256": sha256(res.st_path), "st_bytes": res.st_path.stat().st_size,
            "max_abs_q_diff_vs_torch": float(np.abs(q_st - q_ref).max()),
            "argmax_agreement": float(np.mean(out["ACTION"] == q_ref.argmax(1))),
            "q_bit_identical_to_fbt": bool(np.array_equal(q_st.astype(np.float32), fbt_fb.get("Q").T)),
            "outputs_identical_to_fbt": all(np.array_equal(out[k], sti.lanes(fbt_out[k], len(x))) for k in out),
        }
        print("st_check", result["st_check"], flush=True)

        # deliberate defects
        for target in ("trained_snapshot", "synthetic_seed0"):
            res, policy = exports[target]
            text = res.fbt_path.read_text(encoding="utf-8")
            for mid, desc, old, new in MUTATIONS:
                assert old in text, (target, mid)
                mutant = tmp / f"{target}_{mid}" / "DQN_POLICY.fbt"
                mutant.parent.mkdir()
                mutant.write_text(text.replace(old, new, 1), encoding="utf-8")
                rep = ex.verify_fbt(mutant, policy, x)
                result["mutations"].append({"network": target, "id": mid, "description": desc, **gate_summary(rep)})
                print(f"mutation {target:18s} {mid:9s} {ex.verification_summary(rep)}", flush=True)

        # EAE's FB type compiler: acceptance of two exports, and three broken variants
        if exe is None:
            result["eae"] = {"compiler": None, "note": "FBTypeCompiler.exe not found; compiler checks not run"}
        else:
            version = exe.parents[3].name.rsplit(" ", 1)[-1]
            eae_res = {"compiler": str(exe), "version": version, "options": "default", "accepted": {}, "negative_controls": []}
            for target in ("trained_snapshot", "synthetic_seed0"):
                eae_res["accepted"][target] = compile_report(exports[target][0].fbt_path, tmp / f"eae_{target}", exe)
                print("eae", target, {k: v for k, v in eae_res["accepted"][target].items()
                                      if k != "lines_truncated_to_160_chars"}, flush=True)
            text = exports["trained_snapshot"][0].fbt_path.read_text(encoding="utf-8")
            m = re.search(r'Name="W4_0"[^>]*InitialValue="\[([^"]*)\]"', text)
            short = ",".join(m.group(1).split(",")[:-1])
            broken = [("undeclared variable", text.replace("ACTION := 0;\n", "ACTION := 0;\nNOT_DECLARED := 1.0;\n", 1)),
                      ("REAL assigned to INT", text.replace("ACTION := A;", "ACTION := Q[A];", 1)),
                      ("weight array one value short", text[:m.start(1)] + short + text[m.end(1):])]
            for name, src in broken:
                assert src != text, name
                d = tmp / ("neg_" + name.replace(" ", "_"))
                d.mkdir()
                (d / "DQN_POLICY.fbt").write_text(src, encoding="utf-8")
                rep = compile_report(d / "DQN_POLICY.fbt", d / "out", exe)
                rep["lines_truncated_to_160_chars"] = [ln for ln in rep["lines_truncated_to_160_chars"]
                                                       if "ERR_SYNTAX: '[" not in ln and "SNAPSHOT_GUID" not in ln]
                eae_res["negative_controls"].append({"defect": name, **rep})
                print("eae negative control", name, rep, flush=True)
            result["eae"] = eae_res
    result["wall_seconds"] = round(time.perf_counter() - t_start, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} in {result['wall_seconds']} s")


if __name__ == "__main__":
    main()
