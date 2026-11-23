"""
Standalone IEC 61131-3 (.st) verification on the deterministic synthetic corpus of tests/st_corpus.py.

The generated .st FUNCTION_BLOCK runs in backend/st_interpreter.py (REAL = binary32, every operation
rounded) and iec61499_exporter.verify_st compares it with a float64 forward pass of the same weights:
max |dQ| < 1e-3, the reference action (first maximum) on every state, and that action's setpoints,
with tie_tolerance 0. Coverage of all six actions is asserted before any verdict is used, and
deliberate argmax, tie-break and setpoint mutations must be detected wherever the mutated path is
reachable. Interpreter only: no vendor compilation of the .st and no controller.

Run from the repository root:
    .venv\\Scripts\\python.exe -m pytest tests/test_st_standalone.py -v -s
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
warnings.simplefilter("ignore", UserWarning)  # torch 2.1 TypedStorage notice

import st_corpus  # noqa: E402
from backend import iec61499_exporter as ex  # noqa: E402
from backend import st_interpreter as sti  # noqa: E402

Q_TOL = 1e-3
TABLE = np.array([row[1:] for row in ex.ACTION_SETPOINTS], dtype=np.float32)


def run_st(st_path, x):
    """(Q, ACTION, setpoints) of the .st FUNCTION_BLOCK, one call per state."""
    fb = sti.STFunctionBlock(Path(st_path), n=len(x))
    out = fb(**{nm: x[:, j] for j, nm in enumerate(ex.STATE_NAMES)})
    q = fb.get("Q").T.astype(np.float32)
    action = np.asarray(sti.lanes(out["ACTION"], len(x))).astype(np.int64)
    sp = np.stack([sti.lanes(out[nm], len(x)) for nm in ex.SETPOINT_OUTPUTS], axis=1).astype(np.float32)
    return q, action, sp


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    x = st_corpus.states()
    nets = {}
    for nid, kind, sd in st_corpus.networks():
        policy = ex.policy_from_state_dict(sd)
        res = ex.export(policy, tmp_path_factory.mktemp(nid), ex.ExportOptions(date="2026-11-09"), st=True)
        nets[nid] = dict(kind=kind, policy=policy, st=res.st_path, q_ref=st_corpus.reference_q(sd, x))
    return x, nets


def test_corpus_exercises_every_action_and_the_boundary_inputs(corpus):
    """Preconditions: without them a pass would say little about argmax or the setpoint table."""
    x, nets = corpus
    counts = np.zeros(6, dtype=int)
    for net in nets.values():
        counts += np.bincount(net["q_ref"].argmax(axis=1), minlength=6)
    untrained = set().union(*(set(np.unique(n["q_ref"].argmax(axis=1))) for n in nets.values()
                              if n["kind"] == "untrained"))
    print(f"\n[.st corpus] {len(nets)} networks x {len(x):,} states; reference actions per action: {counts.tolist()}; "
          f"untrained networks alone select {sorted(int(a) for a in untrained)}")
    assert (counts >= 1_000).all(), counts
    assert untrained == set(range(6)), "the state-dependent networks should reach every action by themselves"
    for a in range(6):  # the constructed networks select what they were built to select, on every state
        assert (nets[f"dominant_action{a}"]["q_ref"].argmax(axis=1) == a).all(), a
    for i, j in st_corpus.TIE_PAIRS:
        assert (nets[f"tie_{i}_{j}"]["q_ref"].argmax(axis=1) == i).all(), (i, j)
    seg = st_corpus.segments()
    boundary = x[slice(*seg["boundary"])]
    tiny = np.finfo(np.float32).tiny
    assert np.any(np.signbit(boundary) & (boundary == 0)), "-0.0 inputs"
    assert np.any((boundary != 0) & (np.abs(boundary) < tiny)), "subnormal inputs"
    assert np.abs(x).max() == 100.0 and seg["vertices"][1] - seg["vertices"][0] == 4096


def test_st_matches_float64_reference_on_every_network(corpus):
    x, nets = corpus
    for nid, net in nets.items():
        rep = ex.verify_st(net["st"], net["policy"], x, net=st_corpus.recorded(net["q_ref"]))
        print(f"[.st {nid:18s}] {ex.verification_summary(rep)}")
        assert rep["tie_tolerance"] == 0.0
        assert rep["ok"] and all(rep["checks"].values()), (nid, rep)
        assert rep["max_abs_q_diff"] < Q_TOL and rep["argmax_disagreements"] == rep["setpoint_mismatches"] == 0


def test_every_action_drives_its_own_setpoints(corpus):
    """Action/setpoint correspondence, one action at a time, bit for bit as binary32."""
    x, nets = corpus
    for a in range(6):
        _, action, sp = run_st(nets[f"dominant_action{a}"]["st"], x)
        assert (action == a).all(), a
        np.testing.assert_array_equal(sp, np.broadcast_to(TABLE[a], sp.shape), err_msg=f"action {a}")


def test_exact_ties_go_to_the_first_maximum(corpus):
    x, nets = corpus
    for i, j in st_corpus.TIE_PAIRS:
        net = nets[f"tie_{i}_{j}"]
        q, action, _ = run_st(net["st"], x)
        assert np.array_equal(q[:, i], q[:, j]), "the generated code should tie the pair exactly"
        assert np.array_equal(net["q_ref"][:, i], net["q_ref"][:, j]), "and so should the reference"
        assert (net["q_ref"].argmax(axis=1) == i).all() and (action == i).all(), (i, j)


def test_mutations_are_detected_where_reachable(corpus, tmp_path):
    """Each mutant must fail the predicate on every network that can reach the mutated path; the
    tie-break mutant is invisible without exact ties, which is why the corpus contains tie networks."""
    x, nets = corpus

    def check(nid, kind, action=0):
        net = nets[nid]
        path = tmp_path / f"{nid}_{kind}{action}.st"
        path.write_text(st_corpus.mutate(net["st"].read_text(encoding="utf-8"), kind, action), encoding="utf-8")
        return ex.verify_st(path, net["policy"], x, net=st_corpus.recorded(net["q_ref"]))

    for nid in ["untrained_seed0"] + [f"dominant_action{a}" for a in range(6)]:
        rep = check(nid, "argmin")
        assert not rep["ok"] and not rep["checks"]["argmax"], (nid, rep)
    for i, j in st_corpus.TIE_PAIRS:
        rep = check(f"tie_{i}_{j}", "last_max")
        assert not rep["ok"] and rep["argmax_disagreements"] == len(x), rep
    q0, _, _ = run_st(nets["untrained_seed0"]["st"], x)
    top2 = np.sort(q0, axis=1)[:, -2:]
    assert not np.any(top2[:, 0] == top2[:, 1]), "precondition: no exact ties in untrained_seed0's outputs"
    assert check("untrained_seed0", "last_max")["ok"], "without ties the tie-break mutant is not observable"
    for a in range(6):
        rep = check(f"dominant_action{a}", "setpoint", a)
        assert rep["checks"] == {"q_values": True, "argmax": True, "setpoints": False}, (a, rep)
        assert check(f"dominant_action{(a + 1) % 6}", "setpoint", a)["ok"], "another action's branch is not reached"
