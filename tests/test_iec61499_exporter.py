"""
Tests for backend/iec61499_exporter.py: the generated PLC code must execute
the trained policy.

(a) Fidelity. The ST algorithm inside the generated .fbt is parsed and run by
    backend/st_interpreter.py (IEC REAL = float32 with every multiply and add
    rounded; IEC 61499 event, WITH and ECC semantics) on 12,000 states from
    AutomationEnvironment rollouts, and compared with DQNNetwork in eval mode:
    100 % argmax agreement and max |Q difference| < 1e-3. The trained snapshot
    picks one action everywhere, so the same check also runs on a randomly
    initialised network whose argmax covers all six actions. A recorded near-tie
    (tests/fixtures/near_tie_speed_proportional_seed5.json) must stay rejected by
    the strict gate, although a Q-only check would accept it.
(b) Format. The .fbt is well-formed XML with the expected interface, WITH
    associations, ECC and algorithms; every weight literal reads back
    bit-exactly; it validates against EAE's DTD and compiles with EAE's own
    FBTypeCompiler.exe when EAE Buildtime is installed (skipped otherwise).
(c) Server. POST /api/code/generate and the WebSocket "generate_code" action of
    backend/main.py return the verified FB instead of PLCCodeGenerator's template,
    and refuse an FB whose argmax or setpoint table is wrong (runtime gate,
    iec61499_exporter.verify_fbt). Generated files download from /api/code/files/
    only; CORS and the WebSocket accept only the dashboard's own origin; the
    server starts in a checkout without a data/ folder and binds 127.0.0.1.

Run from the repo root:
    .venv\\Scripts\\python.exe -m pytest tests/test_iec61499_exporter.py -v -s
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib
import inspect
import json
import logging
import random
import re
import subprocess
import sys
import uuid
import warnings
import xml.etree.ElementTree as ET
from collections import deque
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend import eae_fbtype_compiler as eae  # noqa: E402
from backend import iec61499_exporter as ex  # noqa: E402
from backend import st_interpreter as sti  # noqa: E402
from backend.rl_agent import AutomationEnvironment, DQNNetwork  # noqa: E402

SNAPSHOT = REPO / "visual_outputs" / "run" / "policy_snapshot_20261106_190932.pt"
N_RANDOM_ACTION_STATES = 10_000
N_GREEDY_STATES = 2_000
Q_TOL = 1e-3
OUTPUTS = ["ACTION", "SPEED_SP", "PRESSURE_SP", "TEMP_SP"]
REAL_LITERAL = re.compile(r"-?\d+\.\d+(E[+-]?\d+)?")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def env_config() -> dict:
    """The 'automation' config main.py passes to AutomationEnvironment."""
    cfg = {"cycle_time_target": 30.0, "throughput_target": 120.0,
           "safety_threshold": 0.95, "energy_efficiency_target": 0.85}
    path = REPO / "config" / "config.json"
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")).get("automation", {}))
    return cfg


def rollouts(n: int, choose, seed: int) -> np.ndarray:
    """n states visited by rl_agent.AutomationEnvironment (the environment main.py
    trains on) with actions from choose(state, rng). numpy's global RNG, which
    the environment draws from, is restored afterwards."""
    saved = np.random.get_state(), random.getstate()
    try:
        np.random.seed(seed)
        random.seed(seed)
        env = AutomationEnvironment(env_config())
        rng = np.random.default_rng(seed)
        out = []
        while len(out) < n:
            s, done = env.reset(), False
            while not done and len(out) < n:
                out.append(s.copy())
                s, _, done = env.step(choose(s, rng))
        return np.array(out)
    finally:
        np.random.set_state(saved[0])
        random.setstate(saved[1])


def reference_net(state_dict) -> DQNNetwork:
    net = DQNNetwork(12, 6)
    net.load_state_dict(state_dict)
    net.eval()  # dropout off
    return net


def torch_q(net: DQNNetwork, states) -> np.ndarray:
    with torch.no_grad():
        return net(torch.from_numpy(np.asarray(states, dtype=np.float32))).numpy()


def run_fbt(fbt: Path, states: np.ndarray):
    """One Basic FB instance per state: INIT, then REQ. Returns (Q, CNF outputs, INITO outputs)."""
    x = np.asarray(states, dtype=np.float32)
    fb = sti.BasicFB(fbt, n=len(x))
    init = fb.fire("INIT")
    assert [e for e, _ in init] == ["INITO"]
    if "STATE" in fb.decls:
        inputs = {"STATE": x.T}
    else:
        inputs = {nm: x[:, j] for j, nm in enumerate(ex.STATE_NAMES)}
    cnf = fb.fire("REQ", inputs)
    assert [e for e, _ in cnf] == ["CNF"], "REQ must emit exactly one CNF"
    assert fb.state == "START"
    out = {k: sti.lanes(v, len(x)) for k, v in cnf[0][1].items()}
    return fb.get("Q").T.astype(np.float64), out, init[0][1]


def check_fidelity(q_fb: np.ndarray, out: dict, q_ref: np.ndarray, label: str) -> float:
    a_ref = q_ref.argmax(axis=1)  # torch.argmax: first maximum
    dq = np.abs(q_fb - q_ref.astype(np.float64))
    agree = float(np.mean(out["ACTION"] == a_ref))
    print(f"\n[{label}] {len(q_ref):,} states: argmax agreement {agree:.2%}, max |dQ| {dq.max():.3e}, "
          f"Q in [{q_ref.min():.4g}, {q_ref.max():.4g}], actions {np.bincount(a_ref, minlength=6).tolist()}")
    assert agree == 1.0, f"{label}: {int(np.sum(out['ACTION'] != a_ref))} states disagree"
    assert dq.max() < Q_TOL, f"{label}: max |dQ| = {dq.max():.3e}"
    table = np.array([row[1:] for row in ex.ACTION_SETPOINTS], dtype=np.float32)
    for k, name in enumerate(OUTPUTS[1:]):
        np.testing.assert_array_equal(out[name], table[a_ref, k], err_msg=f"{label}: {name}")
    return float(dq.max())


def nearest_float32(lit: str) -> np.float32:
    """Correctly rounded decimal -> binary32 (ties to even), with exact rationals.
    This is what a compiler that parses a REAL literal directly must produce."""
    exact = Fraction(lit)
    c = np.float32(float(exact))
    cands = {c, np.nextafter(c, np.float32(-np.inf)), np.nextafter(c, np.float32(np.inf))}
    return min(cands, key=lambda v: (abs(Fraction(float(v)) - exact),
                                     int(np.array(v, dtype=np.float32).view(np.uint32)) & 1))


def bits(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32)).view(np.uint32)


ARGMAX_LINE = "IF Q[A] > Q[ACTION] THEN"


def inject_bug(text: str, bug: str, action: int) -> str:
    """Generated ST (or .fbt) with one defect: "argmax" makes the action loop an argmin
    (Q-values untouched); "setpoints" changes SPEED_SP in the table entry of ``action``
    (Q-values and argmax untouched)."""
    if bug == "argmax":
        old, new = ARGMAX_LINE, ARGMAX_LINE.replace(">", "<")
    else:
        label, speed = ex.ACTION_SETPOINTS[action][:2]
        branch = f"    {action}: (* {label} *)\n        SPEED_SP := "
        old, new = branch + f"{ex.real_literal(speed)};", branch + f"{ex.real_literal(speed + 0.25)};"
    assert old in text, bug
    return text.replace(old, new, 1)


#: verify_fbt's checks each injected defect must fail ("argmax" also moves the
#: setpoints off the network's action, since they follow the FB's own choice).
FAILED_CHECKS = {"argmax": {"q_values": True, "argmax": False, "setpoints": False},
                 "setpoints": {"q_values": True, "argmax": True, "setpoints": False}}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def snapshot_state_dict():
    if not SNAPSHOT.exists():
        pytest.skip(f"{SNAPSHOT.relative_to(REPO)} not found (a training-run artefact, not in git)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # torch 2.1 TypedStorage deprecation notice
        return torch.load(SNAPSHOT, map_location="cpu", weights_only=True)["state_dict"]


@pytest.fixture(scope="module")
def trained(snapshot_state_dict, tmp_path_factory) -> ex.ExportResult:
    return ex.export(ex.load_policy(SNAPSHOT), tmp_path_factory.mktemp("trained"))


@pytest.fixture(scope="module")
def eval_states(snapshot_state_dict) -> np.ndarray:
    """10,000 states under uniform random actions + 2,000 under the trained greedy policy."""
    net = reference_net(snapshot_state_dict)
    explore = rollouts(N_RANDOM_ACTION_STATES, lambda s, rng: int(rng.integers(6)), seed=20261106)
    greedy = rollouts(N_GREEDY_STATES, lambda s, rng: int(torch_q(net, s[None])[0].argmax()), seed=7)
    return np.concatenate([explore, greedy])


@pytest.fixture(scope="module")
def random_net() -> DQNNetwork:
    saved = torch.get_rng_state()
    try:
        torch.manual_seed(0)
        net = DQNNetwork(12, 6)
    finally:
        torch.set_rng_state(saved)
    net.eval()
    return net


@pytest.fixture(scope="module")
def eae_exe() -> Path:
    exe = eae.find_fbtype_compiler()
    if exe is None:
        pytest.skip("EAE Buildtime (FBTypeCompiler.exe) is not installed")
    return exe


# ---------------------------------------------------------------------------
# (a) Fidelity against the PyTorch network
# ---------------------------------------------------------------------------
def test_fbt_reproduces_trained_network(trained, snapshot_state_dict, eval_states):
    assert len(eval_states) >= 10_000
    q_ref = torch_q(reference_net(snapshot_state_dict), eval_states)
    q_fb, out, init_out = run_fbt(trained.fbt_path, eval_states)
    check_fidelity(q_fb, out, q_ref, "trained snapshot, .fbt")
    _, speed, pressure, temp = ex.ACTION_SETPOINTS[ex.DEFAULT_ACTION]
    assert init_out == {"ACTION": ex.DEFAULT_ACTION, "SPEED_SP": np.float32(speed),
                        "PRESSURE_SP": np.float32(pressure), "TEMP_SP": np.float32(temp)}


def test_st_function_block_reproduces_trained_network(trained, snapshot_state_dict, eval_states):
    x = eval_states.astype(np.float32)
    fb = sti.STFunctionBlock(trained.st_path, n=len(x))
    out = fb(**{nm: x[:, j] for j, nm in enumerate(ex.STATE_NAMES)})
    out = {k: sti.lanes(v, len(x)) for k, v in out.items()}
    q_ref = torch_q(reference_net(snapshot_state_dict), eval_states)
    check_fidelity(fb.get("Q").T.astype(np.float64), out, q_ref, "trained snapshot, .st")


def test_fbt_reproduces_random_network_on_every_action(random_net, tmp_path):
    res = ex.export(ex.policy_from_state_dict(random_net.state_dict()), tmp_path, st=False)
    box = np.random.default_rng(3).uniform(-2.0, 2.0, size=(6_000, 12))  # the environment clips to [-2, 2]
    states = np.concatenate([rollouts(4_000, lambda s, rng: int(rng.integers(6)), seed=99), box])
    q_ref = torch_q(random_net, states)
    assert set(q_ref.argmax(axis=1)) == set(range(6)), "this test needs every action to be the argmax somewhere"
    q_fb, out, _ = run_fbt(res.fbt_path, states)
    check_fidelity(q_fb, out, q_ref, "random-init network, .fbt")


def test_array_input_variant(snapshot_state_dict, eval_states, tmp_path):
    res = ex.export(ex.load_policy(SNAPSHOT), tmp_path, ex.ExportOptions(state_input="array"), st=False)
    root = ET.parse(res.fbt_path).getroot()
    [state] = root.findall("InterfaceList/InputVars/VarDeclaration")
    assert (state.get("Name"), state.get("Type"), state.get("ArraySize")) == ("STATE", "REAL", "12")
    assert [w.get("Var") for w in root.find("InterfaceList/EventInputs/Event[@Name='REQ']")] == ["STATE"]
    states = eval_states[:2_000]
    q_fb, out, _ = run_fbt(res.fbt_path, states)
    check_fidelity(q_fb, out, torch_q(reference_net(snapshot_state_dict), states), "STATE array input, .fbt")


def test_req_works_without_init(trained, snapshot_state_dict, eval_states):
    """The weights are InitialValues, so REQ is valid before any INIT."""
    x = eval_states[:500].astype(np.float32)
    fb = sti.BasicFB(trained.fbt_path, n=len(x))
    cnf = fb.fire("REQ", {nm: x[:, j] for j, nm in enumerate(ex.STATE_NAMES)})
    q_ref = torch_q(reference_net(snapshot_state_dict), x)
    np.testing.assert_array_equal(sti.lanes(cnf[0][1]["ACTION"], len(x)), q_ref.argmax(axis=1))


@pytest.mark.parametrize("bug, old, new", [
    ("REQ does not latch ENERGY", '        <With Var="ENERGY" />\n      </Event>\n    </EventInputs>',
     "      </Event>\n    </EventInputs>"),
    ("ReLU missing in one block of layer 2", "MAX(ACC + B2[I], 0.0)", "ACC + B2[I]"),
    ("last row of a weight block skipped", "FOR I := 16 TO 31 DO", "FOR I := 16 TO 30 DO"),
    ("wrong setpoint for Quality focus", "        TEMP_SP := 0.95;", "        TEMP_SP := 0.9;"),
])
def test_fidelity_check_detects_generator_bugs(trained, snapshot_state_dict, eval_states, tmp_path, bug, old, new):
    """Mutation check: the harness above fails for typical generator mistakes
    (each mutation changes only the first occurrence of the pattern)."""
    text = trained.fbt_path.read_text(encoding="utf-8")
    assert old in text, bug
    mutant = tmp_path / "DQN_POLICY.fbt"
    mutant.write_text(text.replace(old, new, 1), encoding="utf-8")
    states = eval_states[::6]
    q_fb, out, _ = run_fbt(mutant, states)
    with pytest.raises(AssertionError):
        check_fidelity(q_fb, out, torch_q(reference_net(snapshot_state_dict), states), f"mutant: {bug}")


@pytest.mark.parametrize("bug", ["argmax", "setpoints"])
def test_runtime_gate_rejects_argmax_only_and_setpoint_only_bugs(trained, snapshot_state_dict, eval_states,
                                                                 tmp_path, bug):
    """verify_fbt, the check the server runs before it keeps an FB, needs more than
    matching Q-values: the action and the setpoints must be right too."""
    states = eval_states[::6]
    action = int(np.bincount(torch_q(reference_net(snapshot_state_dict), states).argmax(axis=1)).argmax())
    mutant = tmp_path / "DQN_POLICY.fbt"
    mutant.write_text(inject_bug(trained.fbt_path.read_text(encoding="utf-8"), bug, action), encoding="utf-8")
    policy = ex.load_policy(SNAPSHOT)
    clean = ex.verify_fbt(trained.fbt_path, policy, states)
    assert clean["ok"] and all(clean["checks"].values()) and not clean["examples"], clean
    rep = ex.verify_fbt(mutant, policy, states)
    print(f"\n[runtime gate, {bug} bug] {ex.verification_summary(rep)}")
    assert not rep["ok"] and rep["checks"] == FAILED_CHECKS[bug], rep
    assert rep["max_abs_q_diff"] < Q_TOL  # the Q-values alone would have passed it
    assert rep["examples"] and rep["setpoint_mismatches"] > 0


def test_runtime_gate_excuses_near_ties_only_when_asked(random_net, tmp_path):
    """Actions 0 and 1 get identical output rows, so their Q-values tie exactly, and the
    FB's argmax keeps the last maximum instead of the first. With the default tie
    tolerance 0 every such state fails; an explicit tolerance excuses them, counted."""
    sd = {k: v.clone() for k, v in random_net.state_dict().items()}
    sd["fc4.weight"][1], sd["fc4.bias"][1] = sd["fc4.weight"][0], sd["fc4.bias"][0]
    policy = ex.policy_from_state_dict(sd)
    res = ex.export(policy, tmp_path, st=False)
    last_max = tmp_path / "last_max.fbt"
    text = res.fbt_path.read_text(encoding="utf-8")
    last_max.write_text(text.replace(ARGMAX_LINE, ARGMAX_LINE.replace(">", ">="), 1), encoding="utf-8")
    states = np.random.default_rng(5).uniform(-2.0, 2.0, size=(3_000, 12))
    net = ex.reference_network(policy)
    tied = int(np.sum(torch_q(net, states).argmax(axis=1) == 0))
    assert tied > 0, "the test needs states where the tied pair is the best"
    assert ex.verify_fbt(res.fbt_path, policy, states, net=net)["ok"]
    strict = ex.verify_fbt(last_max, policy, states, net=net)
    assert not strict["ok"] and strict["checks"] == FAILED_CHECKS["argmax"], strict
    assert (strict["argmax_disagreements"], strict["near_tie_disagreements"]) == (tied, 0)
    lenient = ex.verify_fbt(last_max, policy, states, net=net, tie_tolerance=1e-6)
    assert lenient["ok"], lenient
    assert lenient["near_tie_disagreements"] == lenient["argmax_disagreements"] == tied
    assert "near-tie disagreements excused" in ex.verification_summary(lenient)


# A real near-tie the strict gate must keep rejecting: the two states of the 2026-11-08
# export check (docs/data/export_verification.json) on which speed_proportional_seed5's
# generated FB picks another action than PyTorch. Written by docs/scripts/extract_near_tie_fixture.py.
NEAR_TIE = REPO / "tests" / "fixtures" / "near_tie_speed_proportional_seed5.json"


@pytest.fixture(scope="module")
def near_tie(tmp_path_factory):
    """(case, policy, .fbt of the checkpoint, the two input states, PyTorch's recorded Q-values)."""
    case = json.loads(NEAR_TIE.read_text(encoding="utf-8"))
    ckpt = REPO / case["checkpoint"]["path"]
    if not ckpt.exists():
        pytest.skip(f"{case['checkpoint']['path']} not found (a multi-seed evaluation artefact)")
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == case["checkpoint"]["sha256"]
    policy = ex.load_policy(ckpt)
    assert policy.sha256 == case["checkpoint"]["weights_sha256"]
    res = ex.export(policy, tmp_path_factory.mktemp("near_tie"), ex.ExportOptions(date=case["export"]["date"]),
                    st=False)
    x = np.array([s["input"] for s in case["states"]], dtype=np.float32)
    q_ref = np.array([s["reference_q"] for s in case["states"]], dtype=np.float32)
    return case, policy, res.fbt_path, x, q_ref


def test_near_tie_fixture_generated_outputs_are_reproduced(near_tie):
    """At float32 the generated FB gives the recorded outputs bit for bit: its two competing
    Q-values round to an exact tie, and the first maximum, Aggressive, wins with its setpoints."""
    case, _, fbt, x, _ = near_tie
    q, out, _ = run_fbt(fbt, x)
    for k, s in enumerate(case["states"]):
        a, b = s["competing_actions"]
        assert bits(q[k]).tolist() == [int(h, 16) for h in s["generated_q_bits"]], s["index"]
        assert q[k, a] == q[k, b], "the generated code should tie the two actions exactly"
        assert out["ACTION"][k] == s["generated_action"] == min(a, b) != s["reference_action"]
        sp = np.array([out[nm][k] for nm in OUTPUTS[1:]], dtype=np.float32)
        np.testing.assert_array_equal(sp, np.array(s["generated_setpoints"], dtype=np.float32))
        assert s["generated_setpoints"] != s["reference_setpoints"]


def test_strict_gate_rejects_near_tie_a_q_only_check_accepts(near_tie):
    """Against PyTorch's recorded outputs, so the verdict does not depend on this machine's
    summation order: every Q-value is within tolerance, the action and setpoints are not."""
    case, policy, fbt, x, q_ref = near_tie

    def recorded(_):
        return torch.from_numpy(q_ref)

    strict = ex.verify_fbt(fbt, policy, x, net=recorded)
    assert strict["max_abs_q_diff"] < Q_TOL  # a check of the Q-values alone would accept this export
    assert not strict["ok"] and strict["checks"] == FAILED_CHECKS["argmax"], strict
    counts = strict["argmax_disagreements"], strict["near_tie_disagreements"], strict["setpoint_mismatches"]
    assert counts == (2, 0, 2), strict
    # Diagnostic only: an explicit tolerance counts both states as near-ties. The FB still picks
    # the other action and its setpoints on both, so this is neither equivalence nor a strict pass.
    diag = ex.verify_fbt(fbt, policy, x, net=recorded, tie_tolerance=2 * Q_TOL)
    assert diag["near_tie_disagreements"] == diag["argmax_disagreements"] == 2
    assert diag["argmax_agreement"] == 0.0


def test_strict_gate_rejects_near_tie_checkpoint_on_measured_states(near_tie):
    """End to end, with the live PyTorch reference on the 20,000 states of the 2026-11-08 check.
    PyTorch's own values for these states move by up to 1.2e-4 with batch size, so another
    machine may round them differently; then this test skips and the recorded-reference test
    above still applies."""
    case, policy, fbt, _, q_ref = near_tie
    st = case["state_set"]
    box = np.random.default_rng(st["seed"]).uniform(-2.0, 2.0, size=(st["n_box"], 12))
    x = np.concatenate([ex.rollout_states(st["n_rollout"], seed=st["seed"]), box]).astype(np.float32)
    assert hashlib.sha256(x.tobytes()).hexdigest() == st["float32_sha256"]
    idx = st["rejected_indices"]
    if not np.array_equal(bits(torch_q(ex.reference_network(policy), x)[idx]), bits(q_ref)):
        pytest.skip("PyTorch on this machine gives other float32 values for the two recorded states")
    rep = ex.verify_fbt(fbt, policy, x)
    assert not rep["ok"] and rep["checks"] == FAILED_CHECKS["argmax"], rep
    assert rep["argmax_disagreements"] == 2 and [e["state"] for e in rep["examples"]] == idx


def test_server_default_keeps_the_strict_gate():
    """Acceptance decision of 2026-11-08: no tie tolerance unless one is set explicitly."""
    tree = ast.parse((REPO / "backend" / "main.py").read_text(encoding="utf-8"))
    defaults = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(getattr(t, "id", None) == "CODE_GENERATION_DEFAULTS" for t in node.targets))
    assert defaults["tie_tolerance"] == 0.0
    cfg = json.loads((REPO / "config" / "config.json").read_text(encoding="utf-8"))
    assert cfg.get("code_generation", {}).get("tie_tolerance", 0.0) == 0.0
    assert inspect.signature(ex.verify_fbt).parameters["tie_tolerance"].default == 0.0


def test_action_table_matches_environment():
    saved = np.random.get_state()
    try:
        env = AutomationEnvironment(env_config())
    finally:
        np.random.set_state(saved)
    assert env.get_state_size() == len(ex.STATE_NAMES)
    assert env.get_action_size() == len(ex.ACTION_SETPOINTS)
    for a, (_, speed, pressure, temp) in enumerate(ex.ACTION_SETPOINTS):
        assert env._apply_action(a) == {"speed": speed, "pressure": pressure, "temp": temp}
    assert env._apply_action(99) == env._apply_action(ex.DEFAULT_ACTION)


# ---------------------------------------------------------------------------
# (b) Format
# ---------------------------------------------------------------------------
def test_fbt_interface_ecc_and_algorithms(trained):
    raw = trained.fbt_path.read_text(encoding="utf-8")
    assert raw.startswith('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE FBType')
    root = ET.fromstring(raw.encode("utf-8"))  # well-formed
    assert root.tag == "FBType"
    assert (root.get("Name"), root.get("Namespace")) == ("DQN_POLICY", "Main")
    uuid.UUID(root.get("GUID"))
    assert root.find("Identification").get("Standard") == "61499-2"

    il = root.find("InterfaceList")

    def events(tag):
        return {e.get("Name"): [w.get("Var") for w in e.findall("With")] for e in il.findall(f"{tag}/Event")}

    def data(tag):
        return [(v.get("Name"), v.get("Type"), v.get("ArraySize")) for v in il.findall(f"{tag}/VarDeclaration")]

    ev_in, ev_out = events("EventInputs"), events("EventOutputs")
    assert list(ev_in) == ["INIT", "REQ"]
    assert list(ev_out) == ["INITO", "CNF"]
    assert data("InputVars") == [(nm, "REAL", None) for nm in ex.STATE_NAMES]
    assert data("OutputVars") == [("ACTION", "INT", None), ("SPEED_SP", "REAL", None),
                                  ("PRESSURE_SP", "REAL", None), ("TEMP_SP", "REAL", None)]
    assert ev_in["INIT"] == []
    assert ev_in["REQ"] == list(ex.STATE_NAMES)
    assert ev_out["INITO"] == OUTPUTS
    assert ev_out["CNF"] == OUTPUTS

    basic = root.find("BasicFB")
    ecc = basic.find("ECC")
    states = [(s.get("Name"), [(a.get("Algorithm"), a.get("Output")) for a in s.findall("ECAction")])
              for s in ecc.findall("ECState")]
    assert states == [("START", []), ("INIT", [("INIT", "INITO")]), ("REQ", [("REQ", "CNF")])]
    assert {(t.get("Source"), t.get("Destination"), t.get("Condition")) for t in ecc.findall("ECTransition")} == {
        ("START", "INIT", "INIT"), ("INIT", "START", "1"), ("START", "REQ", "REQ"), ("REQ", "START", "1")}

    algs = {a.get("Name"): (a.find("ST").text or "") for a in basic.findall("Algorithm")}
    assert set(algs) == {"INIT", "REQ"}
    assert raw.count("<ST><![CDATA[") == 2  # ST as CDATA content, as EAE writes it
    for needle in ("X[11] := ENERGY;", "W2_7[K] * H1[J]", "MAX(ACC + B3[I], 0.0)", "Q[I] := ACC + B4[I];",
                   "IF Q[A] > Q[ACTION] THEN", "CASE ACTION OF"):
        assert needle in algs["REQ"], needle
    assert "ACTION := 1;" in algs["INIT"]
    sti.BasicFB(trained.fbt_path, n=1)  # also checks every WITH / ECAction / variable reference


def test_fbt_validates_against_eae_dtd(trained, eae_exe):
    etree = pytest.importorskip("lxml.etree")
    dtd_path = eae_exe.parents[2] / "data" / "Schemas" / "nxtLibraryElement.dtd"
    if not dtd_path.exists():
        pytest.skip(f"{dtd_path} not found")
    dtd = etree.DTD(str(dtd_path))
    assert dtd.validate(etree.parse(str(trained.fbt_path))), dtd.error_log.filter_from_errors()


def test_weights_are_bit_exact_and_literals_round_correctly(trained, snapshot_state_dict):
    root = ET.parse(trained.fbt_path).getroot()
    decls = {v.get("Name"): v for v in root.findall("BasicFB/InternalVars/VarDeclaration")}
    st_fb = sti.STFunctionBlock(trained.st_path, n=1)

    def fbt_array(name):
        v = decls[name]
        lo, hi = sti.parse_array_size(v.get("ArraySize"))
        return sti.parse_initializer(v.get("InitialValue"), sti.VarDecl(name, "REAL", lo, hi))

    literals = []
    for layer in range(1, 5):
        W = snapshot_state_dict[f"fc{layer}.weight"].numpy()
        b = snapshot_state_dict[f"fc{layer}.bias"].numpy()
        assert np.array_equal(bits(fbt_array(f"B{layer}")), bits(b))
        assert np.array_equal(bits(st_fb.get(f"B{layer}")), bits(b))
        names = sorted((n for n in decls if n.startswith(f"W{layer}_")), key=lambda n: int(n.split("_")[1]))
        assert all(int(decls[n].get("ArraySize")) <= ex.DEFAULT_CHUNK_ELEMS for n in names)
        assert np.array_equal(bits(np.concatenate([fbt_array(n) for n in names])), bits(W.ravel()))
        assert np.array_equal(bits(np.concatenate([st_fb.get(n) for n in names])), bits(W.ravel()))
        for n in [f"B{layer}", *names]:
            literals += decls[n].get("InitialValue")[1:-1].split(",")
    assert len(literals) == 35_462
    assert [s for s in literals if not REAL_LITERAL.fullmatch(s)] == []
    # a compiler that parses REAL literals directly (correctly rounded) gets the same bits
    # as one that goes decimal -> double -> float32, which is what parse_initializer did above
    direct = np.array([nearest_float32(s) for s in literals], dtype=np.float32)
    via_double = np.array([float(s) for s in literals]).astype(np.float32)
    assert np.array_equal(bits(direct), bits(via_double))


def test_st_and_fbt_run_the_same_algorithm(trained):
    root = ET.parse(trained.fbt_path).getroot()
    req = next(a for a in root.iter("Algorithm") if a.get("Name") == "REQ").find("ST").text
    st_text = trained.st_path.read_text(encoding="utf-8")
    body = st_text.rsplit("END_VAR", 1)[1].rsplit("END_FUNCTION_BLOCK", 1)[0]

    def tokens(text):
        return [(t.kind, t.text) for t in sti.tokenize(text)]

    assert tokens(body) == tokens(req)  # identical code once comments are dropped


def test_reported_size_and_cost(trained):
    p = trained.policy
    assert p.sizes == [12, 128, 128, 128, 6]
    assert p.n_params == 35_462
    assert p.macs == 12 * 128 + 128 * 128 + 128 * 128 + 128 * 6 == 35_072
    assert trained.weight_bytes == 4 * 35_462
    assert sum(len(c) for c in trained.chunks) == 18
    assert max(c.size for layer in trained.chunks for c in layer) <= ex.DEFAULT_CHUNK_ELEMS
    print("\n" + trained.summary())


def test_load_policy_accepts_agent_policy_and_bare_state_dict(snapshot_state_dict, tmp_path):
    ref = ex.policy_from_state_dict(snapshot_state_dict)
    for name, obj in {"agent": {"model_state": snapshot_state_dict, "epsilon": 0.01, "training_steps": 5},
                      "bare": snapshot_state_dict}.items():
        path = tmp_path / f"{name}.pt"
        torch.save(obj, path)
        got = ex.load_policy(path)
        assert got.sha256 == ref.sha256
    with pytest.raises(ValueError):
        ex.policy_from_state_dict({**snapshot_state_dict, "extra.running_mean": torch.zeros(3)})


def test_cli_exports_and_verifies(snapshot_state_dict, tmp_path, capsys):
    rc = ex.main([str(SNAPSHOT), "--out-dir", str(tmp_path), "--verify", "300"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert (tmp_path / "DQN_POLICY.fbt").is_file() and (tmp_path / "DQN_POLICY.st").is_file()
    assert "argmax agreement 100.00%" in out


# ---------------------------------------------------------------------------
# (b) EAE's own compiler (skipped when EAE Buildtime is not installed)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("args", [(), ("/disable:AllowKeywords",), ("/enable:OPAFCompatible",)],
                         ids=["default", "keywords-reserved", "OPAF-profile"])
def test_eae_compiler_accepts_fbt(trained, eae_exe, tmp_path, args):
    rep = eae.compile_fbt(trained.fbt_path, tmp_path, eae_exe, extra_args=args)
    print("\n" + rep.summary())
    assert rep.ok, rep.summary()
    initial_values = [v.get("InitialValue") for v in ET.parse(trained.fbt_path).getroot().iter("VarDeclaration")
                      if v.get("ArraySize") and v.get("InitialValue")]
    assert len(rep.context_messages) == 1 + len(initial_values)  # snapshot GUID + one per array initialiser
    # kept whole, as the compiler printed them: each echoes its initialiser completely
    assert all(any(f"'{iv}' is not correct" in m for m in rep.context_messages) for iv in initial_values)


def test_eae_compiler_check_rejects_broken_fbt(trained, eae_exe, tmp_path):
    """Negative controls: the compile check above is not vacuous."""
    text = trained.fbt_path.read_text(encoding="utf-8")
    m = re.search(r'Name="W4_0"[^>]*InitialValue="\[([^"]*)\]"', text)
    short = ",".join(m.group(1).split(",")[:-1])
    broken = {  # name: (source, message code EAE must report)
        "undeclared variable": (text.replace("ACTION := 0;\n", "ACTION := 0;\nNOT_DECLARED := 1.0;\n", 1),
                                "ERR_NO_SUCH_VAR"),
        "REAL assigned to INT": (text.replace("ACTION := A;", "ACTION := Q[A];", 1), "ERR_CAST_UNDEFINED"),
        "short weight array": (text[:m.start(1)] + short + text[m.end(1):], "WRN_ARRAY_NOINIT"),
    }
    for name, (src, code) in broken.items():
        assert src != text
        d = tmp_path / name.replace(" ", "_")
        d.mkdir()
        (d / "DQN_POLICY.fbt").write_text(src, encoding="utf-8")
        rep = eae.compile_fbt(d / "DQN_POLICY.fbt", d / "out", eae_exe)
        assert not rep.ok, f"{name}: {rep.summary()}"
        assert any(code in msg for msg in rep.errors + rep.warnings), f"{name}: {rep.summary()}"


# ---------------------------------------------------------------------------
# (c) The server: backend/main.py
# ---------------------------------------------------------------------------
async def asgi_call(app, method: str, path: str, headers=()):
    """One HTTP request straight into the ASGI app (FastAPI's TestClient needs
    httpx, which is not in requirements.txt). Returns (status, headers, body bytes)."""
    sent, requested = [], False

    async def receive():
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.Event().wait()  # the client never disconnects

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
             "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
             "root_path": "", "headers": [(b"host", b"127.0.0.1:8000"), *headers],
             "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8000)}
    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], {k.decode().lower(): v.decode() for k, v in start["headers"]}, body


async def asgi_request(app, method: str, path: str):
    """Returns (status, JSON body)."""
    status, _, body = await asgi_call(app, method, path)
    return status, json.loads(body)


def asgi_http(app, method: str, path: str):
    return asyncio.run(asgi_request(app, method, path))


def asgi_websocket_handshake(app, origin=None):
    """Open /ws as a browser on the page at ``origin`` would. Returns "accepted", or the
    close code the server answered the handshake with."""
    async def run():
        inbox, outcome = asyncio.Queue(), []
        await inbox.put({"type": "websocket.connect"})

        async def receive():
            return await inbox.get()

        async def send(message):
            if message["type"] == "websocket.accept":
                outcome.append("accepted")
                await inbox.put({"type": "websocket.disconnect", "code": 1000})
            elif message["type"] == "websocket.close":
                outcome.append(message.get("code", 1000))

        headers = [(b"host", b"127.0.0.1:8000")] + ([(b"origin", origin.encode())] if origin else [])
        scope = {"type": "websocket", "asgi": {"version": "3.0"}, "scheme": "ws", "path": "/ws",
                 "raw_path": b"/ws", "query_string": b"", "root_path": "", "headers": headers,
                 "client": ("127.0.0.1", 50002), "server": ("127.0.0.1", 8000), "subprotocols": []}
        await asyncio.wait_for(app(scope, receive, send), timeout=60)
        return outcome[0]

    return asyncio.run(run())


def asgi_websocket(app, actions):
    """Open /ws, send each {"action": ...} once the previous reply has arrived,
    disconnect after the reply to the last one. Returns the JSON messages received."""
    async def run():
        inbox, received = asyncio.Queue(), []
        pending = [json.dumps({"action": a}) for a in actions]
        await inbox.put({"type": "websocket.connect"})

        async def receive():
            return await inbox.get()

        async def send(message):
            if message["type"] == "websocket.accept":
                await inbox.put({"type": "websocket.receive", "text": pending.pop(0)})
            elif message["type"] == "websocket.send":
                received.append(json.loads(message["text"]))
                if received[-1]["type"] != "system_state":
                    nxt = {"type": "websocket.receive", "text": pending.pop(0)} if pending else \
                          {"type": "websocket.disconnect", "code": 1000}
                    await inbox.put(nxt)

        scope = {"type": "websocket", "asgi": {"version": "3.0"}, "scheme": "ws", "path": "/ws",
                 "raw_path": b"/ws", "query_string": b"", "root_path": "", "headers": [(b"host", b"testserver")],
                 "client": ("127.0.0.1", 50001), "server": ("testserver", 80), "subprotocols": []}
        await asyncio.wait_for(app(scope, receive, send), timeout=120)
        return received

    return asyncio.run(run())


@pytest.fixture(scope="module")
def api_server(snapshot_state_dict, eval_states, tmp_path_factory):
    """WMGDigitalTwinServer from backend/main.py, run from a scratch working folder
    (main.py uses paths relative to the working folder), holding the snapshot
    weights and 2,500 environment states in its replay buffer."""
    work = tmp_path_factory.mktemp("server")
    for d in ("frontend/css", "frontend/js"):  # no data/ folder, as in a fresh clone
        (work / d).mkdir(parents=True)
    mp = pytest.MonkeyPatch()
    mp.chdir(work)
    mp.syspath_prepend(str(REPO / "backend"))  # main.py uses flat imports
    saved = np.random.get_state(), random.getstate(), torch.get_rng_state()
    try:
        main = importlib.import_module("main")
        server = main.WMGDigitalTwinServer()
    finally:
        np.random.set_state(saved[0])
        random.setstate(saved[1])
        torch.set_rng_state(saved[2])
    server.rl_agent.q_network.load_state_dict(snapshot_state_dict)
    for s in eval_states[:2_500]:
        server.rl_agent.remember(s, 0, 0.0, s, False)
    yield server, work
    mp.undo()
    root = logging.getLogger()
    for h in list(root.handlers):  # the log file main.py may have opened in the scratch folder
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename).is_relative_to(work):
            root.removeHandler(h)
            h.close()


def test_api_code_generate_returns_verified_fb(api_server, snapshot_state_dict, eval_states):
    server, work = api_server
    memory = server.rl_agent.memory
    full, memory.buffer = memory.buffer, deque(maxlen=memory.buffer.maxlen)
    try:
        assert asgi_http(server.app, "POST", "/api/code/generate") == (400, {"detail": "Model not trained"})
    finally:
        memory.buffer = full

    status, body = asgi_http(server.app, "POST", "/api/code/generate")
    assert status == 200, body
    root = ET.fromstring(body["code"].encode("utf-8"))
    assert (root.tag, root.get("Name"), root.get("Namespace")) == ("FBType", "DQN_POLICY", "Main")
    assert "PROGRAM" not in body["code"] and "FB_RLOptimizer" not in body["code"]  # not the old template
    assert body["fb_type"] == "Main.DQN_POLICY"
    assert body["network"] == [12, 128, 128, 128, 6]
    assert (body["parameters"], body["macs_per_decision"]) == (35_462, 35_072)
    assert body["weights_sha256"] == ex.policy_from_state_dict(snapshot_state_dict).sha256
    check = body["verification"]
    print(f"\n[API] {ex.verification_summary(check)}")
    assert check["ok"] and check["states"] == 2_000 and check["argmax_agreement"] == 1.0
    assert check["checks"] == {"q_values": True, "argmax": True, "setpoints": True}
    assert (check["near_tie_disagreements"], check["setpoint_mismatches"], check["tie_tolerance"]) == (0, 0, 0.0)
    assert check["max_abs_q_diff"] < Q_TOL
    fbt, st = work / body["files"]["fbt"], work / body["files"]["st"]
    assert fbt.read_text(encoding="utf-8") == body["code"]
    assert st.is_file()
    assert fbt.parent.parent == work / "data" / "generated_code" / "iec61499"
    # downloads: only the generated files, through /api/code/files/; data/ itself is not served
    for kind, path in (("fbt", fbt), ("st", st)):
        status, headers, data = asyncio.run(asgi_call(server.app, "GET", body["downloads"][kind]))
        assert status == 200 and data == path.read_bytes(), kind
        assert headers["content-disposition"].startswith("attachment"), headers
    stamp = fbt.parent.name
    for url in ("/data/training_logs/wmg_platform.log", f"/data/generated_code/iec61499/{stamp}/DQN_POLICY.fbt",
                "/api/code/files/../wmg_platform.log", f"/api/code/files/{stamp}/missing.fbt",
                f"/api/code/files/{stamp}/eae_build"):
        assert asyncio.run(asgi_call(server.app, "GET", url))[0] == 404, url
    if eae.find_fbtype_compiler() is None:
        assert body["eae_compile"] is None
    else:
        compiled = body["eae_compile"]
        assert compiled["ok"] and not compiled["errors"] and not compiled["warnings"], compiled["errors"]
        initial_values = [v.get("InitialValue") for v in root.iter("VarDeclaration")
                          if v.get("ArraySize") and v.get("InitialValue")]
        assert len(compiled["context_messages"]) == 1 + len(initial_values)
        assert all(any(f"'{iv}' is not correct" in m for m in compiled["context_messages"]) for iv in initial_values)
    # the file the server wrote runs the trained network
    q_fb, out, _ = run_fbt(fbt, eval_states[:500])
    check_fidelity(q_fb, out, torch_q(reference_net(snapshot_state_dict), eval_states[:500]), "API .fbt")


def test_websocket_generate_code_sends_the_fb(api_server):
    server, _ = api_server
    saved = server.config.get("code_generation")
    server.config["code_generation"] = {"eae_check": False}  # merged with the defaults
    try:
        messages = asgi_websocket(server.app, ["generate_code"])
    finally:
        server.config["code_generation"] = saved
    assert [m["type"] for m in messages] == ["system_state", "plc_code"]
    reply = messages[1]
    assert ET.fromstring(reply["code"].encode("utf-8")).get("Name") == "DQN_POLICY"
    assert reply["verification"]["ok"] and reply["verification"]["argmax_agreement"] == 1.0
    assert reply["eae_compile"] is None


@pytest.mark.parametrize("bug", ["argmax", "setpoints"])
def test_server_refuses_an_fb_that_fails_the_gate(api_server, snapshot_state_dict, eval_states, bug):
    """The exporter is made to emit one defect. POST /api/code/generate answers 500 and
    the WebSocket action answers plc_code_error, both naming the failed checks; no
    files stay behind and the last good export stays the current one."""
    server, work = api_server
    main = sys.modules["main"]
    q = torch_q(reference_net(snapshot_state_dict), eval_states[:2_500])
    action = int(np.bincount(q.argmax(axis=1)).argmax())
    real_algorithm = main.iec61499_exporter.req_algorithm  # main.py imports the exporter as a flat module
    out_root = work / "data" / "generated_code" / "iec61499"
    saved_settings, last_good = server.config.get("code_generation"), server.last_code_export
    mp = pytest.MonkeyPatch()
    mp.setattr(main.iec61499_exporter, "req_algorithm",
               lambda *args, **kwargs: inject_bug(real_algorithm(*args, **kwargs), bug, action))
    server.config["code_generation"] = {"eae_check": False}
    try:
        before = sorted(out_root.iterdir()) if out_root.exists() else []
        status, body = asgi_http(server.app, "POST", "/api/code/generate")
        messages = asgi_websocket(server.app, ["generate_code"])
        after = sorted(out_root.iterdir()) if out_root.exists() else []
    finally:
        mp.undo()
        server.config["code_generation"] = saved_settings
    failed = "FAILED: " + ", ".join(name for name, passed in FAILED_CHECKS[bug].items() if not passed)
    print(f"\n[server, {bug} bug] {body['detail'][:220]}")
    assert status == 500 and failed in body["detail"], body
    assert [m["type"] for m in messages] == ["system_state", "plc_code_error"], messages
    assert failed in messages[1]["detail"]
    assert after == before, "a refused FB must not stay on disk"
    assert server.last_code_export is last_good


def test_cors_and_websocket_accept_only_the_dashboard_origin(api_server):
    server, _ = api_server
    for origin, allowed in (("http://127.0.0.1:8000", True), ("http://localhost:8000", True),
                            ("http://evil.example", False), ("http://127.0.0.1:9999", False)):
        status, headers, _ = asyncio.run(asgi_call(server.app, "GET", "/api/system/status",
                                                   [(b"origin", origin.encode())]))
        assert status == 200
        assert headers.get("access-control-allow-origin") == (origin if allowed else None), origin
        assert asgi_websocket_handshake(server.app, origin) == ("accepted" if allowed else 1008), origin
    assert asgi_websocket_handshake(server.app) == "accepted"  # non-browser clients send no Origin


def test_server_starts_in_a_fresh_checkout_and_binds_loopback(tmp_path):
    """No data/ folder yet: main.py creates its log folder before logging to it."""
    for d in ("frontend/css", "frontend/js"):
        (tmp_path / d).mkdir(parents=True)
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import main; "
            "print('host=' + main.WMGDigitalTwinServer().config['server']['host'])")
    run = subprocess.run([sys.executable, "-c", code, str(REPO / "backend")], cwd=tmp_path,
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    assert "host=127.0.0.1" in run.stdout.splitlines()
    assert (tmp_path / "data" / "training_logs" / "wmg_platform.log").is_file()
    assert json.loads((REPO / "config" / "config.json").read_text(encoding="utf-8"))["server"]["host"] == "127.0.0.1"


def test_code_generation_during_training(api_server):
    """Generate while the training loop runs: the export works on a copy of the
    weights in a worker thread, so training keeps going and the FB still matches
    the network it was taken from."""
    main = sys.modules["main"]
    saved = np.random.get_state(), random.getstate(), torch.get_rng_state()
    try:
        server = main.WMGDigitalTwinServer()  # fresh, untrained; same scratch working folder
        server.config["code_generation"] = {"eae_check": False, "verify_states": 1_000}

        async def scenario():
            assert (await asgi_request(server.app, "POST", "/api/training/start"))[0] == 200
            while len(server.rl_agent.memory) <= 1000:  # is_trained() needs > 1000 transitions
                await asyncio.sleep(0.05)
            before = server.current_episode
            result = await asgi_request(server.app, "POST", "/api/code/generate")
            during = server.current_episode - before
            await asgi_request(server.app, "POST", "/api/training/stop")
            return result, during

        (status, body), during = asyncio.run(scenario())
    finally:
        np.random.set_state(saved[0])
        random.setstate(saved[1])
        torch.set_rng_state(saved[2])
    print(f"\n[during training] {during} episodes ran while the FB was generated; {body.get('verification')}")
    assert status == 200, body
    assert body["verification"]["ok"] and body["verification"]["argmax_agreement"] == 1.0
    assert during >= 1, "training stalled while the FB was generated"
    if during >= 2:  # at least one replay() step certainly ran after the weights were copied
        live = ex.policy_from_state_dict(server.rl_agent.q_network.state_dict()).sha256
        assert body["weights_sha256"] != live, "the export should hold the copy, not the live weights"


# ---------------------------------------------------------------------------
# The interpreter itself
# ---------------------------------------------------------------------------
def st_fb(body: str, decls: str, n: int = 1) -> sti.STFunctionBlock:
    return sti.STFunctionBlock(f"FUNCTION_BLOCK T\n{decls}\n{body}\nEND_FUNCTION_BLOCK\n", n=n)


def test_interpreter_lane_masks_loops_and_argmax_tie_break():
    fb = st_fb("""
        ACTION := 0;
        FOR A := 1 TO 2 DO
            IF Q[A] > Q[ACTION] THEN
                ACTION := A;
            END_IF;
        END_FOR;
        IF X > 0.0 THEN Y := 1.0; ELSIF X < 0.0 THEN Y := -1.0; ELSE Y := 0.0; END_IF;
        CASE ACTION OF
            0: Z := 10;
            1, 2: Z := 20;
        ELSE
            Z := 99;
        END_CASE;
        S := 0;
        FOR I := 10 TO 1 BY -3 DO
            S := S + I;
        END_FOR;
    """, """
        VAR_INPUT X : REAL; Q : ARRAY[0..2] OF REAL; END_VAR
        VAR_OUTPUT ACTION : INT; Y : REAL; Z : INT; S : DINT; END_VAR
        VAR A : INT; I : DINT; END_VAR
    """, n=4)
    q = np.array([[1, 1, 0], [0, 2, 2], [3, 1, 3], [0, 0, 0.5]], dtype=np.float32)  # lanes x actions, with ties
    out = fb(X=np.array([1, -1, 0, 2], dtype=np.float32), Q=q.T)
    np.testing.assert_array_equal(sti.lanes(out["ACTION"], 4), torch.argmax(torch.from_numpy(q), dim=1).numpy())
    np.testing.assert_array_equal(sti.lanes(out["Y"], 4), [1, -1, 0, 1])
    np.testing.assert_array_equal(sti.lanes(out["Z"], 4), [10, 20, 10, 20])
    assert out["S"] == 10 + 7 + 4 + 1


def test_interpreter_is_float32_and_strict():
    out = st_fb("R := R + 1.0E-8; L := L + 1.0E-8;", "VAR_OUTPUT R : REAL := 1.0; L : LREAL := 1.0; END_VAR")()
    assert out["R"] == np.float32(1.0)      # 1e-8 is below half an ulp of 1.0 in binary32
    assert out["L"] == 1.0 + 1e-8
    compile_errors = [
        ("VAR_OUTPUT R : REAL; END_VAR VAR I : DINT; END_VAR", "R := I;"),        # INT -> REAL needs a conversion
        ("VAR_OUTPUT N : INT; END_VAR VAR D : DINT; END_VAR", "N := D;"),         # narrowing
        ("VAR_OUTPUT R : REAL; END_VAR", "R := 1;"),                              # integer literal for a REAL
        ("VAR_OUTPUT R : REAL; END_VAR", "R := NOT_DECLARED;"),
        ("VAR_INPUT X : REAL; END_VAR", "X := 1.0;"),                             # inputs are read-only
        ("VAR_OUTPUT R : REAL; END_VAR VAR CONSTANT C : ARRAY[0..1] OF REAL := [1.0, 2.0]; END_VAR", "C[0] := R;"),
        ("VAR CONSTANT C : ARRAY[0..2] OF REAL := [1.0, 2.0]; END_VAR", ""),     # short initialiser
        ("VAR_OUTPUT R : REAL; END_VAR", "R := MAX(R, 0.0)"),                     # missing ';'
        ("VAR_OUTPUT R : REAL; END_VAR", "WHILE R < 1.0 DO R := 1.0; END_WHILE;"),  # unsupported: refused
    ]
    for decls, body in compile_errors:
        with pytest.raises(sti.STError):
            st_fb(body, decls)()
    runtime_errors = [
        ("VAR_OUTPUT R : REAL; END_VAR VAR A : ARRAY[0..1] OF REAL; I : DINT; END_VAR", "I := 2; R := A[I];"),
        ("VAR_OUTPUT N : INT := 32767; END_VAR", "N := N + 1;"),
        ("VAR_OUTPUT N : INT; END_VAR", "FOR N := 32760 TO 32767 DO ; END_FOR;"),  # would wrap on a PLC
    ]
    for decls, body in runtime_errors:
        fb = st_fb(body, decls)
        with pytest.raises(sti.STError):
            fb()


MINI_FBT = """<?xml version="1.0" encoding="UTF-8"?>
<FBType Name="MINI">
  <InterfaceList>
    <EventInputs>
      <Event Name="REQ"><With Var="A" /></Event>
      <Event Name="SET_B"><With Var="B" /></Event>
    </EventInputs>
    <EventOutputs>
      <Event Name="CNF"><With Var="S" /></Event>
    </EventOutputs>
    <InputVars>
      <VarDeclaration Name="A" Type="REAL" />
      <VarDeclaration Name="B" Type="REAL" />
    </InputVars>
    <OutputVars>
      <VarDeclaration Name="S" Type="REAL" />
    </OutputVars>
  </InterfaceList>
  <BasicFB>
    <ECC>
      <ECState Name="START" />
      <ECState Name="REQ"><ECAction Algorithm="REQ" Output="CNF" /></ECState>
      <ECTransition Source="START" Destination="REQ" Condition="REQ" />
      <ECTransition Source="REQ" Destination="START" Condition="1" />
    </ECC>
    <Algorithm Name="REQ"><ST><![CDATA[S := A + B;]]></ST></Algorithm>
  </BasicFB>
</FBType>
"""


def test_interpreter_with_latching_and_ecc():
    fb = sti.BasicFB(MINI_FBT, n=1)
    assert fb.fire("REQ", {"A": 1.0, "B": 5.0}) == [("CNF", {"S": np.float32(1.0)})]  # B is not WITH REQ
    assert fb.fire("SET_B", {"B": 5.0}) == []  # latched; no transition waits for SET_B
    assert fb.fire("REQ", {"A": 1.0}) == [("CNF", {"S": np.float32(6.0)})]
    assert fb.state == "START"
    with pytest.raises(sti.STError):
        fb.fire("REQ", {})  # REQ latches A, so a value must be supplied
