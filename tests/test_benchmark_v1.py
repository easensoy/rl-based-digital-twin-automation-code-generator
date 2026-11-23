"""
Benchmark v1 (experiments/v1): energy conservation, response directions, ledger against reward, reward
accounting, scenario determinism, disjoint partitions, graders, and the one-change-per-condition design.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

V1 = Path(__file__).resolve().parents[1] / "experiments" / "v1"
sys.path.insert(0, str(V1))

import controllers as C  # noqa: E402
import scenarios as S  # noqa: E402
from evaluate import GRADERS, run  # noqa: E402
from process import P0, TH, V, VecProcess, make_scenarios  # noqa: E402

LOCKED = dict(fast=4, margin=0.3, bridge=True, sat_guard=False)


def quiet(seeds, **spec):
    sc = make_scenarios(list(seeds), spec)
    sc.noise_meas[:] = 0.0
    sc.noise_proc[:] = 0.0
    return sc


def trajectory(sc, a):
    env = VecProcess(sc, log=True)
    env.reset()
    for t in range(P0.steps):
        env.step(np.full(len(sc), a(t) if callable(a) else a))
    return env.trajectory()


def test_energy_balance_closes_without_process_noise():
    sc = quiet(range(7, 17), cold_dT=-15.0, sat=0.6, lag=(0.5, 2.0))
    tr = trajectory(sc, lambda t: t % 6)
    T = np.concatenate([sc.T0[:, None], tr["T"]], axis=1)
    a = tr["action"].astype(int)
    stored = P0.C * sc.lag[:, None] * np.diff(T, axis=1)                      # kJ per step
    flows = (tr["P_h"] - P0.UA * (T[:, :-1] - tr["T_amb"]) - P0.P_prod * V[a]) * P0.dt_min * 60.0
    np.testing.assert_allclose(stored, flows, rtol=0, atol=1e-6)
    assert np.all(tr["P_h"] >= 0.0) and np.all(tr["P_h"] <= tr["P_max"] + 1e-12)
    assert np.any(tr["P_h"] == tr["P_max"])                                   # saturation is exercised


def test_response_directions():
    base = quiet(range(300, 320))
    # a higher heater setpoint gives a hotter oven at steady state
    temps = [trajectory(base, a)["T"][:, 120:].mean() for a in np.argsort(TH)]
    assert np.all(np.diff(temps) > 0)
    # a colder ambient costs heater energy while the heater has headroom ...
    cold = quiet(range(300, 320), cold_dT=-20.0)
    assert np.all(trajectory(cold, 1)["P_h"].sum(1) > trajectory(base, 1)["P_h"].sum(1))
    # ... and cools the oven once the heater is saturated
    cold_sat = quiet(range(300, 320), cold_dT=-20.0, sat=0.6)
    assert np.all(trajectory(cold_sat, 1)["T"].min(1) < trajectory(cold, 1)["T"].min(1) - 1.0)
    # a sensor reading 10 K high makes the loop hold the true temperature about 10 K low
    high = quiet(range(300, 320), bias=10.0)
    drop = trajectory(base, 1)["T"][:, -40:].mean(1) - trajectory(high, 1)["T"][:, -40:].mean(1)
    assert np.all((drop > 9.0) & (drop < 11.0))
    # a slower thermal response heats more slowly from a cold start
    slow, fast = quiet(range(300, 320), T0=150.0, lag=2.0), quiet(range(300, 320), T0=150.0, lag=0.5)
    assert np.all(trajectory(slow, 2)["T"][:, 5] < trajectory(fast, 2)["T"][:, 5])
    # faster presets make more parts per shift
    out = {a: run(C.fixed(a), base) for a in (3, 2)}
    made = {a: out[a]["good_parts"] + out[a]["scrap_parts"] for a in out}
    np.testing.assert_allclose(made[2] / made[3], V[2] / V[3])


@pytest.mark.parametrize("name", ["aggressive", "uniform", "locked_rule", "throughput"])
def test_ledger_margin_equals_intended_return(name):
    sc = make_scenarios(range(100, 140), {"sat": 0.6, "bias": -10.0})
    ctrl = {"aggressive": C.fixed(2), "uniform": C.uniform(3), "locked_rule": C.rule(LOCKED),
            "throughput": C.throughput()}[name]
    out = run(ctrl, sc)
    np.testing.assert_allclose(out["margin"], out["reward_intended"], rtol=0, atol=1e-8)


def test_incremental_accounting_shifts_each_return_by_a_constant():
    sc = make_scenarios(range(200, 230), {"sat": 0.6})
    for make in (lambda: C.fixed(0), lambda: C.fixed(3), lambda: C.uniform(1), lambda: C.rule(LOCKED)):
        totals = {}
        for kind in ("intended", "incremental"):
            ctrl, env = make(), VecProcess(sc, reward=kind)
            obs, tot = env.reset(), np.zeros(len(sc))
            for _ in range(P0.steps):
                obs, r, _ = env.step(ctrl(obs))
                tot += r
            totals[kind] = tot
        np.testing.assert_allclose(totals["incremental"] - totals["intended"], P0.shortfall_cost * sc.demand,
                                   rtol=0, atol=1e-8)


def test_scenarios_are_deterministic_and_independent_of_the_set():
    spec = {"bias": (-3.0, 3.0), "delay": 2}
    a, b, c = make_scenarios([5, 6, 7], spec), make_scenarios([7], spec), make_scenarios([5, 6, 7], spec)
    for k in a.__dict__:
        np.testing.assert_array_equal(getattr(a, k)[2], getattr(b, k)[0])
        np.testing.assert_array_equal(getattr(a, k), getattr(c, k))


def test_partitions_are_disjoint():
    sets = {"train": {s for r in (*range(1, 7), 101) for s in S.train_seeds(r, 1000)},
            "dev": set(S.DEV), "heldout": set(S.HELDOUT),
            "map": {int(s) for i in range(5) for j in range(5) for s in S.map_cell(i, j).seed},
            "faults_heldout": {int(s) for f in S.FAULTS for s in S.fault_family(f, "heldout").seed},
            "faults_dev": {int(s) for f in S.FAULTS for s in S.fault_family(f, "dev").seed}}
    names = list(sets)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            assert not sets[x] & sets[y], (x, y)
    assert len(sets["map"]) == 25 * S.MAP_EPISODES and len(sets["faults_heldout"]) == 6 * S.FAULT_EPISODES


def test_graders():
    o = dict(excursion_steps=np.array([0, 1, 0]), shortfall=np.array([0.0, 13.9, 14.5]),
             demand=np.full(3, 1400.0), scrap_fraction=np.array([0.049, 0.051, 0.0]))
    assert GRADERS["safety"](o).tolist() == [True, False, True]
    assert GRADERS["delivery"](o).tolist() == [True, True, False]
    assert GRADERS["quality"](o).tolist() == [True, False, True]


def test_each_condition_changes_one_thing_relative_to_its_control():
    from train_dqn import CONDITIONS
    controls = {"I1": ("I0", "eval_fix"), "I2": ("I0", "update_every"), "I4": ("I3", "reward"),
                "I5": ("I4", "double"), "R1": ("I3", "reward")}
    for cond, (ctrl, key) in controls.items():
        diff = {k for k in CONDITIONS[cond] if CONDITIONS[cond][k] != CONDITIONS[ctrl][k]}
        assert diff == {key}, (cond, diff)
    assert {k for k in CONDITIONS["I3"] if CONDITIONS["I3"][k] != CONDITIONS["I0"][k]} == {"eval_fix", "update_every"}
