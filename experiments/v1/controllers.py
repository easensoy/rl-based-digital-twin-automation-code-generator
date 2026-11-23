"""
Controllers for benchmark v1. Each maps a batch of observations (n, 11) to preset indices (n,).

  fixed(a)          always preset a
  uniform(seed)     uniform random presets
  rule(cfg)         tariff- and order-aware pacing with a cure-window guard (the non-learning reference)
  dqn(state_dict)   greedy policy of a DQNNetwork in eval mode (first maximum on ties, as torch.argmax)

Deliberately defective controllers for evaluator validation:
  rule(cfg, sensor_blind=True)  ignores the temperature sensor and assumes the oven sits at the current
                                preset's setpoint
  rule(cfg, counter_blind=True) ignores the good-part counter and assumes every commanded part is good
  throughput()                  maximises parts: the fastest preset whose cure window contains the measured
                                temperature, ignoring the order cap and the tariff (it exploits the
                                "throughput" reward specification, which pays for every part)
"""

from __future__ import annotations

import numpy as np
import torch

from process import P0, PRESETS, TH, V, peak_at, t_req

BY_SPEED = np.argsort(V)                     # Energy saving, Conservative, Balanced, Normal, Quality, Aggressive
LO = t_req(V)
HI = LO + P0.window
SP = TH * P0.T_nom
_PEAK = peak_at(np.arange(P0.steps))
_PEAK_LEFT = np.array([_PEAK[t:].sum() for t in range(P0.steps + 1)])
PER_STEP = P0.rate * P0.dt_min               # parts per step at v = 1


def decode(obs):
    T = obs[:, 0] * 30.0 + 170.0
    t = np.clip(np.rint(obs[:, 3] * P0.steps).astype(int), 0, P0.steps)
    D = obs[:, 10] * 1500.0
    return T, t, D, obs[:, 6] * D, obs[:, 9], obs[:, 8], obs[:, 2] * 0.15 + 0.95


def fixed(a: int):
    return lambda obs: np.full(len(obs), a, dtype=int)


def uniform(seed: int):
    rng = np.random.default_rng(seed)
    return lambda obs: rng.integers(0, 6, len(obs))


def _cheapest_at_least(v_target):
    """Slowest preset whose speed meets v_target; the fastest preset if none does."""
    ok = V[BY_SPEED][None, :] >= v_target[:, None] - 1e-9
    first = np.where(ok.any(axis=1), ok.argmax(axis=1), len(V) - 1)
    return BY_SPEED[first]


def rule(cfg: dict, sensor_blind: bool = False, counter_blind: bool = False):
    """cfg: fast (off-peak preset before the peak), margin (relative speed reserve), bridge (avoid scrap during
    transitions by passing through a preset whose cure window contains the current temperature), sat_guard
    (fall back to the slowest preset that can hold its window when the heater saturates and parts scrap)."""
    fast, margin, bridge, sat_guard = cfg["fast"], cfg["margin"], cfg["bridge"], cfg["sat_guard"]
    memory = {}

    def act(obs):
        T, t, D, rem, duty, scrap, theta_prev = decode(obs)
        if sensor_blind:
            T = theta_prev * P0.T_nom
        if counter_blind:                                     # count commanded parts instead of good parts
            memory["made"] = (np.zeros(len(obs)) if t[0] == 0 or "made" not in memory
                              else memory["made"] + PER_STEP * V[memory["last"]])
            rem = D - memory["made"]
        n_peak = _PEAK_LEFT[t]
        n_off = (P0.steps - t) - n_peak
        in_peak = _PEAK[np.minimum(t, P0.steps - 1)]
        after = t >= P0.peak_end
        v_peak = np.maximum(rem - n_off * PER_STEP * V[fast], 0.0) / np.maximum(n_peak * PER_STEP, 1e-9)
        v_after = rem / np.maximum(n_off * PER_STEP, 1e-9)
        v_target = np.where(in_peak, v_peak, np.where(after, v_after, V[fast])) * (1.0 + margin)
        a = _cheapest_at_least(v_target)
        a = np.where(rem <= 0, 3, a)                          # order complete: slowest, cheapest preset
        if bridge:
            out = (T < LO[a]) | (T > HI[a])
            inside = (T[:, None] >= LO[None, :]) & (T[:, None] <= HI[None, :])
            up = inside & (SP[None, :] > T[:, None])          # presets that make good parts while heating
            down = inside & (SP[None, :] < T[:, None])        # ... while cooling
            want_up = T < LO[a]
            choice = np.where(want_up, np.where(up.any(1), np.argmax(np.where(up, SP, -np.inf), 1), a),
                              np.where(down.any(1), np.argmin(np.where(down, SP, np.inf), 1), a))
            a = np.where(out, choice, a)
        if sat_guard:
            hot_enough = (T[:, None] >= LO[None, :]) & (T[:, None] <= HI[None, :])
            fallback = np.where(hot_enough.any(1), np.argmin(np.where(hot_enough, SP, np.inf), 1), 3)
            a = np.where((duty > 0.97) & (scrap > 0.5), fallback, a)
        memory["last"] = a.astype(int)
        return a.astype(int)
    return act


def throughput():
    def act(obs):
        T = obs[:, 0] * 30.0 + 170.0
        inside = (T[:, None] >= LO[None, :]) & (T[:, None] <= HI[None, :])
        return np.where(inside.any(1), np.argmax(np.where(inside, V, -np.inf), 1), 2).astype(int)
    return act


def dqn(state_dict: dict, n_inputs: int = 11):
    from backend.rl_agent import DQNNetwork
    net = DQNNetwork(n_inputs, 6)
    net.load_state_dict(state_dict)
    net.eval()

    def act(obs):
        with torch.no_grad():
            return net(torch.from_numpy(np.asarray(obs, dtype=np.float32))).argmax(1).numpy()
    return act


RULE_GRID = [dict(fast=f, margin=m, bridge=b, sat_guard=g)
             for f in (2, 4) for m in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5) for b in (False, True)
             for g in (False, True)]
assert PRESETS.shape == (6, 3)
