"""
Outcome ledger for benchmark v1, implemented separately from the reward in process.py.

It recomputes every outcome from the logged physical trajectory (true oven temperature, preset, heater power)
and the order quantity, using the process specification's constants but none of the reward code: production,
good and scrap parts, energy and its cost, safety excursions, order shortfall and the operating margin.
For the intended reward, the sum of rewards must equal the ledger margin; tests/test_benchmark_v1.py checks it.
"""

from __future__ import annotations

import numpy as np

from process import P0, PC, V, Params


def outcomes(traj: dict, demand, p: Params = P0) -> dict:
    T, a, P_h = traj["T"], traj["action"].astype(int), traj["P_h"]
    n, steps = T.shape
    v, pc = V[a], PC[a]
    parts = p.rate * v * p.dt_min
    lo = p.T_req0 + p.T_req1 * v
    good = np.where((T >= lo) & (T <= lo + p.window), parts, 0.0)
    kwh = (P_h + p.P_motor * v + p.P_comp * pc) * (p.dt_min / 60.0)
    t = np.arange(steps)
    on_peak = (t >= p.peak_start) & (t < p.peak_end)
    tariff = np.where(on_peak, p.price_peak, p.price_off)
    demand = np.asarray(demand, dtype=float)
    cum = np.cumsum(good, axis=1)
    before = cum - good
    sold = np.clip(demand[:, None] - before, 0.0, good)
    shortfall = np.maximum(demand - cum[:, -1], 0.0)
    excursions = (T > p.T_safe).sum(axis=1)
    revenue = p.part_value * sold.sum(axis=1) + p.salvage * (good - sold).sum(axis=1)
    cost = (kwh * tariff).sum(axis=1)
    margin = revenue - cost - p.safety_cost * excursions - p.shortfall_cost * shortfall
    return {
        "good_parts": cum[:, -1], "scrap_parts": parts.sum(axis=1) - cum[:, -1],
        "scrap_fraction": 1.0 - cum[:, -1] / parts.sum(axis=1), "energy_kwh": kwh.sum(axis=1),
        "peak_energy_kwh": kwh[:, on_peak].sum(axis=1), "energy_cost": cost, "shortfall": shortfall,
        "demand_met": shortfall <= 0.0, "excursion_steps": excursions, "revenue": revenue, "margin": margin,
        "excess_parts": (good - sold).sum(axis=1),
    }
