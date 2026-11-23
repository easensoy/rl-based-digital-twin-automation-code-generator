"""
Scenario partitions of benchmark v1. Frozen in docs/experiments/PROTOCOL_v1.md before the confirmatory runs.

Seed ranges are disjoint, and each episode has its own RNG (process.make_scenarios), so a scenario is
identical for every controller evaluated on it (paired comparisons with common random numbers).
"""

from __future__ import annotations

from process import make_scenarios

# Training distribution (also the development and held-out nominal distribution): order 1,200-1,500 parts,
# initial temperature 150-185 degC, thermal-lag factor 0.8-1.25, a mild cold front of 0 to -5 K for 60 steps
# starting at step 30-150; no sensor bias, no saturation, no measurement delay.
TRAIN_SPEC: dict = {}


def train_seeds(run_seed: int, episodes: int):
    return [10_000_000 + run_seed * 100_000 + e for e in range(episodes)]


DEV = list(range(1_000_000, 1_000_100))        # development: tuning, baseline selection, pilot, calibration
HELDOUT = list(range(2_000_000, 2_000_100))    # held-out nominal: the primary comparison

MAP_DT = (0.0, -5.0, -10.0, -15.0, -20.0)      # cold-front magnitude, K
MAP_LAG = (0.5, 0.8, 1.0, 1.25, 2.0)           # thermal-lag factor
MAP_EPISODES = 50


def in_dev_region(dT: float, lag: float) -> bool:
    return dT >= -5.0 and 0.8 <= lag <= 1.25


def map_cell(i: int, j: int, n: int = MAP_EPISODES):
    seeds = [3_000_000 + (i * len(MAP_LAG) + j) * 1000 + k for k in range(n)]
    return make_scenarios(seeds, {"cold_dT": MAP_DT[i], "lag": MAP_LAG[j]})


# Fault families. Onset is drawn uniformly in steps 30-150; durations are 60 steps unless stated.
FAULTS = {
    "sensor_bias_low": {"bias": -10.0},          # sensor reads 10 K low from onset to shift end
    "sensor_bias_high": {"bias": 10.0},          # sensor reads 10 K high
    "saturation": {"sat": 0.6},                  # heater limited to 60 % for 60 steps
    "delay": {"delay": 5},                       # measurements 5 steps (10 min) old for the whole shift
    "cold_saturation": {"cold_dT": -15.0, "sat": 0.6},   # composed: cold front and saturation together
    "bias_lag": {"bias": -10.0, "lag": 1.5},     # composed: low-reading sensor and slower thermal response
}
FAULT_EPISODES = 100


def fault_family(name: str, split: str = "heldout", n: int = FAULT_EPISODES):
    base = {"heldout": 4_000_000, "dev": 5_000_000}[split] + list(FAULTS).index(name) * 1000
    return make_scenarios([base + k for k in range(n)], FAULTS[name])


def dev():
    return make_scenarios(DEV, TRAIN_SPEC)


def heldout():
    return make_scenarios(HELDOUT, TRAIN_SPEC)
