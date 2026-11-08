"""
Production model of AutomationEnvironment (backend/rl_agent.py).

The simulator in commit 898c112 adds 1/speed to the cycle counter on every step, so slower
settings finish parts faster. That "legacy" behaviour stays the default so the published
results remain reproducible. "speed_proportional" is the corrected model: a part takes
cycle_time_target / speed steps.

Part counts depend only on the action's speed factor, not on the process noise, so these
tests are deterministic.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from rl_agent import PRODUCTION_MODELS, AutomationEnvironment  # noqa: E402

BASE = {"cycle_time_target": 30.0, "throughput_target": 120.0,
        "safety_threshold": 0.95, "energy_efficiency_target": 0.85}
CONSERVATIVE, NORMAL, AGGRESSIVE, ENERGY_SAVING, QUALITY, BALANCED = range(6)
SPEED = {CONSERVATIVE: 0.8, NORMAL: 1.0, AGGRESSIVE: 1.2,
         ENERGY_SAVING: 0.6, QUALITY: 1.1, BALANCED: 0.9}


def parts_after(model, action, steps=200):
    np.random.seed(0)
    cfg = dict(BASE) if model is None else {**BASE, "production_model": model}
    env = AutomationEnvironment(cfg)
    env.reset()
    for _ in range(steps):
        env.step(action)
    return env.production_count


def test_models_are_listed():
    assert PRODUCTION_MODELS == ("legacy", "speed_proportional")


def test_default_is_legacy():
    assert AutomationEnvironment(dict(BASE)).production_model == "legacy"
    for action in SPEED:
        assert parts_after(None, action) == parts_after("legacy", action)


def test_legacy_rewards_slower_settings_with_more_parts():
    # Documents the defect: 0.6x speed makes the most parts, 1.2x the fewest.
    assert parts_after("legacy", ENERGY_SAVING) == 11
    assert parts_after("legacy", NORMAL) == 6
    assert parts_after("legacy", AGGRESSIVE) == 5


def test_speed_proportional_orders_parts_by_speed():
    by_speed = sorted(SPEED, key=SPEED.get)
    counts = [parts_after("speed_proportional", a) for a in by_speed]
    assert counts == sorted(counts), dict(zip(by_speed, counts))
    assert parts_after("speed_proportional", AGGRESSIVE) > parts_after("speed_proportional", NORMAL) \
        > parts_after("speed_proportional", ENERGY_SAVING)


def test_speed_proportional_part_time_matches_target():
    # At 1.0x speed a part takes cycle_time_target steps under both models.
    assert parts_after("speed_proportional", NORMAL, steps=200) == 200 // 30


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="production_model"):
        AutomationEnvironment({**BASE, "production_model": "faster_is_slower"})
