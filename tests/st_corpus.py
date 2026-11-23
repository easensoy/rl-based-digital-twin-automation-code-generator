"""
Deterministic corpus for standalone IEC 61131-3 (.st) verification.

Shared by tests/test_st_standalone.py and docs/scripts/measure_st_verification.py.

Networks (14):
  untrained_seed0..4   DQNNetwork(12, 6) at torch.manual_seed(0..4); the selected action varies with the state
  dominant_action0..5  untrained_seed0 with +100 on one output bias, so that action is selected on every state
  tie_i_j              untrained_seed0 with output row j := row i and +100 on both, so actions i and j tie
                       exactly on every state and the first maximum, i, must be selected
The offset of 100 is about 7x the largest spread of untrained_seed0's six Q-values over the corpus (14.7, at
the +-100 inputs; below 0.4 on every other state).

States (11,128 per network, float32): the 4,096 vertices of [-2, 2]^12; 32 boundary vectors (+0, -0,
+-1e-40 subnormal, +-2 e_k for each k, +-10 and +-100 in every component, outside the trained range);
5,000 uniform draws in [-2, 2]^12; 2,000 states of simulator rollouts under uniform random actions.
Random parts are seeded with STATE_SEED.

Reference: reference_q, a float64 forward pass of the same weights (eval mode, dropout off). It does
not depend on the BLAS kernel's float32 accumulation order, and it computes identical output rows
identically, so a tie network's two Q-values are exactly equal in the reference, as they are in exact
arithmetic.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from backend import iec61499_exporter as ex
from backend.rl_agent import DQNNetwork

STATE_SEED = 20261109
TIE_PAIRS = ((0, 1), (2, 4), (3, 5))
OFFSET = 100.0
N_UNIFORM, N_ROLLOUT = 5_000, 2_000
ARGMAX_LINE = "IF Q[A] > Q[ACTION] THEN"


def _untrained(seed: int) -> dict:
    saved = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        net = DQNNetwork(12, 6)
    finally:
        torch.set_rng_state(saved)
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def networks():
    """[(id, kind, state_dict)] in a fixed order."""
    out = [(f"untrained_seed{s}", "untrained", _untrained(s)) for s in range(5)]
    base = _untrained(0)
    for a in range(6):
        sd = {k: v.clone() for k, v in base.items()}
        sd["fc4.bias"][a] += OFFSET
        out.append((f"dominant_action{a}", "dominant", sd))
    for i, j in TIE_PAIRS:
        sd = {k: v.clone() for k, v in base.items()}
        sd["fc4.weight"][j] = sd["fc4.weight"][i]
        sd["fc4.bias"][j] = sd["fc4.bias"][i]
        sd["fc4.bias"][i] += OFFSET
        sd["fc4.bias"][j] += OFFSET
        out.append((f"tie_{i}_{j}", "tie", sd))
    return out


def states() -> np.ndarray:
    vertices = np.array(list(itertools.product((-2.0, 2.0), repeat=12)))
    eye = np.eye(12)
    boundary = np.vstack([np.zeros(12), -np.zeros(12), np.full(12, 1e-40), np.full(12, -1e-40),
                          2 * eye, -2 * eye, np.full(12, 10.0), np.full(12, -10.0),
                          np.full(12, 100.0), np.full(12, -100.0)])
    uniform = np.random.default_rng(STATE_SEED).uniform(-2.0, 2.0, size=(N_UNIFORM, 12))
    rollout = ex.rollout_states(N_ROLLOUT, seed=STATE_SEED)
    return np.concatenate([vertices, boundary, uniform, rollout]).astype(np.float32)


def segments() -> dict:
    """Index ranges of the parts of states()."""
    n_v, n_b = 2 ** 12, 32
    return {"vertices": (0, n_v), "boundary": (n_v, n_v + n_b), "uniform": (n_v + n_b, n_v + n_b + N_UNIFORM),
            "rollout": (n_v + n_b + N_UNIFORM, n_v + n_b + N_UNIFORM + N_ROLLOUT)}


def reference_q(state_dict: dict, x) -> np.ndarray:
    """float64 forward pass. The output layer sums each row's products with numpy's pairwise
    summation, so identical rows give identical values."""
    w = [state_dict[f"fc{k}.weight"].double().numpy() for k in range(1, 5)]
    b = [state_dict[f"fc{k}.bias"].double().numpy() for k in range(1, 5)]
    h = np.asarray(x, dtype=np.float64)
    for k in range(3):
        h = np.maximum(h @ w[k].T + b[k], 0.0)
    return (h[:, None, :] * w[3][None, :, :]).sum(axis=2) + b[3]


def recorded(q_ref: np.ndarray):
    """A precomputed reference as the ``net`` argument of verify_st / verify_fbt."""
    return lambda _x: torch.from_numpy(q_ref)


def mutate(text: str, kind: str, action: int = 0) -> str:
    """One deliberate defect in generated ST: "argmin" (argmax becomes argmin), "last_max" (ties go to the
    last maximum instead of the first), or "setpoint" (SPEED_SP of ``action``'s branch is changed)."""
    if kind == "argmin":
        old, new = ARGMAX_LINE, ARGMAX_LINE.replace(">", "<")
    elif kind == "last_max":
        old, new = ARGMAX_LINE, ARGMAX_LINE.replace(">", ">=")
    elif kind == "setpoint":
        label, speed = ex.ACTION_SETPOINTS[action][:2]
        branch = f"    {action}: (* {label} *)\n        SPEED_SP := "
        old, new = branch + f"{ex.real_literal(speed)};", branch + f"{ex.real_literal(speed + 0.25)};"
    else:
        raise ValueError(kind)
    assert text.count(old) == 1, (kind, action)
    return text.replace(old, new)
