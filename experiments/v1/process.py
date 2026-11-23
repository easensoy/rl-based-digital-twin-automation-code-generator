"""
Benchmark v1: a heated curing line run for one 8-hour shift under a time-of-use electricity tariff.

A synthetic process model built to ask one control question: can a policy that observes oven temperature,
demand progress, the tariff and heater duty run the shift more cheaply than a tuned rule while meeting the
order? Every parameter below is assumed, not calibrated to a plant, and the model is not a digital twin.
Units: minutes, degC, kW, kWh, EUR, parts. It is isolated from backend/ and from the historical benchmark.

Dynamics, per step of dt = 2 min (240 steps per shift), for each episode (vectorised over episodes):
  preset a -> line speed v, compressor factor p, heater setpoint T_sp = theta * T_nom   (the six presets of
             backend.rl_agent.AutomationEnvironment, reinterpreted for this process)
  heater loop (sees the possibly biased sensor):  P_req = C (T_sp - (T + bias)) / tau + UA (T - T_amb) + P_prod v
  heater power:  P_h = clip(P_req, 0, P_max)          (saturation; no active cooling)
  energy balance:  C (T' - T) = (P_h - UA (T - T_amb) - P_prod v) dt + process noise
  production:  6 v parts per step; a part is good iff T_req(v) <= T' <= T_req(v) + window, T_req(v) = 105 + 55 v
  energy:  E = (P_h + P_motor v + P_comp p) dt / 60 kWh, priced at the off-peak or peak tariff
  safety constraint:  true temperature above T_safe = 205 degC
  objective (EUR):  revenue for good parts up to the order quantity, salvage value above it, minus energy cost,
                    minus a cost per safety-excursion step, minus a shortfall cost per missing part at shift end
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRESETS = np.array([[0.8, 0.9, 0.85], [1.0, 1.0, 1.0], [1.2, 1.1, 1.1],
                    [0.6, 0.8, 0.8], [1.1, 1.05, 0.95], [0.9, 0.95, 0.9]])  # (v, p, theta)
NAMES = ("Conservative", "Normal", "Aggressive", "Energy saving", "Quality focus", "Balanced")
V, PC, TH = PRESETS.T
OBS_NAMES = ("T_meas", "v_prev", "theta_prev", "time", "peak", "to_tariff_change", "demand_left", "pace",
             "scrap_last", "heater_duty", "demand_size")
MEASURED = [0, 6, 7, 8, 9]  # observation entries subject to measurement delay (clock, tariff and preset are not)


@dataclass(frozen=True)
class Params:
    dt_min: float = 2.0
    steps: int = 240
    T_nom: float = 180.0
    C: float = 1000.0          # oven heat capacity, kJ/K
    UA: float = 0.25           # heat-loss coefficient, kW/K
    tau: float = 10.0          # heater-loop time constant, min
    P_max: float = 120.0       # heater power limit, kW
    P_prod: float = 40.0       # heat taken by product at v = 1, kW
    P_motor: float = 10.0      # conveyor at v = 1, kW
    P_comp: float = 8.0        # compressor at p = 1, kW
    rate: float = 3.0          # parts per minute at v = 1
    T_req0: float = 105.0
    T_req1: float = 55.0
    window: float = 40.0       # cure window width above T_req(v), K
    T_safe: float = 205.0
    T_amb: float = 20.0
    price_off: float = 0.10    # EUR/kWh
    price_peak: float = 0.40
    peak_start: int = 90       # steps [90, 180): hours 3-6 of the shift
    peak_end: int = 180
    part_value: float = 0.20   # EUR per good part up to the order quantity
    salvage: float = 0.0       # EUR per good part above it
    shortfall_cost: float = 0.60  # EUR per missing part at shift end
    safety_cost: float = 2.0   # EUR per step above T_safe
    sensor_noise: float = 0.5  # K
    process_noise: float = 0.2  # K per step


P0 = Params()


def t_req(v, p: Params = P0):
    return p.T_req0 + p.T_req1 * v


def peak_at(t, p: Params = P0):
    return (t >= p.peak_start) & (t < p.peak_end)


def price_at(t, p: Params = P0):
    return np.where(peak_at(t, p), p.price_peak, p.price_off)


def steps_to_change(t, p: Params = P0):
    return np.where(t < p.peak_start, p.peak_start - t, np.where(t < p.peak_end, p.peak_end - t, p.steps - t))


@dataclass
class Scenario:
    """Arrays of shape (n,): one entry per episode. Noise arrays (n, steps) give common random numbers."""
    seed: np.ndarray
    demand: np.ndarray
    T0: np.ndarray
    lag: np.ndarray           # multiplies C and tau (slower or faster thermal response)
    cold_dT: np.ndarray
    cold_on: np.ndarray
    cold_len: np.ndarray
    bias: np.ndarray          # sensor offset, K, from bias_on to the end
    bias_on: np.ndarray
    sat: np.ndarray           # P_max multiplier during the saturation window
    sat_on: np.ndarray
    sat_len: np.ndarray
    delay: np.ndarray         # measurement delay, steps
    noise_meas: np.ndarray
    noise_proc: np.ndarray

    def __len__(self):
        return len(self.seed)

    def subset(self, idx):
        return Scenario(**{k: v[idx] for k, v in self.__dict__.items()})


def _draw(rng, spec, key, default):
    v = spec.get(key, default)
    if isinstance(v, tuple):
        lo, hi = v
        return rng.integers(lo, hi + 1) if isinstance(lo, int) and isinstance(hi, int) else rng.uniform(lo, hi)
    return v


def make_scenarios(seeds, spec: dict, p: Params = P0) -> Scenario:
    """One independent RNG per episode seed, so an episode is the same whichever set it is drawn in."""
    cols = {k: [] for k in ("demand", "T0", "lag", "cold_dT", "cold_on", "cold_len", "bias", "bias_on", "sat",
                            "sat_on", "sat_len", "delay", "noise_meas", "noise_proc")}
    for s in seeds:
        rng = np.random.default_rng(int(s))
        cols["demand"].append(_draw(rng, spec, "demand", (1200, 1500)))
        cols["T0"].append(_draw(rng, spec, "T0", (150.0, 185.0)))
        cols["lag"].append(_draw(rng, spec, "lag", (0.8, 1.25)))
        cols["cold_dT"].append(_draw(rng, spec, "cold_dT", (-5.0, 0.0)))
        onset = _draw(rng, spec, "onset", (30, 150))
        cols["cold_on"].append(onset)
        cols["cold_len"].append(_draw(rng, spec, "cold_len", 60))
        cols["bias"].append(_draw(rng, spec, "bias", 0.0))
        cols["bias_on"].append(onset)
        cols["sat"].append(_draw(rng, spec, "sat", 1.0))
        cols["sat_on"].append(onset)
        cols["sat_len"].append(_draw(rng, spec, "sat_len", 60))
        cols["delay"].append(_draw(rng, spec, "delay", 0))
        cols["noise_meas"].append(rng.normal(0.0, p.sensor_noise, p.steps + 1))
        cols["noise_proc"].append(rng.normal(0.0, p.process_noise, p.steps))
    out = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    for k in ("demand", "cold_on", "cold_len", "bias_on", "sat_on", "sat_len", "delay"):
        out[k] = out[k].astype(int)
    return Scenario(seed=np.asarray(seeds, dtype=np.int64), **out)


class VecProcess:
    """Benchmark v1, vectorised over the episodes of a Scenario. reward:
      "intended"     the objective above; the shortfall cost is charged once, at the last step
      "incremental"  the same objective with the shortfall cost settled as the order is filled: each part sold
                     against the order also earns shortfall_cost, and nothing is charged at the end. Per episode
                     the return equals the intended return plus shortfall_cost * demand, a constant that does not
                     depend on the actions, so both rank action sequences identically (undiscounted)
      "throughput"   a deliberately misspecified variant: every part earns part_value, scrap included, with no
                     order cap and no shortfall cost"""

    def __init__(self, sc: Scenario, p: Params = P0, reward: str = "intended", log: bool = False):
        assert reward in ("intended", "incremental", "throughput")
        self.sc, self.p, self.n, self.reward_kind, self.log = sc, p, len(sc), reward, log
        self.reset()

    def reset(self):
        p, sc = self.p, self.sc
        self.t = 0
        self.T = sc.T0.astype(float).copy()
        self.prev = np.full(self.n, 1)
        self.good = np.zeros(self.n)
        self.scrap_last = np.zeros(self.n)
        self.duty = np.zeros(self.n)
        self.hist = []
        self.logs = {k: [] for k in ("T", "T_meas", "action", "P_h", "P_max", "T_amb", "reward")} if self.log else None
        return self.observe()

    def _window(self, start, length):
        return (self.t >= start) & (self.t < start + length)

    def _measured(self):
        sc, t = self.sc, min(self.t, self.p.steps)
        return self.T + sc.bias * (self.t >= sc.bias_on) + sc.noise_meas[:, t]

    def observe(self):
        p, sc, t = self.p, self.sc, self.t
        o = np.stack([(self._measured() - 170.0) / 30.0, (V[self.prev] - 0.9) / 0.3, (TH[self.prev] - 0.95) / 0.15,
                      np.full(self.n, t / p.steps), np.full(self.n, float(peak_at(t, p))),
                      np.full(self.n, steps_to_change(t, p) / p.steps), (sc.demand - self.good) / sc.demand,
                      (self.good - sc.demand * t / p.steps) / sc.demand, self.scrap_last, self.duty,
                      sc.demand / 1500.0], axis=1)
        self.hist.append(o)
        if np.any(sc.delay > 0):
            past = np.stack([self.hist[max(0, len(self.hist) - 1 - d)][i] for i, d in enumerate(sc.delay)])
            o = o.copy()
            o[:, MEASURED] = past[:, MEASURED]
        return o.astype(np.float32)

    def step(self, a):
        p, sc, t = self.p, self.sc, self.t
        a = np.asarray(a, dtype=int)
        v, pc, th = V[a], PC[a], TH[a]
        lag = sc.lag
        C, tau = p.C * lag, p.tau * 60.0 * lag
        T_amb = p.T_amb + sc.cold_dT * self._window(sc.cold_on, sc.cold_len)
        P_max = p.P_max * np.where(self._window(sc.sat_on, sc.sat_len), sc.sat, 1.0)
        T_loop = self.T + sc.bias * (t >= sc.bias_on)
        loss, load = p.UA * (self.T - T_amb), p.P_prod * v
        P_h = np.clip(C * (th * p.T_nom - T_loop) / tau + loss + load, 0.0, P_max)
        self.T = self.T + (P_h - loss - load) / C * (p.dt_min * 60.0) + sc.noise_proc[:, t]
        lo = t_req(v, p)
        parts = p.rate * v * p.dt_min
        good = parts * ((self.T >= lo) & (self.T <= lo + p.window))
        energy = (P_h + p.P_motor * v + p.P_comp * pc) * p.dt_min / 60.0
        cost = energy * price_at(t, p)
        excursion = self.T > p.T_safe
        sold = np.minimum(good, np.maximum(sc.demand - self.good, 0.0))
        self.good = self.good + good
        done = t == p.steps - 1
        if self.reward_kind in ("intended", "incremental"):
            r = p.part_value * sold + p.salvage * (good - sold) - cost - p.safety_cost * excursion
            if self.reward_kind == "incremental":
                r = r + p.shortfall_cost * sold
            elif done:
                r = r - p.shortfall_cost * np.maximum(sc.demand - self.good, 0.0)
        else:
            r = p.part_value * parts - cost - p.safety_cost * excursion
        if self.log:
            for k, val in (("T", self.T), ("T_meas", self._measured()), ("action", a), ("P_h", P_h),
                           ("P_max", P_max), ("T_amb", T_amb), ("reward", r)):
                self.logs[k].append(np.asarray(val, dtype=float).copy())
        self.scrap_last = 1.0 - good / parts
        self.duty = P_h / P_max
        self.prev = a
        self.t += 1
        return self.observe(), r, done

    def trajectory(self):
        return {k: np.stack(v, axis=1) for k, v in self.logs.items()}
