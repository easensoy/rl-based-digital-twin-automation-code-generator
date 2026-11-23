# Evaluation-driven policy improvement: experimental report (benchmark v1)

This report covers the increment that follows the export-verification work. The question is whether a
specified change to the repository's DQN training procedure improves held-out control performance over the
original procedure and over strong non-learning baselines without worsening constraint metrics. The
confirmatory design was frozen in [PROTOCOL_v1.md](PROTOCOL_v1.md) before any confirmatory run.

**Result: policy quality did not improve under the declared criteria.** On 100 held-out episodes, the
proposed procedure I5 earned 88.0 ± 8.7 EUR per shift and the original procedure I0 44.1 ± 92.4 (seed mean ±
95 % t half-width). The difference, +43.9 EUR, has a 95 % interval of [−53.4, +141.2], so H1 is not supported.
I5 meets the feasibility criteria and I0 does not, but the original procedure's seed-to-seed spread (SD 93
EUR) leaves the margin comparison inconclusive. I5 stays 41.6 EUR [32.9, 50.3] below the locked rule (H2) and
level with the best fixed preset: −5.6 [−14.3, +3.2] (H3). Among the single changes, more gradient updates
help most, though without a conclusive interval. The incremental accounting traded 11.9 EUR [1.6, 22.2] of
margin for meeting the order, and Double DQN changed nothing measurable. The exported function blocks
reproduce PyTorch's action at every state the historical networks visit (output F).

## 1. Methodological references

Four Anthropic publications served as methodological references. None of them concerns industrial control,
and nothing here suggests that Anthropic uses this simulator or endorses this repository. Bloom and Petri
simulate language-model interactions. They were not installed, and no part of this benchmark is simulated
by a language model.

| Source | What it demonstrates | Principle carried over | What does not carry over |
|---|---|---|---|
| [Demystifying evals for AI agents](https://www.anthropic.com/research/demystifying-evals-for-ai-agents) | Evaluating agents with repeated trials, outcome checks, inspection of individual trajectories, and separate capability and regression suites | Outcomes are graded by deterministic code on an independent ledger, never by the training reward; many paired episodes per condition; a fixed regression suite (the export checks) kept apart from the capability question (policy quality); selected trajectories read in full (output D) | Model-based and human grading, which a numeric process outcome does not need |
| [Bloom](https://www.anthropic.com/research/bloom) | Generating targeted behavioural scenarios at scale, reporting how often a behaviour occurs with error bars over repeated runs, and checking the evaluator against deliberately altered models | Seeded fault families aimed at specific failure modes, reported as pass rates; deliberately defective controllers that the graders must catch before the confirmatory suite is frozen (PROTOCOL section 5) | Scenario generation and judging by a language model: here a scenario is a parameter draw of a numeric model, and a verdict is a threshold test |
| [Petri 2.0](https://alignment.anthropic.com/2026/petri-v2/) | Multi-turn simulated interactions, with controlled ablations to show which intervention changes the measured behaviour | Multi-step disturbance scenarios with specified onset, duration and magnitude, single and composed; one-change-at-a-time ablations of the training procedure | Auditor and judge agents and simulated tool environments |
| [Reward tampering](https://www.anthropic.com/research/reward-tampering) | In controlled environments, optimisation can exploit the gap between what is rewarded and what is intended | Every policy is scored on outcomes computed independently of the reward it was trained on; a deliberately misspecified reward (R1) is trained and audited against those outcomes; exploitation is distinguished from poor optimisation | The language-model curriculum and the tampering mechanism itself: a policy here cannot reach its reward code |

## 2. Phase 1: why the historical environment cannot answer the question

The audit (`experiments/v1/audit_v0.py`, output `experiments/v1/results/v0_audit.json`) evaluated the
historical environment `backend.rl_agent.AutomationEnvironment` on its 100 published evaluation seeds
(10000-10099, treated here as development and historical evidence). It covered both production models,
the six fixed actions, uniform random actions, the 12 multi-seed checkpoints and the server snapshot. It
also recomputed each reward term at every step and asserted that the terms sum to the environment's reward.

| Audit item | Finding |
|---|---|
| Frozen checkpoints | Every one of the 13 checkpoints selects a single action at all 20,000 states it visits in the 100 episodes (action shares recorded by `boundary_audit.py`). Its mean reward equals that action's fixed-action reward exactly (for example, legacy_seed1 1,682.58, the same as always Quality focus) |
| Fixed action vs schedule | Energy saving is the best fixed action under both production models (1,827.4 legacy, 1,771.6 speed-proportional). The best of 210 two-segment schedules beats it by 6.6 (0.36 %) and 9.0 (0.51 %), and the paired difference has the same value in every episode (95 % half-width about 1e-13): the disturbances do not interact with the actions |
| Reward composition | Per step, the quality term (4.6-4.8) and a constant cycle bonus (+2.0, paid on every step) dominate. The production term contributes 0.10-0.38, and the safety and vibration terms never fire |
| Cumulative vs incremental | The production and energy terms are running averages over the episode so far (`production_count / (steps / 100)`, `1 / (1 + total_energy / steps)`), so a step's reward is not the incremental value of its action |
| Time and production accounting | Under the default legacy model a part completes after `cycle_time += 1 / speed`, so slower settings count more parts: always Energy saving counts 11 parts and always Aggressive 5, against 4.0 and 8.0 in physical speed-time |
| Observation sufficiency | The 12-dimensional state carries no time, no running totals and no order, so the reward is not a function of the observation |
| Delayed trade-offs | None: there is no order, tariff or inventory, and constraint costs are never incurred |
| Constraint reachability | No safety or vibration event in any audited episode, and the `safety_violations > 5` termination is never reached |
| Dropout and train/eval mode | `DQNNetwork` applies Dropout(0.2), and neither `act()` nor the target network ever switches to eval mode, so greedy actions and bootstrap targets are both stochastic |
| Target network | Copied every 10 gradient updates |
| Update-to-interaction ratio | `backend/main.py` calls `replay()` once per 200-step episode: one minibatch update per 200 interactions, about 1,000 over a 1,000-episode run |
| Random-number coupling | Environment noise, the agent's epsilon test and the replay sampling share the global `numpy` and `random` streams, so exploration changes the disturbance sequence an episode sees |

Conclusion: a fixed action essentially solves the historical task. Any learning curve on it measures how
quickly a network collapses onto one action, not control quality, and the environment cannot pose the
question above. Tuning the DQN on it would be meaningless, so a new benchmark version was built for one
stated control question, isolated from `backend/`, from the historical experiments and from the
application.

## 3. Benchmark v1: a heated curing line under a time-of-use tariff

Control question: can a supervisory policy that observes oven temperature, order progress, the tariff and
heater duty run an 8-hour shift more cheaply than a tuned rule while meeting the order? The model is
synthetic. Every parameter is assumed, not identified from a plant, and the model is **not a validated
digital twin**. Code: [`experiments/v1/process.py`](../../experiments/v1/process.py).

**Time step and horizon.** dt = 2 min; one shift is 240 steps. The peak tariff applies to steps 90-179
(hours 3-6).

**Actions.** The six presets of the historical environment, reinterpreted as (line speed v, compressor factor
p, heater setpoint θ·180 °C): Conservative (0.8, 0.9, 153 °C), Normal (1.0, 1.0, 180 °C), Aggressive (1.2,
1.1, 198 °C), Energy saving (0.6, 0.8, 144 °C), Quality focus (1.1, 1.05, 171 °C), Balanced (0.9, 0.95,
162 °C). A preset at speed v makes 6v parts per step, which is 1,440v per shift at constant speed against
orders of 1,200-1,500 parts.

**Dynamics** (per episode, vectorised over episodes):

- heater loop, acting on the possibly biased sensor: P_req = C·(T_sp − (T + b))/τ + UA·(T − T_amb) + P_prod·v
- heater power: P_h = clip(P_req, 0, P_max·s(t)); there is no active cooling
- energy balance: C·(T′ − T) = (P_h − UA·(T − T_amb) − P_prod·v)·dt + process noise
- quality: a part is good iff T_req(v) ≤ T′ ≤ T_req(v) + 40 K, with T_req(v) = 105 + 55·v °C
- energy: E = (P_h + P_motor·v + P_comp·p)·dt/60 kWh, priced at 0.10 or 0.40 EUR/kWh
- safety constraint: true temperature above 205 °C
- objective (EUR per shift): 0.20 per good part up to the order, minus energy cost, minus 2.0 per step above
  205 °C, minus 0.60 per missing part at the end of the shift

**Parameters** (assumed): C = 1,000 kJ/K and τ = 10 min, both scaled by the thermal-lag factor; UA = 0.25 kW/K;
P_max = 120 kW; P_prod = 40 kW; P_motor = 10 kW; P_comp = 8 kW; T_amb = 20 °C; sensor noise 0.5 K; process
noise 0.2 K per step.

**Observation** (11 values, scaled to about unit range): measured temperature, previous preset's speed and
setpoint, time of shift, peak flag, time to the next tariff change, fraction of the order remaining, pace
against a uniform schedule, scrap fraction of the last step, heater duty, order size.

**Disturbances and initial conditions.** Order 1,200-1,500 parts; initial temperature 150-185 °C;
thermal-lag factor 0.8-1.25; cold front 0 to −5 K for 60 steps with onset in steps 30-150 (nominal), plus the
fault families and the generalisation map of PROTOCOL section 2: sensor bias, heater saturation, measurement
delay, stronger cold fronts and lags, and two composed faults.

**Checks** ([`tests/test_benchmark_v1.py`](../../tests/test_benchmark_v1.py)):

- the energy balance closes to 1e-6 kJ per step with process noise off;
- response directions hold: a higher setpoint gives a hotter oven; a colder ambient costs heater energy
  until the heater saturates and then cools the oven; a sensor reading 10 K high holds the oven 9-11 K low;
  a larger lag heats more slowly; a faster preset makes proportionally more parts;
- the ledger margin equals the summed reward to 1e-8 for four controllers;
- the incremental accounting differs from the intended return by exactly 0.60 EUR × order in every episode;
- scenarios are deterministic and independent of the set they are drawn in;
- all seed partitions are disjoint;
- each training condition differs from its control in exactly one setting.

**Reference controller.** A tariff- and order-aware pacing rule
([`experiments/v1/controllers.py`](../../experiments/v1/controllers.py)), tuned on dev over 56
configurations and locked (PROTOCOL section 4). Every speed target carries a 30 % reserve. Before the peak,
the planned off-peak speed (Quality focus, 1.1) plus the reserve exceeds every preset, so the rule runs
Aggressive. During the peak, it picks the cheapest preset that still covers the remaining order, assuming
Quality-focus speed for the off-peak steps after the peak. After the peak, it runs just fast enough to
finish, and it drops to Energy saving once the order is complete. Preset changes are bridged through
presets whose cure window holds the current temperature.

## 4. Design, calibration and evaluator validation

The protocol fixes the hypotheses, conditions, partitions, seeds, budgets, outcome, graders, uncertainty
method, decision rule and stopping rule. Its SHA-256 and the hashes of the code and calibration record it
depends on were written to `experiments/v1/results/protocol_freeze.json` (2026-11-09 00:06 UTC) before the
first confirmatory run. In brief:

- **Interventions**, one change at a time: I1 eval-mode fix, I2 update ratio (one update every 4 steps instead
  of one per episode), I3 both, I4 incremental settlement of the shortfall cost, I5 Double DQN target (the
  proposed intervention). R1 is I3 trained on a deliberately misspecified throughput reward.
- **Budget**: 1,000 episodes of 240 steps per run, the original `max_episodes`, for every condition, so all
  comparisons are matched in interactions. I0 and I1 make about 1,000 gradient updates and the others about
  60,000, so the update ratio is also a compute change. Output A therefore also plots dev margin against
  gradient updates, and the compute table gives CPU time per run.
- **Replication**: training seeds 1-6 are the replication units, and a seed pairs runs across conditions
  (same initialisation, agent random streams and training scenarios). Each controller is evaluated on the
  same 100 held-out nominal episodes, the six held-out fault families (100 episodes each) and the 25-cell
  map (50 episodes per cell).
- **Outcome and decision**: held-out nominal mean margin of the final checkpoint (no checkpoint selection).
  An improvement needs a 95 % t-interval over seeds above 0, constraint failure rates no more than 2
  percentage points worse on each grader, and feasibility.

**Evaluator validation** (dev, before the freeze). Two references expected to pass and four deliberately
defective controllers were run on dev nominal and the six dev fault families (PROTOCOL section 5 has the
full matrix). The suite discriminates between them, and the calibration shows where a defect is harmless:

- *Constant Aggressive* passes every grader on nominal scenarios; only its margin (79 against 132 EUR) gives
  it away. Its setpoint, 198 °C, leaves 7 K to the safety limit, and nominal noise never closes that gap, so
  its safety defect is unreachable there. A sensor reading 10 K low (sensor_bias_low, bias_lag) makes the
  excursion reachable, and the safety grader then fails it. The locked rule shares this exposure: it runs
  Aggressive before the peak, so it fails the safety grader under the same two faults.
- *Throughput maximiser* and the best fixed preset make 1,679 and 1,574 good parts against a mean order of
  1,366. Parts beyond the order earn nothing, and the ledger charges their energy.
- *Rule blind to temperature* fails delivery on nominal scenarios and on every fault family. With the
  first rule grid's best configuration, which had no bridge, the blind variant produced exactly the rule's
  actions: that configuration never read the sensor, so ignoring it was harmless, and no scenario could
  have detected it.
- *Rule blind to the part counter* (plans from commanded rather than counted parts) fails delivery
  wherever scrap occurs, including on nominal scenarios (124 against 132 EUR).
- *Throughput maximiser* passes every grader on nominal scenarios with a margin of 76 EUR, overproducing
  beyond the order. The misspecified throughput reward ranks constant Aggressive first and the locked
  rule last among the feasible controllers, the reverse of the ledger.
- The two saturation families defeat every calibration controller on at least one grader, so there the
  comparison rests on margin and pass rates.

The frozen confirmatory suite is held-out nominal, the map and all six held-out fault families.

## 5. Results

All numbers in this section are held-out unless marked dev, and come from
[`experiments/v1/results/summary.md`](../../experiments/v1/results/summary.md) and the JSON files beside it.

#### Held-out nominal, 100 paired episodes (conditions: mean over 6 seeds ± 95 % t half-width)

| Controller | Margin (EUR/shift) | Safety | Delivery | Quality | Feasible | Dev margin, final checkpoint |
|---|---|---|---|---|---|---|
| uniform | -86.6 | 100 % | 0 % | 0 % | no | |
| fixed: Conservative | -13.2 | 100 % | 0 % | 100 % | no | |
| fixed: Normal | 94.6 | 100 % | 87 % | 100 % | no | |
| fixed: Aggressive | 72.3 | 100 % | 100 % | 100 % | yes | |
| fixed: Energy saving | -221.2 | 100 % | 0 % | 100 % | no | |
| fixed: Quality focus | 93.6 | 100 % | 100 % | 100 % | yes | |
| fixed: Balanced | 70.0 | 100 % | 46 % | 100 % | no | |
| locked rule | 129.6 | 100 % | 100 % | 100 % | yes | |
| I0 | 44.1 ± 92.4 | 100 % | 37 % | 74 % | 0/6 seeds | 29.6 |
| I1 | 58.9 ± 40.3 | 100 % | 43 % | 94 % | 1/6 seeds | 46.3 |
| I2 | 103.3 ± 10.1 | 100 % | 84 % | 95 % | 3/6 seeds | 105.3 |
| I3 | 98.2 ± 14.1 | 100 % | 72 % | 100 % | 1/6 seeds | 96.3 |
| I4 | 86.3 ± 7.8 | 100 % | 98 % | 88 % | 5/6 seeds | 92.0 |
| I5 | 88.0 ± 8.7 | 100 % | 97 % | 99 % | 5/6 seeds | 93.1 |
| R1 | 88.6 ± 21.0 | 100 % | 70 % | 26 % | 0/6 seeds | 83.8 |

#### Preregistered contrasts on held-out nominal (paired by seed; 95 % t-interval, 5 df)

| Contrast | Mean difference (EUR/shift) | 95 % CI | SD between seeds | SD over episodes, within seed | Constraint non-inferiority (S/D/Q) | A feasible | Decision |
|---|---|---|---|---|---|---|---|
| H1: I5 vs I0 | +43.9 | [-53.4, +141.2] | 92.7 | 50.2 | yes/yes/yes | yes | no improvement shown |
| H2: I5 vs locked rule | -41.6 | [-50.3, -32.9] | 8.3 | 11.3 | yes/no/yes | yes | no improvement shown |
| H3: I5 vs best fixed | -5.6 | [-14.3, +3.2] | 8.3 | 7.0 | yes/no/yes | yes | no improvement shown |
| I1 vs I0 | +14.7 | [-83.5, +112.9] | 93.6 | 15.1 | yes/yes/yes | no | no improvement shown |
| I2 vs I0 | +59.1 | [-38.1, +156.3] | 92.6 | 50.7 | yes/yes/yes | no | no improvement shown |
| I3 vs I1 | +39.3 | [-1.4, +80.1] | 38.8 | 45.0 | yes/yes/yes | no | no improvement shown |
| I3 vs I2 | -5.0 | [-19.3, +9.2] | 13.6 | 18.1 | yes/no/yes | no | no improvement shown |
| I4 vs I3 | -11.9 | [-22.2, -1.6] | 9.8 | 25.4 | yes/yes/no | no | no improvement shown |
| I5 vs I4 | +1.7 | [-12.4, +15.8] | 13.4 | 8.5 | yes/yes/yes | yes | no improvement shown |
| R1 vs I3 | -9.6 | [-32.8, +13.5] | 22.1 | 32.1 | yes/yes/no | no | no improvement shown |
| I0 vs best fixed | -49.4 | [-141.8, +43.0] | 88.1 | 52.9 | yes/no/no | no | no improvement shown |
| I0 vs locked rule | -85.5 | [-177.9, +6.9] | 88.1 | 45.5 | yes/no/no | no | no improvement shown |
| I4 vs locked rule | -43.3 | [-51.1, -35.5] | 7.4 | 7.8 | yes/no/no | no | no improvement shown |
| I3 vs locked rule | -31.4 | [-45.5, -17.4] | 13.4 | 20.2 | yes/no/yes | no | no improvement shown |
| interaction (I3-I2)-(I1-I0) | -19.8 | [-119.7, +80.2] | 95.2 | | | | |

#### Held-out fault families (100 episodes each): margin and share of episodes failing any grader

| Family | Locked rule | I0 − rule | I5 − rule [95 % CI] | Fail any: rule | I0 | I5 |
|---|---|---|---|---|---|---|
| sensor_bias_low | 72.0 | -64.9 | -85.3 [-173.1, +2.5] | 85 % | 91 % | 94 % |
| sensor_bias_high | 97.9 | -389.6 | -215.0 [-406.7, -23.3] | 100 % | 100 % | 90 % |
| saturation | 120.2 | -136.9 | -84.9 [-110.2, -59.7] | 20 % | 89 % | 91 % |
| delay | 125.1 | -87.1 | -35.9 [-44.4, -27.5] | 5 % | 68 % | 7 % |
| cold_saturation | 115.2 | -150.2 | -125.0 [-142.2, -107.9] | 27 % | 90 % | 100 % |
| bias_lag | 87.0 | -73.1 | -91.0 [-181.3, -0.7] | 76 % | 90 % | 87 % |

Map, I5 − locked rule: mean over the 6 cells inside the development range -40.7 EUR, over the 19 cells outside -37.7 EUR; best cell (cold front 0 K, lag 2) -29.6, worst cell (0 K, lag 1) -42.8. SD of the I5 − rule difference across fault families 60.4 EUR, across map cells 3.7 EUR.

#### Compute per run (mean over 6 seeds)

| Condition | Interactions | Gradient updates | Training wall time (s) | CPU time (s) |
|---|---|---|---|---|
| I0 | 240,000 | 1,000 | 156 | 131 |
| I1 | 240,000 | 1,000 | 154 | 114 |
| I2 | 240,000 | 59,985 | 1499 | 511 |
| I3 | 240,000 | 59,985 | 1460 | 483 |
| I4 | 240,000 | 59,985 | 1268 | 486 |
| I5 | 240,000 | 59,985 | 1290 | 505 |
| R1 | 240,000 | 59,985 | 1311 | 507 |

Confirmatory training: 42 runs, 0 failed; wall 48.0 min on 18 worker processes; summed process CPU time 4.56 h. Pilot: wall 4.5 min.
Wall time by stage (s): v0 audit 366, calibration 12, boundary audit 617, pilot 268, confirmatory training 2879, held-out evaluation 105; total 70.8 min.

### A. Learning curves and ablation

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/v1/expA_learning-dark.svg">
  <img width="100%" alt="Eight small panels. Seven show the development-set margin against environment interactions (0 to 240,000) for conditions I0 to I5 and R1, with six seed traces, the seed mean and a t-interval band, and reference lines for the locked rule (131.9 EUR), the best fixed preset (100.7) and uniform random (−111.8). I0 and I1 (1,000 updates) swing widely and end at 29.6 and 46.3 EUR on average; I2 and I3 (60,000 updates) settle near the best fixed preset, ending at 105.3 and 96.3; I4, I5 and R1 end at 92.0, 93.1 and 83.8. No condition approaches the locked rule. The eighth panel plots I0 and I3 against gradient updates on a log scale: at about 1,000 updates I3 scores 54.5 EUR after 4,080 interactions, against 29.6 for I0 after 240,000." src="../figures/v1/expA_learning-light.svg">
</picture>

**Figure A.** Data: [run records](../../experiments/v1/results/runs/confirmatory/), [plotted values](../../experiments/v1/results/figure_data/expA_learning.csv).

Under the original procedure (I0), and with only the eval-mode fix (I1), learning is erratic. About 1,000
gradient updates are spread over 240,000 interactions, and a seed's dev margin can move by more than 200 EUR
between checkpoints; the final dev margins average 29.6 and 46.3 EUR. One update every 4 steps (I2, I3) makes
about 60,000 updates. It brings the seed mean to the best fixed preset's level within about 120,000
interactions (final dev 105.3 and 96.3 EUR) and narrows the spread between seeds. Neither the incremental
accounting (I4) nor the Double DQN target (I5) lifts the curves above that plateau, and no condition comes
near the locked rule (dev 131.9 EUR).

Matched in gradient updates rather than interactions, the picture changes. After about 1,000 updates, I3 has
seen 4,080 interactions and scores 54.5 EUR on dev, against 29.6 EUR for I0 after all its 240,000 interactions
and the same number of updates. The update-ratio change is therefore also a 60-fold compute change: 8-9 CPU
minutes per run against about 2 (compute table above).

On held-out episodes the contrasts point the same way, but with wide intervals. I2 − I0 = +59.1 [−38.1, +156.3]
and I3 − I1 = +39.3 [−1.4, +80.1] EUR: both estimates are large, but neither interval excludes zero, because
I0 and I1 vary between seeds by about 90 and 40 EUR (SD). The eval-mode fix shows no effect with either update
ratio: I1 − I0 = +14.7 [−83.5, +112.9] and I3 − I2 = −5.0 [−19.3, +9.2]. The interaction
(I3 − I2) − (I1 − I0) is −19.8 [−119.7, +80.2].

The value estimates show the same pattern from another side. At the final checkpoint on dev, comparing
Q(s0) with the realised discounted return of the greedy policy, the bias is +176 % for I0 and +57 % for I1.
That overestimation fits too few updates plus dropout noise in the targets. It falls to +24 % for I2 and
+1 % for I3. The Double DQN change was justified by overestimation, but by the end of training the accounting
change had already turned the bias negative, to −57 % for I4; with Double DQN it is −36 % (I5).

### B. Generalisation and faults

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/v1/expB_generalisation-dark.svg">
  <img width="100%" alt="Top left: a 5 by 5 map of cold-front magnitude (0 to −20 K) against thermal-lag factor (0.5 to 2) showing I5 minus the locked rule in EUR per shift, between −43 and −30 in every cell, with the development range outlined. Top right: the share of I5 episodes failing any grader per cell, 2 to 8 % for lags up to 1.25 and 12 to 19 % at lag 2, with the locked rule's share in small type (0 to 12 %, and 42 to 52 % at lag 2). Bottom: six held-out fault families with I0 and I5 minus the locked rule and 95 % intervals: I5 is below the rule in all six, from −36 EUR (measurement delay) to −215 EUR (sensor reading 10 K high); the labels give the share of episodes failing any grader for the rule and I5." src="../figures/v1/expB_generalisation-light.svg">
</picture>

**Figure B.** Data: [plotted values, map](../../experiments/v1/results/figure_data/expB_map.csv), [fault families](../../experiments/v1/results/figure_data/expB_faults.csv), [raw episodes](../../experiments/v1/results/episodes.csv.gz).

Across the map, I5's deficit to the locked rule hardly varies: 40.7 EUR on average inside the training
range, 37.7 EUR outside it, and 29.6 to 42.8 EUR per cell (SD across cells 3.7 EUR). The heater loop
compensates cold fronts down to −20 K, and thermal lags from 0.5 to 2, while it has headroom, so these shift
both controllers alike. The slowest response (lag 2.0) is the one place where the rule's constraints suffer.
Its bridging waits for the oven to reach each new cure window, and it delivers the order in only 48-58 % of
those episodes, against 92-96 % for I5, which runs fast throughout. The rule still earns about 30 EUR more
per shift there, because a few missing parts cost less than I5's extra energy.

The fault families separate the controllers far more (SD of the difference across families: 60.4 EUR). I5
trails the rule in every family. The gap runs from 35.9 EUR [27.5, 44.4] with a 10-minute measurement delay
to 215.0 EUR [23.3, 406.7] with a sensor reading 10 K high:

- *Sensor reading 10 K high.* The loop holds the oven 10 K low and scrap follows. The rule passes quality in
  none of these episodes and I5 in 10 %.
- *Sensor reading 10 K low* (alone, or with a slower response). The oven runs 10 K hot, so every phase at the
  Aggressive setpoint breaches the safety limit. The rule passes safety in 15 % and 26 % of episodes, I5 in
  7 % and 15 %. The rule shares the defect that calibration exposed in constant Aggressive.
- *Heater saturation.* The rule's step-down keeps quality at 96 % and delivery at 82 %, against 12 % and 40 %
  for I5.
- *Composed cold front and saturation.* The rule keeps 95 % and 77 %; I5 has 0 % and 19 %.
- *Measurement delay.* Nearly harmless to both: 5 % and 7 % of episodes fail a grader. I0 fails in 68 %.

### C. Reward against outcomes

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/v1/expC_reward_outcome-dark.svg">
  <img width="100%" alt="Four scatter panels against the misspecified throughput reward on held-out nominal episodes, one point per baseline and per training run. R1 runs (trained on the throughput reward) are highlighted against I3 runs (same algorithm, intended reward). Panel a, ledger margin: the locked rule is highest at 129.6 EUR with a throughput reward of 134.7; constant Aggressive has the highest throughput reward, 151.9, and a margin of 72.3. Panel b, good parts beyond the order: the fast fixed presets and runs I4 and I5 overshoot by about 220 to 390 parts. Panel c, energy: rises with the throughput reward. Panel d, scrap: R1 runs reach up to 9.1 % of parts made, against at most 2.8 % for I3." src="../figures/v1/expC_reward_outcome-light.svg">
</picture>

**Figure C.** Data: [plotted values](../../experiments/v1/results/figure_data/expC_reward_outcome.csv).

With the intended reward, training return and ledger margin coincide by construction. I4 and I5 add a
constant that does not depend on the policy. The audit therefore concerns R1, trained on the throughput
reward, which pays for every part, scrap included, with no order cap and no shortfall cost.

On held-out episodes, R1 collects 143.3 EUR of throughput reward against 132.5 for I3, the same algorithm
trained on the intended reward. It gets there by making scrap: 6.7 % of its parts against 2.2 %, and 74 % of
its episodes fail the quality grader against none of I3's. Its ledger margin is 9.6 EUR lower, with an
interval of [−32.8, +13.5] that includes zero. It also misses the order by more (mean shortfall 32 against 23
parts). Under the protocol's definition, this is reward exploitation: R1 is higher on its training reward and
lower on the ledger, and the gap comes from behaviour the misspecified reward pays for, scrap. Excess
production is not the channel: R1 overshoots the order by 80 parts against I3's 71.

On the non-learning baselines, the throughput reward and the ledger rank almost in reverse. Constant
Aggressive scores highest on the throughput reward (151.9) and has the lowest margin of the feasible
controllers (72.3). The locked rule scores 134.7 on it and highest on the ledger (129.6).

The intended-reward runs show a different gap, and it is not exploitation. I4 and I5 overshoot the order by
222 and 227 parts and use 309-315 kWh at the peak tariff, against the rule's 207. Their own training reward
penalises that energy, so they score low on it and on the ledger alike: poor optimisation, not a
specification gap.

### D. Failure and recovery in one episode

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/v1/expD_trajectory-dark.svg">
  <img width="100%" alt="Five panels over the 8-hour shift for one held-out episode with a cold front of −15 K and the heater capped at 60 % from 2 h 58 min to 4 h 58 min. Panel 1, oven temperature with each controller's cure-window lower edge and the 205 °C safety limit: the best fixed preset falls below its window at about 3.2 h; the locked rule steps down through slower presets and stays in its windows; I5 chatters between presets and falls out of its windows from about 3.5 to 5.2 h. Panel 2, presets as heater setpoints. Panel 3, cumulative good parts against the order of 1,434: rule 1,400, I5 1,346, I0 1,187, best fixed 1,142. Panel 4, cumulative energy cost: I5 highest at 181 EUR. Panel 5, cumulative margin at the end: rule 112, I5 35, I0 −39, best fixed −115 EUR." src="../figures/v1/expD_trajectory-light.svg">
</picture>

**Figure D.** Downloadable traces: [cold_saturation_seed4004076.csv](../../experiments/v1/results/trajectories/cold_saturation_seed4004076.csv).

The selection rule picked held-out episode 4004076 of the cold_saturation family: fault onset at step 89
(2 h 58 min), against a family median of 88.5. For the next 60 steps, the ambient drops by 15 K and the
heater is capped at 72 kW.

- *Best fixed preset (Quality focus).* It cools until it falls out of its cure window at about 3 h 15 min,
  makes no good part for two hours and ends 292 parts short, at −115 EUR.
- *Locked rule.* It steps down through Normal, Balanced and Conservative to Energy saving as the oven cools,
  so its window follows the temperature and production continues. It returns to Aggressive after the peak and
  finishes 34 parts short, at 112 EUR.
- *I5 (seed 1).* It chatters between Aggressive and Quality focus or Balanced while the oven cools, and scraps
  from about 3.5 h to 5.2 h. It returns to Aggressive once the fault clears and ends 88 parts short, at
  35 EUR, with the highest energy cost of the four (181 EUR).
- *I0 (seed 1).* It runs Normal before the peak and Energy saving through it. That cures throughout but too
  slowly, and it ends 247 parts short, at −39 EUR.

No controller exceeded the safety limit in this episode (maximum 198.8 °C).

### E. Performance against constraints

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/v1/expE_frontier-dark.svg">
  <img width="100%" alt="Scatter of held-out margin against the share of held-out episodes failing at least one grader, one point per baseline and per training run, with feasible points filled and infeasible ones hollow; the right panel enlarges the region of 0 to 22 % and 60 to 135 EUR. The locked rule sits alone at 129.6 EUR with no failures. Feasible training runs lie between 77 and 113 EUR, mostly I4 and I5 at 0 to 5 % failures, with I2 seed 1 highest at 112.6. The highest-margin run, I3 seed 5 at 114.8 EUR, fails a grader in 7 % of episodes and is infeasible. Runs of I0 and R1 fail a grader in 7-100 % and 27-100 % of episodes." src="../figures/v1/expE_frontier-light.svg">
</picture>

**Figure E.** Data: [plotted values](../../experiments/v1/results/figure_data/expE_frontier.csv).

Of the 42 training runs, 15 are feasible on held-out nominal episodes: 5 of 6 for I4, 5 for I5, 3 for I2,
and 1 each for I1 and I3; none for I0 or R1. The feasible runs span 77-113 EUR per shift. The best of them,
I2 seed 1 at 112.6 EUR, is 17.0 EUR below the locked rule. The highest-margin run, I3 seed 5 at 114.8 EUR,
is infeasible.

Among the baselines, the locked rule (129.6), Quality focus (93.6) and Aggressive (72.3) are feasible.
Normal (94.6) fails delivery in 13 % of episodes, because its 1,440 parts per shift fall short of the larger
orders.

The measured points show two regimes, with no smooth frontier between them:

- runs that slow down at the peak tariff and miss orders (I2, I3);
- runs that meet the order by running fast throughout (I4, I5), whose margins match the fast fixed presets.

### F. Numerical decision boundary of the exported policies

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../figures/v1/expF_decision_boundary-dark.svg">
  <img width="100%" alt="Panel a: states per band of the float64 gap between the two largest Q-values, for 18 networks: corpus-A rollout states, corpus-A synthetic states and the states each network visits under its own greedy policy. The generated .st selects another action than PyTorch on 2 states, both synthetic, in the band 1e-5 to 1e-4; on none of the 360,000 visited states. Panel b: the largest difference between .st and PyTorch Q-values per band, below the gate tolerance of 1e-3 everywhere: up to 1.8e-4 on synthetic states with gaps of 1e-5 to 1e-4, and below 1e-7 on rollout and visited states with gaps under 1e-2." src="../figures/v1/expF_decision_boundary-light.svg">
</picture>

**Figure F.** Data: [boundary_audit.json](../../experiments/v1/results/boundary_audit.json), [disagreements](../../experiments/v1/results/boundary_audit_disagreements.csv), [plotted values](../../experiments/v1/results/figure_data/expF_decision_boundary.csv).

Across the 18 networks the generated `.st` selects a different action from PyTorch float32 on 2 of 360,000
corpus-A inputs: the two near-ties of `speed_proportional_seed5` that the strict gate already rejects. Both
are synthetic uniform draws at float64 gaps of 7.2×10⁻⁵ and 8.9×10⁻⁵, inside the band where the `.st`'s
Q-value error on synthetic states reaches 1.8×10⁻⁴, so the error exceeds the gap. At both, the action moves
from Quality focus to Aggressive (+0.10 speed, +0.05 pressure and +0.15 temperature factor). One further synthetic
input, at a gap of 5.5×10⁻⁶, separates PyTorch float32 from the float64 reference; the `.st` agrees with
PyTorch there. Below a gap of 10⁻², the `.st`'s Q-value error on rollout inputs is at most 6.0×10⁻⁸, and no
rollout input flips. The `.fbt` outputs are bit-identical to the `.st` on all 18 × 20,000 inputs.

On the 360,000 states the networks visit under their own greedy policies in the 100 dev episodes, no
action differs. That includes 114 visited states with gaps below 10⁻⁴, all from untrained networks, whose
Q-value errors there stay below 10⁻⁷. The trained networks never come closer than 0.040 to a tie on a
visited state; for the rejected network that is 65 times its largest Q-value error there, 6.1×10⁻⁴. With the
same action at every visited state, the closed loop with the exported block reproduces the PyTorch
trajectories of these episodes exactly. No closed-loop re-simulation was needed: the script re-simulates
only when a visited state disagrees. The rejection remains correct under the acceptance predicate, which is
about agreement on sampled inputs. On the reachable development trajectories it has no closed-loop
consequence.

## 6. Decisions

| Hypothesis (held-out nominal) | Mean difference, EUR per shift | 95 % CI | Constraints | Decision |
|---|---|---|---|---|
| H1: I5 improves on I0 | +43.9 | [−53.4, +141.2] | non-inferior on all three graders; I5 feasible, I0 not | not supported: no improvement shown |
| H2: I5 improves on the locked rule | −41.6 | [−50.3, −32.9] | delivery 97 % against 100 %: non-inferiority fails | not supported: the rule is better |
| H3: I5 improves on the best fixed preset | −5.6 | [−14.3, +3.2] | delivery 97 % against 100 %: non-inferiority fails | not supported |

**Policy quality did not improve under the declared criteria.** The proposed procedure is feasible where the
original is not. On margin, though, it is indistinguishable from the original at six seeds, 42 EUR per shift
behind a tuned rule, and level with the best fixed preset. It is not worth exporting or deploying, and
nothing trained here was.

Secondary findings, descriptive (no multiplicity correction):

1. The update ratio is the change with the largest estimated effect: I2 − I0 = +59.1 and I3 − I1 = +39.3 EUR
   (intervals include 0). It is confounded with a 60-fold increase in gradient computation.
2. The eval-mode fix has no detectable effect on held-out margin, with or without the extra updates.
3. The incremental shortfall accounting cost 11.9 EUR [1.6, 22.2] of margin against I3. In exchange,
   delivery rose from 72 % to 98 % of episodes and quality fell from 100 % to 88 %. This is the only interval
   among the single-change contrasts that excludes 0. It explains why the four-change package I5 ended below
   the two-change I3 (98.2 EUR, infeasible on delivery). That observation is post hoc and not a confirmatory
   result.
4. Double DQN changed nothing measurable: I5 − I4 = +1.7 [−12.4, +15.8].
5. Training on the misspecified throughput reward produced reward exploitation through scrap (output C).
6. The generalisation gap to the rule is flat across the cold-front × lag map and widens sharply under faults
   (output B). Under the two low-reading-sensor faults, the rule's own Aggressive phases breach the safety
   limit.

Variance components for H1: the SD of the per-seed paired difference is 92.7 EUR between seeds, against
50.2 EUR over episodes within a seed. The replication that limits the conclusion is training seeds, not
evaluation episodes. With the observed between-seed SD, which I0's spread dominates, detecting an
improvement of 44 EUR with 80 % power at the 5 % level would take about 35 seeds per condition. That
estimate is a planning figure for a follow-up, not a result.


## 7. Limitations

- **Synthetic benchmark.** Benchmark v1 has assumed equations and parameters, a first-order thermal model,
  a heater loop with feedforward on the true ambient temperature, and a binary cure window. It was built to pose
  one control question and is not a validated digital twin. Its economics were rebalanced once on dev
  (PROTOCOL section 5) so that adaptation is worth something; with the first parameters a fixed preset almost
  matched the rule.
- **One learner, one budget.** The conclusions concern `backend.rl_agent.DQNNetwork` with the repository's
  hyperparameters, 240,000 interactions per run and six seeds. Other algorithms, network sizes, learning
  rates, reward scaling or longer training were not tried; the protocol allowed one algorithmic change.
- **The rule is a strong, tuned baseline.** It was tuned on dev over 56 configurations, the DQN conditions
  were not tuned at all, and only the final checkpoint is evaluated. That is the intended comparison of a
  deployable procedure, but it is not a tuned-against-tuned comparison.
- **Map resolution.** Map cells have 50 episodes per seed (300 per cell over six seeds), not 100.
- **Multiplicity.** H1 is the only confirmatory test. The other contrasts are reported with intervals and no
  multiplicity correction, and are descriptive.
- **Compute.** All runs used one CPU configuration (Intel Core Ultra 7 165H, Windows 11). Wall times depend on
  how many processes run at once and on Windows power management (`train_dqn.no_power_throttling` opts each
  run out of EcoQoS throttling). Results are deterministic given the seed on this configuration. Bitwise
  reproduction on other hardware or library builds was not tested.
- **Decision-boundary audit.** Reachability was checked on the historical simulator's 100 development episodes
  under each network's own greedy policy. Other episode distributions, or exploration noise, could visit other
  states. Agreement remains sampled-input agreement, not formal equivalence.

## 8. Reproduction and provenance

Commands, file locations, hashes and elapsed times are in
[section 15 of the evidence document](../EVIDENCE.md#15-benchmark-v1-experiment-2026-11-08-to-2026-11-09).
Every number in this report comes from `experiments/v1/results/` (`summary.md` is generated from the JSON
files by `experiments/v1/summarize.py`). Local validation:

- `tests/test_benchmark_v1.py`: 11 passed.
- The repository's CI test selection (`STRICT_SKIPS=1 pytest tests -m "not training and not service"`): 58
  passed, 0 skipped, both in the existing environment and in the clean environment of evidence section 10.

The GitHub workflow has not run remotely.
