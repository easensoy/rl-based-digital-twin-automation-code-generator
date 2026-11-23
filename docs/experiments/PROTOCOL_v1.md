# Protocol v1: evaluation-driven policy improvement on benchmark v1

Status: **frozen** before any confirmatory training run. The SHA-256 of this file, of the benchmark and
training code and of the calibration record are stored in `experiments/v1/results/protocol_freeze.json`,
written when the protocol was frozen. Any later change is listed under "Deviations" at the end; nothing above
that heading changes after the freeze.

## 1. Question

Can an explicitly specified change to the repository's DQN training procedure improve held-out control
performance over the original procedure and over strong non-learning baselines, without worsening the
constraint metrics? A negative answer is a valid outcome.

Benchmark v1 (`experiments/v1/process.py`) exists because the historical environment cannot pose this
question (see `docs/experiments/REPORT_v1.md`, section 2: every historical checkpoint reproduces a fixed
preset, and the best two-segment schedule beats the best fixed action by 0.4-0.5 % with no
episode-to-episode variation). v1 is a synthetic process model with assumed parameters, not a validated
digital twin, and it is isolated from `backend/` and from the historical experiments.

## 2. Benchmark, partitions and seeds

| Set | Seeds | Use |
|---|---|---|
| Training | `10,000,000 + 100,000 * run_seed + episode` | DQN training only |
| Development (dev) | 1,000,000-1,000,099 | rule tuning, baseline selection, evaluator calibration, pilot, learning curves |
| Held-out nominal | 2,000,000-2,000,099 | primary comparison |
| Generalisation map | `3,000,000 + 1,000 * cell + k`, k < 50 | output B, 25 cells x 50 episodes |
| Held-out fault families | `4,000,000 + 1,000 * family + k`, k < 100 | outputs B and D |
| Dev fault families | `5,000,000 + 1,000 * family + k`, k < 100 | evaluator calibration only |

Each episode has its own random generator seeded by its scenario seed, so a scenario is identical for every
controller (common random numbers; all comparisons are paired by episode).

Nominal distribution (training, dev and held-out): order 1,200-1,500 parts, initial oven temperature
150-185 degC, thermal-lag factor 0.8-1.25, a cold front of 0 to -5 K lasting 60 steps from a uniform onset in
steps 30-150, sensor noise 0.5 K, process noise 0.2 K per step.

Generalisation map: cold-front magnitude in {0, -5, -10, -15, -20} K x thermal-lag factor in
{0.5, 0.8, 1.0, 1.25, 2.0}. Cells with magnitude >= -5 K and lag in [0.8, 1.25] lie inside the training and
development range; all others are outside it.

Fault families (onset uniform in steps 30-150; duration 60 steps unless stated):

| Family | Kind | Fault |
|---|---|---|
| sensor_bias_low | single | sensor reads 10 K low from onset to shift end (heater loop and controller both see it) |
| sensor_bias_high | single | sensor reads 10 K high from onset to shift end |
| saturation | single | heater limited to 60 % of P_max for 60 steps |
| delay | single | temperature, counters and duty observed 5 steps (10 min) late, whole shift |
| cold_saturation | composed | cold front of -15 K and heater limited to 60 %, same onset, 60 steps |
| bias_lag | composed | sensor 10 K low from onset and thermal-lag factor 1.5 |

## 3. Outcome ledger, graders and feasibility

The outcome ledger (`experiments/v1/ledger.py`) recomputes every outcome from the logged trajectory,
independently of any training reward: good and scrap parts, energy (kWh, and kWh at the peak tariff), energy
cost, shortfall against the order, safety-excursion steps, revenue and **margin** (EUR per shift: order-capped
revenue minus energy cost, safety cost and shortfall cost).

Deterministic numeric graders, one verdict per episode (`experiments/v1/evaluate.py`):

| Grader | Pass when | Feasibility: minimum pass rate |
|---|---|---|
| safety | no step above 205 degC | 0.99 |
| delivery | shortfall at most 1 % of the order | 0.95 |
| quality | scrap fraction at most 5 % | 0.95 |

A controller is **feasible** on a scenario set when every pass rate meets its minimum. For a training condition
the pass rates are pooled over its six seeds (600 episodes).

## 4. Controllers

Non-learning baselines, all fixed before the confirmatory runs:

- uniform random preset (generator seed 0);
- each of the six fixed presets;
- **best fixed preset**: Quality focus, the feasible preset with the highest dev margin (100.7 EUR);
- **locked rule**: tariff- and order-aware pacing with a temperature-aware transition bridge
  (`controllers.rule`), configuration `fast=4 (Quality focus off-peak), margin=0.3, bridge=on, sat_guard=off`,
  the feasible configuration with the highest dev margin among 56 (131.9 EUR; 3 of 56 are feasible on dev).
  On dev it exceeds the best fixed preset by 31.2 EUR per shift (95 % CI +/- 2.4, 100 paired episodes).

DQN training conditions (`experiments/v1/train_dqn.py`), each differing from its control in one respect:

| Condition | Change | Control |
|---|---|---|
| I0 | original procedure: `backend.rl_agent.DQNAgent.act` and `.replay` unchanged (dropout active when acting and inside the target network; one update at the end of each episode; target copied every 10 updates; MSE loss) | - |
| I1 | eval-mode fix: dropout off when acting and when computing targets | I0 |
| I2 | update ratio: one gradient update every 4 environment steps | I0 |
| I3 | I1 + I2 | I1, I2 |
| I4 | reward accounting: shortfall cost settled as the order is filled (`reward="incremental"`) | I3 |
| I5 | Double DQN target | I4 |
| R1 | I3 trained on the misspecified throughput reward | I3 |

Shared settings: `DQNNetwork(11, 6)` (12-128-128-128 architecture of `backend/rl_agent.py` with 11 inputs),
Adam lr 1e-3, batch 64, buffer 10,000, gamma 0.99, epsilon = max(0.01, 0.995^episode) set at the start of
each episode, 1,000 episodes of 240 steps (240,000 interactions, the original `max_episodes`). Updates per run:
I0 and I1 about 1,000; all others about 60,000. Evaluation is greedy, in eval mode.

**Proposed intervention**: I5 (all four changes). The others are ablations.

Justification of the algorithmic change (Double DQN): the bootstrap target takes a maximum over six noisy
action values in a stochastic process, which biases targets upwards (van Hasselt et al., 2016). The pilot
measured the bias of Q(s0) against the realised discounted return at +5 % for I4 and +3.5 % for I5 after
200 episodes (seed 101), so a small effect is expected; the change was fixed in code before the pilot.

## 5. Evaluator calibration (dev only, done before this freeze)

Six controllers were scored on dev nominal and the six dev fault families (100 episodes each): two references
expected to pass (locked rule, best fixed) and four deliberately defective controllers (constant Aggressive;
the locked rule blind to the temperature sensor; the locked rule blind to the good-part counter; a throughput
maximiser that runs the fastest preset whose cure window holds the measured temperature, ignoring order and
tariff). Mean margin in EUR, then the failed graders (S safety, D delivery, Q quality):

| Family | locked rule | best fixed | const. Aggressive | blind to temperature | blind to counter | throughput max. |
|---|---|---|---|---|---|---|
| nominal | 132 ok | 101 ok | 79 ok | 39 D | 124 D | 76 ok |
| sensor_bias_low | 77 S | 93 ok | -216 S | 17 SD | 78 SD | -205 SD |
| sensor_bias_high | 95 DQ | -493 DQ | 81 ok | -109 DQ | -161 DQ | 81 Q |
| saturation | 122 D | -48 DQ | 44 DQ | 33 DQ | 107 D | 82 D |
| delay | 126 ok | 98 ok | 76 ok | 53 D | 107 D | 70 D |
| cold_saturation | 114 DQ | -69 DQ | 24 DQ | 23 DQ | 96 DQ | 83 D |
| bias_lag | 84 SD | 95 ok | -208 S | 12 SD | 78 SD | -207 S |

Reading: on nominal scenarios the graders catch both blind controllers (delivery), while constant Aggressive
and the throughput maximiser pass every grader and are distinguished only by margin (79 and 76 against 132).
Their defect is harmless on nominal scenarios because it is unreachable there: Aggressive's setpoint, 198 degC,
leaves 7 K to the safety limit, which nominal noise never crosses; a sensor reading 10 K low (sensor_bias_low,
bias_lag) makes the excursion reachable and the safety grader fails them. The misspecified throughput reward
ranks constant Aggressive first (151.8 against 137.6 for the locked rule on dev nominal), the reverse of the
ledger. The saturation families defeat every calibration controller on at least one grader, so there they are
compared on margin and pass rates, not on feasibility.

Calibration changes before the freeze (all on dev): (1) the delivery grader first required the full order;
a 1-part miss then failed an episode, so it became "at least 99 % of the order"; (2) with the first rule grid
(speed reserve up to 0.2) no configuration was feasible, the best being infeasible on quality (scrap) or
delivery, so the grid was extended to reserves of 0.3-0.5; (3) with the first grid's best configuration
(no bridge) the temperature-blind defect produced identical actions to the rule, because that configuration
never read the sensor; the locked configuration reads it. Earlier, benchmark economics were rebalanced once
(heat-loss coefficient 0.40 to 0.25 kW/K, product heat load 20 to 40 kW, tariff 0.12/0.36 to 0.10/0.40
EUR/kWh, part value 0.25 to 0.20 EUR, salvage 0.05 to 0) after a first dev check showed a fixed preset almost
matching the rule, so that adapting to the tariff and the order is worth something. No held-out scenario was
evaluated before this freeze.

Frozen confirmatory suite: held-out nominal, the 25-cell map and all six held-out fault families.

## 6. Pilot

Seed 101 (excluded from all confirmatory analyses), all seven conditions, 200 episodes, dev evaluation only.
Under Windows EcoQoS throttling the jobs ran at about 1.2 s per episode for the 1-update-per-4-steps
conditions; with the per-process opt-out in `train_dqn.no_power_throttling`, 7 concurrent jobs ran at
0.29 s per episode. Budget: 1,000 episodes per run, 7 conditions x 6 seeds = 42 runs on 18 worker processes,
projected at 20-30 min wall-clock.

## 7. Confirmatory design and analysis

- Seeds 1-6 for every condition (training seeds are the replication units). A seed fixes the network
  initialisation, the agent's random streams and the training scenarios, so conditions are paired by seed.
- Model selection: none. The checkpoint after the last episode is evaluated; there is no early stopping.
  Dev learning curves (checkpoints at episodes 5, 10, 17, 25 and every 50) are descriptive only.
- Primary outcome: held-out nominal mean margin (EUR per shift) over 100 paired episodes.
- Constraint metrics: the three grader pass rates, plus excursion steps, shortfall and scrap fraction.
- Effect estimate: for seed s, d_s = (mean margin of A, seed s) - (mean margin of B, seed s), or minus the
  baseline's mean for a non-learning B. Report mean(d) with a 95 % t-interval over the six seeds (5 degrees of
  freedom). Report separately the within-seed episode SD of paired differences, the between-seed SD of the
  per-seed means, and the variation across fault families and map cells. Timesteps are never replicates.
- **H1 (primary)**: I5 improves on I0. **H2**: I5 improves on the locked rule. **H3**: I5 improves on the best
  fixed preset. Secondary contrasts, reported with intervals and no correction for multiplicity:
  I1-I0, I2-I0, I3-I1, I3-I2, the interaction (I3-I2)-(I1-I0), I4-I3, I5-I4, R1-I3, I0-best fixed.
- **Decision rule**: A improves on B only if (i) the lower 95 % bound of mean(d) is above 0, (ii) for every
  grader A's pooled failure rate is at most B's plus 2 percentage points, and (iii) A is feasible on held-out
  nominal. A non-significant difference is reported as "no improvement shown", never as success.
  "Policy quality improved under the declared criteria" means H1 holds; outperforming the non-learning
  baselines additionally requires H2.
- Failed runs: a crashed run is rerun once with the same seed and reported; a second failure is reported as
  missing. Diverged runs are evaluated as they are.
- Maximum runtime for all new experiments: 2 hours wall-clock, CPU only.

## 8. Secondary outputs, fixed in advance

- A: dev margin against interactions and against gradient updates, per seed, with the seed mean and
  t-interval, and the locked rule, best fixed and uniform as references; CPU seconds per run.
- B: map of the paired margin difference (I5 minus locked rule, seed mean) and of grader failure rates per cell,
  dev region outlined, n per cell; fault families as a second panel.
- C: training reward against ledger outcomes (margin, production, energy, scrap, constraints) for every
  controller and run on held-out nominal; R1 against I3 as the controlled reward-specification comparison.
  A policy is called reward-exploiting only if it scores higher than a comparator on its training reward and
  lower on the ledger, with the gap traced to behaviour the misspecified reward pays for (overproduction or
  scrap); low scores on both are poor optimisation.
- D: the held-out cold_saturation episode whose fault onset is closest to the family's median onset (ties:
  lowest seed), for the locked rule, best fixed preset, I0 seed 1 and I5 seed 1: temperature and cure window,
  disturbance onset, presets and setpoints, safety limit, cumulative good parts against the order, cumulative
  energy cost, margin. Traces as CSV.
- E: held-out nominal margin against the share of episodes failing any grader, one point per baseline and per
  run, feasible region marked, infeasible runs shown, no fitted frontier.
- F: decision-boundary audit of the 18 historical networks on the 20,000 corpus-A states (rollout against
  uniform synthetic states), stratified by the float64 top-two Q gap in decades from 1e-6 to 1e-2; PyTorch
  float32, the .fbt gate and the .st interpreter compared; whether any disagreement lies on a reachable
  trajectory and, if so, its closed-loop effect. tie_tolerance = 0; any tolerant re-check is labelled.

## Deviations

None at the time of freezing.

Entries added after the freeze. The SHA-256 in `protocol_freeze.json` is that of this file up to and including
the line above.

1. **Erratum, section 4.** The label `fast=4 (Quality focus off-peak)` describes the parameter, not the
   behaviour. The 0.3 speed reserve applies to every speed target, and 1.1 × 1.3 = 1.43 exceeds every preset,
   so before the peak the locked rule runs the fastest preset, Aggressive. The value of `fast` enters only the
   plan for the peak. The locked configuration and its code are unchanged.
2. **Scripts written after the freeze** implement sections 7 and 8 without changing them:
   `experiments/v1/evaluate_all.py`, `boundary_audit.py`, `summarize.py` and
   `docs/scripts/make_experiment_figures.py`. `evaluate_all.py` was first exercised on the frozen non-learning
   baselines while training was still running, before any checkpoint was evaluated on held-out scenarios.
3. **Integrity.** All 42 confirmatory runs recorded code hashes equal to those in `protocol_freeze.json`, every
   checkpoint matches the SHA-256 in its run record, and no frozen file changed before the held-out
   evaluation ran.
4. **Runtime.** Confirmatory training took 48.0 min wall-clock, against the 20-30 min projected in section 6.
   With 18 concurrent processes, each ran about 4.5 times slower per episode than in the 7-process timing
   check. All new experiment stages together took about 75 min wall-clock, within the 2-hour cap. Training
   used 4.56 h of summed process CPU time.
