# Differential verification of a DQN policy exported to IEC 61499 function blocks

A trained PyTorch DQN is exported as an IEC 61499 Basic function block (EAE `.fbt`) and an IEC 61131-3 function block (`.st`), and an export is accepted only if a binary32 interpreter running the generated code reproduces the network's Q-values, actions and setpoints on every sampled input. EAE's type compiler checks acceptance, not behaviour; nothing has run on a target runtime, and the policy itself performs below a fixed-action baseline.

A second increment asked whether the policy itself can be improved. The historical simulator cannot answer that, because a constant action is close to optimal in it, so a new synthetic benchmark poses one control question: a curing line under a time-of-use tariff, with faults. Under a protocol frozen before the confirmatory runs, four single changes to the training procedure were tested with 6 seeds × 100 held-out episodes against the original procedure, fixed presets and a tuned rule. None improved policy quality under the declared criteria. The proposed procedure is +43.9 EUR per shift against the original, with a 95 % interval of [−53.4, +141.2], 41.6 EUR below the tuned rule, and level with the best fixed preset. On the 360,000 states the historical networks visit in 100 development episodes, the exported blocks never change an action ([experimental report](docs/experiments/REPORT_v1.md)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/fig1_export_verification-dark.svg">
  <img width="100%" alt="Pipeline: frozen PyTorch policy, exporter, generated .fbt and .st. The .fbt run against PyTorch in a binary32 interpreter on 18 networks with 20,000 sampled inputs each: every Q-value within 10^-3 on 18 of 18 networks (worst 7.3e-4); the same action and setpoints on every input for 17 of 18; on the rejected network, 2 of 20,000 inputs where the generated code selects another action while its Q-values pass; 10 of 10 injected defects rejected. Standalone .st: tested on the same 18 networks, 17 passed and 1 rejected (the near-tie); 14 of 14 synthetic networks passed. EAE 25.0 type compiler: 2 of 18 .fbt exports compiled, both accepted with no unfiltered errors; 3 of 3 broken variants rejected. Target runtime: not verified." src="docs/figures/fig1_export_verification-light.svg">
</picture>

**Figure 1.** `.fbt`: 18 networks (13 trained, 5 untrained) × 20,000 sampled inputs. `.st`: the same corpus, plus 14 synthetic networks × 11,128 inputs. Frozen weights, measured 2026-11-08; agreement is established on the sampled inputs only. Data: [export_verification.json](docs/data/export_verification.json), [st_verification.json](docs/data/st_verification.json), [plotted values](docs/data/fig1_export_verification.csv).

| Measured 2026-11-08, frozen weights | Result | Corpus |
|---|---|---|
| Numerical agreement, `.fbt` | 18/18 networks with max ‖q̂ − q‖∞ < 10⁻³; worst 7.3×10⁻⁴ | 18 networks × 20,000 inputs, PyTorch binary32 reference |
| Behavioural agreement, `.fbt` (strict) | 17/18 networks: action and setpoints identical on every input | same |
| Rejected export, `speed_proportional_seed5` | other action and setpoints on 2 of 20,000 inputs, at a top-two gap of 2–4 ulps; ‖q̂ − q‖∞ < 10⁻³, so a Q-value criterion alone accepts it | [the two inputs](docs/EVIDENCE.md#7-the-two-rejected-inputs) |
| Standalone `.st`, same corpus | 17/18 accepted, the same rejection; outputs bit-identical to the `.fbt` for all 18 × 20,000 inputs | same |
| Standalone `.st`, synthetic corpus | 14/14 accepted: all six actions, exact ties, boundary inputs; worst ‖q̂ − q‖∞ 3.9×10⁻⁶ | 14 networks × 11,128 inputs, float64 reference |
| Mutation sensitivity | `.fbt`: 10/10 injected defects rejected. `.st`: detected on 50/50 reachable (mutant, network) pairs, on 0/62 unreachable ones | 5 kinds × 2 networks; 8 mutants × 14 networks |
| Compiler acceptance | EAE 25.0 FBTypeCompiler: 2/2 `.fbt` exports accepted, no unfiltered errors or warnings; 3/3 broken variants rejected | 2 of 18 exports; no vendor compile of the `.st` |
| Target-runtime validation | not performed | |
| **Policy, historical (2026-11-07)** | | |
| Trained DQN against the best fixed action | below it in 12/12 runs; between-seed 95% t-interval of the paired difference excludes 0 | 6 seeds × 2 simulator versions, 100 paired episodes |
| Trained DQN against uniform random actions | above it in 3/12 runs; the mean paired difference is negative in both versions | same |
| **Policy, benchmark v1 (2026-11-09, protocol frozen before the runs)** | | |
| Proposed procedure I5 against the original I0 | +43.9 EUR per shift, 95 % CI [−53.4, +141.2]: no improvement shown | 6 seeds × 100 held-out episodes, paired |
| I5 against the tuned rule; against the best fixed preset | −41.6 [−50.3, −32.9]; −5.6 [−14.3, +3.2] | same |
| Exported block on visited states | PyTorch's action on all 360,000 visited states; the 2 rejected inputs are synthetic near-ties | 18 networks × 100 dev episodes |

**Evidence:** [provenance, commands and hashes](docs/EVIDENCE.md) · [reproduce](#reproduce) · [acceptance predicate](#acceptance-predicate) · [policy evaluation](#policy-evaluation-on-the-historical-simulator) · [the near-tie](#the-near-tie-the-gate-caught) · [coverage](#verification-coverage) · [limitations](#limitations-and-deployment-boundary) · [benchmark-v1 experiment](docs/experiments/REPORT_v1.md)

## Acceptance predicate

The method is differential numerical verification: the generated code and the network are executed on the same inputs, and their outputs are compared. Every export goes through `iec61499_exporter.verify_fbt`, which runs the `.fbt` (INIT, then one REQ per input) in [`backend/st_interpreter.py`](backend/st_interpreter.py). `verify_st` applies the same predicate to the `.st` FUNCTION_BLOCK. The interpreter evaluates `REAL` as IEEE 754 binary32 with every operation rounded, so its accumulation order is the generated code's. For a finite input set S, PyTorch Q-values q(x) ∈ ℝ⁶ (binary32, eval mode, one batch) and generated outputs q̂(x), â(x), ŝ(x):

```math
\mathrm{accept} \iff \max_{x \in S} \lVert \hat q(x) - q(x) \rVert_\infty < 10^{-3}
\;\wedge\; \forall x \in S:\ \hat a(x) = a^{*}(x) \;\wedge\; \hat s(x) = T\!\left[a^{*}(x)\right]
```

Here a\*(x) = min{k : q_k(x) = max_j q_j(x)} is the first maximum, the rule of `torch.argmax` and of the generated code. T maps each action to its speed, pressure and temperature setpoints, compared bit for bit in binary32. The tie tolerance is 0, so no action difference is excused. The server applies `verify_fbt` to every generation; a failing function block returns HTTP 500 and its files are deleted.

S is finite. The predicate establishes sampled-input agreement and nothing about x ∉ S. Because argmax is discontinuous wherever two Q-values are equal, a bound on ‖q̂ − q‖∞ does not imply action agreement near such points.

The checks address five properties separately:

| Property | Measured by | Scope |
|---|---|---|
| Coefficient serialization fidelity | every `REAL` literal reads back to the exact binary32 weight; correctly rounded and decimal → binary64 → binary32 parsing agree | all 35,462 coefficients of the server-run network, `.fbt` and `.st` |
| Numerical agreement | ‖q̂ − q‖∞ per input | sampled inputs |
| Behavioural agreement | action and setpoints per input | sampled inputs |
| Compiler acceptance | EAE FBTypeCompiler accepts the type | 2 `.fbt` exports |
| Deployed execution | none | not performed |

## Policy evaluation on the historical simulator

The multi-seed evaluation of 2026-11-07 ran 6 training seeds × 2 simulator versions on 100 paired episodes, environment seeds 10000–10099 ([Figure 2 of the evidence document](docs/EVIDENCE.md#figures-of-the-verification-increment), [plotted values](docs/data/fig2_policy_evaluation.csv)). It found the trained DQN below the best fixed action (always Energy saving) in 12 of 12 runs. The paired difference was −154.6 ± 62.9 reward units under the default simulator and −102.1 ± 35.4 under the corrected production model (95 % t-intervals over training seeds). Against uniform random actions, 3 of 12 runs were above. The mean paired difference was −43.2 ± 62.9 and −36.6 ± 35.4, which supports no claim that training improves on random actions. Every trained policy selects a single action in every evaluation state. The original template generator (`backend/plc_generator.py`) ignored the network weights and chose Aggressive everywhere.

The benchmark-v1 audit re-evaluated the 13 frozen checkpoints and explains the result. The historical simulator rewards a constant action: its reward is dominated by a constant bonus and running averages, and its constraints are never reached. It therefore cannot show whether training improves control ([report, section 2](docs/experiments/REPORT_v1.md#2-phase-1-why-the-historical-environment-cannot-answer-the-question)).

## Can a better training procedure improve the policy? (benchmark v1)

The historical simulator cannot answer this ([audit](docs/experiments/REPORT_v1.md#2-phase-1-why-the-historical-environment-cannot-answer-the-question)). A new synthetic benchmark therefore poses one control question: run an 8-hour curing shift under a time-of-use tariff and meet the order at the lowest cost, through cold fronts, heater saturation and sensor faults ([specification](docs/experiments/REPORT_v1.md#3-benchmark-v1-a-heated-curing-line-under-a-time-of-use-tariff)). It is isolated from the application and is not a digital twin.

Under a [protocol](docs/experiments/PROTOCOL_v1.md) frozen before the confirmatory runs, the original training procedure (I0) and five variants, each one change away from its control, were trained for 1,000 episodes with 6 seeds each. They were scored on 100 held-out episodes by an outcome ledger kept apart from the training reward, against fixed presets and a rule tuned on development scenarios. The proposed procedure, I5, combines four changes:
- dropout off when acting and when bootstrapping;
- one gradient update every 4 steps instead of one per episode;
- the shortfall cost settled as the order fills;
- a Double DQN target.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/v1/expA_learning-dark.svg">
  <img width="100%" alt="Eight small panels. Seven show the development-set margin against environment interactions (0 to 240,000) for conditions I0 to I5 and R1, with six seed traces, the seed mean and a t-interval band, and reference lines for the locked rule (131.9 EUR), the best fixed preset (100.7) and uniform random (−111.8). I0 and I1 (1,000 updates) swing widely and end at 29.6 and 46.3 EUR on average; I2 and I3 (60,000 updates) settle near the best fixed preset, ending at 105.3 and 96.3; I4, I5 and R1 end at 92.0, 93.1 and 83.8. No condition approaches the locked rule. The eighth panel plots I0 and I3 against gradient updates on a log scale: at about 1,000 updates I3 scores 54.5 EUR after 4,080 interactions, against 29.6 for I0 after 240,000." src="docs/figures/v1/expA_learning-light.svg">
</picture>

**Figure A.** Learning curves on the 100 development episodes; the held-out results are in the table below. Data: [run records](experiments/v1/results/runs/confirmatory/), [plotted values](experiments/v1/results/figure_data/expA_learning.csv).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/v1/expE_frontier-dark.svg">
  <img width="100%" alt="Scatter of held-out margin against the share of held-out episodes failing at least one grader, one point per baseline and per training run, with feasible points filled and infeasible ones hollow; the right panel enlarges the region of 0 to 22 % and 60 to 135 EUR. The locked rule sits alone at 129.6 EUR with no failures. Feasible training runs lie between 77 and 113 EUR, mostly I4 and I5 at 0 to 5 % failures, with I2 seed 1 highest at 112.6. The highest-margin run, I3 seed 5 at 114.8 EUR, fails a grader in 7 % of episodes and is infeasible. Runs of I0 and R1 fail a grader in 7-100 % and 27-100 % of episodes." src="docs/figures/v1/expE_frontier-light.svg">
</picture>

**Figure E.** Held-out margin against constraint failures, for every baseline and training run. Data: [plotted values](experiments/v1/results/figure_data/expE_frontier.csv), [raw episodes](experiments/v1/results/episodes.csv.gz).

| Preregistered comparison, held-out nominal | Difference, EUR per shift [95 % CI] | Decision |
|---|---|---|
| H1: I5 against the original procedure I0 | +43.9 [−53.4, +141.2] | no improvement shown |
| H2: I5 against the locked rule | −41.6 [−50.3, −32.9] | the rule is better |
| H3: I5 against the best fixed preset (Quality focus) | −5.6 [−14.3, +3.2] | no improvement shown |

- **Policy quality did not improve under the declared criteria.** I5 meets the constraint criteria where I0 does not. Its margin advantage over I0 is inconclusive at six seeds, because the original procedure's seed-to-seed SD is 93 EUR.
- **What helped and what did not.** One update every 4 steps instead of one per episode had the largest estimated effect: +59.1 EUR against I0, and +39.3 with the eval-mode fix; both intervals include 0. It also costs 60 times the gradient computation. The eval-mode fix alone showed no effect. The incremental shortfall accounting cost 11.9 EUR [1.6, 22.2] of margin in exchange for meeting the order. Double DQN changed nothing measurable.
- **Reward against outcomes.** Trained on a deliberately misspecified throughput reward, the same algorithm earned 10.8 EUR more of that reward by tripling scrap, and gained no margin. That is reward exploitation, and it shows only because outcomes are scored independently of the reward ([output C](docs/experiments/REPORT_v1.md#c-reward-against-outcomes)).
- **Generalisation.** I5 trails the rule by 30–43 EUR in every cell of a cold-front × thermal-lag map, and by 36–215 EUR in every fault family ([output B](docs/experiments/REPORT_v1.md#b-generalisation-and-faults)). A selected failure-and-recovery episode is traced in [output D](docs/experiments/REPORT_v1.md#d-failure-and-recovery-in-one-episode).

Nothing trained here was exported or deployed. The method follows published evaluation practice: repeated paired trials, deterministic graders on an independent ledger, an evaluator calibrated against deliberately defective controllers, controlled ablations and an audit of reward against outcomes. How each source was used, and what was not carried over, is in [section 1 of the report](docs/experiments/REPORT_v1.md#1-methodological-references).

## The near-tie the gate caught

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/v1/expF_decision_boundary-dark.svg">
  <img width="100%" alt="Panel a: states per band of the float64 gap between the two largest Q-values, for 18 networks: corpus-A rollout states, corpus-A synthetic states and the states each network visits under its own greedy policy. The generated .st selects another action than PyTorch on 2 states, both synthetic, in the band 1e-5 to 1e-4; on none of the 360,000 visited states. Panel b: the largest difference between .st and PyTorch Q-values per band, below the gate tolerance of 1e-3 everywhere: up to 1.8e-4 on synthetic states with gaps of 1e-5 to 1e-4, and below 1e-7 on rollout and visited states with gaps under 1e-2." src="docs/figures/v1/expF_decision_boundary-light.svg">
</picture>

**Figure F.** Decision-boundary audit (output F of the [benchmark-v1 report](docs/experiments/REPORT_v1.md)): the 18 networks of the export check on their 20,000 corpus-A inputs each, and on the 20,000 states each visits under its own greedy policy in the 100 development episodes of the historical simulator, stratified by the float64 gap between the two largest Q-values. Tie tolerance 0. The two rejected inputs are shown in detail in [Figure 3 of the evidence document](docs/EVIDENCE.md#figures-of-the-verification-increment). Data: [boundary_audit.json](experiments/v1/results/boundary_audit.json), [plotted values](experiments/v1/results/figure_data/expF_decision_boundary.csv).

The verification gate caught an action mismatch that Q-value checks alone would have accepted. At these two inputs the network's two largest Q-values, Quality focus and Aggressive, differ by 1.22×10⁻⁴: 4 and 2 binary32 ulps at |Q| ≈ 456 and 705. The generated code accumulates the output-layer products in a different order from PyTorch's GEMM kernel. Both values round to the same binary32 number, and deterministic first-maximum tie-breaking selects Aggressive. The setpoints follow the action: 1.20, 1.10 and 1.10 times nominal for speed, pressure and temperature, instead of 1.10, 1.05 and 0.95. Every Q-value of this network is within 7.3×10⁻⁴ of PyTorch, so a criterion on Q-values alone would accept the export.

At this gap the selected action depends on accumulation order. PyTorch's own Q-values for the two inputs change by up to 1.2×10⁻⁴ with batch size, although it selects Quality focus at each batch size tried. The default therefore stays strict. A re-check with an explicit tie tolerance of 2×10⁻³ counts both inputs as near-ties and passes, but the generated action and setpoints still differ there: it is a diagnostic, not behavioural agreement. Four tests in [`tests/test_iec61499_exporter.py`](tests/test_iec61499_exporter.py) pin the case:
- the generated outputs bit for bit;
- the strict verdict against PyTorch's recorded outputs;
- the end-to-end rejection, run only where PyTorch reproduces the recorded reference;
- the strict server default.

**Reachability.** Both flips lie on synthetic inputs, at float64 gaps of 7.2×10⁻⁵ and 8.9×10⁻⁵, where the generated code's Q-value error (up to 1.8×10⁻⁴ in that band) exceeds the gap. On the 360,000 states that the 18 networks visit under their own greedy policies, the `.st` selects PyTorch's action every time. That includes 114 visited states with gaps below 10⁻⁴, all from untrained networks, whose Q-value errors there stay below 10⁻⁷. The rejected network never comes closer to a tie than 0.040 on its visited states, 65 times its largest Q-value error there. With the same action at every visited state, the exported block reproduces the PyTorch closed-loop trajectories of those episodes exactly. The rejection stands under the strict predicate, but on these episodes it has no closed-loop consequence. One further synthetic input, at a gap of 5.5×10⁻⁶, shows PyTorch float32 itself disagreeing with the float64 reference; the `.st` agrees with PyTorch there.

## Verification coverage

What the checks exercised, measured 2026-11-08 ([Figure 4 of the evidence document](docs/EVIDENCE.md#figures-of-the-verification-increment)):
- the `.fbt` and the standalone `.st` on 18 networks × 20,000 sampled inputs;
- the `.st` on a synthetic corpus of 14 networks × 11,128 inputs: box vertices, ±0, subnormal and ±100 inputs, simulator rollouts and uniform draws;
- 10 injected `.fbt` defects and 112 `.st` mutant–network pairs;
- 3 broken files for the compiler check.

Nothing ran on a target runtime. Data: [plotted values](docs/data/fig4_verification_coverage.csv), [st_verification.json](docs/data/st_verification.json).

The trained policies select one or two actions over the 20,000 inputs, so on them the action check has little discriminating power. On the server-run network it catches 1 of the 5 injected `.fbt` defects; the Q-value and setpoint checks catch the other four.

The synthetic `.st` corpus is built to make the behavioural checks informative, and coverage of all six actions is asserted before any verdict is used:
- untrained networks, which select all six actions between them;
- one network per action with a dominating output bias, which tests the action/setpoint correspondence for every action;
- three networks with identical output rows for one action pair, which tests first-maximum tie-breaking on exact ties.

Mutation sensitivity is reported against reachability. Argmax → argmin, first → last maximum, and a wrong setpoint literal for each action make 8 mutants × 14 networks = 112 pairs. A mutant is reachable on a network when it changes an output there, which holds for 50 pairs; all 50 are detected. The first → last maximum mutant is observable only on the tie networks, which is why the corpus contains them.

The `.st` is checked in the interpreter only. EAE's FBTypeCompiler does not take raw `.st` files, and no other IEC 61131-3 compiler was run.

## What did not work

- **The learning result is negative, twice.** On the historical simulator the trained DQN never beat the best fixed action. On benchmark v1, no training change improved held-out policy quality under the preregistered criteria, and the best procedure stayed 42 EUR per shift below a tuned rule.
- **The original training budget is small:** one gradient step per 200-step episode, 1,000 updates in total. On benchmark v1, sixty times more updates reached the best fixed preset's level, not the rule's.
- **The simulator limits interpretation.** By default, slower settings finish parts faster, because each step adds 1/speed to the cycle counter. A corrected speed-proportional model, now selectable, does not change the outcome.
- **Training log.** A replay-minibatch TD-residual log of one run is kept in [docs/TRAINING_DIAGNOSTIC.md](docs/TRAINING_DIAGNOSTIC.md). It does not bound the action-value error and is not used as evidence here.

What was learned: compare a policy with fixed-action baselines before exporting it, and verify behaviour as well as Q-values. A collapsed policy makes action agreement almost uninformative, and a near-tie makes Q-value agreement insufficient. Score outcomes on a ledger kept apart from the training reward: here it caught a policy that earned more reward by making scrap.

## Reproduce

Python 3.11. The commands below, except `pip install`, were run on 2026-11-08 in this repository's existing `.venv`, with torch 2.1.0+cpu, numpy 1.24.3 and matplotlib 3.7.2.

```bash
pip install -r requirements.txt
python scripts/export_iec61499_fb.py visual_outputs/run/policy_snapshot_20261106_190932.pt --out-dir out --verify 12000
python docs/scripts/measure_export_verification.py   # .fbt: 18 networks, injected defects, EAE compiler if installed
python docs/scripts/measure_st_verification.py       # .st: both corpora and the mutation matrix (~5 min)
python docs/scripts/extract_near_tie_fixture.py      # the two rejected inputs, checked against the measurement
python docs/scripts/make_figures.py                  # the figures, from stored evidence
python -m pytest tests -m "not training"             # 59 tests; the training test is excluded
```

The benchmark-v1 experiment (CPU only; 48 min of wall-clock training on 18 worker processes here, 4.6 h of summed CPU time):

```bash
python experiments/v1/audit_v0.py                    # Phase 1: the historical environment and checkpoints
python experiments/v1/calibrate.py                   # dev only: rule grid, best fixed preset, evaluator validation
python experiments/v1/run_training.py --tag confirmatory --seeds 1,2,3,4,5,6 --episodes 1000 --workers 18
python experiments/v1/evaluate_all.py                # held-out nominal, map, fault families, trajectory
python experiments/v1/boundary_audit.py              # decision-boundary audit (Figure F)
python experiments/v1/summarize.py                   # tables in experiments/v1/results/summary.md
python docs/scripts/make_experiment_figures.py       # Figures A-F
```

Test markers: `training` starts DQN training; `service` launches the server in a separate process; `vendor` needs EAE Buildtime and skips without it; `platform_reference` runs only where PyTorch reproduces the recorded float32 reference. Adding `--eae-check` to the export command also compiles the `.fbt` with EAE's FBTypeCompiler, on Windows with EAE Buildtime installed.

**Clean environment, one configuration.** On Windows 11 (10.0.26200) with Python 3.11.14 and pip 24.0, a fresh virtual environment outside the source tree installed `requirements.txt` without conflicts: 86 packages, and `pip check` passed.
- From a fresh clone of `6c895bd` it reproduced every numerical result of the export check, the near-tie fixture and the figure data. The documented tests gave 43 passed, 1 deselected.
- With this revision's changes, including the 11 benchmark-v1 tests, it passed the CI test selection on 2026-11-09 (58 passed, 0 skipped), as did the existing environment.

EAE Buildtime was present on this machine, so its tests ran. Other operating systems, Python versions and hardware are untested.

**Continuous verification.** [`.github/workflows/portable-verification.yml`](.github/workflows/portable-verification.yml) runs the portable path on Ubuntu 22.04 with Python 3.11:
- the tests marked neither `training` nor `service`, with `STRICT_SKIPS=1`, so that any skip other than `vendor` or `platform_reference` fails the job;
- the export command;
- regeneration of the figure data, which must not change.

Status: validated locally, pending remote execution. These steps pass on Windows in the existing and the clean environment, but the workflow itself, including its Linux CPU-wheel install, has not run on GitHub.

Commands, seeds, tolerances, environments and hashes are in [docs/EVIDENCE.md](docs/EVIDENCE.md).

## Limitations and deployment boundary

- **Sampled-input agreement only.** Agreement holds on the sampled inputs: 20,000 per network for the `.fbt`, plus the synthetic `.st` corpus. It is not formal equivalence.
- **Near-ties outside the sample.** Where two Q-values differ by less than the accumulation error, the generated code can select another action and send that action's setpoints. The strict gate rejects an export when this happens on a sampled input, as it did for 1 of 18 here. On an unsampled input it can still happen.
- **No target-runtime validation.** No function block has been imported into an EAE solution, deployed or timed. The analysis assumes the target computes `REAL` as IEEE 754 binary32 with round-to-nearest; that is unverified. Acceptance by the FB type compiler is not evidence of correct execution.
- **Inputs are not conditioned.** The function block expects the simulator's dimensionless state (trained range [−2, 2]) and does no scaling or clipping.
- **Not a validated digital twin.** The process model is a synthetic 12-variable simulator, with no link to a plant, PLC or OPC UA server. The 3D robot view is not driven by it.
- **The policy is not worth deploying.** It never beats the best fixed action. The exporter and its verification are the contribution, not the policy.
- **No safety claim and no IEC 61499 conformance claim** beyond acceptance by EAE's FB type compiler.
- **Benchmark v1 is synthetic.** Its equations, parameters and disturbances are assumed, not identified from a plant, and it is not a digital twin. Its results speak to this benchmark, this DQN, this budget (240,000 interactions per run) and six seeds, not to reinforcement learning for process control in general. No newly trained policy was exported to the application or deployed.

## Artifact provenance and authorship

Every number above comes from a script and a stored data file, with hashes of the checkpoints, input corpora and generated files in [docs/EVIDENCE.md](docs/EVIDENCE.md). The original application is by Emin Alper Sensoy; the git history records 26 commits from 2025-08-01 to 2025-08-05. The exporter, interpreter, compiler wrapper, tests, evaluation harness, figures and documentation were written in November 2026 by Claude Code agents directed by the author. The author also made the acceptance decisions recorded in the evidence document, such as the zero tie tolerance. A read-only review by a second AI tool (OpenAI Codex) was requested with the prompt in [`visual_outputs/CODEX_REVIEW_PROMPT.md`](visual_outputs/CODEX_REVIEW_PROMPT.md); its report is not stored in this repository. `visual_outputs/REPORT.md` and `FINDINGS.md` are the 6 November 2026 assessment of the original code and are marked as historical. The benchmark-v1 experiment (8-9 November 2026) was designed, run and written up by a Claude Code agent directed by the author, under a protocol frozen before its confirmatory runs; its provenance is in [section 15 of the evidence document](docs/EVIDENCE.md#15-benchmark-v1-experiment-2026-11-08-to-2026-11-09).

## References and licence

- V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature* 518, 529–533 (2015).
- H. van Hasselt, A. Guez and D. Silver, "Deep reinforcement learning with double Q-learning," *AAAI* (2016).
- Methodological references for the benchmark-v1 evaluation (none concerns industrial control): Anthropic, [Demystifying evals for AI agents](https://www.anthropic.com/research/demystifying-evals-for-ai-agents); [Bloom](https://www.anthropic.com/research/bloom); [Petri 2.0](https://alignment.anthropic.com/2026/petri-v2/); [Reward tampering](https://www.anthropic.com/research/reward-tampering). How each was used is in [section 1 of the report](docs/experiments/REPORT_v1.md#1-methodological-references).
- IEC 61499-1:2012, *Function blocks — Part 1: Architecture*. IEC 61131-3:2013, *Programmable controllers — Part 3: Programming languages*.
- IEEE 754-2019, *Standard for Floating-Point Arithmetic*.

Released under the [MIT licence](LICENSE). EcoStruxure Automation Expert is Schneider Electric software and is not included.
