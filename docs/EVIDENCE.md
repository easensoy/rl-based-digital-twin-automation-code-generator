# Evidence and reproduction

The numbers in [README.md](../README.md), where each comes from, and where the evidence stops. Measurements were made on 2026-11-08 unless marked historical.

## 1. Code revision

- Repository: `https://github.com/easensoy/rl-based-digital-twin-automation-code-generator.git`, branch `main`.
- **Measured before it was committed.** All measurements were made on 2026-11-08 on the working tree on top of `898c112` (2025-08-05). The new and changed files were committed to `main` later the same day, one file per commit; `git log` lists them. The hashes below identify the code that was measured.
- SHA-256 of the code at measurement time; the first four are also recorded in `docs/data/export_verification.json`. All five still had these hashes after the last check on 2026-11-08, so the README describes the code that was measured:

```
9d647826dbeb2906001c24046f03f25d1167860e3a00e0b536c15967488b934f  backend/iec61499_exporter.py
9169b62ed8499e1b215dba4d05a13695be7381c0d603a61d0ece9b7bfe7e15f5  backend/st_interpreter.py
b2b48d729ffb1dbeef6173a93bb4ed98206e6d7c220ac709fc2f8b8eadbd7903  backend/eae_fbtype_compiler.py
6aa619a463ddfaaaadcd8a51a8b357e408d64c8d553815bca3895bb0bc8e244a  backend/rl_agent.py
0a7c354e0f6681c6e191f5a40bb915cb054c6bde1f81d8962499885e7bd37447  backend/main.py
```

- `tests/test_iec61499_exporter.py` changed after the measurement: four near-tie regression tests were added (section 6). Its hash went from `0a133784…` to `0b1f0bea…7ee53734`.

## 2. What kind of evidence each claim rests on

| Kind | What | Files |
|---|---|---|
| **Historical, single run** (2026-11-06) | One training run of the unmodified server (seed 42, 1,000 episodes, 188.8 s), then 100 evaluation episodes per policy. Source of the untrained-network, server-run and old-generator rows, and of the [training diagnostic](TRAINING_DIAGNOSTIC.md). | `visual_outputs/run/evaluation.json`, `visual_outputs/data/training_loss.csv`, `visual_outputs/run/policy_snapshot_20261106_190932.pt` |
| **Historical, multi-seed** (2026-11-07) | Seeds 42 and 1–5, each trained for 1,000 episodes under both production models, evaluated greedily on the same 100 environment seeds as the baselines, so every comparison is paired. Source of figure 2. | `visual_outputs/run/eval_v2/summary_*.json`, `episodes_*.csv`, `policies/*.pt` |
| **Verified 2026-11-08** | Export, interpreter run of the `.fbt` and gate on 18 frozen networks; interpreter run of the `.st` for the server-run network only; 10 injected defects; EAE compiler on 2 `.fbt` exports and 3 broken variants; the two rejected states, re-extracted and pinned as a regression fixture; the bounded test run; the README's export command. | `docs/data/export_verification.json`, `tests/fixtures/near_tie_speed_proportional_seed5.json`, section 3 |
| **Test defined, not run 2026-11-08** | `test_code_generation_during_training` starts training. It passed in the full run of 2026-11-07 (40 passed). | `tests/test_iec61499_exporter.py` |
| **Proposed, not implemented** | Import into an EAE solution and deployment; training against a Visual Components or OPC UA model of a real rig; interlocks as action masks. | `visual_outputs/REPORT.md` ("Three next improvements") |
| **Not claimed** | Real-time digital twin, plant connection, GPU use, OPC UA/Modbus, safety function, general IEC 61499 conformance, beneficial learning. | n/a |

Training was not re-run for this write-up. `visual_outputs/REPORT.md`, `FINDINGS.md` and `screenshots/` describe `898c112` before the exporter, gate and dashboard fixes. Treat them as historical. `REPORT.md` rounds the untrained network to 1,713.7; `evaluation.json` holds 1,713.65, shown here as 1,713.6. The 2026-11-06 run evaluated on the same 100 environment seeds as the 2026-11-07 baselines, and its uniform-random and fixed-action results are identical to theirs, so its untrained-network row is comparable with the default-simulator results.

The server snapshot is not the network that offline seed 42 produces. Retrained offline with seed 42, the network picks "Aggressive"; the server's snapshot picks "Quality focus". `visual_outputs/run/eval_v2/rng_leak_check.json` traces the difference to WebSocket keep-alive pings drawing on Python's global `random` stream during the server run. That cause is likely but not proven end to end.

## 3. Commands executed on 2026-11-08

All were run from the repository root with `.venv\Scripts\python.exe` (Python 3.11.14, torch 2.1.0+cpu, numpy 1.24.3, matplotlib 3.7.2, pytest 7.4.3). Nothing was installed. No server was started and nothing was trained.

| Command | Outcome |
|---|---|
| `docs/scripts/measure_export_verification.py` | 24 s; wrote `docs/data/export_verification.json` |
| `docs/scripts/extract_near_tie_fixture.py` | 15 s; reproduced the state-set, checkpoint, weight and `.fbt` hashes and the 2 stored disagreements; wrote the fixture and `docs/data/near_tie_states.csv`. Two runs gave byte-identical files. |
| `docs/scripts/make_figures.py` | 10 SVG + 10 PNG in `docs/figures/`; 4 CSV in `docs/data/` |
| `python -m pytest tests/test_iec61499_exporter.py -k "near_tie or strict_gate"` | 5 passed: the 4 near-tie regression tests and the older synthetic near-tie test. The end-to-end test ran; it did not skip. |
| `python -m pytest tests -q -p no:cacheprovider -k "not during_training" -rs` | 43 passed, 1 deselected, 0 skipped, 0 failed (151 s; 167 s in a second run that profiled durations); EAE Buildtime 25.0 present, so no compiler test skipped. Earlier the same day the suite took 26 s. The slowdown is spread across older tests too, such as the server-start test at 29 s, and its cause was not investigated. |
| `scripts/export_iec61499_fb.py visual_outputs/run/policy_snapshot_20261106_190932.pt --out-dir <scratch> --verify 12000` | exit 0; 12,000 states, 100% agreement, max \|ΔQ\| 5.49×10⁻⁴ |
| `node <dataviz skill>/scripts/validate_palette.js` on the figure colours (calculated) | all checks pass on `#ffffff` and `#0d1117`. Aqua is 2.8:1 on white, so every aqua mark carries a label or a distinct marker shape. |
| WCAG 2 contrast of every text and status colour (calculated) | body text ≥ 7.4:1 on page and card colours in both themes; the muted grey (3.6:1 on white) is used only for large text and symbols; status icons ≥ 3.3:1 |
| Visual inspection at 830 px on GitHub's light and dark page colours (qualitative) | all four figures; label collisions and overflow fixed, every figure kept to the same 8-inch canvas so text renders at one size |

## 4. States, seeds and networks for the export check

- **States:** 20,000 per network, SHA-256 `2b7cfc12…3e21066`:
  - 10,000 from `iec61499_exporter.rollout_states(seed=20261108)`: the default ("legacy") simulator under uniform random actions;
  - 10,000 drawn uniformly from [−2, 2]¹² (`numpy.random.default_rng(20261108)`). The simulator clips states to that range.
- **Networks:**
  - the server snapshot above;
  - the 12 multi-seed policies;
  - `DQNNetwork(12, 6)` at `torch.manual_seed(0…4)`, untrained. On these states they use 1 to 6 actions each and all 6 between them.
- **Reference:** `rl_agent.DQNNetwork` in eval mode (dropout off), float32, CPU, all 20,000 states in one batch.
- **Export date:** pinned to 2026-11-08, so file hashes repeat: `.fbt` `5f528b23…701f0d00` (425,597 B), `.st` `c328ec3d…32458963` (491,841 B) for the snapshot.

## 5. Acceptance criteria and tolerances

`verify_fbt` returns `ok` only when all three checks pass on every state:

1. **Q-values:** max |Q_FB − Q_torch| < `q_tolerance` = 10⁻³.
2. **Action:** the FB's `ACTION` output equals `argmax` of the network. Ties go to the first maximum, as `torch.argmax` does.
3. **Setpoints:** `SPEED_SP`, `PRESSURE_SP` and `TEMP_SP` equal the `ACTION_SETPOINTS` row of the network's action exactly, as float32.

Why the Q-values differ at all. The generated ST adds the bias after the dot product, as `torch.addmm` does. But it accumulates terms in a different order from PyTorch's kernels, so results differ by a few ulps: at most 7.3×10⁻⁴, or 6 ulps, at |Q| ≤ 1,472.

**Near-ties.** If two actions' Q-values are closer than that error, the argmax can flip. `tie_tolerance`, set in the server under `code_generation.tie_tolerance`, defaults to 0, and then nothing is excused. Set above 0, it excuses a disagreement only where the FB's action is within the tolerance of the network's best Q-value. Excused states are counted in `near_tie_disagreements`, so they are never hidden. In this measurement `speed_proportional_seed5` disagrees on 2 states, so the strict gate rejects it (section 6). A separately recorded re-check with `tie_tolerance` = 2×10⁻³ (2 × `q_tolerance`) passes with both states counted. That re-check is a diagnostic, not a strict pass: the generated action and setpoints still differ on both states. The headline uses the strict result. On 2026-11-08 the repository owner decided to keep the default at 0, so a tolerance applies only when set explicitly; `test_server_default_keeps_the_strict_gate` checks the default in `backend/main.py` and `config/config.json`.

## 6. The two rejected states

Network `speed_proportional_seed5`: checkpoint `visual_outputs/run/eval_v2/policies/speed_proportional_seed5.pt`, SHA-256 `738ba0ca…0fff0b56` (weights `5048b068…3848c02f`). It was trained on 2026-11-07 with seed 5 under the corrected production model. Its export dated 2026-11-08 has `.fbt` SHA-256 `1cdad527…11df174ce`. Both states come from the uniform half of the state set: they are synthetic inputs inside the simulator's clipping range, not states the simulator visited.

| | State 13327 | State 14848 |
|---|---|---|
| Source | uniform draw 3,327 of 10,000 | uniform draw 4,848 of 10,000 |
| POS_X, POS_Y, POS_Z | −0.7379468, −0.2419244, −1.611551 | 0.9115475, −0.4681237, −1.669782 |
| VEL_X, VEL_Y, VEL_Z | 1.399571, 0.7049694, −1.932122 | −1.210441, 0.07155652, 1.885886 |
| TEMP, PRESSURE, VIBRATION | 1.09585, −1.23343, −1.41869 | −0.604011, −1.779255, −1.609841 |
| LOAD, QUALITY, ENERGY | −1.655823, −1.573928, −1.047497 | −1.016477, 0.5219349, −1.921787 |
| PyTorch Q: Quality focus / Aggressive | 456.453949 / 456.453827 | 704.549011 / 704.548889 |
| PyTorch gap | 1.22×10⁻⁴ = 4 ulps (ulp 3.05×10⁻⁵) | 1.22×10⁻⁴ = 2 ulps (ulp 6.10×10⁻⁵) |
| Generated Q: Quality focus / Aggressive | 456.453918 / 456.453918, an exact tie | 704.548889 / 704.548889, an exact tie |
| Largest \|ΔQ\| over all six actions | 1.22×10⁻⁴ | 1.83×10⁻⁴ |
| Action: PyTorch → generated | Quality focus → Aggressive | Quality focus → Aggressive |
| Setpoints (speed, pressure, temperature) | 1.10, 1.05, 0.95 → 1.20, 1.10, 1.10 | 1.10, 1.05, 0.95 → 1.20, 1.10, 1.10 |
| Q-only check / strict gate | accept / reject | accept / reject |

Inputs are shown to 7 significant digits, Q-values to 6 decimals. The fixture [`tests/fixtures/near_tie_speed_proportional_seed5.json`](../tests/fixtures/near_tie_speed_proportional_seed5.json) holds the inputs and all six Q-values from both sides exactly, with their binary32 bit patterns; [`docs/data/near_tie_states.csv`](data/near_tie_states.csv) has the same values as one row per state. Both argmax implementations keep the first maximum. In the generated code the pair is tied, so action 2, Aggressive, wins. In PyTorch, Quality focus is strictly larger, so action 4 wins.

PyTorch's own result for these rows depends on the batch they are computed in. The gate evaluates all 20,000 states in one batch. In a batch of 2 the gap is 1.53×10⁻⁴ (state 13327) and 6.10×10⁻⁵ (state 14848); one state at a time it is 9.16×10⁻⁵ and 1.22×10⁻⁴. PyTorch picks Quality focus in every case. The `.st` gives Q-values bit-identical to the `.fbt` on both states.

The fixture is written by `docs/scripts/extract_near_tie_fixture.py`. Before writing, the script reproduces the state-set hash, the checkpoint and weight hashes, the `.fbt` hash and the stored disagreement indices, so the fixture cannot silently drift from the measurement. Four tests in `tests/test_iec61499_exporter.py` use it:

- `test_near_tie_fixture_generated_outputs_are_reproduced`: the generated code's Q-values bit for bit, the exact tie, its action and its setpoints;
- `test_strict_gate_rejects_near_tie_a_q_only_check_accepts`: the strict verdict against PyTorch's recorded outputs, so it does not depend on the machine's summation order; it also checks that the diagnostic re-check counts both states;
- `test_strict_gate_rejects_near_tie_checkpoint_on_measured_states`: end to end with live PyTorch on the 20,000 states; it skips if this machine's PyTorch gives other bits for the two states;
- `test_server_default_keeps_the_strict_gate`: the default tie tolerance is 0 in `backend/main.py`, `config/config.json` and `verify_fbt`.

## 7. Compiler diagnostics: raw against classified

`FBTypeCompiler.exe` (EAE 25.0 Buildtime, default options) prints two kinds of message for every standalone compile. The same messages appear for Schneider Electric's own library type `SE.App2WWW/rotationArrayValidation.fbt`:

- `ERR_XML_SNAPSHOT_GUID_INCONSISTENT` (no base snapshot for a non-incremental compile), once per type;
- `ERR_SYNTAX: '[...]' is not correct structured text syntax`, once per array `InitialValue`. Each such line repeats that whole array, up to about 18 kB.

`eae_fbtype_compiler.classify` puts exactly these into `context_messages`. It matches the `ERR_SYNTAX` text against the file's own initialisers, and treats every other error or warning as a failure. The API returns all three groups as raw lines, and the dashboard shows them on request. Because these known messages are classified out, the README reports "no unfiltered errors or warnings" rather than "no errors".

| Input | Exit code | Unfiltered errors / warnings | Known messages | Verdict |
|---|---|---|---|---|
| Server snapshot export | 0 | 0 / 0 | 23 (1 + 22 arrays) | accepted, 449,289 B `.bin` |
| Untrained seed-0 export | 0 | 0 / 0 | 23 | accepted |
| Undeclared variable added | 1 | `ERR_NO_SUCH_VAR` / 0 | 23 | rejected |
| `REAL` assigned to `INT` | 1 | `ERR_CAST_UNDEFINED` / 0 | 23 | rejected |
| Weight array one value short | 0 | 0 / `WRN_FBTYPE_XMLPARSER … WRN_ARRAY_NOINIT` | 23 | rejected by the wrapper (the compiler still writes a `.bin`) |

Acceptance means only that the FB type compiler accepted the type. No EAE solution build, deployment or runtime test was done, and the other 16 exports were not compiled.

## 8. Figure sources

All figures are drawn by `docs/scripts/make_figures.py` from stored files only.

| Figure | Inputs (SHA-256 prefix) | Plotted values |
|---|---|---|
| 1, export and checks (README hero) | `docs/data/export_verification.json` (`ed28e8ed`) | `docs/data/fig1_export_verification.csv` |
| 2, policy evaluation | `eval_v2/summary_legacy.json` (`d49f37f1`), `summary_speed_proportional.json` (`00b8a6d1`), `evaluation.json` (`67be5953`) | `docs/data/fig2_policy_evaluation.csv` |
| 3, the near-tie | `tests/fixtures/near_tie_speed_proportional_seed5.json` (`875e9585`), checked against `docs/data/near_tie_states.csv` (`40a8f0b3`) | `docs/data/near_tie_states.csv` |
| 4, coverage | `docs/data/export_verification.json` | `docs/data/fig4_verification_coverage.csv` |
| Supporting, [training diagnostic](TRAINING_DIAGNOSTIC.md) | `visual_outputs/data/training_loss.csv` (`99f5b9ff`), `evaluation.json` | `docs/data/supp_td_error_vs_margin.csv` |

The policy snapshot is `d79f4d10…375e2990`. Colours come from the validated palette, with a light and a dark file per figure; status icons always carry a text label. Text is converted to paths in the SVGs, so they render the same without fonts installed. Figure 2 asserts that the 2026-11-06 run used the same evaluation episodes as the 2026-11-07 baselines before it plots the untrained network beside them.

## 9. Dependencies the presented workflow needs

| Need | Packages |
|---|---|
| Export, interpreter, gate | Python 3.11, `numpy==1.24.3`, `torch==2.1.0` (CPU) |
| Figures | `matplotlib==3.7.2` |
| Tests | `pytest==7.4.3`; `fastapi`, `uvicorn` and `starlette` for the server tests |
| DTD test | `lxml==6.1.3`, the version installed in `.venv`, pinned on 2026-11-08. Before that it came in only as an unpinned dependency of `opcua`; the test skips without it. |
| Compiler checks | EAE Buildtime 25.0 on Windows (optional; tests skip without it) |

`requirements.txt` also pins packages this workflow does not use, such as OPC UA, Modbus, Celery and Redis.

## 10. Not verified

- Agreement on inputs outside the 20,000 sampled states per network.
- Import of the `.fbt` into an EAE solution, build, deployment to a controller, execution and cycle time.
- Float behaviour of a target runtime. IEEE 754 binary32 with round-to-nearest is assumed.
- Whether PyTorch on other hardware rounds the two near-tie states the same way; the end-to-end regression test skips if not.
- Behaviour outside the simulator's state range; the FB does not scale or clip inputs.
- Any link to a plant, PLC, OPC UA server or Visual Components model; the 3D view is not driven by the simulator.
- A clean `pip install -r requirements.txt` on a new machine.
- The dashboard beyond one manual browser check on 2026-11-07 (live telemetry, generation, downloads, stale state on disconnect).
