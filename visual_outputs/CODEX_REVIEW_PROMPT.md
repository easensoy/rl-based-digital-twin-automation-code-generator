# Codex review prompt (read-only)

Start Codex inside the repository folder, not anywhere else:
`cd C:\rl-based-digital-twin-automation-code-generator`, then `codex --sandbox read-only`.
Then paste everything below the line.

---

You are a review panel of three people. Review only: you are not implementing anything.

## Step 0: confirm the repository before anything else

Review exactly one repository, and nothing else:

| | |
|---|---|
| Local folder | `C:\rl-based-digital-twin-automation-code-generator` |
| GitHub | `https://github.com/easensoy/rl-based-digital-twin-automation-code-generator` |
| Branch | `main` |
| Base commit | `898c112` (2025-08-05, "Implemented configuration-driven approach allows for easier customization of sensor parameters and action mappings") |

Most of the work to review is **uncommitted** and exists only in the local folder. Review the local
folder, not GitHub, because GitHub only has commit `898c112`.

Run these read-only checks first. Each must print exactly the expected value:
- `git rev-parse --show-toplevel` → `C:/rl-based-digital-twin-automation-code-generator`
- `git remote get-url origin` → `https://github.com/easensoy/rl-based-digital-twin-automation-code-generator.git`
- `git rev-parse --short HEAD` → `898c112`
- `backend/rl_agent.py`, `backend/iec61499_exporter.py` and `visual_outputs/REPORT.md` must all exist.

If any check fails, stop and reply only: `Wrong repository: <what you found>`. Do not review any other
repository, and do not adapt this review to another project.

1. **A staff ML / RL engineer at a frontier AI lab.** You care about evaluation rigour, baselines,
   seeds, uncertainty, and whether the claims survive contact with the code.
2. **A principal controls engineer.** You know IEC 61131-3, IEC 61499, PLC real-time execution,
   EcoStruxure Automation Expert (EAE), TIA Portal and CODESYS. You judge whether a plant engineer
   would trust this.
3. **A hiring manager and technical recruiter.** You screen GitHub portfolios for roles at
   Anthropic, NVIDIA (robotics, Isaac / Omniverse digital twins) and Tesla (manufacturing automation,
   Optimus). You decide in 60 seconds whether to reach out.

## Hard rules

- **Do not modify anything.** Do not create, edit, delete, move or format any file. Do not stage,
  commit, branch or push. Do not write patches or diffs.
- **Stay inside `C:\rl-based-digital-twin-automation-code-generator`.** Do not open any file outside it.
- **Do not run anything that writes files or starts services.** That includes training, servers,
  the scripts in `visual_outputs/scripts/` and `scripts/`, and package installs. You may read files
  and run read-only commands (`git status/log/show/diff`, `rg`, directory listings).
- **Running the tests is optional.** You may run
  `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests` once, with
  `PYTHONDONTWRITEBYTECODE=1`, if your sandbox allows writing to the system temp folder. If it does
  not, do not count sandbox failures as project failures.
- **Skip the bulk folders.** Ignore `.venv/`, `wmg_env/` and `__pycache__/`. Only sample the head of
  the large generated files in `data/generated_code/iec61499/`.
- **Label your evidence.** Back every finding with evidence as `file:line`, a command and its output,
  or a figure name. Mark anything you could not check as **UNVERIFIED**. If you cannot open images,
  judge the figures from `visual_outputs/scripts/make_plots.py` and the CSVs in
  `visual_outputs/data/`, and say so.
- **No generic advice.** "Add tests" or "improve the README" does not count. Say which test, which
  section, and why a top-tier reader would care.

## Context (treat as claims to verify, not facts)

**The owner.** A Lead Engineer in an industrial automation research group (WMG, University of
Warwick). Their day job is IEC 61499 and EAE: generating function-block systems from VueOne models
through CAT templates.

**The goal.** Make this repo a GitHub portfolio piece that makes engineers and recruiters at
Anthropic, NVIDIA and Tesla want to talk to them. The outputs and visuals must reach that bar, and
the repo should visibly use Anthropic-style engineering practices:
- evaluation-first development with explicit success criteria and baselines;
- honest reporting of negative results;
- independent verification loops;
- safety constraints by construction;
- transparent agentic development with Claude Code.

**Original repo.** Commit `898c112`: a FastAPI + PyTorch DQN "digital twin" that claims to generate
"IEC 61499 compliant" PLC code. A review of that commit found:
- `README.md` is empty and both test files were empty.
- The frontend never connected: ES-module files were loaded as classic scripts, and an undefined
  `createSimpleRobot` was called. It showed hard-coded values.
- The twin is a 12-variable stochastic toy (`backend/rl_agent.py`). The richer environment in
  `backend/automation_environment.py` is never imported.
- `backend/plc_generator.py` ignores the network weights. Its fixed template is IEC 61131-3 ST with
  nested POUs, not IEC 61499, and it always selects action 2 (0% agreement with the trained network).
- The DQN does not learn this task. Reward per episode over 100 evaluation episodes on identical
  seeds: untrained 1,713.7, trained 1,682.6, uniform random 1,716.0, best fixed action ("Energy
  saving") 1,827.4. Over 6 seeds, every trained policy collapses to one fixed action. TD loss rises
  from 72 to 7,847, with one gradient step per 200 environment steps.
- The evidence is in `visual_outputs/`: `REPORT.md`, `REPORT.html`, `FINDINGS.md`, figures `01`–`07`,
  `screenshots/`, `plc_code/`, `run/evaluation.json`, `run/multi_seed_check.json`, and the
  evaluation harness in `visual_outputs/scripts/`.

**New, uncommitted work** (see `git status`, and `git diff` for the modified files):
- `backend/iec61499_exporter.py`: writes the trained weights into an EAE-dialect IEC 61499 Basic FB
  (`.fbt`) and an IEC 61131-3 FUNCTION_BLOCK (`.st`).
- `backend/st_interpreter.py`: a strict float32 interpreter that runs the generated ST and the
  Basic FB execution model, so the generated code can be checked against PyTorch.
- `backend/eae_fbtype_compiler.py`: runs EAE's own `FBTypeCompiler.exe`.
- `scripts/export_iec61499_fb.py`: the command-line entry point.
- `tests/test_iec61499_exporter.py`: about 22 tests, including fidelity checks, mutation-style
  "detects generator bugs" checks, and EAE compiler acceptance and rejection.
- Edits to `backend/main.py`, `frontend/main.js`, `frontend/js/core/websocket-manager.js`,
  `frontend/css/components/dashboard.css` and `wmg-digital-twin.html`, so that the dashboard's CODE
  panel generates, checks and downloads the FB.
- `REPORT.md` and `FINDINGS.md` were written **before** this work. Judge how stale they now are.

**Housekeeping as found.**
- There is no `.gitignore`.
- Bytecode is tracked.
- A broken old virtualenv `wmg_env/` sits in the root.
- Three PDFs sit in the root: a Sutton & Barto draft textbook, a CC BY 4.0 article, and a second copy
  of that article carrying Springer terms of use.
- The owner's commit convention is one file per commit.

## What to deliver

Deliver one markdown review, at most about 2,500 words, in this order.

1. **Verdict (5 lines).** Would a recruiter or an Anthropic, NVIDIA or Tesla engineer reach out
   today? Why or why not? What is the single strongest asset, and the single biggest liability?

2. **Scorecard.** Score each row 1–10, calibrated against strong public portfolios, not average
   student repos. Give each a one-line justification and the one change that would raise it most.

   | Dimension |
   |---|
   | First impression (README, root, 60-second skim) |
   | ML / RL technical depth |
   | Evaluation rigour (success criteria, baselines, seeds, uncertainty, held-out data) |
   | Industrial / PLC credibility (IEC 61131-3/61499, EAE, real-time, safety) |
   | Software engineering (structure, tests, typing, CI, packaging, configuration) |
   | Verification design (interpreter, vendor compiler, mutation checks): is it convincing or circular? |
   | Visual and written communication |
   | Honesty and calibration |
   | Legibility of Anthropic-style practices |
   | Reproducibility |
   | Security and licensing hygiene |

3. **Top 10 issues.** Rank them by how much each would change a hiring reader's opinion, divided by
   effort. For each, give:
   - the evidence;
   - why a top-tier reader cares;
   - the exact change, described in prose (no code);
   - the effort, S, M or L.

4. **Overclaims and interview red flags.** List anything in docstrings, UI strings, the report,
   commit messages or file names that claims more than the code does, or that would be awkward to
   defend in a technical interview. Include the old "IEC 61499 compliant" wording and the pre-fix
   descriptions.

5. **Figures and visuals.** For each of `visual_outputs/01`–`07`, the screenshots and `REPORT.html`,
   give:
   - the 5-second takeaway as it stands;
   - whether the title states the conclusion;
   - axes, units, baselines and uncertainty;
   - consistency with the other figures;
   - colour-blind safety;
   - **keep / fix / cut**, with the precise fixes.

   Then propose:
   - the README hero figure and the 3 supporting figures;
   - an architecture diagram: what it must show, as a list of boxes and arrows, not a drawing;
   - a short demo clip of the dashboard: what it should show, step by step.

   The bar is a figure an NVIDIA technical blog, a Tesla AI Day slide or an Anthropic research post
   would ship: one message per figure, a takeaway title, baselines and uncertainty visible, no chart
   junk.

6. **Anthropic-style practices: shown, claimed or missing.** For each practice listed under the goal
   above, say whether the repo shows it with evidence, merely claims it, or lacks it. Then say how to
   make it legible to an Anthropic engineer without buzzwords or overclaiming. Cover:
   - a policy "model card" (intended use, evaluation, limitations);
   - a "What did not work" section;
   - documenting the human/agent split honestly.

7. **README blueprint.** An outline only: section order, what sits above the fold, and the one
   results table that must appear. No finished prose.

8. **Five interview questions.** Questions an Anthropic, NVIDIA or Tesla engineer would ask about
   this repo. For each, rate how well the repo currently supports a strong answer (strong, weak or
   missing) and what is needed.

9. **Two-week plan.** Milestones with acceptance criteria that can be checked. Separate
   "must before going public" from "nice to have". Include what to delete or ignore before
   publishing: environments, bytecode, logs, compiled `.bin` files, PDFs under restrictive terms.
   Include how to structure the commits so the history reads well to a reviewer, given the
   one-file-per-commit convention.

Be blunt and specific, and do not flatter. If the strongest story here is the honest negative result
plus the verified exporter rather than the RL itself, say so and explain how to lead with it.
