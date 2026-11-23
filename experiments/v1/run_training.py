"""
Run train_dqn.py jobs in parallel subprocesses, one CPU thread each, and write a manifest with every job's
return code and elapsed time (failed runs are listed, not dropped).

Run:  .venv\\Scripts\\python.exe experiments\\v1\\run_training.py --tag confirmatory --seeds 1,2,3,4,5,6
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SLOW_FIRST = ["I5", "I4", "R1", "I3", "I2", "I1", "I0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="I0,I1,I2,I3,I4,I5,R1")
    ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--tag", default="confirmatory")
    a = ap.parse_args()
    conds = [c for c in SLOW_FIRST if c in a.conditions.split(",")]
    jobs = [(c, int(s)) for c in conds for s in a.seeds.split(",")]
    logs = HERE / "results" / "logs" / a.tag
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}

    def one(job):
        c, s = job
        cmd = [sys.executable, str(HERE / "train_dqn.py"), "--condition", c, "--seed", str(s),
               "--episodes", str(a.episodes), "--tag", a.tag]
        t = time.perf_counter()
        with open(logs / f"{c}_seed{s}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
        print(f"{c} seed {s}: rc {rc}, {time.perf_counter() - t:.0f} s", flush=True)
        return dict(condition=c, seed=s, returncode=rc, wall_s=round(time.perf_counter() - t, 1))

    start = dt.datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(one, jobs))
    manifest = dict(tag=a.tag, episodes=a.episodes, workers=a.workers, started=start,
                    finished=dt.datetime.now().isoformat(timespec="seconds"),
                    wall_s=round(time.perf_counter() - t0, 1), cpu_count=os.cpu_count(), jobs=results,
                    failed=[r for r in results if r["returncode"] != 0])
    (HERE / "results" / f"manifest_{a.tag}.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                                             encoding="utf-8")
    print(f"{len(results)} jobs, {len(manifest['failed'])} failed, {manifest['wall_s']} s")


if __name__ == "__main__":
    main()
