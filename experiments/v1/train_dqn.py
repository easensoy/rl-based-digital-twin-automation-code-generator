"""
Train one DQN run on benchmark v1 under one condition of docs/experiments/PROTOCOL_v1.md and record its
learning curve on the development scenarios.

Conditions (each differs from its control in one respect):
  I0  original procedure: backend.rl_agent.DQNAgent.act and .replay unchanged (dropout active when acting and
      inside the target network, one gradient update at the end of each episode, target copied every 10
      updates, MSE loss); intended reward with the shortfall charged at the last step
  I1  I0 + eval-mode fix: dropout off when acting and when computing bootstrap targets
  I2  I0 + update ratio: one gradient update every 4 environment steps instead of one per episode
  I3  I1 + I2
  I4  I3 + reward accounting: the shortfall cost settled as the order is filled (process.VecProcess
      reward="incremental"; same ranking of action sequences, credit no longer delayed to the last step)
  I5  I4 + Double DQN target (the online network chooses the bootstrap action, the target network values it)
  R1  I3 trained on the misspecified "throughput" reward (reward-specification comparison, output C)

All conditions: network DQNNetwork(11, 6) from backend/rl_agent.py, Adam lr 1e-3, batch 64, buffer 10,000,
gamma 0.99, and epsilon = max(0.01, 0.995 ** episode) set at the start of each episode. That is the original
schedule when updates happen once per episode; fixing it per episode keeps exploration identical when the
update ratio changes (the original decays epsilon inside replay(), so 60 updates per episode would also
shorten exploration 60-fold). A seed fixes the network initialisation, the agent's random streams and the
sequence of training scenarios, so runs with the same seed are paired across conditions.

Checkpoints are evaluated greedily, in eval mode, on the 100 development scenarios and scored by the outcome
ledger. Held-out scenarios are never evaluated here.

Run:  .venv\\Scripts\\python.exe experiments\\v1\\train_dqn.py --condition I3 --seed 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO)]

import controllers as C  # noqa: E402
import scenarios as S  # noqa: E402
from backend.rl_agent import DQNAgent, DQNNetwork  # noqa: E402
from evaluate import run, summary  # noqa: E402
from process import P0, VecProcess, make_scenarios  # noqa: E402

AGENT_CFG = json.loads((REPO / "config" / "config.json").read_text(encoding="utf-8"))["rl_agent"]
BASE = dict(eval_fix=False, update_every=0, reward="intended", double=False)
CONDITIONS = {
    "I0": dict(BASE),
    "I1": dict(BASE, eval_fix=True),
    "I2": dict(BASE, update_every=4),
    "I3": dict(BASE, eval_fix=True, update_every=4),
    "I4": dict(BASE, eval_fix=True, update_every=4, reward="incremental"),
    "I5": dict(BASE, eval_fix=True, update_every=4, reward="incremental", double=True),
    "R1": dict(BASE, eval_fix=True, update_every=4, reward="throughput"),
}
EPISODES = 1000
CODE_FILES = ["experiments/v1/process.py", "experiments/v1/ledger.py", "experiments/v1/scenarios.py",
              "experiments/v1/controllers.py", "experiments/v1/evaluate.py", "experiments/v1/train_dqn.py",
              "backend/rl_agent.py"]


def eval_points(episodes: int):
    pts = {5, 10, 17, 25} | set(range(50, episodes + 1, 50)) | {episodes}
    return sorted(p for p in pts if p <= episodes)


def code_hashes() -> dict:
    return {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in CODE_FILES}


class Agent(DQNAgent):
    """With eval_fix=False and double=False, act() and replay() are the original methods."""

    def __init__(self, eval_fix: bool, double: bool):
        super().__init__(11, 6, AGENT_CFG)
        self.eval_fix, self.double = eval_fix, double
        self.epsilon_decay = 1.0          # epsilon is set per episode by the training loop
        if eval_fix:
            self.target_network.eval()    # load_state_dict() keeps the mode, so this lasts

    def act(self, state):
        if not self.eval_fix:
            return int(super().act(state))
        if np.random.random() <= self.epsilon:           # same random-stream use as the original
            return random.randrange(self.action_size)
        self.q_network.eval()
        with torch.no_grad():
            q = self.q_network(torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0))
        self.q_network.train()
        return int(np.argmax(q.numpy()))

    def replay(self):
        if not (self.eval_fix or self.double):
            return super().replay()
        if len(self.memory) < self.batch_size:
            return
        s, a, r, s2, d = self.memory.sample(self.batch_size)
        s, s2 = torch.FloatTensor(s), torch.FloatTensor(s2)
        a, r, d = torch.LongTensor(a), torch.FloatTensor(r), torch.BoolTensor(d)
        q = self.q_network(s).gather(1, a.unsqueeze(1))
        with torch.no_grad():
            if self.double:
                self.q_network.eval()
                a2 = self.q_network(s2).argmax(1, keepdim=True)
                self.q_network.train()
                q2 = self.target_network(s2).gather(1, a2).squeeze(1)
            else:
                q2 = self.target_network(s2).max(1)[0]
        loss = F.mse_loss(q.squeeze(), r + self.gamma * q2 * ~d)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.update_target_network()


def evaluate_checkpoint(state_dict: dict, dev, reward_kind: str, gamma: float) -> dict:
    out = run(C.dqn(state_dict), dev, keep_traj=True)
    acts = out.pop("traj")["action"].astype(int)
    res = summary(out)
    res["action_share"] = (np.bincount(acts.ravel(), minlength=6) / acts.size).round(4).tolist()
    res["switches_per_episode"] = float((np.diff(acts, axis=1) != 0).sum(axis=1).mean())
    # Q(s0, greedy) against the realised discounted return of the greedy policy (training reward)
    net = DQNNetwork(11, 6)
    net.load_state_dict(state_dict)
    net.eval()
    env = VecProcess(dev, reward=reward_kind)
    obs = env.reset()
    with torch.no_grad():
        q0 = net(torch.from_numpy(obs)).max(1)[0].numpy()
    G, disc = np.zeros(len(dev)), 1.0
    for _ in range(P0.steps):
        with torch.no_grad():
            a = net(torch.from_numpy(obs)).argmax(1).numpy()
        obs, r, _ = env.step(a)
        G += disc * r
        disc *= gamma
    res["q0_mean"], res["g0_mean"] = float(q0.mean()), float(G.mean())
    return res


def train(condition: str, seed: int, episodes: int, tag: str) -> dict:
    cfg = CONDITIONS[condition]
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = Agent(cfg["eval_fix"], cfg["double"])
    train_sc = make_scenarios(S.train_seeds(seed, episodes), S.TRAIN_SPEC)
    dev = S.dev()
    at = set(eval_points(episodes))
    curve, returns, interactions = [], [], 0
    t0, c0, eval_s = time.perf_counter(), time.process_time(), 0.0
    for ep in range(episodes):
        agent.epsilon = max(agent.epsilon_min, 0.995 ** ep)
        env = VecProcess(train_sc.subset(np.array([ep])), reward=cfg["reward"])
        obs, total = env.reset()[0], 0.0
        for _ in range(P0.steps):
            a = agent.act(obs)
            nxt, r, done = env.step(np.array([a]))
            nxt, r, done = nxt[0], float(r[0]), bool(done)
            agent.remember(obs, a, r, nxt, done)
            obs, total, interactions = nxt, total + r, interactions + 1
            if cfg["update_every"] and interactions % cfg["update_every"] == 0:
                agent.replay()
        if not cfg["update_every"] and len(agent.memory) > agent.batch_size:   # as backend/main.py
            agent.replay()
        returns.append(round(total, 3))
        if ep + 1 in at:
            e0 = time.perf_counter()
            sd = {k: v.detach().clone() for k, v in agent.q_network.state_dict().items()}
            point = dict(episode=ep + 1, interactions=interactions, updates=agent.step_count,
                         wall_s=round(e0 - t0 - eval_s, 2), cpu_s=round(time.process_time() - c0, 2),
                         epsilon=round(agent.epsilon, 4))
            point.update(evaluate_checkpoint(sd, dev, cfg["reward"], agent.gamma))
            curve.append(point)
            eval_s += time.perf_counter() - e0
            print(f"{condition} seed {seed} ep {ep + 1:4d} upd {agent.step_count:6d} dev margin "
                  f"{point['margin']:8.2f} eps {agent.epsilon:.3f} q0 {point['q0_mean']:8.1f} "
                  f"g0 {point['g0_mean']:8.1f}", flush=True)
    ckpt = REPO / "experiments" / "v1" / "checkpoints" / tag / f"{condition}_seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.q_network.state_dict(), ckpt)
    return dict(condition=condition, config=cfg, agent_config=AGENT_CFG, seed=seed, episodes=episodes,
                tag=tag, interactions=interactions, updates=agent.step_count,
                train_wall_s=round(time.perf_counter() - t0 - eval_s, 2), eval_wall_s=round(eval_s, 2),
                cpu_s=round(time.process_time() - c0, 2), checkpoint=str(ckpt.relative_to(REPO).as_posix()),
                checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(), curve=curve,
                train_returns=returns, code_sha256=code_hashes(),
                versions=dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__))


def no_power_throttling():
    """Windows only: opt this process out of EcoQoS execution-speed throttling, which otherwise lets the
    scheduler park long-running background jobs on efficiency cores. Affects this process only."""
    if sys.platform != "win32":
        return
    import ctypes

    class State(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
    st = State(1, 0x1, 0x0)    # PROCESS_POWER_THROTTLING_EXECUTION_SPEED controlled, and off
    k32 = ctypes.windll.kernel32
    k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))


def main():
    no_power_throttling()
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--tag", default="confirmatory")
    a = ap.parse_args()
    res = train(a.condition, a.seed, a.episodes, a.tag)
    out = HERE / "results" / "runs" / a.tag / f"{a.condition}_seed{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}: {res['train_wall_s']} s training, {res['updates']} updates")


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    main()
