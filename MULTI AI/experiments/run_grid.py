import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import random
from dataclasses import asdict
from typing import List, Dict, Any

from envs.commons_env import CommonsEnvConfig
from protocols.protocols import NoChatProtocol, RoundtableProtocol, MediatorProtocol
from agents.agents import GreedyAgent, CooperativeAgent, NormFollowingAgent
from runner.run_episode import run_episode
from metrics.episode_metrics import compute_episode_metrics

def build_agents(n: int, action_max: int, memory_mode: str, rng: random.Random):
    # 60% norm, 20% coop, 20% greedy
    n_norm = int(round(0.6 * n))
    n_coop = int(round(0.2 * n))
    n_greedy = n - n_norm - n_coop
    # adjust if rounding caused issues
    while n_norm + n_coop + n_greedy != n:
        n_norm = max(0, n_norm - 1)
        n_greedy = n - n_norm - n_coop

    agents = []
    aid = 0
    for _ in range(n_norm):
        agents.append(NormFollowingAgent(aid, action_max=action_max, memory_mode=memory_mode))
        aid += 1
    for _ in range(n_coop):
        agents.append(CooperativeAgent(aid, action_max=action_max))
        aid += 1
    for _ in range(n_greedy):
        agents.append(GreedyAgent(aid, action_max=action_max))
        aid += 1

    # shuffle ids? keep consistent for now; but shuffle order of agent list for fairness
    rng.shuffle(agents)
    return agents

def get_protocol(name: str):
    if name == "no_chat":
        return NoChatProtocol()
    if name == "roundtable":
        return RoundtableProtocol()
    if name == "mediator":
        return MediatorProtocol()
    raise ValueError("Unknown protocol: " + name)

def run_grid():
    Ns = [3,5,7]
    regens = [8,5,3]
    protocols = ["no_chat", "roundtable", "mediator"]
    memories = ["M0", "M1"]
    seeds = list(range(5))

    rows: List[Dict[str, Any]] = []

    for N in Ns:
        for regen in regens:
            for prot in protocols:
                for mem in memories:
                    for seed in seeds:
                        rng = random.Random(1000 + seed + 31*N + 7*regen)

                        env_cfg = CommonsEnvConfig(n_agents=N, regen=regen, action_max=3)
                        agents = build_agents(N, action_max=env_cfg.action_max, memory_mode=mem, rng=rng)
                        protocol = get_protocol(prot)

                        ep = run_episode(env_cfg, protocol, agents)
                        m = compute_episode_metrics(ep.history, n_agents=N, action_max=env_cfg.action_max)

                        row = {
                            "N": N,
                            "regen": regen,
                            "protocol": prot,
                            "memory": mem,
                            "seed": seed,
                            "U_total": m.U_total,
                            "cooperation": m.cooperation_index,
                            "gini": m.gini_fairness,
                            "stability_var": m.stability_var,
                            "collapsed": int(m.collapsed),
                            "t_collapse": -1 if m.t_collapse is None else m.t_collapse,
                            "U_per_agent": m.U_per_agent,
                        }
                        rows.append(row)

    # save as CSV
    os.makedirs("results", exist_ok=True)
    outpath = os.path.join("results", "phase1_metrics.csv")
    import csv
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("Saved:", outpath, "rows=", len(rows))

if __name__ == "__main__":
    run_grid()
