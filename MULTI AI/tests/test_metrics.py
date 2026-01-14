import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from envs.commons_env import CommonsEnvConfig
from protocols.protocols import NoChatProtocol
from agents.agents import GreedyAgent, CooperativeAgent, NormFollowingAgent
from runner.run_episode import run_episode
from metrics.episode_metrics import compute_episode_metrics

env_cfg = CommonsEnvConfig(n_agents=3, regen=5, action_max=3)
agents = [
    NormFollowingAgent(0, action_max=3, memory_mode="M0"),
    CooperativeAgent(1, action_max=3),
    GreedyAgent(2, action_max=3),
]
res = run_episode(env_cfg, NoChatProtocol(), agents)

m = compute_episode_metrics(res.history, n_agents=env_cfg.n_agents, action_max=env_cfg.action_max)
print("U_total:", m.U_total)
print("U_per_agent:", m.U_per_agent)
print("Coop:", round(m.cooperation_index, 4))
print("Gini:", round(m.gini_fairness, 4))
print("StabilityVar:", round(m.stability_var, 4))
print("Collapsed:", m.collapsed, "t_collapse:", m.t_collapse)
