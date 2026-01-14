import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from envs.commons_env import CommonsEnvConfig
from protocols.protocols import NoChatProtocol, RoundtableProtocol, MediatorProtocol
from agents.agents import GreedyAgent, CooperativeAgent, NormFollowingAgent
from runner.run_episode import run_episode

def run_one(protocol):
    env_cfg = CommonsEnvConfig(n_agents=3, regen=5, action_max=3)
    agents = [
        NormFollowingAgent(0, action_max=3, memory_mode="M0"),
        CooperativeAgent(1, action_max=3),
        GreedyAgent(2, action_max=3),
    ]
    res = run_episode(env_cfg, protocol, agents)
    print("CONFIG:", res.config)
    print("Last step:", res.history[-1])
    print("Steps:", len(res.history))
    print("Collapsed?", any(h["collapsed"] for h in res.history))
    print("-"*40)

run_one(NoChatProtocol())
run_one(RoundtableProtocol())
run_one(MediatorProtocol())
