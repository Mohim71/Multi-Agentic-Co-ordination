import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.agents import GreedyAgent, CooperativeAgent, NormFollowingAgent

obs = {"t": 0, "R_before": 20, "n_agents": 3}
msgs = [{"sender": 0, "text": "Agent 0: I propose we take 1."},
        {"sender": 1, "text": "Agent 1: I propose we take 2."},
        {"sender": 2, "text": "Agent 2: I propose we take 1."}]

agents = [
    GreedyAgent(0, action_max=3),
    CooperativeAgent(1, action_max=3),
    NormFollowingAgent(2, action_max=3, memory_mode="M0"),
]

for a in agents:
    print(type(a).__name__, "decides take =", a.decide_take(obs, msgs))
