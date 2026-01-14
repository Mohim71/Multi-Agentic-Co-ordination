import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from protocols.protocols import NoChatProtocol, RoundtableProtocol, MediatorProtocol

def dummy_say(agent_id, ctx):
    if agent_id == -1:
        return "Chair: Let's all take 1 to keep the pool stable."
    return f"Agent {agent_id}: I propose we all take 1."

obs = {"R": 20, "t": 0}
agent_ids = [0,1,2]

for P in [NoChatProtocol(), RoundtableProtocol(), MediatorProtocol()]:
    out = P.run_chat(agent_ids, obs, dummy_say)
    print(P.name, out.public_messages)
