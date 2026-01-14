import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import List, Dict, Any

from envs.commons_env import CommonsEnv, CommonsEnvConfig
from protocols.protocols import BaseProtocol, ProtocolOutput
from agents.agents import AgentBase

@dataclass
class EpisodeResult:
    config: Dict[str, Any]
    history: List[Dict[str, Any]]  # env step logs
    chat: List[Dict[str, Any]]     # per round: {"t":..., "public":..., "private":...}

def run_episode(
    env_cfg: CommonsEnvConfig,
    protocol: BaseProtocol,
    agents: List[AgentBase],
) -> EpisodeResult:
    env = CommonsEnv(env_cfg)
    env.reset()

    chat_log = []

    agent_ids = [a.agent_id for a in agents]
    id_to_agent = {a.agent_id: a for a in agents}

    def agent_say_fn(agent_id: int, ctx: Dict[str, Any]) -> str:
        # chair agent: we’ll implement as a simple fixed-policy message for now
        if agent_id == -1:
            # naive chair message (safe default): suggest take 1
            return "Chair: Let's all take 1 to keep the pool stable."
        return id_to_agent[agent_id].say(ctx)

    while not env.done:
        obs = {
            "t": env.t,
            "R_before": env.R,
            "n_agents": env_cfg.n_agents,
            "regen": env_cfg.regen,
            "r_max": env_cfg.r_max,
        }

        # protocol chat phase
        pout: ProtocolOutput = protocol.run_chat(agent_ids, obs, agent_say_fn)
        public_msgs = pout.public_messages or []

        chat_log.append({
            "t": env.t,
            "public": public_msgs,
            "private": pout.private_messages,
        })

        # action decision phase
        actions = []
        for a in agents:
            take = a.decide_take(obs, public_msgs)
            actions.append(int(take))

        # env step
        env.step(actions)

    cfg_dict = {
        "n_agents": env_cfg.n_agents,
        "T": env_cfg.T,
        "r_max": env_cfg.r_max,
        "r_init": env_cfg.r_init,
        "regen": env_cfg.regen,
        "action_max": env_cfg.action_max,
        "protocol": getattr(protocol, "name", type(protocol).__name__),
        "agents": [type(a).__name__ for a in agents],
    }

    return EpisodeResult(config=cfg_dict, history=env.history, chat=chat_log)
