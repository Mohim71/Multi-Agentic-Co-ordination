from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ProtocolOutput:
    public_messages: List[Dict[str, Any]]   # list of {"sender":..., "text":...}
    private_messages: Optional[List[Dict[str, Any]]] = None  # for mediator variant if needed

class BaseProtocol:
    name: str = "base"

    def run_chat(self, agent_ids: List[int], obs: Dict[str, Any], agent_say_fn) -> ProtocolOutput:
        """
        agent_say_fn(agent_id, prompt_context) -> str
        """
        raise NotImplementedError


class NoChatProtocol(BaseProtocol):
    name = "no_chat"

    def run_chat(self, agent_ids, obs, agent_say_fn) -> ProtocolOutput:
        return ProtocolOutput(public_messages=[])


class RoundtableProtocol(BaseProtocol):
    name = "roundtable"

    def __init__(self, max_words: int = 40):
        self.max_words = max_words

    def run_chat(self, agent_ids, obs, agent_say_fn) -> ProtocolOutput:
        msgs = []
        for aid in agent_ids:
            text = agent_say_fn(aid, {"obs": obs, "history": msgs})
            text = " ".join(text.split()[: self.max_words])
            msgs.append({"sender": aid, "text": text})
        return ProtocolOutput(public_messages=msgs)


class MediatorProtocol(BaseProtocol):
    name = "mediator"

    def __init__(self, chair_id: int = -1, max_words_agents: int = 30, max_words_chair: int = 60):
        self.chair_id = chair_id
        self.max_words_agents = max_words_agents
        self.max_words_chair = max_words_chair

    def run_chat(self, agent_ids, obs, agent_say_fn) -> ProtocolOutput:
        private = []
        for aid in agent_ids:
            text = agent_say_fn(aid, {"obs": obs, "to": "chair"})
            text = " ".join(text.split()[: self.max_words_agents])
            private.append({"sender": aid, "text": text})

        chair_text = agent_say_fn(self.chair_id, {"obs": obs, "private_messages": private})
        chair_text = " ".join(chair_text.split()[: self.max_words_chair])
        public = [{"sender": self.chair_id, "text": chair_text}]

        return ProtocolOutput(public_messages=public, private_messages=private)
