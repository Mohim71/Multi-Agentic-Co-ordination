from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import re
from collections import Counter

TAKE_RE = re.compile(r"\btake\s+(\d+)\b", re.IGNORECASE)

def extract_proposed_takes(messages: List[Dict[str, Any]]) -> List[int]:
    """Extract integers from phrases like 'take 1' in message text."""
    takes = []
    for m in messages:
        text = m.get("text", "")
        for match in TAKE_RE.finditer(text):
            takes.append(int(match.group(1)))
    return takes

@dataclass
class AgentBase:
    agent_id: int
    action_max: int = 3

    def decide_take(self, obs: Dict[str, Any], public_messages: List[Dict[str, Any]]) -> int:
        raise NotImplementedError

    def say(self, ctx: Dict[str, Any]) -> str:
        """Return a short message used by protocols. Default: propose own intended take."""
        # Override per agent for richer behavior
        return f"Agent {self.agent_id}: I propose we take 1."

@dataclass
class GreedyAgent(AgentBase):
    def decide_take(self, obs: Dict[str, Any], public_messages: List[Dict[str, Any]]) -> int:
        return self.action_max

    def say(self, ctx: Dict[str, Any]) -> str:
        return f"Agent {self.agent_id}: I will take {self.action_max}."

@dataclass
class CooperativeAgent(AgentBase):
    def decide_take(self, obs: Dict[str, Any], public_messages: List[Dict[str, Any]]) -> int:
        R = int(obs["R_before"])
        N = int(obs["n_agents"])
        safe_total = R // 2
        base = safe_total // N
        return max(0, min(self.action_max, base))

    def say(self, ctx: Dict[str, Any]) -> str:
        return f"Agent {self.agent_id}: I propose we all take 1."

@dataclass
class NormFollowingAgent(AgentBase):
    memory_mode: str = "M0"  # M0, M1, M2
    k: int = 3
    # For M1: store last rounds' public messages
    msg_memory: List[List[Dict[str, Any]]] = field(default_factory=list)
    # For M2: running tally of proposed takes
    take_tally: Counter = field(default_factory=Counter)

    def _update_memory(self, public_messages: List[Dict[str, Any]]):
        if self.memory_mode == "M0":
            return
        if self.memory_mode == "M1":
            self.msg_memory.append(public_messages)
            if len(self.msg_memory) > self.k:
                self.msg_memory.pop(0)
        elif self.memory_mode == "M2":
            takes = extract_proposed_takes(public_messages)
            self.take_tally.update(takes)

    def _get_norm_take(self, current_public: List[Dict[str, Any]]) -> Optional[int]:
        if self.memory_mode == "M0":
            takes = extract_proposed_takes(current_public)
            if not takes:
                return None
            return Counter(takes).most_common(1)[0][0]

        if self.memory_mode == "M1":
            all_msgs = []
            for round_msgs in self.msg_memory[-self.k:]:
                all_msgs.extend(round_msgs)
            takes = extract_proposed_takes(all_msgs)
            if not takes:
                return None
            return Counter(takes).most_common(1)[0][0]

        if self.memory_mode == "M2":
            if not self.take_tally:
                return None
            return self.take_tally.most_common(1)[0][0]

        return None

    def decide_take(self, obs: Dict[str, Any], public_messages: List[Dict[str, Any]]) -> int:
        # update memory with messages from this round
        self._update_memory(public_messages)

        norm = self._get_norm_take(public_messages)
        if norm is not None:
            return max(0, min(self.action_max, int(norm)))

        # fallback to cooperative rule
        R = int(obs["R_before"])
        N = int(obs["n_agents"])
        safe_total = R // 2
        base = safe_total // N
        return max(0, min(self.action_max, base))

    def say(self, ctx: Dict[str, Any]) -> str:
        # norm proposal: try to coordinate on take 1 by default
        return f"Agent {self.agent_id}: I propose we take 1."
