from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class CommonsEnvConfig:
    n_agents: int
    T: int = 30
    r_max: int = 30
    r_init: int = 20
    regen: int = 5
    action_max: int = 3  # set to 3 or 5 based on Step 1.1

@dataclass
class CommonsEnv:
    cfg: CommonsEnvConfig
    t: int = 0
    R: int = 0
    done: bool = False
    history: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self) -> Dict[str, Any]:
        self.t = 0
        self.R = self.cfg.r_init
        self.done = False
        self.history.clear()
        return {"t": self.t, "R": self.R, "done": self.done}

    def step(self, actions: List[int]) -> Dict[str, Any]:
        if self.done:
            raise RuntimeError("Episode already ended. Call reset().")
        if len(actions) != self.cfg.n_agents:
            raise ValueError(f"Expected {self.cfg.n_agents} actions, got {len(actions)}.")

        # validate action range
        for a in actions:
            if not isinstance(a, int):
                raise TypeError("Actions must be integers.")
            if a < 0 or a > self.cfg.action_max:
                raise ValueError(f"Action {a} out of range [0, {self.cfg.action_max}].")

        log = {
            "t": self.t,
            "R_before": self.R,
            "actions": actions.copy(),
        }

        total_take = sum(actions)
        self.R -= total_take

        collapsed = False
        if self.R <= 0:
            self.R = 0
            collapsed = True
            self.done = True

        if not self.done:
            self.R = min(self.cfg.r_max, self.R + self.cfg.regen)

        self.t += 1
        if self.t >= self.cfg.T:
            self.done = True

        log.update({
            "total_take": total_take,
            "R_after": self.R,
            "collapsed": collapsed,
            "done": self.done,
        })
        self.history.append(log)
        return log
