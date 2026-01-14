from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math

def gini(values: List[float]) -> float:
    """Gini coefficient. Returns 0..1. Handles all-zero safely."""
    n = len(values)
    if n == 0:
        return 0.0
    s = sum(values)
    if s == 0:
        return 0.0
    vals = sorted(values)
    cum = 0.0
    for i, v in enumerate(vals, start=1):
        cum += i * v
    return (2.0 * cum) / (n * s) - (n + 1.0) / n

def variance(xs: List[float]) -> float:
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)

@dataclass
class EpisodeMetrics:
    n_agents: int
    steps: int
    collapsed: bool
    t_collapse: Optional[int]

    U_total: int
    U_per_agent: List[int]

    cooperation_index: float
    gini_fairness: float
    stability_var: float  # avg per-agent variance of actions over time

def compute_episode_metrics(history: List[Dict[str, Any]], n_agents: int, action_max: int) -> EpisodeMetrics:
    steps = len(history)
    collapsed = any(h.get("collapsed", False) for h in history)
    t_collapse = None
    for h in history:
        if h.get("collapsed", False):
            t_collapse = int(h["t"])
            break

    # Per-agent utilities
    U_per_agent = [0] * n_agents
    for h in history:
        acts = h["actions"]
        for i in range(n_agents):
            U_per_agent[i] += int(acts[i])

    U_total = sum(U_per_agent)

    # Cooperation index: 1 - normalized extraction
    denom = max(1, steps * n_agents * action_max)
    cooperation_index = 1.0 - (U_total / denom)

    # Fairness via gini
    gini_fairness = gini([float(u) for u in U_per_agent])

    # Stability: average per-agent variance over time
    per_agent_series = [[] for _ in range(n_agents)]
    for h in history:
        acts = h["actions"]
        for i in range(n_agents):
            per_agent_series[i].append(float(acts[i]))
    stability_var = sum(variance(s) for s in per_agent_series) / max(1, n_agents)

    return EpisodeMetrics(
        n_agents=n_agents,
        steps=steps,
        collapsed=collapsed,
        t_collapse=t_collapse,
        U_total=U_total,
        U_per_agent=U_per_agent,
        cooperation_index=float(cooperation_index),
        gini_fairness=float(gini_fairness),
        stability_var=float(stability_var),
    )
