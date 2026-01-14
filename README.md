# Multi-Agent Coordination and Social Dynamics under Resource Scarcity

This repository contains the code, experiments, and analysis for a research-oriented simulation study on multi-agent coordination in shared-resource environments. The work investigates how heterogeneous agents interact, cooperate, and fail under resource scarcity when coordination mechanisms are non-binding.

The project focuses on understanding long-horizon social dynamics rather than task completion, with particular attention to cooperation, inequality, stability, and collapse.

---

## Motivation

Many recent multi-agent AI systems rely on communication, norms, or mediators to enable coordination. However, it remains unclear whether such mechanisms are sufficient when agents face strong individual incentives and shared resource constraints. This project studies the limits of voluntary coordination in the absence of enforcement.

---

## Core Research Questions

- Can communication alone stabilize cooperation under resource scarcity?
- How do different coordination protocols affect inequality and system stability?
- How does population size interact with scarcity to induce collapse?
- What are the failure modes of non-binding multi-agent governance?

---

## Environment Overview

- Agents share a single regenerating resource.
- At each timestep, agents independently choose how much resource to extract.
- The environment does not enforce norms or restrict actions.
- If over-extraction occurs, the resource collapses and the episode terminates.

This setup models classic social dilemmas such as the Tragedy of the Commons.

---

## Agent Types

The system uses heterogeneous agents with fixed behavior profiles per episode:

- **Norm-Following Agents**: Tend to comply with suggested norms or recommendations.
- **Cooperative Agents**: Prefer conservative extraction to preserve the resource.
- **Greedy Agents**: Maximize short-term extraction regardless of long-term impact.

Agent composition is fixed at the start of each episode and shuffled.

---

## Coordination Protocols

Three non-binding coordination protocols are implemented:

- **No Chat**: Agents act independently without communication.
- **Roundtable**: Agents share proposals before acting.
- **Mediator**: A central mediator suggests a collective norm, which agents may ignore.

All protocols are advisory only; no enforcement or penalties exist.

---

## Experimental Phases

### Phase 1: Baseline Evaluation
- Population sizes: 3, 5, 7 agents
- Varying scarcity levels via regeneration rate
- Metrics: cooperation index, Gini coefficient, stability variance, collapse rate

### Phase 2: Stress Testing
- Larger populations: 7 and 9 agents
- Fixed high scarcity
- Memory ablation studies (M0, M1, M2)
- Focus on robustness and failure regimes

---

## Metrics

The following quantitative metrics are used:

- **Cooperation Index**: Degree of collective restraint
- **Gini Coefficient**: Inequality in cumulative rewards
- **Stability Variance**: Fluctuations in system behavior over time
- **Collapse Rate**: Frequency of resource collapse
- **Time-to-Collapse** (when applicable)

---

## Repository Structure

