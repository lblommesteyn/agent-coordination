"""Agent model and assignment policies."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from dag_utils import TaskDAG, Task, compute_critical_path


@dataclass
class Agent:
    """An agent with capability vector and speed multiplier."""
    id: int
    speed: float = 1.0
    capability: Optional[np.ndarray] = None

    def effective_duration(self, task: Task) -> float:
        """Compute task duration when executed by this agent."""
        return task.duration / self.speed


def create_agents(k: int, speed_range: Tuple[float, float] = (0.7, 1.5),
                  rng: Optional[np.random.RandomState] = None) -> List[Agent]:
    """Create k agents with random speeds."""
    if rng is None:
        rng = np.random.RandomState(42)
    agents = []
    for i in range(k):
        speed = rng.uniform(*speed_range)
        agents.append(Agent(id=i, speed=speed))
    return agents


def create_uniform_agents(k: int, speed: float = 1.0) -> List[Agent]:
    """Create k identical agents."""
    return [Agent(id=i, speed=speed) for i in range(k)]


# ── Assignment Policies ──────────────────────────────────────────────────────


def assign_random(dag: TaskDAG, agents: List[Agent],
                  rng: Optional[np.random.RandomState] = None) -> Dict[int, int]:
    """Random assignment: each task assigned to a random agent.

    Returns:
        Dict mapping task_id -> agent_id.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    k = len(agents)
    return {t.id: int(rng.randint(0, k)) for t in dag.tasks}


def assign_round_robin(dag: TaskDAG, agents: List[Agent]) -> Dict[int, int]:
    """Round-robin in topological order."""
    topo = dag.topological_sort()
    k = len(agents)
    assignment = {}
    for idx, tid in enumerate(topo):
        assignment[tid] = idx % k
    return assignment


def assign_cp_first(dag: TaskDAG, agents: List[Agent]) -> Dict[int, int]:
    """Critical-path first: assign the fastest agent to CP tasks,
    then greedily assign remaining tasks.

    Strategy:
    1. Compute critical path.
    2. Sort agents by speed (descending).
    3. Assign fastest agent(s) to CP tasks.
    4. Assign remaining tasks round-robin among remaining agents.
    """
    # Compute CP with default durations
    _, cp_tasks, _ = compute_critical_path(dag)
    cp_set = set(cp_tasks)

    # Sort agents by speed, fastest first
    sorted_agents = sorted(agents, key=lambda a: a.speed, reverse=True)

    assignment = {}

    # Assign fastest agent to all critical path tasks
    fastest = sorted_agents[0]
    for tid in cp_tasks:
        assignment[tid] = fastest.id

    # Assign remaining tasks round-robin among other agents
    other_agents = sorted_agents[1:] if len(sorted_agents) > 1 else sorted_agents
    topo = dag.topological_sort()
    non_cp_tasks = [tid for tid in topo if tid not in cp_set]

    for idx, tid in enumerate(non_cp_tasks):
        agent = other_agents[idx % len(other_agents)]
        assignment[tid] = agent.id

    return assignment


def assign_cp_first_bundled(dag: TaskDAG, agents: List[Agent]) -> Dict[int, int]:
    """CP-first with bundling: like cp_first, but also bundles
    dependent subtask chains to the same agent to minimize handoffs.

    Strategy:
    1. Assign CP tasks to fastest agent.
    2. For each non-CP task, prefer the agent assigned to its predecessor
       (to minimize handoffs), breaking ties by agent load.
    """
    _, cp_tasks, _ = compute_critical_path(dag)
    cp_set = set(cp_tasks)

    sorted_agents = sorted(agents, key=lambda a: a.speed, reverse=True)
    fastest = sorted_agents[0]

    assignment = {}
    agent_load = {a.id: 0.0 for a in agents}

    # Assign CP tasks to fastest agent
    for tid in cp_tasks:
        assignment[tid] = fastest.id
        agent_load[fastest.id] += dag.get_task(tid).duration

    # Assign non-CP tasks, preferring predecessor's agent
    topo = dag.topological_sort()
    non_cp_tasks = [tid for tid in topo if tid not in cp_set]

    for tid in non_cp_tasks:
        preds = dag.predecessors[tid]
        assigned_preds = [assignment[p] for p in preds if p in assignment]

        # Candidate: least-loaded agent overall
        least_loaded = min(agent_load, key=agent_load.get)
        min_load = agent_load[least_loaded]

        if assigned_preds:
            # Prefer predecessor's agent only if it won't exceed 1.5x the min load
            pred_agents = set(assigned_preds)
            best_pred_agent = min(pred_agents, key=lambda aid: agent_load[aid])
            if agent_load[best_pred_agent] <= min_load * 1.5 + dag.get_task(tid).duration:
                best_agent = best_pred_agent
            else:
                best_agent = least_loaded
        else:
            best_agent = least_loaded

        assignment[tid] = best_agent
        agent_load[best_agent] += dag.get_task(tid).duration

    return assignment


# ── Makespan computation under assignment ────────────────────────────────────

def assign_cp_first_dynamic(
    dag: TaskDAG,
    agents: List[Agent],
    agent_available_time: Optional[Dict[int, float]] = None,
    locked_assignment: Optional[Dict[int, int]] = None,
    preferred_assignment: Optional[Dict[int, int]] = None,
    handoff_penalty: float = 0.5,
    switch_penalty: float = 1.0,
) -> Dict[int, int]:
    """Replanning-oriented CP-first assignment.

    This heuristic accounts for current agent availability and applies a
    small handoff penalty so replanning can trade off speed against queueing
    delay and coordination cost on the residual DAG.
    """
    agent_available_time = dict(agent_available_time or {})
    locked_assignment = dict(locked_assignment or {})
    preferred_assignment = dict(preferred_assignment or {})
    for agent in agents:
        agent_available_time.setdefault(agent.id, 0.0)

    _, cp_tasks, _ = compute_critical_path(dag)
    cp_set = set(cp_tasks)
    topo = dag.topological_sort()
    projected_free = dict(agent_available_time)
    assignment: Dict[int, int] = {}
    agent_map = {agent.id: agent for agent in agents}

    for tid in topo:
        task = dag.get_task(tid)
        pred_agents = []
        for pred in dag.predecessors[tid]:
            if pred in assignment:
                pred_agents.append(assignment[pred])
            elif pred in locked_assignment:
                pred_agents.append(locked_assignment[pred])

        best_agent_id = agents[0].id
        best_score = float("inf")
        for aid, agent in agent_map.items():
            exec_time = agent.effective_duration(task)
            handoffs = sum(1 for pred_aid in pred_agents if pred_aid != aid)
            start_penalty = projected_free[aid]
            move_penalty = 0.0
            if preferred_assignment.get(tid) is not None and preferred_assignment[tid] != aid:
                move_penalty = switch_penalty

            if tid in cp_set:
                score = start_penalty + exec_time + handoff_penalty * handoffs + move_penalty
            else:
                score = (
                    start_penalty
                    + 0.7 * exec_time
                    + 0.6 * handoff_penalty * handoffs
                    + 0.5 * move_penalty
                )

            if score < best_score:
                best_score = score
                best_agent_id = aid

        assignment[tid] = best_agent_id
        projected_free[best_agent_id] += agent_map[best_agent_id].effective_duration(task)

    return assignment


def compute_makespan_with_assignment(dag: TaskDAG, agents: List[Agent],
                                     assignment: Dict[int, int]) -> float:
    """Compute the makespan of a DAG under a given assignment.

    Uses list scheduling: tasks are scheduled in topological order,
    each on its assigned agent. Agent becomes available after its
    current task finishes.

    Returns:
        The makespan (completion time of the last task).
    """
    agent_map = {a.id: a for a in agents}
    topo = dag.topological_sort()

    # Track when each agent becomes free
    agent_free = {a.id: 0.0 for a in agents}
    # Track finish time of each task
    finish_time = {}

    for tid in topo:
        task = dag.get_task(tid)
        aid = assignment[tid]
        agent = agent_map[aid]

        # Earliest start: max of agent availability and all predecessors finished
        pred_finish = 0.0
        if dag.predecessors[tid]:
            pred_finish = max(finish_time[p] for p in dag.predecessors[tid])

        start = max(agent_free[aid], pred_finish)
        duration = agent.effective_duration(task)
        finish = start + duration

        finish_time[tid] = finish
        agent_free[aid] = finish

    return max(finish_time.values()) if finish_time else 0.0


def count_handoffs(dag: TaskDAG, assignment: Dict[int, int]) -> int:
    """Count number of edges where assigned agents differ (handoffs)."""
    count = 0
    for src, dst in dag.edges:
        if assignment.get(src) != assignment.get(dst):
            count += 1
    return count


def compute_overhead(dag: TaskDAG, assignment: Dict[int, int],
                     gamma: float = 0.5) -> float:
    """Compute total coordination overhead.

    overhead = gamma * number_of_handoffs
    """
    return gamma * count_handoffs(dag, assignment)
