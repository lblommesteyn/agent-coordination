"""Event-driven simulation engine for stochastic DAG execution."""

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from agents import Agent, assign_cp_first_dynamic
from dag_utils import Task, TaskDAG, compute_critical_path


@dataclass(order=True)
class Event:
    """Simulation event."""

    time: float
    task_id: int = field(compare=False)
    event_type: str = field(compare=False)  # "complete" or "fail"
    agent_id: int = field(compare=False)


@dataclass
class TaskResult:
    """Result of executing a single task."""

    task_id: int
    agent_id: int
    start_time: float
    end_time: float
    error: float
    succeeded: bool
    attempts: int


@dataclass
class EpisodeResult:
    """Result of a complete DAG execution episode."""

    makespan: float
    total_overhead: float
    final_error: float
    total_cost: float
    task_results: List[TaskResult]
    total_rework_cost: float = 0.0


class StochasticScheduler:
    """Event-driven simulation of stochastic DAG execution."""

    def __init__(
        self,
        dag: TaskDAG,
        agents: List[Agent],
        assignment: Dict[int, int],
        a_ij: Optional[Dict[Tuple[int, int], float]] = None,
        validator_edges: Optional[Set[Tuple[int, int]]] = None,
        gamma: float = 0.5,
        alpha: float = 0.3,
        beta: float = 0.5,
        duration_noise_sigma: float = 0.0,
        a_ij_drift_sigma: float = 0.0,
        representation_format: str = "structured_json",
        epsilon_0: float = 0.1,
        error_threshold: float = 1.0,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.dag = dag
        self.agents = {a.id: a for a in agents}
        self.agent_list = list(agents)
        self.assignment = dict(assignment)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.duration_noise_sigma = duration_noise_sigma
        self.a_ij_drift_sigma = a_ij_drift_sigma
        self.representation_format = representation_format
        self.epsilon_0 = epsilon_0
        self.error_threshold = error_threshold
        self.rng = rng or np.random.RandomState(42)
        self.topo_rank = {
            tid: idx for idx, tid in enumerate(self.dag.topological_sort())
        }

        if a_ij is None:
            self.a_ij = {edge: 1.0 for edge in dag.edges}
        else:
            self.a_ij = dict(a_ij)

        self.validator_edges = set(validator_edges or set())
        self.repr_modifiers = {
            "prose": {"cost_mult": 1.0, "error_mult": 1.4},
            "structured_json": {"cost_mult": 0.6, "error_mult": 0.9},
            "compressed_summary": {"cost_mult": 0.3, "error_mult": 1.1},
        }

    def _sample_duration(self, task: Task, agent: Agent) -> float:
        """Sample a stochastic duration for a task executed by an agent."""

        base = agent.effective_duration(task)
        if self.duration_noise_sigma > 0:
            noise = self.rng.lognormal(0.0, self.duration_noise_sigma)
            return base * noise
        return base

    def _task_priority(self, tid: int) -> Tuple[int, float, int]:
        """Order ready tasks deterministically for dispatch."""

        task = self.dag.get_task(tid)
        return (self.topo_rank[tid], -task.duration, tid)

    def _choose_representation(
        self,
        active_ids: Set[int],
        assignment: Dict[int, int],
        a_ij: Dict[Tuple[int, int], float],
        validator_edges: Set[Tuple[int, int]],
        current_representation: str,
    ) -> str:
        """Select a representation for future handoffs on the residual graph."""

        if len(active_ids) <= 1:
            return current_representation

        residual_dag = self.dag.induced_subdag(active_ids)
        if not residual_dag.edges:
            return current_representation

        handoff_edges = [
            edge
            for edge in residual_dag.edges
            if assignment.get(edge[0]) != assignment.get(edge[1])
        ]
        residual_validators = {
            edge for edge in validator_edges if edge in set(residual_dag.edges)
        }

        best_format = current_representation
        best_score = float("inf")
        for fmt, mods in self.repr_modifiers.items():
            est_overhead = self.gamma * mods["cost_mult"] * len(handoff_edges)
            est_error = compute_final_error(
                residual_dag,
                a_ij,
                residual_validators,
                epsilon_0=self.epsilon_0,
                repr_error_mult=mods["error_mult"],
            )
            score = self.alpha * est_overhead + self.beta * est_error
            if score < best_score:
                best_score = score
                best_format = fmt

        return best_format

    def _replan(
        self,
        completed: Set[int],
        running: Dict[int, int],
        current_assignment: Dict[int, int],
        agent_free_time: Dict[int, float],
        current_a_ij: Dict[Tuple[int, int], float],
        current_validator_edges: Set[Tuple[int, int]],
        current_representation: str,
    ) -> Tuple[Dict[int, int], Set[Tuple[int, int]], str]:
        """Static scheduler: no replanning."""

        return current_assignment, current_validator_edges, current_representation

    def run_episode(self) -> EpisodeResult:
        """Run one complete episode of the DAG execution."""

        completed: Set[int] = set()
        running: Dict[int, int] = {}
        ready: Set[int] = set(self.dag.source_tasks())
        task_errors: Dict[int, float] = {}
        task_results: List[TaskResult] = []
        task_start_times: Dict[int, float] = {}
        finish_times: Dict[int, float] = {}
        attempts = {task.id: 1 for task in self.dag.tasks}
        agent_free_time = {agent.id: 0.0 for agent in self.agent_list}

        accumulated_overhead = 0.0
        edge_representation: Dict[Tuple[int, int], str] = {}
        current_a_ij = dict(self.a_ij)
        current_assignment = dict(self.assignment)
        current_validator_edges = set(self.validator_edges)
        current_representation = self.representation_format
        event_queue: List[Event] = []

        def schedule_task(tid: int, aid: int, now: float) -> None:
            nonlocal accumulated_overhead

            task = self.dag.get_task(tid)
            agent = self.agents[aid]
            duration = self._sample_duration(task, agent)
            end_time = now + duration

            running[tid] = aid
            agent_free_time[aid] = end_time
            task_start_times[tid] = now
            task_errors.setdefault(tid, self.epsilon_0)

            for pred in self.dag.predecessors[tid]:
                edge = (pred, tid)
                if edge not in edge_representation:
                    edge_representation[edge] = current_representation
                    if current_assignment.get(pred) != aid:
                        mods = self.repr_modifiers[current_representation]
                        accumulated_overhead += self.gamma * mods["cost_mult"]

            event_type = "fail" if self.rng.random() < task.failure_prob else "complete"
            heapq.heappush(event_queue, Event(end_time, tid, event_type, aid))

        def dispatch_ready_tasks(now: float) -> None:
            while True:
                scheduled_any = False
                for aid in sorted(self.agents):
                    if agent_free_time[aid] > now + 1e-9:
                        continue
                    candidates = [
                        tid for tid in ready if current_assignment.get(tid) == aid
                    ]
                    if not candidates:
                        continue
                    tid = min(candidates, key=self._task_priority)
                    ready.remove(tid)
                    schedule_task(tid, aid, now)
                    scheduled_any = True
                if not scheduled_any:
                    break

        dispatch_ready_tasks(0.0)

        while event_queue:
            event = heapq.heappop(event_queue)
            now = event.time
            tid = event.task_id
            aid = event.agent_id

            if event.event_type == "fail":
                task = self.dag.get_task(tid)
                agent = self.agents[aid]
                attempts[tid] += 1
                task_errors[tid] = self.epsilon_0
                task_start_times[tid] = now

                duration = self._sample_duration(task, agent) * 2.0
                end_time = now + duration
                agent_free_time[aid] = end_time
                next_event = "fail" if self.rng.random() < task.failure_prob else "complete"
                heapq.heappush(event_queue, Event(end_time, tid, next_event, aid))
                continue

            running.pop(tid, None)
            completed.add(tid)
            finish_times[tid] = now

            preds = self.dag.predecessors[tid]
            if not preds:
                task_errors[tid] = self.epsilon_0
            else:
                propagated_error = 0.0
                for pred in preds:
                    edge = (pred, tid)
                    fmt = edge_representation.get(edge, current_representation)
                    aij = current_a_ij.get(edge, 1.0)
                    effective_aij = aij * self.repr_modifiers[fmt]["error_mult"]
                    if edge in current_validator_edges:
                        err = self.epsilon_0
                    else:
                        err = task_errors.get(pred, self.epsilon_0) * effective_aij
                    propagated_error = max(propagated_error, err)
                task_errors[tid] = propagated_error

            task_results.append(
                TaskResult(
                    task_id=tid,
                    agent_id=aid,
                    start_time=task_start_times.get(tid, 0.0),
                    end_time=now,
                    error=task_errors.get(tid, self.epsilon_0),
                    succeeded=True,
                    attempts=attempts[tid],
                )
            )

            if self.a_ij_drift_sigma > 0:
                for edge in current_a_ij:
                    current_a_ij[edge] = max(
                        0.1,
                        current_a_ij[edge] + self.rng.normal(0.0, self.a_ij_drift_sigma),
                    )

            for succ in self.dag.successors[tid]:
                if succ in completed or succ in running or succ in ready:
                    continue
                if all(pred in completed for pred in self.dag.predecessors[succ]):
                    ready.add(succ)
                    task_errors.setdefault(succ, self.epsilon_0)

            (
                current_assignment,
                current_validator_edges,
                current_representation,
            ) = self._replan(
                completed,
                running,
                current_assignment,
                agent_free_time,
                current_a_ij,
                current_validator_edges,
                current_representation,
            )

            dispatch_ready_tasks(now)

        makespan = max(finish_times.values()) if finish_times else 0.0
        sink_tasks = self.dag.sink_tasks()
        final_error = (
            max(task_errors.get(tid, self.epsilon_0) for tid in sink_tasks)
            if sink_tasks
            else self.epsilon_0
        )
        rework_cost = sum(
            self.dag.get_task(tid).duration * (attempts.get(tid, 1) - 1)
            for tid in range(self.dag.n)
            if task_errors.get(tid, 0.0) > self.error_threshold
        )
        total_cost = makespan + self.alpha * accumulated_overhead + self.beta * final_error

        return EpisodeResult(
            makespan=makespan,
            total_overhead=accumulated_overhead,
            final_error=final_error,
            total_cost=total_cost,
            task_results=task_results,
            total_rework_cost=rework_cost,
        )


class AdaptiveScheduler(StochasticScheduler):
    """Scheduler that replans assignment, validator placement, and format."""

    def __init__(self, *args, select_representation: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.select_representation = select_representation

    def _replan(
        self,
        completed: Set[int],
        running: Dict[int, int],
        current_assignment: Dict[int, int],
        agent_free_time: Dict[int, float],
        current_a_ij: Dict[Tuple[int, int], float],
        current_validator_edges: Set[Tuple[int, int]],
        current_representation: str,
    ) -> Tuple[Dict[int, int], Set[Tuple[int, int]], str]:
        active_ids = {task.id for task in self.dag.tasks if task.id not in completed}
        pending_ids = active_ids - set(running)

        updated_assignment = dict(current_assignment)
        if pending_ids:
            residual_dag = self.dag.induced_subdag(pending_ids)
            dynamic_assignment = assign_cp_first_dynamic(
                residual_dag,
                self.agent_list,
                agent_available_time=agent_free_time,
                locked_assignment={
                    tid: aid
                    for tid, aid in current_assignment.items()
                    if tid in completed or tid in running
                },
                preferred_assignment=current_assignment,
                handoff_penalty=0.5 * self.gamma,
                switch_penalty=100.0,
            )
            updated_assignment.update(dynamic_assignment)

        updated_validator_edges = set(current_validator_edges)
        if len(active_ids) > 1:
            active_dag = self.dag.induced_subdag(active_ids)
            best_edge = optimal_validator_placement(active_dag, current_a_ij)
            if best_edge is not None:
                updated_validator_edges.add(best_edge)

        updated_representation = current_representation
        if self.select_representation:
            updated_representation = self._choose_representation(
                active_ids,
                updated_assignment,
                current_a_ij,
                updated_validator_edges,
                current_representation,
            )

        return updated_assignment, updated_validator_edges, updated_representation


def optimal_validator_placement(
    dag: TaskDAG, a_ij: Dict[Tuple[int, int], float]
) -> Optional[Tuple[int, int]]:
    """Find optimal single validator placement on the critical path.

    Under the multiplicative reset model, placing the validator after CP edge k
    gives final_error = eps0 * prod(A_ij for edges after k).  Minimising this
    is equivalent to maximising the prefix product prod(A_ij for edges up to k).
    This is NOT generally the same as placing at argmax A_ij.
    """
    _, cp_tasks, cp_edges = compute_critical_path(dag)
    if not cp_edges:
        return None

    # Reconstruct CP edges in topological order using the ordered cp_tasks list
    ordered_edges = [
        (cp_tasks[i], cp_tasks[i + 1])
        for i in range(len(cp_tasks) - 1)
        if (cp_tasks[i], cp_tasks[i + 1]) in cp_edges
    ]
    if not ordered_edges:
        return None

    # Constrain to interior CP edges: the validator must be placed before the
    # final CP task so that at least one post-validator edge exists to benefit.
    # Placing at the last CP edge gives final_error = eps_0 trivially (no
    # downstream CP tasks), which is degenerate when all A_ij > 1.
    candidates = ordered_edges[:-1] if len(ordered_edges) > 1 else ordered_edges

    # Argmax of running prefix product over interior candidates
    best_edge = candidates[0]
    best_prefix = a_ij.get(candidates[0], 1.0)
    running = best_prefix
    for edge in candidates[1:]:
        running *= a_ij.get(edge, 1.0)
        if running > best_prefix:
            best_prefix = running
            best_edge = edge
    return best_edge


def compute_final_error(
    dag: TaskDAG,
    a_ij: Dict[Tuple[int, int], float],
    validator_edges: Set[Tuple[int, int]],
    epsilon_0: float = 0.1,
    repr_error_mult: float = 1.0,
) -> float:
    """Compute deterministic final error along the highest-error path."""

    topo = dag.topological_sort()
    errors: Dict[int, float] = {}

    for tid in topo:
        preds = dag.predecessors[tid]
        if not preds:
            errors[tid] = epsilon_0
            continue

        max_error = 0.0
        for pred in preds:
            edge = (pred, tid)
            if edge in validator_edges:
                err = epsilon_0
            else:
                err = errors[pred] * a_ij.get(edge, 1.0) * repr_error_mult
            max_error = max(max_error, err)
        errors[tid] = max_error

    sinks = dag.sink_tasks()
    return max(errors[tid] for tid in sinks) if sinks else epsilon_0
