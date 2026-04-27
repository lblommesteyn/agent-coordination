"""Helper simulations for more novel orchestration control ideas."""

from typing import Dict, Iterable, Tuple

import numpy as np

from dag_utils import Task, TaskDAG


def sample_chain_durations(
    length: int,
    duration_range: Tuple[float, float] = (2.0, 6.0),
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Sample durations for a synthetic serial critical path."""

    rng = rng or np.random.RandomState(42)
    return rng.uniform(duration_range[0], duration_range[1], size=length)


def simulate_speculative_chain(
    durations: np.ndarray,
    hit_rate: float,
    predictable_fraction: float,
    waste_weight: float,
    launch_cost_per_edge: float = 0.0,
    rng: np.random.RandomState | None = None,
) -> Dict[str, float]:
    """Approximate one speculative-execution episode on a linear chain.

    Each edge can overlap part of the successor work with its predecessor.
    A hit keeps that overlap as useful wall-clock savings; a miss discards it
    as wasted speculative compute.
    """

    rng = rng or np.random.RandomState(42)
    baseline_makespan = float(np.sum(durations))
    overlaps = predictable_fraction * np.minimum(durations[:-1], durations[1:])
    hits = rng.binomial(1, hit_rate, size=len(overlaps))

    time_saved = float(np.dot(hits, overlaps))
    wasted_work = float(np.dot(1 - hits, overlaps))
    speculative_makespan = max(float(np.max(durations)), baseline_makespan - time_saved)
    launch_cost = launch_cost_per_edge * len(overlaps)
    total_cost = speculative_makespan + waste_weight * wasted_work + launch_cost

    return {
        "baseline_makespan": baseline_makespan,
        "makespan": speculative_makespan,
        "time_saved": time_saved,
        "wasted_work": wasted_work,
        "launch_cost": launch_cost,
        "total_cost": total_cost,
    }


def compute_descendant_work(dag: TaskDAG) -> Dict[int, float]:
    """Sum descendant durations for each task."""

    descendant_work: Dict[int, float] = {}
    for task in dag.tasks:
        seen = set()
        frontier = list(dag.successors[task.id])
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(dag.successors[current])
        descendant_work[task.id] = sum(dag.get_task(tid).duration for tid in seen)
    return descendant_work


def simulate_early_stopping_episode(
    dag: TaskDAG,
    stop_fraction: float,
    descendant_work: Dict[int, float],
    predictor_strength: float,
    rerun_factor: float,
    rework_scale: float,
    rng: np.random.RandomState | None = None,
) -> Dict[str, float]:
    """Simulate one episode of predictive early stopping on the 7-task DAG."""

    rng = rng or np.random.RandomState(42)
    baseline_cost = 0.0
    stop_cost = 0.0
    detected_bad = 0
    false_stops = 0

    true_positive_rate = min(0.05 + predictor_strength * stop_fraction, 0.82)
    false_positive_rate = max(0.26 - 0.16 * stop_fraction, 0.04)

    for task in dag.tasks:
        bad_prob = min(0.45, task.failure_prob + 0.08)
        bad = rng.random() < bad_prob
        downstream_penalty = rework_scale * descendant_work[task.id]

        baseline_cost += task.duration
        if bad:
            baseline_cost += downstream_penalty

        if bad:
            if rng.random() < true_positive_rate:
                detected_bad += 1
                residual_rework = 0.20 * stop_fraction * downstream_penalty
                stop_cost += (
                    stop_fraction * task.duration
                    + rerun_factor * task.duration
                    + residual_rework
                )
            else:
                stop_cost += task.duration + downstream_penalty
        else:
            if rng.random() < false_positive_rate:
                false_stops += 1
                stop_cost += stop_fraction * task.duration + rerun_factor * task.duration
            else:
                stop_cost += task.duration

    return {
        "baseline_cost": baseline_cost,
        "stop_cost": stop_cost,
        "detected_bad": float(detected_bad),
        "false_stops": float(false_stops),
    }


def clone_with_task_duration(
    dag: TaskDAG,
    task_id: int,
    duration: float,
    failure_prob: float = 0.0,
) -> TaskDAG:
    """Clone a DAG while overriding one task's duration."""

    cloned_tasks = []
    for task in dag.tasks:
        if task.id == task_id:
            cloned_tasks.append(
                Task(id=task.id, duration=duration, failure_prob=failure_prob)
            )
        else:
            cloned_tasks.append(
                Task(id=task.id, duration=task.duration, failure_prob=task.failure_prob)
            )
    return TaskDAG(tasks=cloned_tasks, edges=list(dag.edges))


def simulate_dag_morphing_episode(
    quality_threshold: float,
    safe_quality_threshold: float,
    quality_noise_sigma: float,
    skip_task_duration: float,
    error_penalty_scale: float,
    rng: np.random.RandomState | None = None,
) -> Dict[str, float]:
    """Simulate one conditional-skipping decision for a verification task."""

    rng = rng or np.random.RandomState(42)
    true_quality = float(rng.beta(5.0, 2.2))
    observed_quality = float(
        np.clip(true_quality + rng.normal(0.0, quality_noise_sigma), 0.0, 1.0)
    )
    should_skip = observed_quality >= quality_threshold

    baseline_cost = skip_task_duration
    morph_cost = 0.0 if should_skip else skip_task_duration
    penalty = 0.0

    if should_skip and true_quality < safe_quality_threshold:
        penalty = error_penalty_scale * (safe_quality_threshold - true_quality)
        morph_cost += penalty

    return {
        "baseline_cost": baseline_cost,
        "morph_cost": morph_cost,
        "skipped": float(should_skip),
        "true_quality": true_quality,
        "observed_quality": observed_quality,
        "penalty": penalty,
    }


def expected_prewarm_overhead(
    base_cost: float,
    tau: float,
    prewarm_fraction: float,
    residual_fraction: float,
    ready_rate: float,
    stale_rate: float,
) -> float:
    """Expected handoff overhead after pre-warming with lead time tau."""

    if tau <= 0:
        return base_cost

    p_ready = 1.0 - np.exp(-ready_rate * tau)
    p_stale = 1.0 - np.exp(-stale_rate * tau)
    prewarm_cost = p_ready * prewarm_fraction * base_cost
    fresh_success = p_ready * (1.0 - p_stale)
    residual_cost = residual_fraction * base_cost

    return prewarm_cost + fresh_success * residual_cost + (1.0 - fresh_success) * base_cost


def summarize_threshold(
    xs: Iterable[float],
    ys: Iterable[float],
    baseline: float = 1.0,
) -> float | None:
    """Return the first x where y improves over baseline."""

    for x_val, y_val in zip(xs, ys):
        if y_val < baseline:
            return float(x_val)
    return None
