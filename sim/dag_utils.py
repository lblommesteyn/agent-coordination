"""DAG generation, critical path analysis, and topological sort utilities."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional


@dataclass
class Task:
    """A single task node in the DAG."""
    id: int
    duration: float
    failure_prob: float = 0.0
    # Computed by critical path analysis
    est: float = 0.0   # earliest start time
    eft: float = 0.0   # earliest finish time
    lst: float = np.inf  # latest start time
    lft: float = np.inf  # latest finish time
    slack: float = np.inf
    on_critical_path: bool = False


@dataclass
class TaskDAG:
    """Directed acyclic graph of tasks."""
    tasks: List[Task]
    edges: List[Tuple[int, int]]  # (src, dst) pairs
    n: int = 0

    def __post_init__(self):
        self.n = len(self.tasks)
        self._task_map: Dict[int, int] = {}  # task_id -> index in self.tasks
        for idx, t in enumerate(self.tasks):
            self._task_map[t.id] = idx
        self._build_adjacency()

    def _build_adjacency(self):
        """Build predecessor/successor adjacency lists."""
        self.successors: Dict[int, List[int]] = {t.id: [] for t in self.tasks}
        self.predecessors: Dict[int, List[int]] = {t.id: [] for t in self.tasks}
        for src, dst in self.edges:
            self.successors[src].append(dst)
            self.predecessors[dst].append(src)

    def topological_sort(self) -> List[int]:
        """Kahn's algorithm for topological sort."""
        in_degree = {t.id: 0 for t in self.tasks}
        for _, dst in self.edges:
            in_degree[dst] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            # Pick smallest id for determinism
            queue.sort()
            node = queue.pop(0)
            order.append(node)
            for succ in self.successors[node]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        assert len(order) == self.n, "Cycle detected in DAG"
        return order

    def get_task(self, tid: int) -> Task:
        """Get task by id."""
        return self.tasks[self._task_map[tid]]

    def source_tasks(self) -> List[int]:
        """Tasks with no predecessors."""
        return [t.id for t in self.tasks if len(self.predecessors[t.id]) == 0]

    def sink_tasks(self) -> List[int]:
        """Tasks with no successors."""
        return [t.id for t in self.tasks if len(self.successors[t.id]) == 0]

    def induced_subdag(self, task_ids: Set[int]) -> "TaskDAG":
        """Return the induced sub-DAG on a subset of task IDs."""
        sub_tasks = [
            Task(id=t.id, duration=t.duration, failure_prob=t.failure_prob)
            for t in self.tasks
            if t.id in task_ids
        ]
        sub_edges = [
            (src, dst)
            for src, dst in self.edges
            if src in task_ids and dst in task_ids
        ]
        return TaskDAG(tasks=sub_tasks, edges=sub_edges)

    def width(self) -> int:
        """Maximum antichain size (max number of tasks that can run in parallel).
        Approximated via level-based decomposition."""
        levels = self._compute_levels()
        if not levels:
            return 0
        level_counts = {}
        for tid, level in levels.items():
            level_counts[level] = level_counts.get(level, 0) + 1
        return max(level_counts.values())

    def _compute_levels(self) -> Dict[int, int]:
        """Assign each task to a level (longest path from a source)."""
        topo = self.topological_sort()
        levels = {}
        for tid in topo:
            preds = self.predecessors[tid]
            if not preds:
                levels[tid] = 0
            else:
                levels[tid] = max(levels[p] for p in preds) + 1
        return levels


def compute_critical_path(dag: TaskDAG, durations: Optional[Dict[int, float]] = None) -> Tuple[float, List[int], Set[Tuple[int, int]]]:
    """Compute critical path via forward + backward DP passes.

    Args:
        dag: The task DAG.
        durations: Optional override durations (e.g., adjusted by agent speed).
                   If None, uses task.duration.

    Returns:
        makespan: The critical path length (L*).
        cp_tasks: List of task IDs on the critical path.
        cp_edges: Set of edges on the critical path.
    """
    topo = dag.topological_sort()

    # Get durations
    dur = {}
    for t in dag.tasks:
        dur[t.id] = durations[t.id] if durations and t.id in durations else t.duration

    # Forward pass: compute EST and EFT
    est = {}
    eft = {}
    for tid in topo:
        preds = dag.predecessors[tid]
        if not preds:
            est[tid] = 0.0
        else:
            est[tid] = max(eft[p] for p in preds)
        eft[tid] = est[tid] + dur[tid]

    # Makespan
    makespan = max(eft[tid] for tid in topo)

    # Backward pass: compute LST and LFT
    lst = {}
    lft = {}
    for tid in reversed(topo):
        succs = dag.successors[tid]
        if not succs:
            lft[tid] = makespan
        else:
            lft[tid] = min(lst[s] for s in succs)
        lst[tid] = lft[tid] - dur[tid]

    # Identify critical path
    slack = {}
    cp_tasks = []
    for tid in topo:
        slack[tid] = lst[tid] - est[tid]
        dag.get_task(tid).est = est[tid]
        dag.get_task(tid).eft = eft[tid]
        dag.get_task(tid).lst = lst[tid]
        dag.get_task(tid).lft = lft[tid]
        dag.get_task(tid).slack = slack[tid]
        if abs(slack[tid]) < 1e-9:
            dag.get_task(tid).on_critical_path = True
            cp_tasks.append(tid)
        else:
            dag.get_task(tid).on_critical_path = False

    # CP edges: edges where both endpoints are on CP and the edge is tight
    cp_edges = set()
    cp_set = set(cp_tasks)
    for src, dst in dag.edges:
        if src in cp_set and dst in cp_set:
            if abs(eft[src] - est[dst]) < 1e-9:
                cp_edges.add((src, dst))

    return makespan, cp_tasks, cp_edges


def generate_random_dag(n: int, edge_prob: float = 0.3,
                        duration_range: Tuple[float, float] = (1.0, 10.0),
                        failure_prob_range: Tuple[float, float] = (0.0, 0.2),
                        rng: Optional[np.random.RandomState] = None) -> TaskDAG:
    """Generate a random DAG with n tasks.

    Acyclicity enforced by only allowing edges from lower to higher index.

    Args:
        n: Number of tasks.
        edge_prob: Probability of edge between any valid pair.
        duration_range: (min, max) for uniform duration sampling.
        failure_prob_range: (min, max) for uniform failure probability.
        rng: Random state for reproducibility.

    Returns:
        TaskDAG instance.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    tasks = []
    for i in range(n):
        d = rng.uniform(*duration_range)
        p = rng.uniform(*failure_prob_range)
        tasks.append(Task(id=i, duration=d, failure_prob=p))

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                edges.append((i, j))

    # Ensure connectivity: if any node has no predecessors (except node 0)
    # and no successors, add an edge from a random earlier node
    for j in range(1, n):
        has_pred = any(dst == j for _, dst in edges)
        if not has_pred:
            src = rng.randint(0, j)
            edges.append((src, j))

    return TaskDAG(tasks=tasks, edges=edges)


def generate_linear_chain(m: int,
                          duration_range: Tuple[float, float] = (1.0, 5.0),
                          failure_prob_range: Tuple[float, float] = (0.05, 0.15),
                          rng: Optional[np.random.RandomState] = None) -> TaskDAG:
    """Generate a linear chain DAG of m tasks."""
    if rng is None:
        rng = np.random.RandomState(42)

    tasks = []
    for i in range(m):
        d = rng.uniform(*duration_range)
        p = rng.uniform(*failure_prob_range)
        tasks.append(Task(id=i, duration=d, failure_prob=p))

    edges = [(i, i + 1) for i in range(m - 1)]
    return TaskDAG(tasks=tasks, edges=edges)


def generate_research_pipeline_dag(rng: Optional[np.random.RandomState] = None) -> TaskDAG:
    """Generate the fixed 7-task research pipeline DAG.

    Structure:
     0: Literature Review    (d=3)
     1: Data Collection      (d=5)
     2: Problem Framing      (d=4)
     3: Method Design        (d=4, depends on 0, 2)
     4: Implementation       (d=6, depends on 1, 3)
     5: Evaluation           (d=5, depends on 4)
     6: Paper Writing        (d=4, depends on 3, 5)

    CP: 2->3->4->5->6 (len = 4+4+6+5+4 = 23)
    Width: 3 (tasks 0, 1, 2 can run in parallel initially)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    tasks = [
        Task(id=0, duration=3.0, failure_prob=0.05),  # Literature Review
        Task(id=1, duration=5.0, failure_prob=0.10),  # Data Collection
        Task(id=2, duration=4.0, failure_prob=0.08),  # Problem Framing
        Task(id=3, duration=4.0, failure_prob=0.12),  # Method Design
        Task(id=4, duration=6.0, failure_prob=0.15),  # Implementation
        Task(id=5, duration=5.0, failure_prob=0.10),  # Evaluation
        Task(id=6, duration=4.0, failure_prob=0.05),  # Paper Writing
    ]

    edges = [
        (0, 3),  # Literature Review -> Method Design
        (2, 3),  # Problem Framing -> Method Design
        (1, 4),  # Data Collection -> Implementation
        (3, 4),  # Method Design -> Implementation
        (4, 5),  # Implementation -> Evaluation
        (3, 6),  # Method Design -> Paper Writing
        (5, 6),  # Evaluation -> Paper Writing
    ]

    return TaskDAG(tasks=tasks, edges=edges)


def compute_lower_bound_makespan(dag: TaskDAG, num_agents: int) -> float:
    """Compute a lower bound on makespan.

    Lower bound = max(critical_path_length, total_work / num_agents).
    """
    cp_length, _, _ = compute_critical_path(dag)
    total_work = sum(t.duration for t in dag.tasks)
    return max(cp_length, total_work / num_agents)
