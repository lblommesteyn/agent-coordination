# Experiment Improvements for DAG-Based Multi-Agent Coordination

## Purpose

The current seven experiments are useful, but they mainly establish internal validity:

> Under controlled simulations and explicit assumptions, DAG-aware assignment, validation, representation choice, and proactive controls reduce cost in multi-agent workflows.

That is not the same as showing that the framework improves real agent workloads. The corrected experimental story should separate three evidence types:

1. **Mechanistic evidence:** simulations explain why the controller works under known assumptions.
2. **Calibration evidence:** real traces show whether intermediate handoff-risk signals predict final failure.
3. **Intervention evidence:** real benchmarks show whether acting on those signals improves success, cost, or latency.

The existing seven simulation experiments should be presented as mechanistic analysis. The main external-validity evidence should come from benchmarked real or production-like workloads with objective final evaluators.

## Revised Claims

The experimental section should support these claims in order:

| Claim | Required evidence |
| --- | --- |
| The DAG abstraction improves multi-agent orchestration under controlled assumptions. | Existing simulator experiments 1-7. |
| Intermediate handoff-risk signals are meaningful on real workloads. | New Experiment 8: real-trace error calibration. |
| A risk-aware adaptive controller improves real benchmark outcomes under fixed budgets. | Intervention studies on SWE-bench Verified and MLE-bench. |
| The scheduling contribution is not specific to LLM agents. | WfCommons/WfBench non-LLM workflow experiments. |

This avoids overclaiming from simulation while giving the framework a path to stronger empirical support.

## Current Experiments as Mechanistic Analysis

The current experiments remain valuable, but their role should be reframed:

1. **Critical path assignment:** tests why critical-path-first assignment can beat random and round-robin assignment on synthetic DAGs.
2. **Validator placement:** tests how error compounds and where validators help most in linear chains.
3. **Optimal agent count:** tests the tradeoff between parallelism and coordination overhead.
4. **Representation format:** tests how prose, structured JSON, and compressed summaries change cost under different error-amplification regimes.
5. **Adaptive vs static policy:** tests closed-loop replanning under stochastic execution and drifting amplification factors.
6. **Proactive control primitives:** isolates speculation, early stopping, DAG morphing, and pre-warming.
7. **Composed proactive controller:** tests whether proactive controls compose on top of the adaptive controller.

These should be described as controlled stress tests, not as the main evidence that the framework works in deployed agent systems.

## New Experiment 8: Real-Trace Error Calibration and Intervention

### Motivation

The framework depends on edge-level handoff risk. Validators, early stopping, skipping, representation selection, and replanning are only useful if the controller can estimate when an intermediate output is likely to damage downstream execution.

The missing question is:

> Can the controller's estimated intermediate handoff-risk signal predict final task failure on real workloads, and does acting on that signal improve outcomes?

This should be split into two claims:

1. **Prediction claim:** intermediate signals predict downstream or final failure.
2. **Control claim:** a controller that acts on those signals improves benchmark outcomes under the same budget.

Do not describe observational correlations as "signals influence failure." They predict failure. Influence is only shown in the intervention comparison.

### Operational definition of handoff risk

Use an estimated risk signal:

```text
hat_A_ij = P(downstream failure | handoff state on edge i -> j,
             task metadata, executor metadata, representation,
             validator signals, runtime signals)
```

For compatibility with the existing multiplicative error model, this can also be reported as an expected downstream loss multiplier:

```text
hat_A_ij = E(loss_after_j / loss_before_j | handoff state)
```

The paper should be explicit that `hat_A_ij` is an estimator, not a directly observed primitive.

### Candidate intermediate signals

Use concrete, auditable signals rather than vague "quality" labels:

| Signal | Applies to | Notes |
| --- | --- | --- |
| Unit-test delta | SWE-bench | Number of newly passing/failing visible tests; do not use held-out gold tests for control decisions. |
| Generated-test pass rate | SWE-bench | Useful when no visible tests cover the suspected bug; final grading must remain held out. |
| Static-analysis warnings | SWE-bench | Lint/type/security warnings, changed warning count, syntax validity. |
| Patch-risk features | SWE-bench | Files touched, diff size, number of localized symbols, whether patch touches tests only. |
| Validation-score delta | MLE-bench | Public/local validation score change, cross-validation stability, leakage checks. |
| Runtime/resource anomalies | MLE-bench, WfBench | Timeouts, memory spikes, missing artifacts, failed subprocesses. |
| Submission/schema validity | MLE-bench, WorkArena, tau-bench | Required columns, JSON schema, API argument validity, database-state constraints. |
| LLM judge score | All LLM benchmarks | Should be calibrated against objective outcomes, not used alone as ground truth. |
| Executor confidence | All LLM benchmarks | Useful only after calibration; raw self-confidence is often miscalibrated. |

### Prediction study

Run many traces on real tasks and log every handoff:

```text
task id
edge i -> j
executor/model/tool used
representation format
intermediate artifacts
validator outputs
runtime metadata
hat_A_ij
final success/failure
downstream recovery events
cost, latency, tool calls, tokens
```

Evaluate whether `hat_A_ij` predicts downstream or final failure:

| Metric | Purpose |
| --- | --- |
| AUROC | Ranking quality for bad outcomes. |
| AUPRC | Better when failures are rare. |
| Brier score | Probabilistic calibration. |
| Expected calibration error | Whether predicted risks match observed frequencies. |
| Spearman/Pearson correlation | Relationship between handoff risk and downstream loss. |
| Time-to-detection | How early the risk becomes visible before final failure. |

Report calibration curves by benchmark and by edge type. A useful controller needs not only high AUROC; it also needs calibrated thresholds so it can decide when to validate, retry, skip, or escalate.

### Intervention study

After fitting or selecting the risk estimator, freeze thresholds and compare:

| Controller | Description |
| --- | --- |
| Single-agent ReAct-style loop | No explicit DAG scheduling; standard agent baseline. |
| Static DAG | Fixed decomposition and schedule, no adaptive risk response. |
| Round-robin multi-agent | Multi-agent but no critical-path or risk-aware assignment. |
| Adaptive DAG without risk signals | Replans from task completion and runtime status only. |
| Risk-aware adaptive DAG | Uses `hat_A_ij` for validators, retries, representation selection, and reassignment. |
| Full adaptive + proactive | Adds speculation, early stopping, skipping, and pre-warming when calibrated thresholds allow. |

Primary metrics:

| Metric | Definition |
| --- | --- |
| Success rate | Benchmark-specific final success. |
| Cost per success | Total tokens, wall-clock, tool calls, and compute divided by successes. |
| Latency | End-to-end time and critical-path waiting time. |
| Failure recovery rate | Fraction of initially bad traces recovered by validators/retries/replanning. |
| Regret vs static budget | Outcome loss at equal cost or equal latency budget. |

Use the same task set, model set, token budget, time budget, and tool limits for all controllers.

### Leakage controls

The most important validity risk is leaking final evaluator information into intermediate control decisions.

For SWE-bench:

- Do not use hidden/gold tests during controller execution.
- Use visible repository tests, generated tests, syntax checks, static analysis, and patch-level features as intermediate signals.
- Use the official final harness only for final scoring.

For MLE-bench:

- Use local train/validation splits and public-style validation during execution.
- Keep final grading scripts and held-out splits as the outcome evaluator.
- Add leakage checks because feature engineering mistakes can produce misleading validation gains.

For WorkArena, TheAgentCompany, and tau-bench:

- Use schema/API/state checks available to the agent as validators.
- Keep final database state, task-specific evaluator, or official scoring function as the outcome.
- Separate LLM judge outputs from objective final labels.

## Proposed Benchmark Suite

### Core external-validity benchmarks

| Benchmark | Claim | DAG mapping | Why it matters | Main caveat |
| --- | --- | --- | --- | --- |
| [SWE-bench Verified](https://www.swebench.com/SWE-bench/guides/datasets/) | Real AI agent workload | Issue understanding -> repo search -> bug localization -> patch generation -> test generation -> visible test execution -> patch repair -> final diff review | Objective software-engineering benchmark with real GitHub issues and held-out tests. Strong fit for validators, patch repair, and error-signal calibration. | Avoid hidden-test leakage; many tasks are expensive to run. |
| [MLE-bench](https://github.com/openai/mle-bench) | Real AI agent workload | Inspect data -> build baseline -> feature engineering -> train -> validate -> tune -> submit | ML workflows are naturally DAG-like and errors propagate through data, features, models, and submissions. Validation score, leakage checks, runtime checks, and submission format checks make strong validators. | Expensive; needs careful budget normalization across controllers. |
| [WfCommons/WfBench](https://wfcommons.org/) | Non-LLM scheduler contribution | Treat workflow tasks as services/executors; compare assignment, retry, validator/checkpoint placement, and pre-warming on scientific workflow DAGs | Separates the scheduling contribution from LLM uncertainty and tests whether the framework is useful outside LLM agents. | Needs strong classical scheduling baselines, not just random and round-robin. |

### Secondary transfer benchmarks

| Benchmark | Claim | DAG mapping | Why it matters | Main caveat |
| --- | --- | --- | --- | --- |
| [WorkArena](https://github.com/ServiceNow/WorkArena) | Enterprise web-agent workflow | Understand request -> gather records -> update fields -> trigger workflow -> verify state | Tests representation format, schema validation, and auditable enterprise workflows. | Browser/UI control adds confounds unrelated to orchestration. |
| [TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany) | Professional multi-tool work | Workplace request -> research/coding/documentation subtasks -> validate outputs -> integrate deliverable | Strong test for heterogeneous agents and multi-agent coordination on professional tasks. | Heavy environment and many sources of variance. |
| [tau-bench](https://taubench.com/) | Tool-agent conversations | User request -> policy interpretation -> API calls -> state verification -> correction | Relevant for schema validity, policy constraints, and final database-state validation. | Less naturally a multi-agent DAG; use for calibration and state-validation experiments rather than the main scheduling claim. |

### Benchmarks to defer

| Benchmark | Reason to defer |
| --- | --- |
| GAIA | Strong real-world assistant benchmark, but multimodality and web search add confounds. |
| AgentBench | Mainly evaluates single-agent decision making across environments, not multi-agent decomposition. |
| OSWorld | Valuable but introduces computer-use and multimodal UI control as dominant confounds. |
| WebArena/VisualWebArena | Realistic web tasks, but browser/UI fragility can swamp orchestration effects. |

## Stronger Experimental Design

| Study | Setup | Baselines | Metrics |
| --- | --- | --- | --- |
| External benchmark comparison | Run the controller on SWE-bench Verified and MLE-bench under fixed budgets. | Single-agent ReAct, static DAG, round-robin multi-agent, adaptive DAG without risk, full adaptive + proactive. | Success rate, cost per success, latency, tokens, tool calls, failures recovered. |
| Component ablation | Remove one controller component at a time. | Full model vs no CP assignment, no validators, no representation selection, no speculation, no early stop, no DAG morphing. | Contribution of each component to success/cost/latency. |
| `hat_A_ij` calibration | Estimate handoff risk from real traces instead of hand-specifying it. | Hand-specified, LLM-judge, deterministic-test, hybrid, learned predictor. | AUROC, AUPRC, Brier score, calibration error, time-to-detection. |
| Cross-benchmark generalization | Tune thresholds on one split or benchmark, test on another. | Same controller with frozen parameters. | Measures overfitting to one DAG family or benchmark. |
| Human-readable audit study | Log every controller decision and ask reviewers to diagnose failures. | Raw agent transcript vs DAG audit trace. | Reviewer time, error detection rate, decision trust/usability. |
| Executor heterogeneity | Use cheap model, strong model, retrieval agent, code runner, and judge as distinct executors. | Homogeneous agents vs heterogeneous assigned agents. | Cost-quality tradeoff and where expensive executors are worth using. |
| DAG-construction sensitivity | Compare hand-written DAGs, LLM-generated DAGs, and fixed templates. | Same controller, different DAG source. | Robustness to imperfect task decomposition. |
| Non-LLM workflow scheduling | Run WfBench workflows with service-like executors. | HEFT/list scheduling, CP scheduling, static workflow-engine defaults, retry-only policies. | Makespan, cost, retry count, checkpoint overhead, SLA violations. |

## Recommended Implementation Order

1. **Create trace schema.** Define a common event log for tasks, handoffs, validators, artifacts, costs, and final outcomes.
2. **Run passive traces first.** Collect traces without changing the controller so calibration is not confounded by intervention.
3. **Fit simple risk estimators.** Start with logistic regression or gradient-boosted trees over deterministic signals, then compare against LLM judges and hybrid predictors.
4. **Freeze thresholds.** Select thresholds on a dev split before running intervention experiments.
5. **Run intervention experiments.** Compare risk-aware and risk-blind controllers under equal budgets.
6. **Add ablations.** Remove one mechanism at a time to identify what actually contributes.
7. **Test generalization.** Transfer thresholds across task families, benchmarks, and executor mixes.

## Minimal Viable External-Validity Package

If compute is limited, the smallest credible package is:

1. SWE-bench Verified subset, 50-100 tasks.
2. Passive trace calibration of `hat_A_ij`.
3. Risk-aware adaptive controller vs static DAG and single-agent baseline.
4. Ablations: no validators, no representation selection, no risk-aware repair.
5. Report success rate, cost per success, AUROC/AUPRC, Brier score, and calibration curves.

The next-best expansion is MLE-bench Lite or a curated MLE-bench subset, because it tests a different domain where DAG structure is natural and validation signals are objective.

## Revised Paper Framing

The experimental narrative should be:

1. **Simulation experiments 1-7:** mechanistic evidence that the controller's components behave as predicted.
2. **Experiment 8 calibration:** real traces show whether `hat_A_ij` is predictive and calibrated.
3. **Benchmark intervention:** risk-aware adaptive control improves real workload outcomes at fixed budget.
4. **WfBench generalization:** the framework contributes beyond LLM-specific agent behavior.

A concise claim for the paper:

> The simulations establish internal validity of the controller mechanisms. Real-trace calibration tests whether edge-level handoff risk is measurable. Benchmark interventions then test whether using that risk signal improves real-world-like agent workflows under fixed cost and latency budgets.

