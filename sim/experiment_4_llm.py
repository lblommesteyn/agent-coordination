"""Experiment 4 (LLM): real Gemini pipeline for representation format study.

Topic: "impact of sleep on athletic performance", 7-task research DAG,
three formats (prose / structured_json / compressed_summary).
Returns a DataFrame compatible with experiment_4_representation().

Usage:
  export GEMINI_API_KEY=<your_key>
  python experiment_4_llm.py
"""

import os, re, sys, time
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

sys.path.insert(0, os.path.dirname(__file__))

FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

TOPIC = "impact of sleep on athletic performance"

TASK_NAMES = [
    "Literature Review", "Data Collection", "Problem Framing",
    "Method Design", "Implementation", "Evaluation", "Paper Writing",
]
TASK_KEY = {
    0: "lit_review", 1: "data_collection", 2: "problem_framing",
    3: "method_design", 4: "implementation", 5: "evaluation", 6: "paper_writing",
}
DAG_EDGES = [(0,3),(2,3),(1,4),(3,4),(4,5),(3,6),(5,6)]
CP_EDGES  = frozenset([(2,3),(3,4),(4,5),(5,6)])
COST_MULT = {"prose": 1.0, "structured_json": 0.6, "compressed_summary": 0.3}

FORMAT_INSTRUCTIONS = {
    "prose": (
        "Write your entire response as clear, flowing prose paragraphs. "
        "No bullet points, numbered lists, or JSON."
    ),
    "structured_json": (
        "Respond ONLY with a valid JSON object using descriptive snake_case keys. "
        "No prose outside the JSON braces."
    ),
    "compressed_summary": (
        "Respond ONLY as concise bullet points prefixed with '- '. "
        "One key fact per bullet. Maximise information density."
    ),
}

TASK_PROMPTS = [
    # 0 — Literature Review
    f"""You are an academic research assistant. Conduct a literature review on: "{TOPIC}".
Summarise the 5-6 most important empirical findings. For each: state the claim,
study design, and quantitative effect size if available. Under 350 words.""",

    # 1 — Data Collection
    f"""You are a sports-science data analyst. Specify data requirements for a study on "{TOPIC}".
Cover: measurement instruments (polysomnography, actigraphy, wearables), performance tests
(reaction time, VO2max, sprint, strength), recommended sample, and confounders to control.
Under 300 words.""",

    # 2 — Problem Framing
    f"""You are a research methodologist. Frame the research problem for "{TOPIC}".
Define: primary research question, three testable null hypotheses, the theoretical mechanism
linking sleep to athletic performance, and the scope of the study. Under 300 words.""",

    # 3 — Method Design  (preds: 0, 2)
    """You are a study-design specialist. Design a methodology for studying sleep and athletic performance.

=== LITERATURE REVIEW ===
{lit_review}

=== PROBLEM FRAMING ===
{problem_framing}

Design: study type with justification, participant criteria and sample-size calculation
(power=0.80, alpha=0.05), sleep protocol, performance battery, and statistical analysis plan.
Under 400 words.""",

    # 4 — Implementation  (preds: 1, 3)
    """You are a data scientist. Implement the analysis pipeline for the sleep-performance study.

=== DATA COLLECTION PLAN ===
{data_collection}

=== METHOD DESIGN ===
{method_design}

Provide: preprocessing steps, primary statistical test with justification, effect-size
calculation, visualisation plan, and Python pseudocode (10-15 lines). Under 400 words.""",

    # 5 — Evaluation  (pred: 4)
    """You are a peer reviewer. Evaluate the analysis pipeline below.

=== IMPLEMENTATION ===
{implementation}

Assess: statistical validity, plausible results with p-values and Cohen's d,
internal/external validity, and an overall rigour rating 1-10. Under 400 words.""",

    # 6 — Paper Writing  (preds: 3, 5)
    """You are an academic writer. Write a research abstract and conclusion.

=== METHOD DESIGN ===
{method_design}

=== EVALUATION ===
{evaluation}

Write: a 200-word abstract (Background/Objective/Methods/Results/Conclusion),
three key contributions, practical recommendations, and two future directions.
Under 500 words.""",
]

_AIJ_PROMPT = """Score information degradation at this agent handoff (0.0-2.0):
  1.0 = no change, <1.0 = improved, >1.0 = information lost.

"{src}" output:
{src_output}

"{dst}" output:
{dst_output}

Reply with a single decimal number only."""


def _build_model(api_key=None):
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    genai.configure(api_key=key)
    return genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config={"temperature": 0.2, "max_output_tokens": 1024},
    )


def _call(model, prompt, max_retries=3, extra_cfg=None):
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = (model.generate_content(prompt, generation_config=extra_cfg)
                    if extra_cfg else model.generate_content(prompt))
            return resp.text.strip(), time.time() - t0
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return "", 0.0


def _measure_aij(model, src_name, src_output, dst_name, dst_output):
    prompt = _AIJ_PROMPT.format(
        src=src_name, src_output=src_output[:2000],
        dst=dst_name, dst_output=dst_output[:2000],
    )
    text, _ = _call(model, prompt, extra_cfg={"temperature": 0.0, "max_output_tokens": 16})
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", text.strip())
    return max(0.0, min(2.0, float(m.group(1)))) if m else 1.0


def _run_pipeline(model, format_name, verbose=True):
    fmt_instr = FORMAT_INSTRUCTIONS[format_name]
    task_outputs, task_durations, edge_aij = {}, {}, {}
    preds_of = {}
    for src, dst in DAG_EDGES:
        preds_of.setdefault(dst, []).append(src)

    for tid in range(7):
        preds = sorted(preds_of.get(tid, []))
        if not preds:
            prompt = f"{TASK_PROMPTS[tid]}\n\n{fmt_instr}"
        else:
            ctx = {TASK_KEY[p]: task_outputs[p] for p in preds}
            prompt = f"{TASK_PROMPTS[tid].format(**ctx)}\n\n{fmt_instr}"

        if verbose:
            print(f"    [{format_name}] Task {tid} ({TASK_NAMES[tid]}) ...", end="", flush=True)
        output, duration = _call(model, prompt)
        task_outputs[tid] = output
        task_durations[tid] = duration
        if verbose:
            print(f" {duration:.2f}s")
        time.sleep(1.0)

        for pred_id in preds:
            edge = (pred_id, tid)
            if verbose:
                print(f"      Aij {pred_id}->{tid} ...", end="", flush=True)
            aij = _measure_aij(model, TASK_NAMES[pred_id], task_outputs[pred_id],
                               TASK_NAMES[tid], output)
            edge_aij[edge] = aij
            if verbose:
                print(f" {aij:.3f}")
            time.sleep(0.5)

    return {"task_outputs": task_outputs, "task_durations": task_durations, "edge_aij": edge_aij}


def _plot(df, pipeline_results, formats):
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind", len(formats))
    fmt_labels = {"prose": "Prose", "structured_json": "Structured JSON",
                  "compressed_summary": "Compressed Summary"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    width = 0.25

    ax = axes[0]
    edge_strs = [f"{s}->{d}" for s, d in DAG_EDGES]
    x = np.arange(len(DAG_EDGES))
    for i, fmt in enumerate(formats):
        idx = df[df["format"] == fmt].set_index("edge")
        vals = [float(idx.loc[e, "a_ij_base"]) if e in idx.index else 1.0 for e in edge_strs]
        ax.bar(x + i*width, vals, width, label=fmt_labels[fmt], color=palette[i], alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{s}->{d}" for s,d in DAG_EDGES], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("$A_{ij}$")
    ax.set_title("Measured $A_{ij}$ per Handoff")
    ax.legend(fontsize=7)

    ax = axes[1]
    x2 = np.arange(7)
    for i, fmt in enumerate(formats):
        vals = [pipeline_results[fmt]["task_durations"].get(t, 0.0) for t in range(7)]
        ax.bar(x2 + i*width, vals, width, label=fmt_labels[fmt], color=palette[i], alpha=0.85)
    ax.set_xticks(x2 + width)
    ax.set_xticklabels([n[:9] for n in TASK_NAMES], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Latency (s)")
    ax.set_title("Task Latency $d_i$")
    ax.legend(fontsize=7)

    ax = axes[2]
    summary = df.groupby("format")[["makespan","overhead","final_error"]].first().reindex(formats)
    x3 = np.arange(len(formats))
    ax.bar(x3, summary["makespan"], 0.5, label="Makespan", color="steelblue", alpha=0.75)
    ax.bar(x3, summary["overhead"], 0.5, bottom=summary["makespan"],
           label="Overhead", color="orange", alpha=0.75)
    ax.bar(x3, summary["final_error"], 0.5,
           bottom=summary["makespan"]+summary["overhead"],
           label="Final Error", color="firebrick", alpha=0.75)
    ax.set_xticks(x3)
    ax.set_xticklabels([fmt_labels[f] for f in formats], rotation=12, ha="right")
    ax.set_ylabel("Cost")
    ax.set_title("Total Cost by Format")
    ax.legend(fontsize=8)

    fig.suptitle(f'Experiment 4 (LLM): "{TOPIC}"', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig4_representation_llm.pdf"), bbox_inches="tight")
    plt.close(fig)


def experiment_4_llm_representation(api_key=None, verbose=True):
    """Run the 7-task Gemini pipeline three times (one per format).

    Returns a DataFrame compatible with experiment_4_representation().
    """
    if verbose:
        print(f"Experiment 4 (LLM) — topic: {TOPIC}")
    model = _build_model(api_key)
    formats = ["prose", "structured_json", "compressed_summary"]
    alpha, beta, gamma, epsilon_0 = 0.3, 0.5, 0.5, 0.1

    pipeline_results = {}
    for fmt in formats:
        if verbose:
            print(f"\n  Format: {fmt}")
        pipeline_results[fmt] = _run_pipeline(model, fmt, verbose=verbose)

    rows = []
    for fmt in formats:
        dur = pipeline_results[fmt]["task_durations"]
        edge_aij = pipeline_results[fmt]["edge_aij"]
        makespan = max(dur[0], dur[1], dur[2]) + dur[3] + dur[4] + dur[5] + dur[6]
        final_error = epsilon_0 * float(np.prod([edge_aij.get(e, 1.0) for e in sorted(CP_EDGES)]))
        overhead = gamma * len(DAG_EDGES) * COST_MULT[fmt]
        total_cost = makespan + alpha * overhead + beta * final_error
        for src, dst in DAG_EDGES:
            rows.append({
                "a_ij_base":   edge_aij.get((src, dst), 1.0),
                "format":      fmt,
                "makespan":    makespan,
                "overhead":    overhead,
                "final_error": final_error,
                "total_cost":  total_cost,
                "edge":        f"{src}->{dst}",
                "src_task":    TASK_NAMES[src],
                "dst_task":    TASK_NAMES[dst],
                "duration_src": dur[src],
                "duration_dst": dur[dst],
            })

    df = pd.DataFrame(rows)
    _plot(df, pipeline_results, formats)
    if verbose:
        print("\n", df.groupby("format")[["makespan","total_cost","a_ij_base"]].agg(
            {"makespan":"first","total_cost":"first","a_ij_base":"mean"}
        ).round(3))
        print(f"\nFigure -> {FIGURE_DIR}/fig4_representation_llm.pdf")
    return df


if __name__ == "__main__":
    df = experiment_4_llm_representation(verbose=True)
    print(df[["format","edge","a_ij_base","duration_src","total_cost"]].to_string(index=False))
