"""Evaluation run: exercises all 3 domain agents plus the human-approval gate
across a small set of representative queries and reports the metrics listed
in config/settings.yaml (evaluation.metrics).

Usage: .venv\\Scripts\\python.exe -m src.harness.evaluate
"""

import json
import statistics
import time
from pathlib import Path

from src.agents import supervisor
from src.harness.runner import run_turn

_RESULTS_PATH = Path("data/logs/evaluation_results.json")

# One query per domain agent (auto-approved), one that requires human approval,
# and a follow-up retrieval query to demonstrate RAG grounding (Checkpoint 3.1).
_QUERIES = [
    {"text": "Add a to-do to buy milk tomorrow", "expect_approval": False},
    {"text": "What's the current weather in Paris?", "expect_approval": False},
    {"text": "Save a note that my favorite color is teal", "expect_approval": False},
    {"text": "What is my favorite color, based on my notes?", "expect_approval": False},
    {"text": "Cancel my event for tomorrow", "expect_approval": True},
]


def run_evaluation() -> dict:
    turns = []
    for query in _QUERIES:
        start = time.time()
        result = run_turn(query["text"], approval_prompt=lambda _: "approve")
        latency = time.time() - start
        turns.append(
            {
                "query": query["text"],
                "expected_approval": query["expect_approval"],
                "actual_approval_required": result.get("requires_human_approval"),
                "response": result["response"],
                "latency_seconds": latency,
                "qa_score": result.get("qa_score"),
                "escalation_count": result.get("escalation_count"),
            }
        )

    qa_scores = [t["qa_score"] for t in turns if t["qa_score"] is not None]
    latencies = [t["latency_seconds"] for t in turns]
    escalations = sum(t["escalation_count"] or 0 for t in turns)

    summary = {
        "action_correctness_avg_qa_score": statistics.mean(qa_scores) if qa_scores else None,
        "action_correctness_pass_rate": (
            sum(1 for s in qa_scores if supervisor.passes_qa(s)) / len(qa_scores) if qa_scores else None
        ),
        "system_latency_avg_seconds": statistics.mean(latencies),
        "system_latency_max_seconds": max(latencies),
        "escalation_rate": escalations / len(turns),
        "turns": turns,
    }
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run_evaluation()
    print(json.dumps(result, indent=2, default=str))
