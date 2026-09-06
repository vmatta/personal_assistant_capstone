"""Clean console trace for 2 Tree-of-Thought demo prompts - suitable for live presentation.

These two prompts are hand-picked to showcase genuine multi-path ToT reasoning
(a rare case, ~5% of requests by design - see src/agents/supervisor.py
classify_complexity()). Everything else should route through the fast "direct"
path with a single LLM call.

Demo 1: Constraint-satisfaction scheduling (3 meetings, no overlap, limited
availability window) - ToT explores different arrangements and executes the
winning one.

Demo 2: Multi-constraint itinerary exploration (3 cities, driving limit,
indoor+outdoor mix) - ToT explores different "framings" of the request and
returns multiple concrete, comparable options with a final recommendation.

Suppresses noisy third-party logging (httpx, huggingface_hub, etc.) so only
the assistant's own reasoning trace (🧠 PLANNING, branch candidates, QA
scoring, dispatch, compose) is visible on the console.

Usage:
    From the project root, with the venv active:
        $env:PYTHONPATH="."
        .venv\\Scripts\\python.exe demo\\tot_demo_presentation.py

Each run uses a fresh random thread_id, so re-running during rehearsal never
picks up stale conversation state from a previous run.
"""
import sys, io, time, logging, uuid

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Silence noisy third-party loggers before anything else initializes them.
for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "urllib3", "openai", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from src.harness.runner import run_turn
from src.graph.workflow import workflow

# Make sure our own logger prints INFO-level messages (the emoji trace lines).
logging.getLogger("personal_assistant").setLevel(logging.INFO)

DEMOS = [
    (
        "ToT Demo 1: Constraint-Satisfaction Scheduling",
        "I need to schedule 3 meetings: one that needs 2 hours, one that needs 1 hour, and one "
        "that needs 30 minutes. They can't overlap and I'm only available Tuesday-Thursday 9-5. "
        "Find the best way to schedule all three.",
    ),
    (
        "ToT Demo 2: Multi-Constraint Itinerary Exploration",
        "Plan my weekend: I want to visit 3 different cities, stay under 4 hours driving, and "
        "include both indoor and outdoor activities. Show me different ways to organize this.",
    ),
]

for title, prompt in DEMOS:
    thread_id = f"demo-{uuid.uuid4()}"
    print("\n" + "#" * 100)
    print(f"# {title}")
    print("#" * 100)
    print(f"\nUSER: {prompt}\n")
    print("-" * 100)

    t0 = time.time()
    result = run_turn(prompt, thread_id=thread_id)
    elapsed = time.time() - t0

    config = {"configurable": {"thread_id": thread_id}}
    snap = workflow.get_state(config)
    thoughts = snap.values.get("thoughts", [])
    reasoning_mode = snap.values.get("reasoning_mode")

    print("-" * 100)
    print(f"\n[REASONING MODE: {(reasoning_mode or 'unknown').upper()}]  [ELAPSED: {elapsed:.1f}s]  [QA SCORE: {result['qa_score']}]")
    print(f"\n[CANDIDATE BRANCHES GENERATED: {len(thoughts)}]")
    for i, t in enumerate(thoughts, 1):
        status = "PRUNED" if t.get("pruned") else "SURVIVED"
        score = t.get("score")
        score_str = f"{score:.2f}" if score is not None else "n/a"
        print(f"  Branch {i} [{status}] (confidence: {score_str}) -> agent={t.get('target_agent')}")
        print(f"      {t.get('thought')}")

    print(f"\n[FINAL ASSISTANT RESPONSE]")
    print(result["response"])
    print()

print("\n" + "#" * 100)
print("# END OF DEMO")
print("#" * 100)
