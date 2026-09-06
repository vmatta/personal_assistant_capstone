"""Clean console trace for 2 multi-agent coordination demo prompts - suitable for live presentation.

These two prompts are hand-picked to showcase reliable SEQUENTIAL multi-agent dispatch
(see CAPSTONE_FINAL_REPORT.md: parallel ThreadPoolExecutor dispatch deadlocked in
LangGraph, so the system dispatches secondary agents sequentially within one turn).

Demo 1: Schedule + Weather (scheduler_todo + weather_external) - the primary agent
schedules an outdoor event, then a secondary agent automatically checks the weather
for that event, with location extracted directly from the user's request.

Demo 2: Schedule + Save Note (scheduler_todo + notes_knowledge) - the primary agent
schedules an event, then a secondary agent saves a related preference/fact, each
agent correctly scoped to ONLY its own slice of the compound request.

NOTE: A third candidate ("save a note that my project deadline is next month, then
schedule a reminder...") was tested and REJECTED for this demo - it reliably scored
QA 0.1 due to a genuine date-ambiguity issue ("next month" vs. the resolved date),
triggering QA retries and an escalation. Not reliable enough for a live demo.

Suppresses noisy third-party logging (httpx, huggingface_hub, etc.) so only the
assistant's own reasoning trace (dispatch, secondary dispatch, QA scoring, compose)
is visible on the console.

Usage:
    From the project root, with the venv active:
        $env:PYTHONPATH="."
        .venv\\Scripts\\python.exe demo\\multiagent_demo_presentation.py

Each run uses a fresh random thread_id, so re-running during rehearsal never picks
up stale conversation state from a previous run.
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
        "Multi-Agent Demo 1: Schedule + Weather (scheduler_todo -> weather_external)",
        "Schedule an outdoor picnic in Denver on Friday at 2 PM, and check if the weather "
        "will be suitable for an outdoor event.",
    ),
    (
        "Multi-Agent Demo 2: Schedule + Save Note (scheduler_todo -> notes_knowledge)",
        "Schedule volleyball in Miami on Saturday at 4 PM and save it as my favorite sport.",
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
    subagent_results = snap.values.get("subagent_results", [])

    print("-" * 100)
    print(f"\n[ELAPSED: {elapsed:.1f}s]  [QA SCORE: {result['qa_score']}]")
    print(f"\n[AGENTS DISPATCHED: {len(subagent_results)}]")
    for i, r in enumerate(subagent_results, 1):
        print(f"  Agent {i}: {r.get('agent')}  (success={r.get('success')})")
        print(f"      output: {r.get('output')}")

    print(f"\n[FINAL COMPOSED RESPONSE]")
    print(result["response"])
    print()

print("\n" + "#" * 100)
print("# END OF DEMO")
print("#" * 100)
