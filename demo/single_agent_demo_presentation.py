"""Clean console trace for single-agent demo prompts covering all 3 domain agents.

2 prompts each for scheduler_todo, notes_knowledge, and weather_external - suitable
for a live presentation showing each domain agent working independently before
moving on to the multi-agent and Tree-of-Thought demos.

IMPORTANT NOTE ON notes_knowledge candidates: the "save" and "query" prompts are run
on the SAME conversation thread (same thread_id) so the query prompt is properly
grounded in a fact this exact demo run just saved. Running the query prompt on an
unrelated/fresh thread was tested and found unreliable for a live demo - the vector
store persists across the whole process/session, so a fresh thread's query can
retrieve a stale note from an EARLIER unrelated run and confuse the QA critic (it saw
this fail with QA score 0.0 - "unsupported assumption" - even though retrieval
technically worked, because the critic has no visibility into retrieval evidence).

NOTE ON scheduler_todo candidate #2: avoid the word "list" in isolation (e.g. "List
all my tasks for this week") - this was tested and found to occasionally misroute to
notes_knowledge instead of scheduler_todo. "Show me all my scheduled events for
today" is a more reliable phrasing that consistently routes correctly.

Suppresses noisy third-party logging (httpx, huggingface_hub, etc.) so only the
assistant's own reasoning trace is visible on the console.

Usage:
    From the project root, with the venv active:
        $env:PYTHONPATH="."
        .venv\\Scripts\\python.exe demo\\single_agent_demo_presentation.py
"""
import sys, io, time, logging, uuid

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "urllib3", "openai", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from src.harness.runner import run_turn
from src.graph.workflow import workflow

logging.getLogger("personal_assistant").setLevel(logging.INFO)


def run_demo(title, prompt, thread_id=None):
    thread_id = thread_id or f"demo-{uuid.uuid4()}"
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
    print(f"[AGENT DISPATCHED: {[r.get('agent') for r in subagent_results]}]")
    print(f"\n[FINAL RESPONSE]")
    print(result["response"])
    print()
    return thread_id


# --- scheduler_todo ---
run_demo("scheduler_todo Demo 1: Schedule an appointment", "Schedule a dentist appointment on Tuesday at 3 PM.")
run_demo("scheduler_todo Demo 2: List events for today", "Show me all my scheduled events for today.")

# --- notes_knowledge (save + query chained on the SAME thread for proper grounding) ---
notes_thread = f"demo-notes-{uuid.uuid4()}"
run_demo("notes_knowledge Demo 1: Save a preference", "Save that my favorite hobby is painting.", thread_id=notes_thread)
run_demo("notes_knowledge Demo 2: Query the saved preference", "What is my favorite hobby?", thread_id=notes_thread)

# --- weather_external ---
run_demo("weather_external Demo 1: Current weather", "What's the current weather in Pittsburgh?")
run_demo("weather_external Demo 2: Current temperature", "What's the temperature in Miami right now?")

print("\n" + "#" * 100)
print("# END OF DEMO")
print("#" * 100)
