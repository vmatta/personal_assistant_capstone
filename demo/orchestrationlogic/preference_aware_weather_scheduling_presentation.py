"""Clean console trace for the preference-aware, weather-conditional multi-day
scheduling demo - suitable for a live presentation.

This showcases NEW orchestration logic (not just a prompt tweak) added to the
system: src/graph/nodes.py::_weather_permitting_multi_day_response() and
src/tools/external.py::get_daily_forecast().

USER PROMPT (the original ask this demo is built around):
    "If weather permits can you schedule an outdoor Saturday or Sunday"

What makes this hard for a naive system: the request implicitly requires THREE
different capabilities to work together in one turn:
  1. Recognize "Saturday or Sunday" as two independent candidate days to evaluate
     (not a single ambiguous date).
  2. Cross-reference SAVED PREFERENCES (long-term memory / RAG over notes) to see
     if either day should be ruled out before even checking weather - in this
     demo, a previously-saved note "On Sundays, I only relax at home" rules out
     Sunday as a candidate.
  3. For the remaining candidate day(s), fetch the ACTUAL FUTURE-DATE forecast
     (not just "current weather", which only answers questions about right now)
     and apply a suitability rule to decide.

Demo flow (2 turns, same conversation thread):
  Turn 1: "If weather permits can you schedule an outdoor activity in New York
           City on Saturday or Sunday"
          -> System rules out Sunday (preference), checks Saturday's forecast,
             recommends Saturday, asks for a time.
  Turn 2: "Saturday at 2 PM"
          -> Falls through to the normal scheduler and ACTUALLY BOOKS the event.

Also demonstrates a "no location given" edge case first (the exact original
prompt with no city named), and the design decision that the system asks for a
location instead of guessing one - consistent with the rest of the codebase's
"never assume a specific detail; ask" philosophy (see
src/agents/context.py's DO NOT guess/default location rules for weather_external).

Suppresses noisy third-party logging (httpx, huggingface_hub, etc.) so only the
assistant's own reasoning trace is visible on the console.

Usage:
    From the project root, with the venv active:
        $env:PYTHONPATH="."
        .venv\\Scripts\\python.exe demo\\orchestrationlogic\\preference_aware_weather_scheduling_presentation.py

PREREQUISITE: this demo depends on a saved note already existing in the vector
store: "Preference: On Sundays, I only relax at home." If that note is not
present (e.g. a fresh/reset ChromaDB), run the SETUP step below first, or the
"rule out Sunday" behavior will not trigger (the system will instead say both
days look suitable, or ask a clarifying question if it can't resolve a decision).
"""
import sys, io, time, logging, uuid

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "urllib3", "openai", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from src.harness.runner import run_turn
from src.tools.memory import search_notes, save_note

logging.getLogger("personal_assistant").setLevel(logging.INFO)


def ensure_sunday_preference_saved():
    """SETUP: make sure the demo's required preference note exists before running.
    Idempotent - if the note is already there, this is a no-op (semantic search
    for an exact/near-duplicate match before saving again).
    """
    existing = search_notes("Sunday preference relax", k=3)
    already_saved = any("sunday" in n.lower() and "relax" in n.lower() for n in existing)
    if not already_saved:
        print("[SETUP] Saving required preference note for this demo...")
        save_note("On Sundays, I only relax at home.")
    else:
        print("[SETUP] Required preference note already present.")


def run_demo(title, prompt, thread_id):
    print("\n" + "#" * 100)
    print(f"# {title}")
    print("#" * 100)
    print(f"\nUSER: {prompt}\n")
    print("-" * 100)

    t0 = time.time()
    result = run_turn(prompt, thread_id=thread_id)
    elapsed = time.time() - t0

    print("-" * 100)
    print(f"\n[ELAPSED: {elapsed:.1f}s]")
    print(f"\n[FINAL RESPONSE]")
    print(result["response"])
    print()


ensure_sunday_preference_saved()

# --- Edge case: original exact prompt with NO location given ---
run_demo(
    "Edge Case: No location given (system asks instead of guessing)",
    "If weather permits can you schedule an outdoor Saturday or Sunday",
    thread_id=f"demo-{uuid.uuid4()}",
)

# --- Full flow: location given, 2-turn conversation ---
main_thread = f"demo-{uuid.uuid4()}"
run_demo(
    "Main Demo Turn 1: Preference rules out Sunday, weather confirms Saturday",
    "If weather permits can you schedule an outdoor activity in New York City on Saturday or Sunday",
    thread_id=main_thread,
)
run_demo(
    "Main Demo Turn 2: Follow-up with a time -> actually books the event",
    "Saturday at 2 PM",
    thread_id=main_thread,
)

print("\n" + "#" * 100)
print("# END OF DEMO")
print("#" * 100)
