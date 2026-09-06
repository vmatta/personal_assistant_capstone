"""CLI entry point for the personal assistant orchestrator."""

from src.harness.runner import run_turn
from src.tools.memory import warm_up_async

_DIVIDER = "-" * 60


def main() -> None:
    # Kick off ChromaDB/embedding-model init in the background so the first
    # scheduling/notes request doesn't pay the ~10s cold-start cost inline.
    warm_up_async()

    thread_id = None
    print("Personal Assistant (type 'exit' to quit)")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue
        result = run_turn(user_text, thread_id=thread_id)
        thread_id = result["thread_id"]
        # Reasoning/harness trace is printed above this point; the final answer stands alone below.
        print(f"\n{_DIVIDER}\nFinal Response:\n{result['response']}\n{_DIVIDER}\n")


if __name__ == "__main__":
    main()
