"""Harness: drives the orchestrator graph with a step/timeout guardrail
(Guardrail 1), resumes human-in-the-loop interrupts, and logs each turn.
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Callable, Optional

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from config import SETTINGS
from src.graph.state import new_state
from src.graph.workflow import workflow
from src.harness.logger import log_event

_HARNESS_CFG = SETTINGS.get("harness", {})
STEP_TIMEOUT_SECONDS = _HARNESS_CFG.get("step_timeout_seconds", 30)
MAX_SYSTEM_STEPS = _HARNESS_CFG.get("max_system_steps", 6)

_executor_pool = ThreadPoolExecutor(max_workers=1)


def invoke_with_timeout(payload, config: dict) -> dict:
    """Run one workflow.invoke() call under the overall harness timeout guardrail
    (Guardrail 1: STEP_TIMEOUT_SECONDS * MAX_SYSTEM_STEPS). Public so any caller
    (CLI harness, Streamlit UI, demos) gets the same hard cutoff instead of
    calling workflow.invoke() directly with no timeout protection.
    """
    future = _executor_pool.submit(workflow.invoke, payload, config)
    try:
        return future.result(timeout=STEP_TIMEOUT_SECONDS * MAX_SYSTEM_STEPS)
    except FutureTimeoutError as exc:
        raise TimeoutError(
            f"Workflow exceeded overall timeout of {STEP_TIMEOUT_SECONDS * MAX_SYSTEM_STEPS}s"
        ) from exc


# Backwards-compatible private alias (kept in case anything still imports the old name).
_invoke_with_timeout = invoke_with_timeout


def build_turn_payload(user_text: str, config: dict):
    """Build the graph input for a new turn.

    On a brand-new thread (no prior checkpoint), send a full fresh AssistantState via
    new_state(). On a thread that already has checkpointed state, send an update that:
      1. Appends the new user message.
      2. EXPLICITLY resets every turn-local field to its fresh-turn default (thoughts,
         subagent_results, selected_branch_id, qa_score, qa_feedback, requires_human_approval,
         human_decision, final_response, error, retry_count, step_count). Without this,
         LangGraph's partial-update merge would leave these fields holding STALE values
         from the previous turn — most dangerously, a previous turn's
         human_decision="approved" would silently leak into an unrelated new destructive
         request on the same thread, skipping Guardrail 3's approval gate entirely.
      3. PRESERVES only the fields intended to persist across the whole conversation:
         escalation_count and consecutive_ambiguous_turns (Guardrail 3b), by simply
         omitting them from this update so the checkpointer keeps their existing values.
    """
    existing = workflow.get_state(config)
    if not existing.values:
        return new_state(user_text)
    return {
        "messages": [HumanMessage(content=user_text)],
        "reasoning_mode": None,
        "thoughts": [],
        "selected_branch_id": None,
        "subagent_results": [],
        "qa_score": None,
        "qa_feedback": None,
        "retry_count": 0,
        "step_count": 0,
        "requires_human_approval": False,
        "human_decision": None,
        "final_response": None,
        "error": None,
    }


def run_turn(
    user_text: str,
    thread_id: Optional[str] = None,
    approval_prompt: Callable[[str], str] = input,
) -> dict:
    """Run one user turn through the orchestrator graph, handling any human-in-the-loop pause."""
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    start = time.time()

    log_event("turn_start", thread_id=thread_id, user_text=user_text)
    result = _invoke_with_timeout(build_turn_payload(user_text, config), config)

    approval_was_requested = False
    while "__interrupt__" in result:
        approval_was_requested = True
        interrupt_payload = result["__interrupt__"][0].value
        log_event("human_approval_requested", thread_id=thread_id, payload=interrupt_payload)
        answer = approval_prompt(
            f"Approval needed for: {interrupt_payload.get('pending_action')} [approve/reject]: "
        ).strip().lower()
        decision = "approved" if answer.startswith("a") else "rejected"
        log_event("human_approval_decision", thread_id=thread_id, decision=decision)
        result = _invoke_with_timeout(Command(resume=decision), config)

    latency = time.time() - start
    log_event(
        "turn_end",
        thread_id=thread_id,
        latency_seconds=latency,
        qa_score=result.get("qa_score"),
        escalation_count=result.get("escalation_count"),
        final_response=result.get("final_response"),
    )
    return {
        "thread_id": thread_id,
        "response": result.get("final_response"),
        "latency_seconds": latency,
        "qa_score": result.get("qa_score"),
        "escalation_count": result.get("escalation_count"),
        # Final graph state's flag: always False once approved (by design, so QA
        # retries don't re-prompt). Use approval_was_requested to check whether the
        # gate actually fired at any point during this turn.
        "requires_human_approval": result.get("requires_human_approval"),
        "approval_was_requested": approval_was_requested,
    }
