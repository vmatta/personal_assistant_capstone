"""Streamlit UI for the personal assistant capstone."""

from __future__ import annotations

import json
import uuid
import logging
import re
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from langgraph.types import Command
from openai import APIConnectionError, OpenAIError

from config import LLM_MODEL, SETTINGS
from src.graph.state import new_state
from src.harness.logger import log_event
from src.harness.runner import STEP_TIMEOUT_SECONDS, MAX_SYSTEM_STEPS, build_turn_payload, invoke_with_timeout
from src.tools.memory import warm_up_async


LOG_FILE = Path("data/logs/run_events.jsonl")
MAX_CONTEXT_MESSAGES = 12
USER_TIMEZONE = ZoneInfo("America/New_York")
TOOL_SCOPES = {
    "scheduler_todo": ["add_todo", "list_todos", "complete_todo"],
    "notes_knowledge": ["save_note", "search_notes", "search_knowledge_base"],
    "weather_external": ["get_current_weather"],
}

# Global container for displaying logs
_log_container = None
_current_logs = []  # Accumulate logs for persistent display
_log_container_active = False


class StreamlitLogHandler(logging.Handler):
    """Custom logging handler that writes to Streamlit UI.

    Streamlit widgets are tied to the active script context; background threads
    (for example, timeouts or warm-up tasks) do not have a valid ScriptRunContext.
    In that case we still accumulate the log line for later display, but we must
    never call .write() on a Streamlit container from a non-main-thread context.
    """
    def emit(self, record):
        global _log_container, _current_logs, _log_container_active
        try:
            msg = self.format(record)
            # Always store in accumulator for later UI inspection.
            _current_logs.append(msg)

            # Guard against background-thread writes: only the main Streamlit script
            # thread has a valid ScriptRunContext. If missing, skip UI writes.
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx
            except Exception:
                get_script_run_ctx = None

            script_ctx = get_script_run_ctx() if get_script_run_ctx is not None else None
            if script_ctx is None:
                return

            if _log_container_active and _log_container is not None:
                _log_container.write(msg)
        except Exception:
            # Silently fail if container no longer exists or a background thread
            # calls into Streamlit without a script context.
            pass


def setup_logging():
    """Configure logging to display in Streamlit."""
    try:
        logger = logging.getLogger("personal_assistant")
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        
        # Add Streamlit handler
        handler = StreamlitLogHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        
        return logger
    except Exception:
        # Fallback: just use the logger without Streamlit handler
        logger = logging.getLogger("personal_assistant")
        logger.setLevel(logging.INFO)
        return logger


def provider_error_message(exc: Exception) -> str:
    return (
        "I couldn't reach the configured LLM provider. Please check your network connection, "
        f"API key/provider settings, and try again. Details: {exc.__class__.__name__}"
    )


def init_session() -> None:
    defaults = {
        "thread_id": str(uuid.uuid4()),
        "chat_messages": [],
        "pending_interrupt": None,
        "last_result": None,
        "memory_notes": [],
        "current_turn_agents": [],  # Track agents called for current prompt
        "last_reasoning_logs": [],  # Keep reasoning steps visible
        "conditional_action_ready": False,
        "conditional_execute_command": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner=False)
def get_workflow():
    from src.graph.workflow import workflow

    return workflow


def build_reasoning_trace(result: dict) -> list[str]:
    """Create a compact, user-visible summary from the graph's actual state."""
    trace = []
    handled_directly = result.get("handled_directly")
    reasoning_mode = result.get("reasoning_mode")
    if reasoning_mode:
        trace.append(f"Reasoning mode: {reasoning_mode}")
    if handled_directly == "capability_response":
        trace.append("Answered directly: assistant capability summary")
        return trace
    if handled_directly == "out_of_scope":
        trace.append("Answered directly: request outside configured assistant scope")
        return trace

    thoughts = result.get("thoughts") or []
    selected_branch_id = result.get("selected_branch_id")
    selected = next((thought for thought in thoughts if thought.get("node_id") == selected_branch_id), None)
    if selected:
        trace.append(f"Selected subagent: {selected.get('target_agent')}")
        trace.append(f"Plan: {selected.get('thought')}")
    elif thoughts:
        best = max(thoughts, key=lambda thought: thought.get("score") or 0.0)
        trace.append(f"Best candidate: {best.get('thought')}")

    if result.get("requires_human_approval"):
        trace.append("Human approval gate: required")
    if result.get("human_decision"):
        trace.append(f"Human approval decision: {result.get('human_decision')}")

    subagent_results = result.get("subagent_results") or []
    if subagent_results:
        latest = subagent_results[-1]
        trace.append(f"Dispatched to: {latest.get('agent')}")
        trace.append(f"Tool execution: {'succeeded' if latest.get('success') else 'failed'}")
    if result.get("qa_score") is not None:
        trace.append(f"QA score: {result.get('qa_score')}")
    if result.get("qa_feedback"):
        trace.append(f"QA feedback: {result.get('qa_feedback')}")
    if result.get("escalation_count"):
        trace.append(f"Escalations: {result.get('escalation_count')}")
    return trace


def _derive_conditional_execute_command(messages: list[dict]) -> str | None:
    """Build a one-shot execution command from prior user turns in a conditional-weather thread."""
    user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
    if not user_turns:
        return None

    joined = " ".join(user_turns)
    lower = joined.lower()
    if not ("weather" in lower and ("picnic" in lower or "schedule" in lower)):
        return None

    # Location from explicit preposition first.
    location = None
    loc_match = re.search(r"\b(?:in|at|near|around)\s+([A-Za-z][A-Za-z\s]{1,40})", joined)
    if loc_match:
        candidate = loc_match.group(1).strip(" .,!?:;\"'")
        low_candidate = candidate.lower()
        invalid = (
            low_candidate.startswith("least")
            or low_candidate.startswith("next")
            or low_candidate.startswith("suitable")
            or re.match(r"^\d", low_candidate) is not None
        )
        if not invalid:
            location = candidate

    # Fallback: short user turn like "Seattle" / "Seattle Washington".
    if not location:
        intent_words = {
            "proceed", "cancel", "yes", "no", "tomorrow", "today", "morning", "afternoon", "evening",
            "suitable weather", "weather", "nice", "sunny",
        }
        for text in reversed(user_turns):
            t = text.strip().strip('"\'“”‘’.,!? ')
            low = t.lower()
            if low in intent_words:
                continue
            if 1 <= len(t.split()) <= 3 and all(ch.isalpha() or ch.isspace() or ch in "-." for ch in t):
                location = t
                break

    threshold = "sunny, at least 70F, and no rain"
    if "above" in lower and "f" in lower:
        threshold = "sunny, above 70F, and no rain"

    fallback = "reschedule to the next suitable day at 3 PM"
    if "cancel" in lower:
        fallback = "cancel the picnic"
    elif "proceed anyway" in lower:
        fallback = "proceed anyway"

    date_hint = "tomorrow"
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        if day in lower:
            date_hint = day
            break

    time_hint = "3 PM"
    time_match = re.search(r"\b(1[0-2]|0?[1-9])\s*(am|pm)\b", lower)
    if time_match:
        time_hint = f"{int(time_match.group(1))} {time_match.group(2).upper()}"
    elif "evening" in lower:
        time_hint = "6 PM"
    elif "afternoon" in lower:
        time_hint = "3 PM"
    elif "morning" in lower:
        time_hint = "9 AM"

    if not location:
        return None

    return (
        f"Check weather in {location} and if {threshold}, schedule a picnic {date_hint} at {time_hint}; "
        f"otherwise {fallback}."
    )


def _next_weekday(base: datetime, weekday_name: str) -> datetime:
    weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    target = weekday_map.get(weekday_name.lower(), base.weekday())
    days_ahead = (target - base.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return base + timedelta(days=days_ahead)


def _extract_schedule_when(command: str) -> tuple[str, str]:
    lower = command.lower()
    when = "tomorrow"
    for token in ["today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        if token in lower:
            when = token
            break

    time_label = "3 PM"
    match = re.search(r"\b(1[0-2]|0?[1-9])\s*(am|pm)\b", lower)
    if match:
        time_label = f"{int(match.group(1))} {match.group(2).upper()}"
    elif "evening" in lower:
        time_label = "6 PM"
    elif "morning" in lower:
        time_label = "9 AM"
    return when, time_label


def _parse_time_label(time_label: str) -> tuple[int, int]:
    m = re.match(r"^\s*(1[0-2]|0?[1-9])\s*(AM|PM)\s*$", time_label.upper())
    if not m:
        return 15, 0
    hour = int(m.group(1))
    ampm = m.group(2)
    if ampm == "PM" and hour != 12:
        hour += 12
    if ampm == "AM" and hour == 12:
        hour = 0
    return hour, 0


def _extract_location_from_command(command: str) -> str | None:
    match = re.search(r"check weather in\s+(.+?)\s+and\s+if", command, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip(" .,!?:;\"'")


def _execute_conditional_plan(command: str) -> str:
    """Execute conditional weather scheduling in one step from a normalized command."""
    from src.tools import memory
    from src.tools.external import get_current_weather, reset_api_call_counters

    location = _extract_location_from_command(command)
    if not location:
        return "I could not determine the location to check weather. Please provide the city and try Proceed again."

    when, time_label = _extract_schedule_when(command)
    hour, minute = _parse_time_label(time_label)

    # Keep this execution isolated to avoid stale test counters causing false failures.
    reset_api_call_counters()
    weather = None
    location_candidates = [location]
    parts = location.split()
    if len(parts) >= 2:
        # Common user pattern: "Seattle Washington" -> try "Seattle, Washington"
        location_candidates.append(f"{parts[0]}, {' '.join(parts[1:])}")
        # Final fallback: city token only
        location_candidates.append(parts[0])

    last_exc = None
    for candidate in location_candidates:
        try:
            weather = get_current_weather(candidate)
            break
        except Exception as exc:
            last_exc = exc
            continue

    if weather is None:
        raise ValueError(f"Could not resolve location for weather lookup: {location}") from last_exc

    temp_c = weather.get("temperature_c")
    temp_f = (temp_c * 9 / 5 + 32) if isinstance(temp_c, (int, float)) else None
    precip = weather.get("precipitation_mm") or 0
    condition = (weather.get("condition") or "").lower()

    sunny_like = any(token in condition for token in ["clear", "mainly clear", "partly cloudy", "sunny"])
    warm_enough = temp_f is not None and temp_f >= 70
    dry_enough = precip == 0
    suitable = sunny_like and warm_enough and dry_enough

    now = datetime.now(USER_TIMEZONE)
    if when == "today":
        target_date = now
    elif when == "tomorrow":
        target_date = now + timedelta(days=1)
    elif when in {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}:
        target_date = _next_weekday(now, when)
    else:
        target_date = now + timedelta(days=1)
    target_due = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if suitable:
        item = memory.add_todo(
            description="Picnic (weather condition met)",
            due=target_due.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        return (
            f"Weather in {weather.get('location')}: {weather.get('condition')}, "
            f"{temp_f:.1f}F, precipitation {precip} mm. Conditions are suitable. "
            f"Scheduled picnic for {target_due.strftime('%Y-%m-%d %I:%M %p')}."
        )

    # Fallback path: schedule next day at the same time as a practical interpretation
    # of "next suitable day" in this build.
    fallback_due = (target_due + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    memory.add_todo(
        description="Picnic (fallback schedule: weather not suitable)",
        due=fallback_due.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    temp_text = f"{temp_f:.1f}F" if temp_f is not None else "unknown temp"
    return (
        f"Weather in {weather.get('location')}: {weather.get('condition')}, {temp_text}, precipitation {precip} mm. "
        f"Conditions are not suitable now. I scheduled the fallback picnic for {fallback_due.strftime('%Y-%m-%d %I:%M %p')}."
    )


def _append_chat_message(role: str, content: str, trace: list[str] | None = None) -> None:
    """Append a chat message with a wall-clock timestamp for display (e.g. '10:24 AM')."""
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(USER_TIMEZONE).strftime("%I:%M %p").lstrip("0"),
    }
    if trace:
        message["trace"] = trace
    st.session_state.chat_messages.append(message)


def run_user_turn(user_text: str) -> None:
    global _log_container, _current_logs, _log_container_active
    # Ensure session state is initialized
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    
    # Reset logs and agents for new turn
    st.session_state.current_turn_agents = []
    _current_logs = []  # Clear logs for this new turn
    _log_container = None
    _log_container_active = False

    entered = user_text.strip()

    # One-click/one-word proceed path: execute conditional plan directly to avoid
    # sending another natural-language turn back into the clarification loop.
    if entered.lower() == "proceed" and st.session_state.get("conditional_action_ready"):
        execute_command = st.session_state.get("conditional_execute_command")
        if not execute_command:
            execute_command = _derive_conditional_execute_command(st.session_state.get("chat_messages", []))

        if not execute_command:
            final_response = (
                "I couldn't reconstruct the conditional command from this thread. "
                "Please provide the full weather-and-scheduling command once, and I will execute it directly."
            )
            _append_chat_message("user", entered)
            _append_chat_message(
                "assistant",
                final_response,
                trace=["Conditional execution path", "Could not derive command from thread state"],
            )
            st.session_state.conditional_action_ready = False
            st.session_state.conditional_execute_command = None
            st.session_state.last_result = {
                "handled_directly": "conditional_execute",
                "final_response": final_response,
                "qa_score": None,
                "escalation_count": 0,
            }
            log_event("turn_end", thread_id=st.session_state.thread_id, final_response=final_response)
            return

        _append_chat_message("user", entered)
        try:
            final_response = _execute_conditional_plan(execute_command)
        except Exception as exc:
            final_response = (
                f"I could not execute the conditional plan automatically ({exc.__class__.__name__}: {exc}). "
                "Please try again or provide the full command in one message."
            )

        st.session_state.conditional_action_ready = False
        st.session_state.conditional_execute_command = None
        st.session_state.last_result = {
            "handled_directly": "conditional_execute",
            "final_response": final_response,
            "qa_score": None,
            "escalation_count": 0,
        }
        _append_chat_message(
            "assistant",
            final_response,
            trace=["Conditional execution path", "Triggered by Proceed button or proceed text"],
        )
        log_event("turn_end", thread_id=st.session_state.thread_id, final_response=final_response)
        return

    user_text = entered
    _append_chat_message("user", user_text)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.status("🧠 Thinking through the request...", expanded=True) as status:
        _log_container = status
        _log_container_active = True
        setup_logging()

        try:
            _log_container.write("📖 Reading conversation context...")
            _log_container.write("🤔 Planning the best approach...")
            log_event("turn_start", thread_id=st.session_state.thread_id, user_text=user_text)
            _log_container.write("🚀 Running orchestrator and agents...")
            try:
                # Use build_turn_payload (not a bare new_state()) so cross-turn guardrail
                # counters (escalation_count, consecutive_ambiguous_turns) actually persist,
                # and so a previous turn's human_decision/requires_human_approval can never
                # leak into an unrelated new turn on the same Streamlit session thread.
                # invoke_with_timeout() (not a bare workflow.invoke()) enforces the same
                # hard step/timeout guardrail as the CLI harness, so a slow/stuck LLM or
                # tool call fails gracefully here instead of hanging indefinitely.
                result = invoke_with_timeout(build_turn_payload(user_text, config), config)
            except (APIConnectionError, OpenAIError) as exc:
                status.update(label="❌ Provider connection failed", state="error")
                # Save logs before returning
                st.session_state.last_reasoning_logs = _current_logs.copy()
                handle_graph_result(
                    {
                        "final_response": provider_error_message(exc),
                        "qa_score": 0.0,
                        "qa_feedback": "LLM provider connection failed.",
                        "escalation_count": 1,
                    }
                )
                return
            except TimeoutError as exc:
                status.update(label="⏱️ Request timed out", state="error")
                st.session_state.last_reasoning_logs = _current_logs.copy()
                handle_graph_result(
                    {
                        "final_response": (
                            "This request took too long and was stopped by the timeout "
                            f"guardrail ({STEP_TIMEOUT_SECONDS * MAX_SYSTEM_STEPS}s). "
                            "Please try again, or try a simpler/more specific request."
                        ),
                        "qa_score": 0.0,
                        "qa_feedback": str(exc),
                        "escalation_count": 1,
                    }
                )
                return
            if "__interrupt__" in result:
                _log_container.write("⏸️ Waiting for human approval...")
                status.update(label="✋ Approval needed", state="complete")
            else:
                _log_container.write("✅ Response ready!")
                status.update(label="✅ Answer ready", state="complete")
            # Save reasoning logs to session state so they persist
            st.session_state.last_reasoning_logs = _current_logs.copy()
            handle_graph_result(result)
        finally:
            _log_container_active = False
            _log_container = None


def resume_after_approval(decision: str) -> bool:
    """Resume workflow after human approval. Returns True if successful, False if error occurred."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    try:
        with st.status("Resuming after human approval...", expanded=True) as status:
            st.write(f"Recorded decision: {decision}.")
            log_event("human_approval_decision", thread_id=st.session_state.thread_id, decision=decision)
            st.write("Continuing the graph from the approval checkpoint.")
            try:
                result = invoke_with_timeout(Command(resume=decision), config)
                st.write("Running dispatch, QA scoring, and final response composition.")
                status.update(label="Approval handled", state="complete")
                st.session_state.pending_interrupt = None
                handle_graph_result(result)
                return True
            except (APIConnectionError, OpenAIError) as exc:
                status.update(label="Provider connection failed", state="error")
                st.session_state.pending_interrupt = None
                error_msg = provider_error_message(exc)
                st.error(f"❌ Provider Error: {error_msg}")
                log_event("resume_provider_error", thread_id=st.session_state.thread_id, error=str(exc))
                handle_graph_result(
                    {
                        "final_response": error_msg,
                        "qa_score": 0.0,
                        "qa_feedback": "LLM provider connection failed after approval.",
                        "human_decision": decision,
                        "escalation_count": 1,
                    }
                )
                return False
            except TimeoutError as exc:
                status.update(label="⏱️ Request timed out", state="error")
                st.session_state.pending_interrupt = None
                error_msg = (
                    "This request took too long and was stopped by the timeout "
                    f"guardrail ({STEP_TIMEOUT_SECONDS * MAX_SYSTEM_STEPS}s)."
                )
                st.error(f"⏱️ {error_msg}")
                log_event("resume_timeout", thread_id=st.session_state.thread_id, error=str(exc))
                handle_graph_result(
                    {
                        "final_response": error_msg,
                        "qa_score": 0.0,
                        "qa_feedback": str(exc),
                        "human_decision": decision,
                        "escalation_count": 1,
                    }
                )
                return False
            except Exception as exc:
                # Catch any other unexpected errors
                import traceback
                status.update(label="Error resuming workflow", state="error")
                st.session_state.pending_interrupt = None
                error_msg = f"{exc.__class__.__name__}: {str(exc)}"
                st.error(f"❌ Unexpected Error: {error_msg}")
                error_traceback = traceback.format_exc()
                st.error("Traceback:")
                st.code(error_traceback, language="python")
                log_event("resume_error", thread_id=st.session_state.thread_id, error=error_msg, traceback=error_traceback)
                handle_graph_result(
                    {
                        "final_response": f"An error occurred while processing your approval decision. {error_msg} Please try again or contact support.",
                        "qa_score": 0.0,
                        "qa_feedback": f"Error during approval resume: {str(exc)}",
                        "human_decision": decision,
                        "escalation_count": 1,
                    }
                )
                return False
    except Exception as outer_exc:
        # Outermost catch-all to prevent blank screen
        import traceback
        st.error(f"❌ Critical Error in approval handler: {outer_exc.__class__.__name__}: {str(outer_exc)}")
        st.error("Full traceback:")
        st.code(traceback.format_exc(), language="python")
        st.session_state.pending_interrupt = None
        log_event("resume_critical_error", thread_id=st.session_state.thread_id, error=str(outer_exc))
        return False


def handle_graph_result(result: dict) -> None:
    # Ensure session state is initialized
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "pending_interrupt" not in st.session_state:
        st.session_state.pending_interrupt = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "current_turn_agents" not in st.session_state:
        st.session_state.current_turn_agents = []
    
    st.session_state.last_result = result
    
    # Capture agents called in this turn
    subagent_results = result.get("subagent_results") or []
    if subagent_results:
        st.session_state.current_turn_agents = subagent_results
    
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        st.session_state.pending_interrupt = payload
        log_event("human_approval_requested", thread_id=st.session_state.thread_id, payload=payload)
        return

    response = result.get("final_response") or "I couldn't produce a final response."
    response_lower = response.lower()
    st.session_state.conditional_action_ready = (
        "reply 'proceed' to execute this now" in response_lower
        or "reply 'proceed' to continue with this plan" in response_lower
    )
    if st.session_state.conditional_action_ready:
        st.session_state.conditional_execute_command = _derive_conditional_execute_command(st.session_state.chat_messages)
    else:
        st.session_state.conditional_execute_command = None
    _append_chat_message("assistant", response, trace=build_reasoning_trace(result))
    log_event(
        "turn_end",
        thread_id=st.session_state.thread_id,
        qa_score=result.get("qa_score"),
        escalation_count=result.get("escalation_count"),
        final_response=response,
    )


def reset_conversation() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_messages = []
    st.session_state.pending_interrupt = None
    st.session_state.last_result = None
    st.session_state.current_turn_agents = []
    st.session_state.conditional_action_ready = False
    st.session_state.conditional_execute_command = None


def fill_context_window() -> None:
    examples = [
        "Remember that my favorite color is teal.",
        "Add a to-do to buy milk tomorrow.",
        "What's the current weather in Pittsburgh?",
        "Give me that temperature in Fahrenheit.",
    ]
    for text in examples:
        _append_chat_message("user", text)
        _append_chat_message("assistant", "Demo context message.")


def wipe_long_term_memory() -> None:
    from src.tools import memory

    persist_directory = Path(SETTINGS.get("vector_db", {}).get("persist_directory", "data/chromadb"))
    for collection_name in SETTINGS.get("vector_db", {}).get("collections", {}).values():
        try:
            memory._client.delete_collection(collection_name)  # noqa: SLF001 - maintenance UI action
        except Exception:
            pass
    st.toast(f"Cleared Chroma collections in {persist_directory}")


def load_trace_events(limit: int = 60) -> list[dict]:
    if not LOG_FILE.exists():
        return []
    events = []
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def render_chat() -> None:
    # Handle action controls first so any state changes are reflected when
    # rendering the chat below in the same script run.
    if st.session_state.pending_interrupt:
        pending_action = st.session_state.pending_interrupt.get("pending_action", "the planned action")
        st.warning(f"Approval needed for: {pending_action}")
        approve_col, reject_col = st.columns(2)
        if approve_col.button("Approve", type="primary", use_container_width=True):
            success = resume_after_approval("approved")
            if success:
                st.rerun()
        if reject_col.button("Reject", use_container_width=True):
            success = resume_after_approval("rejected")
            if success:
                st.rerun()

    if st.session_state.get("conditional_action_ready") and not st.session_state.pending_interrupt:
        st.info("Conditional plan is ready. You can continue with one click.")
        proceed_col, cancel_col = st.columns(2)
        if proceed_col.button("Proceed", type="primary", use_container_width=True):
            run_user_turn("proceed")
        if cancel_col.button("Cancel", use_container_width=True):
            cancel_msg = "Okay, cancelled this conditional request."
            _append_chat_message("assistant", cancel_msg, trace=["Conditional flow cancelled from UI button"])
            st.session_state.conditional_action_ready = False
            st.session_state.conditional_execute_command = None
            log_event("conditional_cancel", thread_id=st.session_state.thread_id)

    chat_box = st.container(height=435, border=False)
    with chat_box:
        if not st.session_state.chat_messages:
            st.markdown('<p class="empty-chat">Start by asking about a task, note, event, or weather.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chat-date-divider"><span>Today</span></div>', unsafe_allow_html=True)
        for message in st.session_state.chat_messages:
            role = message["role"]
            bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-assistant"
            timestamp = message.get("timestamp", "")
            with st.chat_message(role):
                st.markdown(
                    f'<div class="chat-bubble {bubble_class}">{message["content"]}</div>'
                    + (f'<div class="chat-timestamp">{timestamp}</div>' if timestamp else ""),
                    unsafe_allow_html=True,
                )
                trace = message.get("trace") or []
                if trace:
                    with st.expander("Reasoning trace", expanded=False):
                        for item in trace:
                            st.markdown(f"- {item}")

        # Display reasoning logs from the last response (if any)
        if st.session_state.last_reasoning_logs and st.session_state.chat_messages:
            last_msg = st.session_state.chat_messages[-1]
            if last_msg["role"] == "assistant":
                with st.expander("🧠 Tree-of-Thought Reasoning (All Steps)", expanded=False):
                    for log in st.session_state.last_reasoning_logs:
                        st.markdown(f"`{log}`")

    with st.form("chat_form", clear_on_submit=True):
        user_text = st.text_area(
            "Message",
            placeholder="Type your message...",
            height=68,
            label_visibility="collapsed",
        )
        spacer_col, send_col = st.columns([8.3, 1.4])
        submitted = send_col.form_submit_button(
            "Send", type="primary", use_container_width=True, icon=":material/send:"
        )
    if submitted and user_text.strip():
        run_user_turn(user_text.strip())
        st.rerun()


def render_context_tab() -> None:
    messages = st.session_state.chat_messages[-MAX_CONTEXT_MESSAGES:]
    title_col, info_col = st.columns([5, 0.4])
    with title_col:
        st.markdown('<div class="rail-card-title">Short-term memory</div>', unsafe_allow_html=True)
    with info_col:
        st.markdown(
            ':material/info:',
            help="The rolling window of recent turns sent to the model as conversation context.",
        )
    st.caption(f"{len(messages)} / {MAX_CONTEXT_MESSAGES} messages")
    st.progress(min(len(messages) / MAX_CONTEXT_MESSAGES, 1.0) if MAX_CONTEXT_MESSAGES else 0.0)

    if not messages:
        st.markdown(
            """
            <div class="rail-empty">
                <div class="rail-empty-icon">\U0001F4AC</div>
                <div style="font-weight:600; color:#3d4658;">Nothing said yet</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.write("")
    for index, message in enumerate(messages, start=1):
        role = "User" if message["role"] == "user" else "Assistant"
        st.markdown(f"**{index}. {role}:** {message['content']}")


def render_memory_tab() -> None:
    from src.tools import memory

    st.markdown('<div class="rail-card-title">Long-term memory</div>', unsafe_allow_html=True)
    st.write("")
    if st.button("Load recent saved notes", use_container_width=True):
        with st.spinner("Loading Chroma memory..."):
            st.session_state.memory_notes = memory.list_notes(limit=10)

    with st.expander("Recent saved notes", expanded=True):
        notes = st.session_state.memory_notes
        if notes:
            for note in notes:
                st.info(note)
        else:
            st.caption("Click Load recent saved notes to inspect Chroma memory.")

    query = st.text_input("Search saved notes", placeholder="favorite color, project deadline, meeting preference...")
    if query:
        matches = memory.search_notes(query, k=5)
        if matches:
            st.markdown("**Semantic matches**")
            for match in matches:
                st.info(match)
        else:
            st.caption("No matching notes found.")
    st.caption("Notes are embedded with the configured sentence-transformer model and stored in ChromaDB.")


def render_tools_tab() -> None:
    st.markdown('<div class="rail-card-title">Scoped tools</div>', unsafe_allow_html=True)
    st.write("")
    for agent_name, tools in TOOL_SCOPES.items():
        st.markdown(f"**{agent_name}**")
        st.write(", ".join(tools))


def render_trace_tab() -> None:
    st.markdown('<div class="rail-card-title">Trace</div>', unsafe_allow_html=True)
    st.write("")
    events = [event for event in load_trace_events() if event.get("thread_id") == st.session_state.thread_id]
    if not events:
        st.markdown(
            """
            <div class="rail-empty">
                <div class="rail-empty-icon">\U0001F4C8</div>
                <div style="font-weight:600; color:#3d4658;">No trace events yet</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    for event in events[-20:]:
        st.json(event, expanded=False)


def render_subagents_tab() -> None:
    st.markdown('<div class="rail-card-title">Subagents</div>', unsafe_allow_html=True)
    st.write("")
    specs = {
        "Scheduler & To-Do": "Creates, lists, and completes to-dos, reminders, and scheduled items.",
        "Notes & Knowledge": "Saves notes and retrieves grounded context from ChromaDB memory.",
        "Weather & External Info": "Fetches current weather from external APIs and summarizes it.",
    }
    for name, description in specs.items():
        st.markdown(f"**{name}**")
        st.caption(description)
    result = st.session_state.last_result or {}
    subagent_results = result.get("subagent_results") or []
    if subagent_results:
        st.divider()
        st.markdown("**Latest dispatch**")
        for item in subagent_results[-3:]:
            st.json(item, expanded=False)


def render_agents_called_tab() -> None:
    st.markdown('<div class="rail-card-title">Agents Called</div>', unsafe_allow_html=True)
    st.write("")

    # Helper function to remove task IDs from output
    def sanitize_output(output: str) -> str:
        """Remove task IDs and UUID patterns from agent output for cleaner display."""
        import re
        # Remove "Task ID: <uuid>" patterns
        sanitized = re.sub(r'[Tt]ask\s+[Ii][Dd]:\s*[a-f0-9\-]{36}', '', output)
        # Remove remaining UUID patterns that might appear standalone
        sanitized = re.sub(r'\b[a-f0-9\-]{36}\b', '', sanitized)
        # Clean up extra spaces
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        return sanitized
    
    # Show only agents from current turn (tracked in session state)
    current_turn_agents = st.session_state.get("current_turn_agents", [])
    last_result = st.session_state.get("last_result", {})
    
    if not current_turn_agents:
        st.markdown(
            """
            <div class="rail-empty">
                <div class="rail-empty-icon">\U0001F4DE</div>
                <div style="font-weight:600; color:#3d4658;">No agents dispatched yet</div>
                <div style="font-size:0.8rem;">Try asking a question to see the dispatch flow.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    
    # Get the user's current prompt from chat history
    chat_messages = st.session_state.get("chat_messages", [])
    current_prompt = None
    if chat_messages:
        # Find the last user message
        for message in reversed(chat_messages):
            if message.get("role") == "user":
                current_prompt = message.get("content")
                break
    
    # Display current prompt
    if current_prompt:
        st.markdown("**Your Request:**")
        with st.container(border=True):
            st.markdown(f'> "{current_prompt}"')
        st.markdown("")
    
    # Agent Check Descriptions
    AGENT_CHECKS = {
        "scheduler_todo": {
            "icon": "📅",
            "role": "Scheduler & To-Do Agent",
            "checks": [
                "✓ Validates task/event details (date, time, description)",
                "✓ Checks for scheduling conflicts",
                "✓ Detects preference violations (e.g., 'relax on Sundays')",
                "✓ Asks clarifying questions for ambiguous requests",
                "✓ Requires human approval for destructive actions",
            ]
        },
        "notes_knowledge": {
            "icon": "🧠",
            "role": "Notes & Knowledge Agent",
            "checks": [
                "✓ Searches saved notes and preferences",
                "✓ Retrieves grounded context from memory (ChromaDB)",
                "✓ Applies semantic search for relevant information",
                "✓ Cites sources when answering from memory",
                "✓ Saves new notes for future reference",
            ]
        },
        "weather_external": {
            "icon": "🌤️",
            "role": "Weather & External Info Agent",
            "checks": [
                "✓ Fetches current weather for specified location",
                "✓ Validates location (clarifies if ambiguous)",
                "✓ Provides: temperature, humidity, wind, precipitation",
                "✓ Reports source and time of data",
                "✓ Integrates weather into scheduling decisions",
            ]
        }
    }
    
    # Display agent flow
    st.markdown("**Agent Dispatch Flow:**")
    st.markdown("")
    
    agent_count = len(current_turn_agents)
    
    for idx, agent_result in enumerate(current_turn_agents, start=1):
        agent_name = agent_result.get("agent", "unknown")
        success = agent_result.get("success", False)
        status_icon = "✅" if success else "❌"
        
        # Get agent info from mapping
        agent_info = AGENT_CHECKS.get(agent_name, {})
        agent_icon = agent_info.get("icon", "🤖")
        agent_role = agent_info.get("role", agent_name)
        agent_checks = agent_info.get("checks", [])
        
        # Display agent with expandable details
        col1, col2 = st.columns([0.15, 0.85])
        with col1:
            st.markdown(f"### {agent_icon}")
        with col2:
            st.markdown(f"**Step {idx}: {status_icon} {agent_role}**")
        
        # Expandable "What it checks" section
        with st.expander(f"What {agent_name.split('_')[0].title()} checks", expanded=(idx == 1)):
            for check in agent_checks:
                st.caption(check)
        
        # Show output in collapsible section (with task IDs hidden)
        output = agent_result.get("output", "")
        error = agent_result.get("error")
        
        if output:
            # Sanitize output to remove task IDs and UUIDs
            clean_output = sanitize_output(str(output))
            with st.expander("Output", expanded=False):
                st.text(clean_output if len(clean_output) < 500 else clean_output[:500] + "...")
        
        if error and not success:
            st.error(f"Error: {error}")
        
        # Add separator between agents (except after last one)
        if idx < agent_count:
            st.markdown("↓")
            st.markdown("")
    
    # Summary
    st.divider()
    st.markdown("**Flow Summary:**")
    agent_names = [r.get("agent", "unknown") for r in current_turn_agents]
    flow_str = " → ".join([AGENT_CHECKS.get(name, {}).get("icon", "🤖") + " " + name.split("_")[0].title() 
                           for name in agent_names])
    st.markdown(flow_str)
    st.caption(f"{agent_count} agent(s) dispatched in sequence for this prompt")
    
    # Show reasoning mode if available
    reasoning_mode = last_result.get("reasoning_mode")
    if reasoning_mode:
        st.caption(f"Reasoning mode: {reasoning_mode.replace('_', ' ').title()}")



def render_sidebar_controls() -> None:
    with st.container(border=True, key="quick_actions_strip"):
        st.markdown(
            '<div class="quick-actions-title">Quick Actions</div>',
            unsafe_allow_html=True,
        )
        reset_col, wipe_col, fill_col = st.columns(3)
        with reset_col:
            if st.button(
                "Reset conversation", use_container_width=True, icon=":material/refresh:"
            ):
                reset_conversation()
                st.rerun()
        with wipe_col:
            if st.button(
                "Wipe long-term memory", use_container_width=True, icon=":material/delete:"
            ):
                wipe_long_term_memory()
        with fill_col:
            if st.button(
                "Fill context window", use_container_width=True, icon=":material/database:"
            ):
                fill_context_window()
                st.rerun()


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --cmu-red: #C41230;
            --cmu-red-dark: #9E0E26;
            --cmu-red-tint: #FBEAEC;
            --ink: #1F2430;
            --ink-soft: #5B6270;
            --line: #ECEAE6;
            --panel: #F7F5F1;
        }

        .stApp { background: #FBFAF8; color: var(--ink); }
        .block-container { max-width: 1420px; padding-top: 0.6rem; padding-bottom: 1rem; }
        [data-testid="stHeader"] { background: transparent; }

        /* ---------- Top app bar ---------- */
        .app-topbar {
            display: flex; align-items: center; justify-content: space-between;
            padding: 0.65rem 0.25rem 0.9rem 0.25rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1rem;
        }
        .app-topbar-left { display: flex; align-items: center; gap: 0.75rem; }
        .app-logo-mark {
            width: 40px; height: 40px; border-radius: 10px;
            background: var(--cmu-red);
            display: flex; align-items: center; justify-content: center;
            color: #fff; font-weight: 800; font-size: 1.15rem; font-family: Georgia, serif;
            flex-shrink: 0;
        }
        .app-title { font-size: 1.28rem; font-weight: 700; color: var(--ink); line-height: 1.2; }
        .app-subtitle { font-size: 0.8rem; color: var(--ink-soft); margin-top: 1px; }
        .model-pill {
            display: inline-flex; align-items: center; gap: 0.35rem;
            padding: 0.45rem 0.85rem; border: 1px solid var(--line);
            border-radius: 8px; background: #fff; font-size: 0.85rem; color: var(--ink);
            font-weight: 600; white-space: nowrap; width: 100%; box-sizing: border-box;
        }
        /* Settings popover trigger: square icon button, aligned with the model pill */
        div[data-testid="stPopover"] button[data-testid="stPopoverButton"] {
            border-radius: 8px; padding: 0.45rem 0.6rem;
        }

        /* ---------- Chat surface ---------- */
        .chat-date-divider { text-align: center; margin: 0.4rem 0 1rem 0; }
        .chat-date-divider span {
            display: inline-block; padding: 0.25rem 0.9rem; border-radius: 999px;
            background: var(--panel); color: var(--ink-soft); font-size: 0.78rem; font-weight: 600;
        }
        .empty-chat { color: #9aa0ad; font-style: italic; padding: 1.4rem; text-align: center; }

        div[data-testid="stChatMessage"] {
            background: transparent; border: none; padding: 0.15rem 0; box-shadow: none;
        }
        /* User bubble = light CMU-red tint; assistant bubble = soft grey */
        div[data-testid="stChatMessageAvatarUser"] { background: var(--cmu-red) !important; }
        div[data-testid="stChatMessageAvatarAssistant"] { background: var(--ink) !important; }
        .chat-bubble {
            display: inline-block; padding: 0.75rem 1rem; border-radius: 12px;
            font-size: 0.93rem; line-height: 1.45; max-width: 100%;
        }
        .chat-bubble-user { background: var(--cmu-red-tint); color: var(--ink); }
        .chat-bubble-assistant { background: var(--panel); color: var(--ink); }
        .chat-timestamp { font-size: 0.72rem; color: #a9aeb9; margin-top: 0.25rem; padding-left: 0.1rem; }

        /* ---------- Composer ---------- */
        div[data-testid="stTextArea"] textarea {
            border-radius: 10px !important; border-color: var(--line) !important;
        }
        .stButton > button[kind="primary"] {
            background: var(--cmu-red); border-color: var(--cmu-red); border-radius: 10px;
            font-weight: 600;
        }
        .stButton > button[kind="primary"]:hover { background: var(--cmu-red-dark); border-color: var(--cmu-red-dark); }
        .stButton > button[kind="secondary"] { border-radius: 10px; }

        /* Memory-strategy segmented control pills */
        div[data-testid="stSegmentedControl"] button {
            border-radius: 999px !important;
        }

        /* ---------- Right rail tabs styled like an icon nav bar ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.6rem; border-bottom: 1px solid var(--line); flex-wrap: nowrap; overflow-x: auto;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.25rem 0.1rem 0.7rem 0.1rem; color: var(--ink-soft); font-size: 0.8rem;
        }
        .stTabs [aria-selected="true"] { color: var(--cmu-red) !important; font-weight: 700; }
        .stTabs [data-baseweb="tab-highlight"] { background-color: var(--cmu-red) !important; }
        .stTabs [data-baseweb="tab-border"] { background-color: var(--line) !important; }

        /* ---------- Right-rail cards ---------- */
        .rail-card {
            background: #fff; border: 1px solid var(--line); border-radius: 12px;
            padding: 1.1rem 1.25rem; margin-top: 0.75rem;
        }
        .rail-card-title { font-weight: 700; font-size: 0.95rem; color: var(--ink); }
        .rail-empty {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 2.5rem 1rem; color: var(--ink-soft); text-align: center; gap: 0.5rem;
        }
        .rail-empty .rail-empty-icon { font-size: 2rem; opacity: 0.4; }

        /* ---------- Quick actions strip ---------- */
        .st-key-quick_actions_strip > div {
            border: 1px solid var(--line) !important; background: #fff !important;
        }
        .quick-actions-title {
            color: var(--ink); font-weight: 700; font-size: 0.85rem;
            margin-bottom: 0.6rem;
        }

        /* ---------- Misc ---------- */
        div[data-testid="stExpander"] { border-radius: 10px; border-color: var(--line); }
        .stCaption, [data-testid="stCaptionContainer"] { color: var(--ink-soft) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_render(section_name: str, render_fn) -> None:
    """Render a UI section safely so one failing panel does not blank the page."""
    try:
        render_fn()
    except Exception as exc:
        st.error(f"{section_name} failed: {exc.__class__.__name__}: {exc}")
        st.code(traceback.format_exc(), language="python")


def render_topbar() -> None:
    left, right = st.columns([3, 1.4], vertical_alignment="center")
    with left:
        st.markdown(
            f"""
            <div class="app-topbar-left">
                <div class="app-logo-mark">A</div>
                <div>
                    <div class="app-title">Personal Assistant Multi-Agent System</div>
                    <div class="app-subtitle">Tasks · notes · weather · events — with transparent reasoning</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        model_col, settings_col = st.columns([3, 1], gap="small", vertical_alignment="center")
        with model_col:
            st.markdown(
                f'<div class="model-pill">\U0001F916 {LLM_MODEL}</div>',
                unsafe_allow_html=True,
            )
        with settings_col:
            with st.popover("", icon=":material/settings:"):
                st.caption(f"Thread: `{st.session_state.thread_id}`")
                st.caption(f"Provider: `{SETTINGS.get('llm', {}).get('provider', 'unknown')}`")
    st.markdown('<div style="border-bottom:1px solid var(--line); margin: 0.4rem 0 1rem 0;"></div>', unsafe_allow_html=True)


def main() -> None:
    try:
        st.set_page_config(page_title="Personal Assistant", layout="wide")
        init_session()
        apply_styles()

        # Warm up ChromaDB + the sentence-transformer embedding model on a
        # background thread as soon as the page loads. Without this, the
        # ~10s cold-start cost of loading the embedding model lands inline
        # inside the user's first scheduling/notes request (any add_todo
        # call triggers a preference-conflict notes search). Calling this
        # here means the user can start reading/typing immediately while it
        # loads in the background; runs at most once per process.
        warm_up_async()

        _safe_render("Top bar", render_topbar)

        left, right = st.columns([1.35, 1], gap="large")
        with left:
            with st.container(border=True):
                _safe_render("Chat panel", render_chat)
            _safe_render("Sidebar controls", render_sidebar_controls)
        with right:
            with st.container(border=True):
                context_tab, memory_tab, tools_tab, trace_tab, subagents_tab, agents_tab = st.tabs(
                    [
                        ":material/description: Context",
                        ":material/database: Memory",
                        ":material/build: Tools",
                        ":material/trending_up: Trace",
                        ":material/groups: Subagents",
                        ":material/call: Agents Called",
                    ]
                )
                with context_tab:
                    _safe_render("Context tab", render_context_tab)
                with memory_tab:
                    _safe_render("Memory tab", render_memory_tab)
                with tools_tab:
                    _safe_render("Tools tab", render_tools_tab)
                with trace_tab:
                    _safe_render("Trace tab", render_trace_tab)
                with subagents_tab:
                    _safe_render("Subagents tab", render_subagents_tab)
                with agents_tab:
                    _safe_render("Agents Called tab", render_agents_called_tab)
    except Exception as exc:
        st.error(f"Application error: {exc.__class__.__name__}: {exc}")
        st.code(traceback.format_exc(), language="python")


if __name__ == "__main__":
    main()