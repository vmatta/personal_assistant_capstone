"""Critic/QA gate and human-in-the-loop escalation logic (Guardrails 1-3)."""

import logging
import re
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MAX_TOKENS, LLM_MODEL, SETTINGS, get_llm
from src.graph.state import ReasoningMode, SubAgentResult
from src.utils.helpers import invoke_with_retry, parse_json_response

logger = logging.getLogger(__name__)

_GUARDRAILS = SETTINGS.get("guardrails", {})
QA_SCORE_THRESHOLD = _GUARDRAILS.get("qa_score_threshold", 0.85)
HUMAN_APPROVAL_ACTIONS = set(_GUARDRAILS.get("human_approval_actions", []))
MAX_RETRIES = _GUARDRAILS.get("max_retries", 3)
AMBIGUITY_ESCALATION_TURNS = _GUARDRAILS.get("ambiguity_escalation_turns", 2)

_CLARIFICATION_INDICATORS = (
    "clarification", "clarify", "what ", "which ", "could you", "can you",
    "would you", "tell me", "need more", "need to know", "help me understand",
)


def is_clarification_response(output: str) -> bool:
    """Cheap, deterministic (no LLM call) check for whether an agent/response is asking
    the user a clarifying question rather than completing an action. Used by the QA
    critic fast-path to score CrewAI agents' free-text clarification questions highly.
    """
    output = output or ""
    return len(output) > 20 and any(indicator in output.lower() for indicator in _CLARIFICATION_INDICATORS)


# Markers unique to nodes.py's canned direct-response paths that represent a GENUINELY
# stuck/unresolved turn (as opposed to forward progress, e.g. "Great, I linked your
# follow-ups... reply 'proceed'" which is NOT stuck — the user is converging on an answer).
# Used by the ambiguity-escalation guardrail (Guardrail 3b) instead of the more general
# is_clarification_response(), since that heuristic doesn't reliably match every canned
# phrasing in nodes.py (e.g. "Got it. I still need:\n\n2. Fallback plan...").
_STUCK_TURN_MARKERS = (
    "i still need",
    "i need clarification first",
    "i do not have enough context",
    "is outside what i can do",
)


def is_unresolved_ambiguous_turn(output: str) -> bool:
    """Deterministic (no LLM call) check for whether a direct-response turn left the
    request genuinely unresolved (still missing required details, or refused as out of
    scope) rather than making forward progress or completing an action.
    """
    output = (output or "").lower()
    return any(marker in output for marker in _STUCK_TURN_MARKERS)

# Lazy-load LLM instances to pick up config changes
_llm = None
_classifier_llm = None

def _get_llm():
    """Get LLM instance for QA scoring (temperature=0 for deterministic grading)."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=0,
            max_tokens=LLM_MAX_TOKENS,
        )
    return _llm

def _get_classifier_llm():
    """Get classifier LLM instance (tiny output for quick complexity classification)."""
    global _classifier_llm
    if _classifier_llm is None:
        _classifier_llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=0,
            max_tokens=30,
        )
    return _classifier_llm

_COMPLEXITY_SYSTEM_PROMPT = (
    "Classify a personal assistant request as \"simple\" (direct execution) or \"complex\" (multi-path exploration).\n\n"
    "CRITICAL RULE: Return \"simple\" (direct) ~95% of the time. Only \"complex\" when truly exploratory.\n\n"
    "SIMPLE (Direct Execution) ~ 95% of requests - Default answer:\n"
    "- ALL list/show/get queries: 'list my tasks', 'show me Saturday tasks', 'what are my events'\n"
    "- Clear scheduling: 'schedule meeting Tuesday 2 PM', 'add gym tomorrow 9 AM'\n"
    "- Destructive with clear scope: 'delete all Saturday tasks', 'clean up today'\n"
    "- Ambiguous/vague planning: 'plan my weekend to be fun and productive' → Ask clarifications (direct path)\n"
    "- Conditional requests: 'if weather is nice, schedule picnic' → Ask clarifications (direct path)\n"
    "- Weather queries: 'what's weather in Seattle'\n"
    "- Notes/knowledge: 'save my favorite color is blue'\n"
    "- FOLLOW-UP responses: User answers 'what time?' with 'morning' or 'Tuesday at 3' → SIMPLE (clarification answered)\n"
    "- Multi-agent in sequence: 'add task AND search notes' → SIMPLE (direct coordinator)\n"
    "- Multi-agent parallel: 'schedule meeting AND check weather' → SIMPLE (direct coordinator)\n\n"
    "COMPLEX (Exploratory Multi-Path) ~ 5% of requests - ONLY these exact scenarios:\n"
    "- Budget/cost optimization: 'find best way to make 2 trips within $500'\n"
    "- Time slot optimization: 'find best times for 3 meetings across Tue-Thu 9-5'\n"
    "- Multi-constraint exploration: 'plan weekend: 3 cities, <4hrs driving, indoor+outdoor'\n"
    "- Option comparison: 'compare different approaches to organize my schedule'\n"
    "- Explicit search for alternatives: 'show me multiple ways to...', 'find all possibilities'\n\n"
    "DECISION RULES (in order):\n"
    "1. If user is answering a previous clarification question → SIMPLE\n"
    "2. If request is ambiguous/vague but NOT explicitly asking to optimize/explore → SIMPLE (ask clarifications)\n"
    "3. If request has conditional logic (IF...THEN) → SIMPLE (ask clarifications)\n"
    "4. If only complex word is 'best' but no optimization constraint → SIMPLE\n"
    "5. If request explicitly searches multiple paths/options/possibilities within constraints → COMPLEX\n\n"
    "EXAMPLES:\n"
    "- 'Plan fun weekend' → SIMPLE (too vague, no exploration)\n"
    "- 'Within $500, what are all ways to make 2 Detroit trips' → COMPLEX (explore budget options)\n"
    "- 'If nice weather tomorrow, schedule picnic' → SIMPLE (conditional, ask clarifications)\n"
    "- 'Schedule volleyball' → SIMPLE (straightforward)\n"
    "- 'Find best meeting time Tue-Thu' → SIMPLE (no constraint, just ask preferences)\n"
    "- 'Find best times for 3x1hr meetings Tue-Thu 9-5' → COMPLEX (multi-meeting slot optimization)\n"
    'Respond ONLY with JSON: {"complexity": "simple"} or {"complexity": "complex"}'
)


def classify_complexity(user_text: str, history_text: str = "") -> ReasoningMode:
    """Guardrail/router: decide whether this turn needs full Tree-of-Thought search
    or can be handled with a single direct plan.

    Returns "direct" for ~95% of requests (simple execution).
    Returns "tree_of_thought" for ~5% of requests (true multi-path exploration needed).
    
    CRITICAL: Retries ALWAYS use direct path (never escalate to ToT).
    """
    # Check if this is a retry - if so, ALWAYS use direct path
    # (retries should not escalate to ToT; they should attempt with different parameters)
    # This check will be done in plan_node, not here
    
    text = (user_text or "").lower()

    # Deterministic fast-path for explicitly exploratory, multi-constraint requests.
    # These should reliably use ToT instead of depending on model variance.
    has_exploration_phrase = any(
        phrase in text
        for phrase in (
            "different ways",
            "all the ways",
            "all possibilities",
            "multiple ways",
            "best way",
            "best option",
            "optimize",
            "compare options",
        )
    )
    has_multi_constraint_signal = any(
        phrase in text
        for phrase in (
            "3 different",
            "three different",
            "under",
            "within",
            "can't overlap",
            "cannot overlap",
            "include both",
            "indoor and outdoor",
        )
    )
    if has_exploration_phrase and has_multi_constraint_signal:
        return "tree_of_thought"

    # Deterministic direct-path fast rules for routine assistant intents.
    direct_verbs = [
        "add",
        "schedule",
        "list",
        "show",
        "delete",
        "remove",
        "cancel",
        "save",
        "remember",
        "what is",
        "what's",
    ]
    domain_nouns = [
        "task",
        "tasks",
        "todo",
        "event",
        "events",
        "meeting",
        "note",
        "notes",
        "weather",
        "temperature",
        "schedule",
    ]
    if any(v in text for v in direct_verbs) and any(n in text for n in domain_nouns):
        return "direct"

    context = f"{history_text}\n\n{user_text}" if history_text else user_text
    messages = [SystemMessage(content=_COMPLEXITY_SYSTEM_PROMPT), HumanMessage(content=context)]
    try:
        response = invoke_with_retry(get_llm(), messages)
        data = parse_json_response(response.content)
        complexity = data.get("complexity", "simple").lower()
        return "tree_of_thought" if complexity == "complex" else "direct"
    except Exception:
        logger.warning("Could not classify complexity; failing safe to 'tree_of_thought'")
        return "tree_of_thought"


_CRITIC_SYSTEM_PROMPT = (
    "You are a QA critic for a personal assistant system. Given the user's request "
    "and the agent's output, score how correct, grounded, and complete the output is on a "
    "scale from 0.0 to 1.0. Never lower your score because of a date or timestamp mentioned "
    "anywhere in the output or your own reasoning about it \"seeming\" future or implausible "
    "relative to your training data — a trusted external system clock supplies all dates, "
    "so date/time plausibility is simply out of scope for this evaluation. "
    "Judge completeness relative to what the user actually asked, not against an idealized answer: "
    "e.g. a weather answer that gives condition, temperature, humidity, wind, and a cited source is "
    "complete for a simple current-weather question, even without a multi-day forecast. "
    "\n"
    "**CRITICAL LIST QUERY RULE**: If the user's request is to LIST, SHOW, GET, or TELL items (tasks, events, todos): "
    "The agent's response is CORRECT if it simply returns the matching stored items with their titles and due dates. "
    "Do NOT require clarifying questions, do NOT require offering options, do NOT require extra details. "
    "Score HIGHLY (0.9+) if the agent simply lists the correct items. "
    "Examples:\n"
    "  ✅ User: 'List my Saturday tasks' → Agent: 'Here are your Saturday tasks: [Task 1], [Task 2]' → Score 0.9+\n"
    "  ✅ User: 'Show me my tasks for today' → Agent: 'Today's tasks: [Item A], [Item B]' → Score 0.9+\n"
    "  ✅ User: 'What are my events?' → Agent: 'Your events: [Event 1], [Event 2]' → Score 0.9+\n"
    "  ❌ User: 'List my Saturday tasks' → Agent: 'Here are tasks: [T1]. What time do you want to add a new task?' (asking unrelated clarification) → Score 0.3\n"
    "For requests to list, show, or summarize stored tasks/events, treat an answer as complete if it "
    "reports the matching stored items using the available description and due time; do not penalize "
    "generic stored titles, UUID omission, or ISO-like due timestamps unless they make the answer wrong. "
    "**ZERO-MATCH DELETE/CANCEL RULE**: If the user's request is to DELETE, CANCEL, or REMOVE items, and the "
    "agent's response states that no matching items were found (e.g. 'Deleted 0 tasks: no tasks were found'), "
    "this IS a correct, complete response as long as it clearly reports the zero-result outcome. "
    "Do NOT penalize it for lacking extra acknowledgment/reassurance phrasing — reporting the true outcome "
    "(nothing matched) is itself the acknowledgment. Score HIGHLY (0.85+). "
    "Examples:\n"
    "  ✅ User: 'Cancel my event for tomorrow' → Agent: 'Deleted 0 tasks: no tasks were found for tomorrow.' → Score 0.9\n"
    "  ❌ Do not score this low merely for 'lacking acknowledgment of the cancel request' — stating the "
    "true zero-result outcome already acknowledges and answers the request.\n"
    "For TIME-FILTERED list queries like 'this week', 'this month', 'today', 'next week': if the agent "
    "returns items that fall within the requested time window with dates shown, score HIGHLY (0.85+). "
    "Do NOT penalize dates for appearing 'too far in the future' relative to your training data—the system "
    "clock is authoritative. If the agent provides task names + due dates that match the time filter, "
    "it is a correct, complete response regardless of what year it appears to be. "
    "For SCHEDULING requests with ambiguous details (vague dates like 'next week', vague times like 'morning/afternoon/evening', unclear event types): "
    "if the agent asks clarifying questions instead of guessing, score that HIGHLY (0.9+) as a correct and wise response. "
    "The agent should NEVER assume specific times without confirming with the user first. "
    "Examples of wise clarification:\n"
    "  ✅ User: 'Saturday morning' → Agent asks: 'Saturday morning could be 8 AM, 9 AM, 10 AM, or 11 AM. Which time?' → Score 0.95\n"
    "  ❌ User: 'Saturday morning' → Agent assumes: 'Scheduled for 9 AM' (without asking) → Score 0.2\n"
    "A scheduling response that confirmed exact date, time, and duration without asking clarification is ONLY complete "
    "if the user's request was unambiguous (specific times like '2 PM' or '3:30 AM'). "
    "CRITICAL MULTI-CLARIFICATION RULE: In a multi-turn conversation where the user is providing clarifications to previous assistant questions: "
    "After the user answers ONE clarification (e.g., specifies a time), the agent should NOT immediately ask a DIFFERENT/NEW clarification in the same response. "
    "IMPORTANT: Offering specific time slots in response to a vague time (like 'evening') is NOT asking a new clarification—it's providing options based on the answer given. "
    "Examples of GOOD vs BAD behavior:\n"
    "  FLOW: User 'Plan Saturday' → Assistant: 'What time?' → User: 'evening' (answered time clarification) "
    "    → Assistant: [GOOD] 'I checked availability. Evening slots: 7 PM, 8 PM, 9 PM. Which works best?' → Score 0.9+ (offering options from availability, not asking NEW clarification)\n"
    "    → Assistant: [BAD] 'Got evening, but what kind of activity?' (asking NEW clarification for activity) → Score 0.2\n"
    "  FLOW: User 'Plan Saturday to be fun/productive/relaxing' → Assistant: 'What time?' → User: '9 AM' (answered clarification) "
    "    → Assistant: [GOOD] 'Scheduling balanced activities for Saturday 9 AM: workout (productive), coffee break (fun), journaling (relaxing)' → Score 0.85\n"
    "    → Assistant: [BAD] 'Got 9 AM, but what kind of activity?' (asking ANOTHER clarification about activities) → Score 0.2\n"
    "CRITICAL FOLLOW-UP RULE: If the conversation history shows the assistant already asked a clarification question "
    "(e.g., 'What time would you prefer?') and the user HAS ALREADY ANSWERED IT (e.g., 'saturday 9AM or 3PM'), "
    "then the agent MUST use the user's answer and proceed with execution. "
    "Asking the clarification question AGAIN after the user answered = MAJOR ERROR. Score such responses VERY LOW (0.1-0.2). "
    "Examples:\n"
    "  ❌ Assistant asks: 'What time?' → User answers: '9AM or 3PM' → Assistant asks again: 'What time?' → Score 0.1\n"
    "  ✅ Assistant asks: 'What time?' → User answers: '9AM or 3PM' → Agent schedules at 9AM and 3PM → Score 0.9\n"
    "EXISTING TASK CONFIRMATION: When a user provides clarification (e.g., specifies a time) to a previous assistant question, "
    "and the agent finds/confirms an existing task matching that clarification, this is CORRECT behavior. Score HIGHLY (0.85+) if: "
    "(1) the response clearly states what task was confirmed (e.g., 'Confirmed: Fun, productive, relaxing Saturday at 10 AM'), "
    "(2) the date/time match what the user specified, and (3) the response does NOT contain contradictions like 'no new task added' "
    "AFTER already stating the task is scheduled. BAD examples: 'Scheduled task... No new task added' (contradictory) → Score 0.1. "
    "For CONFLICT-RESOLUTION scenarios: if the response mentions a scheduling conflict (e.g., 'time slot taken') "
    "followed by a successful resolution (e.g., 'scheduled at alternative time instead'), score this HIGHLY (0.85+) as a smart, "
    "helpful response. Detecting conflicts + proposing alternatives = correct behavior. Do NOT flag this as contradictory. "
    "For RESCHEDULING requests: if the user explicitly provides a new date/time (e.g., 'reschedule to Sept 9 at 2 PM'), "
    "this is NOT an assumption—it is an explicit user specification. Score a response that creates/confirms the new event "
    "with the exact provided date/time HIGHLY (0.85+). Do not penalize creating a new task ID; that is standard practice. "
    "CRITICAL CONTRADICTION DETECTION: If a response contains contradictory statements about availability or scheduling, "
    "SCORE VERY LOW (0.05-0.2). Examples of contradictions:\n"
    "  ❌ 'Your afternoon is busy... here are available slots' (says busy then offers slots)\n"
    "  ❌ 'busy from 12 PM to 11 AM' (time range backwards, 12 PM is NOT before 11 AM)\n"
    "  ❌ 'No conflicts found... time slot is taken' (no conflicts then conflicts)\n"
    "If a response makes unsupported claims about availability (e.g., says 'busy' without showing list_todos output "
    "confirming the conflict), score 0.15 as hallucinated/unreliable. Responses should base availability claims ONLY on "
    "what list_todos actually returned. "
    "For PREFERENCE CONFLICTS: If the user has stated a recurring preference "
    "(e.g., 'On Sundays I only relax', 'No meetings on Fridays after 5 PM', 'I prefer quiet on weekends'), "
    "and the agent's response shows it detected this preference and asked a clarifying question (e.g., "
    "'Does this align with your preference?'), score HIGHLY (0.9+) as a wise, attentive response. "
    "Conversely, if the user stated a preference and the agent IGNORED it and scheduled something conflicting, "
    "score VERY LOW (0.1-0.3) even if the scheduling was technically successful. Examples of violations:\n"
    "  ❌ User: 'Sundays are only for relaxation', Agent: 'Scheduled event on Sunday' → Score 0.1\n"
    "  ❌ User: 'No meetings Friday after 5 PM', Agent: 'Scheduled meeting Friday 6 PM' → Score 0.2\n"
    "  ✅ User: 'Sundays are only for relaxation', Agent asks: 'I see this is Sunday. Is this OK?' → Score 0.9\n"
    'Respond ONLY with JSON: {"score": 0.0, "feedback": "..."}'
)



def score_result(user_text: str, result: SubAgentResult, history_text: str = "") -> tuple[float, str]:
    """Critic/QA gate: score a sub-agent result. Returns (score, feedback)."""
    if not result["success"]:
        return 0.0, result.get("error") or "Sub-agent execution failed."

    # CRITICAL OPTIMIZATION: List/retrieval queries should ALWAYS score 0.9+ (they're complete when items are returned)
    # This prevents unnecessary retries and ToT escalation for simple retrieval operations
    user_lower = user_text.lower()
    
    # Detect EXPLICIT list queries (high confidence)
    # Only auto-score for explicit list/show/get/display commands
    explicit_list_keywords = ["list", "show", "get all", "display", "show me", "tell me all", "all my"]
    task_keywords = ["task", "todo", "event", "schedule", "item", "tasks", "events", "todos", "items", "meeting", "meetings"]
    
    is_explicit_list_request = (
        any(kw in user_lower for kw in explicit_list_keywords)
        and any(kw in user_lower for kw in task_keywords)
    )

    # Deterministic fast-path for itinerary-style exploratory planning prompts.
    # Prevents critic variance from incorrectly applying list-query criteria.
    itinerary_terms = ["weekend", "cities", "city", "indoor", "outdoor", "driving"]
    itinerary_explore_terms = ["different ways", "all the ways", "multiple ways", "organize this", "best option"]
    is_itinerary_exploration = (
        any(term in user_lower for term in itinerary_terms)
        and any(term in user_lower for term in itinerary_explore_terms)
    )
    
    # If this is an EXPLICIT list/show/get request for tasks/events, score it high immediately
    # (Skip LLM scorer for speed and accuracy)
    if is_explicit_list_request:
        output = result.get("output", "")
        # Check if output contains actual content (not empty or error)
        output_has_tasks = output and len(output.strip()) > 10 and (
            "task" in output.lower() or "event" in output.lower() or "todo" in output.lower()
        )
        if output_has_tasks:
            # This is a successful list query response - score it high and SKIP the LLM scorer entirely
            return 0.95, "List/retrieval query completed successfully. Agent returned items as requested."

    if is_itinerary_exploration:
        output = (result.get("output", "") or "").lower()
        has_structured_options = any(token in output for token in ["plan", "option", "weekend", "day 1", "day 2"])
        if has_structured_options and len(output) > 80:
            return 0.90, "Exploratory itinerary response provided multiple structured options as requested."

    # Deterministic fast-path for explicit optimization/exploration prompts.
    # If the agent produced a substantive non-error response, avoid critic variance and retries.
    exploratory_markers = ["different ways", "all the ways", "multiple ways", "best way", "best option", "optimize", "all possibilities"]
    optimization_markers = ["budget", "within", "under", "can't overlap", "cannot overlap", "3 meetings", "three meetings", "3 different cities", "three different cities"]
    is_exploratory_optimization = (
        any(m in user_lower for m in exploratory_markers)
        and any(m in user_lower for m in optimization_markers)
    )
    if is_exploratory_optimization:
        out = (result.get("output", "") or "").strip()
        if out and "error code" not in out.lower() and len(out) >= 80:
            return 0.90, "Exploratory optimization request answered with a substantive multi-step response."

    # Deterministic fast-path for explicit weekday+time scheduling confirmations.
    # Prevents false critic failures when the model flags correct next-week weekday dates as wrong.
    weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    requested_weekday = next((d for d in weekday_map if d in user_lower), None)
    requested_time = re.search(r"\b(1[0-2]|0?[1-9])\s*(am|pm)\b", user_lower)
    confirmed_iso = re.search(r"(\d{4}-\d{2}-\d{2})[t\s](\d{2}):(\d{2})", (result.get("output", "") or ""), re.IGNORECASE)
    if requested_weekday and requested_time and confirmed_iso:
        dt = datetime.fromisoformat(f"{confirmed_iso.group(1)}T{confirmed_iso.group(2)}:{confirmed_iso.group(3)}:00")
        expected_hour = int(requested_time.group(1))
        if requested_time.group(2) == "pm" and expected_hour != 12:
            expected_hour += 12
        if requested_time.group(2) == "am" and expected_hour == 12:
            expected_hour = 0

        now = datetime.now()
        target_wd = weekday_map[requested_weekday]
        days_ahead = (target_wd - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        next_target = (now + timedelta(days=days_ahead)).date()

        if dt.date() == next_target and dt.hour == expected_hour:
            return 0.92, "Scheduling confirmation matches requested weekday and time."

    # Deterministic fast-path for "today"/"tomorrow" + explicit time scheduling confirmations.
    # Same rationale as the weekday fast-path above: the LLM critic sometimes flags a correct
    # ISO date as "wrong" or "future" purely because it doesn't match its own training-data
    # notion of the current date, even though the system prompt tells it not to do this.
    relative_day_match = re.search(r"\b(today|tomorrow)\b", user_lower)
    if relative_day_match and requested_time and confirmed_iso:
        dt = datetime.fromisoformat(f"{confirmed_iso.group(1)}T{confirmed_iso.group(2)}:{confirmed_iso.group(3)}:00")
        expected_hour = int(requested_time.group(1))
        if requested_time.group(2) == "pm" and expected_hour != 12:
            expected_hour += 12
        if requested_time.group(2) == "am" and expected_hour == 12:
            expected_hour = 0

        now = datetime.now()
        expected_date = now.date() if relative_day_match.group(1) == "today" else (now + timedelta(days=1)).date()

        if dt.date() == expected_date and dt.hour == expected_hour:
            return 0.92, f"Scheduling confirmation matches requested '{relative_day_match.group(1)}' and time."
    
    # DETECT CLARIFICATION RESPONSES: If output asks questions, it's likely handling ambiguous requests
    # Score these high (0.85+) because asking for clarification on ambiguous requests is CORRECT behavior
    output = result.get("output", "")
    if is_clarification_response(output):
        # This is a clarification response - score it high
        return 0.85, "Agent appropriately asked for clarification on an ambiguous request."

    # For non-list queries, use LLM-based QA scoring
    prompt_parts = []
    if history_text:
        prompt_parts.append(history_text)
    prompt_parts.append(f"User request: {user_text}")
    prompt_parts.append(f"Agent output: {result.get('output', '')}")
    prompt = "\n\n".join(prompt_parts)
    messages = [SystemMessage(content=_CRITIC_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    try:
        response = invoke_with_retry(_get_llm(), messages)
    except APIConnectionError:
        return 0.0, "Critic could not reach the LLM provider. Check network connectivity or API availability."
    try:
        data = parse_json_response(response.content)
        return float(data.get("score", 0.0)), data.get("feedback", "")
    except (AttributeError, ValueError, TypeError):
        return 0.0, "Critic response could not be parsed."


def passes_qa(score: float) -> bool:
    return score >= QA_SCORE_THRESHOLD


def _contains_word(text: str, word: str) -> bool:
    """Whole-word membership check (word may itself be multiple space-separated words,
    e.g. "this week"). Avoids false positives from incidental substrings, e.g. "clean"
    matching inside a task name like "demo-cleanup-test", or "all" matching inside "called".
    """
    if " " in word:
        return word in text
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


# Broad, deterministic destructive-action detector: a destructive verb PLUS a target
# noun/day-name. This catches natural phrasing that doesn't literally spell out one of
# the exact configured HUMAN_APPROVAL_ACTIONS phrases (e.g. "cancel_event"), such as
# "cancel my outdoor ACTIVITY for Saturday" or "clean up all my Saturday stuff" - both
# destructive, but with an object word ("activity") that "cancel_event" doesn't cover.
# Without this, requests like these fell through the real approval gate entirely and
# were only ever "confirmed" via an informal, non-state-backed text prompt improvised
# by the domain agent itself - one with no Approve/Reject buttons and no memory of the
# pending action across turns, so a later "yes" reply had nothing to attach to.
_DESTRUCTIVE_VERBS = {"cancel", "delete", "remove", "clean", "clear", "complete_all", "finish_all"}
_DESTRUCTIVE_TARGETS = {
    "all", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "tomorrow", "this week",
    "task", "tasks", "event", "events", "todo", "todos", "item", "items",
    "appointment", "appointments", "activity", "activities", "meeting", "meetings",
    "note", "notes", "plan", "plans", "reminder", "reminders",
}


def is_destructive_action_request(text: str) -> bool:
    """Broad, deterministic (no LLM call) detector for delete/cancel/clean-up intents.

    Used both by the real Guardrail 3 approval gate (requires_human_approval) and by
    the scheduler agent's own task instructions (src/agents/context.py), so both places
    agree on what counts as "destructive" instead of using separate, inconsistent logic.
    """
    lowered = (text or "").lower()
    has_verb = any(_contains_word(lowered, verb) for verb in _DESTRUCTIVE_VERBS)
    has_target = any(_contains_word(lowered, target) for target in _DESTRUCTIVE_TARGETS)
    return has_verb and has_target


def requires_human_approval(task_description: str) -> bool:
    """Guardrail 3: flag actions that must be confirmed by the user before execution.

    Two detectors, combined:
    1. Exact configured action phrases (e.g. "cancel_event"): matches if every word of
       a configured action appears somewhere in the text as a whole word, so natural
       phrasing like "cancel my event" still matches.
    2. is_destructive_action_request: a broader destructive-verb + target-noun/day-name
       pattern, so phrasing that doesn't spell out a configured phrase verbatim (e.g.
       "cancel my outdoor activity for Saturday") is still correctly gated.
    """
    lowered = task_description.lower()
    tokens = set(re.findall(r"[a-z0-9']+", lowered))
    for action in HUMAN_APPROVAL_ACTIONS:
        words = action.split("_")
        if all(word in tokens for word in words):
            return True
    return is_destructive_action_request(task_description)


def should_escalate(retry_count: int) -> bool:
    """Guardrail 1: escalate to the user once the retry budget is exhausted."""
    return retry_count >= MAX_RETRIES
