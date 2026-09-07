"""Tree-of-Thought planner used by the LangGraph orchestrator (Checkpoint 4.1).

Generates candidate reasoning branches, scores them, and prunes weak paths
using a beam-search strategy bounded by branching_factor/beam_width/max_depth
from config/settings.yaml.
"""

import uuid
import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, SETTINGS, get_llm
from src.graph.state import ThoughtBranch
from src.utils.helpers import invoke_with_retry, parse_json_response

logger = logging.getLogger("personal_assistant")

_TOT_CFG = SETTINGS.get("tree_of_thought", {})
BRANCHING_FACTOR = _TOT_CFG.get("branching_factor", 2)
BEAM_WIDTH = _TOT_CFG.get("beam_width", 1)
MAX_DEPTH = _TOT_CFG.get("max_depth", 2)
PRUNE_THRESHOLD = _TOT_CFG.get("prune_threshold", 0.50)

_VALID_AGENTS = ("scheduler_todo", "notes_knowledge", "weather_external")
OUT_OF_SCOPE = "out_of_scope"
_ALL_TARGETS = _VALID_AGENTS + (OUT_OF_SCOPE,)

# Lazy-load LLM to pick up config changes without module cache issues
_llm = None

def _get_llm():
    """Get LLM instance, lazily initializing on first use."""
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm

_SYSTEM_PROMPT = (
    "You are the planning module of a personal assistant orchestrator. "
    "You may be given prior conversation turns for context — use them to resolve "
    "follow-up requests (e.g. \"give me that in fahrenheit\" refers back to whatever city/topic "
    "was just discussed), rather than guessing something unrelated. "
    "CRITICAL: When the user is responding to an earlier clarification question WITH the requested information, "
    "recognize this as CLEAR INTENT and route directly to the appropriate agent. "
    "Example: If you asked 'What time?' and user responds '9AM or 3PM', this is CLEAR—use scheduler_todo with those times. "
    "Do NOT ask for clarification again if the user has already answered it. "
    f"Given the user's request (and optionally a prior partial thought to refine), propose up to "
    f"{BRANCHING_FACTOR} distinct candidate interpretations/plans for how to handle it.\n\n"
    f"Each candidate must pick exactly one target agent from: {', '.join(_VALID_AGENTS)}. "
    f'If the request does not fit any of these three domains at all (e.g. general knowledge, '
    f'math, or unrelated small talk), use target_agent "{OUT_OF_SCOPE}" instead, with a high '
    f"score reflecting how confident you are that it is genuinely out of scope.\n\n"
    "CRITICAL ROUTING RULES:\n"
    "1. If the request contains action verbs like 'schedule', 'add', 'create', 'plan', 'list', 'show', 'view', 'check' + 'event/task/todo', "
    "pick 'scheduler_todo' as the target_agent. This includes queries like 'list my tasks this week', 'show my schedule', 'what events do I have'. "
    "Weather checking for outdoor events is AUTOMATIC and happens after scheduling, so do NOT make weather_external the target.\n"
    "2. If the request is ONLY about weather/forecast (no scheduling/saving), pick 'weather_external'.\n"
    "3. If the request asks to save/remember/note something, pick 'notes_knowledge'.\n"
    "4. If the request is a QUESTION asking what the user wants/likes/prefers to do (e.g. 'what do I want "
    "to do on a Sunday', 'what should I do this weekend', 'what's my preference for X') and does NOT contain "
    "an explicit scheduling verb ('schedule', 'add', 'create', 'book'), this is a RETRIEVAL of a previously "
    "stated preference/note, NOT a request to create a new scheduled event. Pick 'notes_knowledge' and never "
    "invent or schedule an activity on the user's behalf just because a related preference exists in history.\n"
    "Score each candidate from 0.0 to 1.0 for how confidently it correctly and completely "
    "addresses the PRIMARY user intent (the main action verb).\n\n"
    'Respond ONLY with JSON of the form: {"candidates": [{"thought": "...", '
    '"target_agent": "...", "score": 0.0}]}'
)


def _call_llm(user_text: str, history_text: str, parent_thought: Optional[str]) -> list[dict]:
    context = f"{history_text}\n\n{user_text}" if history_text else user_text
    if parent_thought:
        context = f"{context}\n\nRefine this prior thought: {parent_thought}"
    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=context)]
    try:
        response = invoke_with_retry(_get_llm(), messages)
    except Exception as exc:
        logger.warning(f"Planner LLM unavailable; using heuristic candidates ({exc.__class__.__name__})")
        return _heuristic_candidates(user_text)
    try:
        data = parse_json_response(response.content)
        return data.get("candidates", [])
    except (ValueError, AttributeError, TypeError):
        return _heuristic_candidates(user_text)


def _heuristic_target_agent(user_text: str) -> str:
    text = (user_text or "").lower()
    schedule_terms = ["schedule", "task", "tasks", "todo", "event", "events", "meeting", "meetings", "list", "show", "delete", "remove", "cancel"]
    note_terms = ["save", "note", "remember", "preference", "favorite"]
    weather_terms = ["weather", "forecast", "temperature", "rain", "sunny", "humidity", "wind"]

    if any(term in text for term in note_terms):
        return "notes_knowledge"

    if any(term in text for term in schedule_terms):
        return "scheduler_todo"

    if any(term in text for term in weather_terms):
        return "weather_external"

    return OUT_OF_SCOPE


def _is_schedule_list_query(text_lower: str) -> bool:
    """True for requests to list/show scheduled tasks/events/todos (never notes/preferences).

    Without this, the direct-path LLM can occasionally misclassify a plain list request
    (e.g. "list all scheduled events") as notes_knowledge, which then searches saved notes
    (finding nothing) instead of calling list_todos, producing a confusing "I couldn't find
    any scheduled events in your saved notes" reply even when open to-dos exist.
    """
    list_verbs = ("list", "show", "get all", "what are", "tell me", "which", "view")
    schedule_nouns = (
        "task", "tasks", "todo", "todos", "event", "events", "item", "items",
        "schedule", "scheduled", "meeting", "meetings", "appointment", "appointments",
    )
    note_nouns = ("note", "notes", "preference", "preferences", "favorite", "favorites")
    has_list_verb = any(v in text_lower for v in list_verbs)
    has_schedule_noun = any(n in text_lower for n in schedule_nouns)
    has_note_noun = any(n in text_lower for n in note_nouns)
    return has_list_verb and has_schedule_noun and not has_note_noun


def _heuristic_candidates(user_text: str) -> list[dict]:
    text = (user_text or "").lower()
    target = _heuristic_target_agent(user_text)

    if target == "scheduler_todo" and any(k in text for k in ["different ways", "all the ways", "best way", "3 meetings", "cannot overlap", "can't overlap"]):
        thought = "Explore multiple feasible scheduling arrangements and present the best option with trade-offs."
        return [{"thought": thought, "target_agent": "scheduler_todo", "score": 0.92}]

    if target == "scheduler_todo" and any(k in text for k in ["cities", "city", "weekend", "indoor", "outdoor", "driving"]):
        thought = "Provide 2-4 weekend itinerary options that satisfy location and activity constraints, with a recommendation."
        return [{"thought": thought, "target_agent": "notes_knowledge", "score": 0.90}]

    default_thought = f"Handle the user request with {target}."
    return [{"thought": default_thought, "target_agent": target, "score": 0.85}]


def _expand(user_text: str, history_text: str, parent: Optional[ThoughtBranch], depth: int) -> list[ThoughtBranch]:
    candidates = _call_llm(user_text, history_text, parent["thought"] if parent else None)
    branches: list[ThoughtBranch] = []
    for candidate in candidates[:BRANCHING_FACTOR]:
        target_agent = candidate.get("target_agent")
        if target_agent not in _ALL_TARGETS:
            continue
        branches.append(
            ThoughtBranch(
                node_id=str(uuid.uuid4()),
                parent_id=parent["node_id"] if parent else None,
                depth=depth,
                thought=candidate.get("thought", ""),
                target_agent=target_agent,
                score=float(candidate.get("score", 0.0)),
                pruned=False,
            )
        )
    return branches


def run_tree_of_thought(
    user_text: str, history_text: str = ""
) -> tuple[list[ThoughtBranch], Optional[ThoughtBranch]]:
    """Beam-search over candidate plans; returns (all_branches_seen, best_branch)."""
    logger.info("🧠 PLANNING: Starting Tree-of-Thought reasoning...")
    all_branches: list[ThoughtBranch] = _expand(user_text, history_text, None, depth=0)
    logger.info(f"   Depth 0: Generated {len(all_branches)} candidate(s)")
    for i, b in enumerate(all_branches, 1):
        logger.info(f"     Candidate {i}: {b['target_agent']} (confidence: {b['score']:.2f})")
        logger.info(f"       💭 Thought: {b['thought']}")
    
    frontier = list(all_branches)

    depth = 0
    while depth < MAX_DEPTH - 1:
        frontier = [b for b in frontier if (b["score"] or 0.0) >= PRUNE_THRESHOLD]
        frontier.sort(key=lambda b: b["score"] or 0.0, reverse=True)
        frontier = frontier[:BEAM_WIDTH]
        if not frontier or frontier[0]["score"] >= 0.9:
            logger.info(f"   ✓ Confident solution found at depth {depth}")
            break  # confident enough to terminate early
        depth += 1
        next_frontier: list[ThoughtBranch] = []
        for parent in frontier:
            next_frontier.extend(_expand(user_text, history_text, parent, depth))
        logger.info(f"   Depth {depth}: Generated {len(next_frontier)} candidate(s)")
        for i, b in enumerate(next_frontier, 1):
            logger.info(f"     Candidate {i}: {b['target_agent']} (confidence: {b['score']:.2f})")
            logger.info(f"       💭 Thought: {b['thought']}")
        all_branches.extend(next_frontier)
        frontier = next_frontier

    for branch in all_branches:
        if (branch["score"] or 0.0) < PRUNE_THRESHOLD:
            branch["pruned"] = True

    survivors = [b for b in all_branches if not b["pruned"]]
    best = max(survivors, key=lambda b: b["score"] or 0.0) if survivors else None
    if best:
        logger.info(f"   🎯 SELECTED: {best['target_agent']} (confidence: {best['score']:.2f})")
        logger.info(f"       💭 Thought: {best['thought']}")
    return all_branches, best


# Cheap, single-candidate instance for the "simple" fast path (Checkpoint 4.1: not every
# request needs branch-and-prune search; a single direct call is far fewer tokens/latency).
_direct_llm = None

def _get_direct_llm():
    """Get direct LLM instance, lazily initializing on first use."""
    global _direct_llm
    if _direct_llm is None:
        _direct_llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=200,
        )
    return _direct_llm

_DIRECT_SYSTEM_PROMPT = (
    "You are the planning module of a personal assistant orchestrator, handling a request "
    "already classified as simple and unambiguous. Propose exactly ONE plan: pick the single "
    "target agent from: " + ", ".join(_VALID_AGENTS) + f'. If the request does not fit any of '
    f'these domains, use target_agent "{OUT_OF_SCOPE}" instead. You may be given prior '
    "conversation turns for context — use them to resolve follow-up requests. "
    "\n\nCRITICAL RULE: Distinguish a GENUINE conditional (two different outcomes, a real fork) from "
    "a plain compound request that merely mentions weather alongside scheduling/notes.\n"
    "- GENUINE conditional (AMBIGUOUS, needs clarification): the user names a threshold/fork with an "
    "explicit alternative outcome, e.g. 'IF the weather is nice, schedule a picnic, OTHERWISE cancel it' "
    "or 'if it rains, reschedule instead'. Route these as target_agent "
    f'"{OUT_OF_SCOPE}" with a note that clarification is needed.\n'
    "- NOT a genuine conditional (proceed normally, single agent, no clarification): the user asks to "
    "schedule/save something AND also check/verify the weather for it, with only ONE outcome implied "
    "(e.g. 'schedule the picnic and check if the weather will be suitable', 'check if it'll be nice this "
    "weekend and put something fun on my calendar'). Here 'if'/'whether' just means "
    "'find out whether', not a fork — treat the scheduling/notes verb as the PRIMARY user intent and "
    "pick that agent (scheduler_todo or notes_knowledge); weather is checked automatically afterward.\n"
    "- QUESTION ABOUT A STORED PREFERENCE (route to notes_knowledge, NEVER schedule anything): if the "
    "user asks what they want/like/prefer to do (e.g. 'what do I want to do on a Sunday', 'what should "
    "I do this weekend') and does NOT use an explicit scheduling verb ('schedule', 'add', 'create', "
    "'book'), this is a RETRIEVAL question about a previously stated preference/note. Pick "
    "'notes_knowledge' to look up and answer from stored notes. NEVER invent, assume, or create a new "
    "scheduled event just because a related preference exists in the conversation history. The 'thought' "
    "you write for this case MUST describe LOOKING UP and ANSWERING the question from stored notes "
    "(e.g. 'Search saved notes for the user's Sunday preference and answer the question') — it must NOT "
    "use words like 'schedule', 'book', or 'create an event', since that would wrongly instruct the "
    "downstream agent to fabricate a scheduling action it has no tool to perform.\n"
    'Respond ONLY with JSON: {"thought": "...", "target_agent": "..."}'
)


def _is_genuine_conditional(text_lower: str) -> bool:
    """Distinguish a real IF/THEN/ELSE conditional from plain 'check if/whether' phrasing.

    Plain statements like "check if the weather will be suitable" or "see if it'll be
    nice" use "if" as a subordinating conjunction meaning "whether" — there is only ONE
    outcome (do the thing, then report on weather). A genuine conditional has TWO
    outcomes and an explicit fork, e.g. "schedule the picnic if it's nice, otherwise
    cancel it" or "if it rains, reschedule instead". We require evidence of that fork
    (an "else/otherwise" branch, or an explicit negative-outcome action) before treating
    the request as ambiguous and asking for clarification.
    """
    if_trigger = any(kw in text_lower for kw in ("if ", "unless ", "only if ", "provided that", "in case"))
    if not if_trigger:
        return False

    # "check if" / "see if" / "verify if" / "confirm if" is "whether", not a real branch.
    whether_phrasing = any(
        phrase in text_lower
        for phrase in ("check if", "check whether", "see if", "verify if", "confirm if", "find out if")
    )

    fork_signal = any(
        kw in text_lower
        for kw in (
            "otherwise",
            "else",
            "if not",
            "if it's not",
            "if it is not",
            "in that case",
            "if it rains",
            "if the weather is bad",
            "if the weather isn't",
            "cancel it",
            "reschedule instead",
            "proceed anyway",
            "skip it",
        )
    )

    if whether_phrasing and not fork_signal:
        return False

    return fork_signal or not whether_phrasing


def detect_multi_agent_dispatch(user_text: str) -> list[str]:
    """Detect if a request requires multiple agents and return their names.
    
    Returns a list of agent names that should be dispatched in parallel, or empty list
    if single-agent dispatch should proceed.
    
    NOTE: Conditional patterns (IF...THEN...) are NOT multi-agent dispatch targets.
    They should be routed as single-agent with clarification questions.
    """
    text_lower = user_text.lower()

    # BLOCK: itinerary-style exploration should stay single-agent (planning response),
    # not scheduler+notes combined execution.
    itinerary_travel_terms = ["visit", "itinerary", "cities", "city", "driving", "weekend"]
    itinerary_explore_terms = ["different ways", "all the ways", "multiple ways", "organize this", "best option"]
    if any(t in text_lower for t in itinerary_travel_terms) and any(t in text_lower for t in itinerary_explore_terms):
        return []
    
    # BLOCK: Conditional patterns - NOT multi-agent dispatch
    # These should ask clarifying questions, not dispatch in parallel
    is_conditional = (
        ("schedule" in text_lower or "activity" in text_lower or "plan" in text_lower) and
        ("weather" in text_lower or "rain" in text_lower or "sunny" in text_lower or "nice" in text_lower) and
        _is_genuine_conditional(text_lower)
    )
    
    if is_conditional:
        # Conditional weather requests are ambiguous - ask for clarification instead
        logger.info(f"   ℹ Conditional pattern detected (IF weather THEN action) - not multi-agent dispatch")
        return []  # Don't dispatch as multi-agent; let it route to single agent with clarification
    
    # Multi-agent patterns: user explicitly requests multiple INDEPENDENT actions
    multi_agent_keywords = {
        # Schedule + Notes (save + schedule are separate)
        ("schedule", ("event", "meeting"), ("save", "note", "document", "remember")):
            ["scheduler_todo", "notes_knowledge"],
        # Notes + Schedule
        ("my", ("preferences", "favorites", "activities"), ("schedule", "plan", "book")):
            ["notes_knowledge", "scheduler_todo"],
    }
    
    # Check for explicit multi-agent patterns
    for pattern_tuple, agent_list in multi_agent_keywords.items():
        primary_keyword = pattern_tuple[0]
        secondary_keywords = pattern_tuple[1]
        tertiary_keywords = pattern_tuple[2]
        
        if primary_keyword in text_lower:
            if any(kw in text_lower for kw in secondary_keywords) and \
               any(kw in text_lower for kw in tertiary_keywords):
                return agent_list
    
    # Check for explicit "and" / "also" / "simultaneously" connectors with agent-domain keywords
    domain_keywords = {
        "scheduler_todo": ["schedule", "event", "meeting", "task", "reminder", "calendar"],
        "notes_knowledge": ["save", "note", "remember", "document", "preference", "favorite"],
        "weather_external": ["weather", "forecast", "temperature", "rain", "sunny", "suitable"],
    }
    
    agent_matches = []
    for agent, keywords in domain_keywords.items():
        if any(kw in text_lower for kw in keywords):
            agent_matches.append(agent)
    
    # If we found 2+ agents AND there's an explicit AND/OR/ALSO connector (not just "check"), assume multi-agent
    # "check" alone is not a connector - it's just a word (e.g., "weather check" is not multi-agent)
    connectors = [" and ", " also ", " or ", "simultaneously", " plus ", ", then"]
    has_connector = any(conn in text_lower for conn in connectors)
    
    if len(agent_matches) >= 2 and has_connector:
        return agent_matches
    
    return []


def run_direct(user_text: str, history_text: str = "") -> tuple[list[ThoughtBranch], Optional[ThoughtBranch]]:
    """Single-call fast path for requests already classified as \"simple\" (no branch/prune search)."""
    
    # PRE-FILTER: Detect conditional weather patterns that ONLY need clarification
    # (not exploratory multi-path scenarios - those go to ToT)
    text_lower = user_text.lower()
    weather_keywords = ["weather", "rain", "sunny", "nice", "forecast", "temperature"]
    action_keywords = ["schedule", "activity", "plan", "book", "arrange"]
    
    is_conditional_weather = (
        any(weather_kw in text_lower for weather_kw in weather_keywords) and
        any(action_kw in text_lower for action_kw in action_keywords) and
        _is_genuine_conditional(text_lower)
    )
    
    if is_conditional_weather:
        # Conditional weather request - route as OUT_OF_SCOPE (requires clarification first)
        logger.info("🧠 PLANNING: Conditional weather pattern detected - needs clarification...")
        branch = ThoughtBranch(
            node_id=str(uuid.uuid4()),
            parent_id=None,
            depth=0,
            thought="Conditional scheduling based on weather requires clarification before proceeding.",
            target_agent=OUT_OF_SCOPE,  # Mark as ambiguous/needs clarification
            score=1.0,
            pruned=False,
        )
        logger.info(f"   🎯 SELECTED: {OUT_OF_SCOPE} (conditional weather - ask clarifications first)")
        return [branch], branch
    
    logger.info("🧠 PLANNING: Simple/Direct reasoning (routine task)...")
    context = f"{history_text}\n\n{user_text}" if history_text else user_text
    messages = [SystemMessage(content=_DIRECT_SYSTEM_PROMPT), HumanMessage(content=context)]
    try:
        response = invoke_with_retry(_get_direct_llm(), messages)
    except Exception as exc:
        logger.warning(f"Direct planner LLM unavailable; using heuristic routing ({exc.__class__.__name__})")
        target_agent = _heuristic_target_agent(user_text)
        thought = f"Handle the user request with {target_agent}."
        branch = ThoughtBranch(
            node_id=str(uuid.uuid4()),
            parent_id=None,
            depth=0,
            thought=thought,
            target_agent=target_agent,
            score=0.9,
            pruned=False,
        )
        logger.info(f"   🎯 SELECTED: {target_agent}")
        return [branch], branch

    try:
        data = parse_json_response(response.content)
        target_agent = data.get("target_agent")
        thought = data.get("thought", user_text)
    except (ValueError, AttributeError, TypeError):
        target_agent = _heuristic_target_agent(user_text)
        thought = f"Handle the user request with {target_agent}."

    if target_agent not in _ALL_TARGETS:
        logger.warning(f"   ⚠ Invalid target agent: {target_agent}")
        return [], None  # fall back to caller deciding what to do (e.g. escalate/retry)

    # SAFETY NET: the direct-path LLM occasionally misclassifies a plain, unambiguous
    # single-domain request (e.g. "What is the current weather in Seattle?") as
    # out_of_scope. If a fast heuristic confidently resolves the request to a real
    # domain and there's no genuine conditional in play, trust the heuristic instead
    # of forcing the user through an unnecessary clarification round.
    if target_agent == OUT_OF_SCOPE and not _is_genuine_conditional(text_lower):
        heuristic_agent = _heuristic_target_agent(user_text)
        if heuristic_agent != OUT_OF_SCOPE:
            logger.info(
                f"   ⚠ Direct planner said out_of_scope but heuristic confidently resolves to "
                f"'{heuristic_agent}'; overriding to avoid unnecessary clarification."
            )
            target_agent = heuristic_agent
            thought = f"Handle the user request with {target_agent}."

    # SAFETY NET: same idea, but for list/show requests about tasks/events/todos being
    # misrouted to notes_knowledge instead of scheduler_todo (see _is_schedule_list_query).
    if target_agent != "scheduler_todo" and _is_schedule_list_query(text_lower):
        logger.info(
            f"   ⚠ Direct planner said '{target_agent}' for a schedule list/show request; "
            "overriding to 'scheduler_todo' so it calls list_todos instead of searching notes."
        )
        target_agent = "scheduler_todo"
        thought = "List the user's scheduled tasks/events by calling list_todos."

    branch = ThoughtBranch(
        node_id=str(uuid.uuid4()),
        parent_id=None,
        depth=0,
        thought=thought,
        target_agent=target_agent,
        score=1.0,  # trusted, not scored/pruned since this bypasses beam search entirely
        pruned=False,
    )
    logger.info(f"   🎯 SELECTED: {target_agent}")
    return [branch], branch
