"""Adapters between LangGraph state and CrewAI task inputs/outputs."""

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage

from src.graph.state import AssistantState, SubAgentResult, ThoughtBranch


def _contains_word(text: str, word: str) -> bool:
    """Whole-word membership check (word may itself be multiple space-separated words,
    e.g. "this week"). Avoids false positives from incidental substrings, e.g. "clean"
    matching inside a task name like "demo-cleanup-test", or "all" matching inside "called".
    """
    if " " in word:
        return word in text  # multi-word phrases are fine to substring-match
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


USER_TIMEZONE = ZoneInfo("America/New_York")


def _get_next_weekday_date(day_name: str, reference_date: datetime) -> str | None:
    """Calculate the next occurrence of a weekday name (e.g., 'Sunday') from reference_date.
    
    Returns ISO date string (YYYY-MM-DD) or None if day_name is invalid.
    Example: If today is Monday 2026-09-01 and day_name is 'Sunday', returns '2026-09-07'
    """
    weekday_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }
    
    day_lower = day_name.lower().strip()
    if day_lower not in weekday_map:
        return None
    
    target_weekday = weekday_map[day_lower]
    current_weekday = reference_date.weekday()
    
    # Days to add to get to target weekday (next occurrence, or today if it's today)
    days_ahead = (target_weekday - current_weekday) % 7
    
    # If it's the same day, schedule for next week's occurrence (unless it's earlier today)
    if days_ahead == 0:
        days_ahead = 7  # Always next week for day-name-only requests
    
    target_date = reference_date + timedelta(days=days_ahead)
    return target_date.strftime("%Y-%m-%d")


_LOCATION_HINT_EXCLUDE = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}


def _extract_location_hint(user_text: str) -> str | None:
    """Best-effort extraction of a place name from phrases like 'picnic in Denver'.

    Used to give the weather_external sub-agent a concrete location when it's dispatched
    as a secondary agent in a multi-agent turn, so it doesn't have to guess/hallucinate
    a city when the user's request never actually named one.
    """
    import re

    pattern = re.compile(r"\b(?:in|at|near)\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)")
    for match in pattern.findall(user_text or ""):
        first_word = match.split()[0].lower()
        if first_word in _LOCATION_HINT_EXCLUDE:
            continue
        return match.strip()
    return None


def latest_user_text(state: AssistantState) -> str:
    """Return the most recent human message text from the conversation."""
    for message in reversed(state["messages"]):
        if message.type == "human":
            return message.content
    return ""


def conversation_history_text(state: AssistantState, max_messages: int = 20) -> str:
    """Render prior turns (excluding the current, still-unanswered request) as a transcript.

    Without this, follow-ups like "give me in fahrenheit" have no way to resolve what
    "it" refers to, since planning/dispatch would otherwise only ever see the latest message.
    """
    # The current request is the last human message; everything before it is prior context.
    prior_messages = state["messages"][:-1] if state["messages"] else []
    prior_messages = prior_messages[-max_messages:]
    if not prior_messages:
        return ""
    lines = ["Conversation so far:"]
    for message in prior_messages:
        speaker = "User" if message.type == "human" else "Assistant"
        lines.append(f"{speaker}: {message.content}")
    return "\n".join(lines)


def _should_include_history_for_request(user_text: str) -> bool:
    """Only include prior turns when the latest request is a likely follow-up reference.

    This avoids stale-context leakage where explicit new commands get mixed with
    previous requests (for example, a new scheduling command being treated like
    an earlier list request).
    """
    text = (user_text or "").strip().lower()
    if not text:
        return False

    # Very short replies are usually follow-ups to the previous assistant question.
    if len(text.split()) <= 4:
        return True

    followup_markers = (
        "that",
        "it",
        "them",
        "those",
        "same",
        "instead",
        "also",
        "as well",
        "the one",
        "previous",
        "earlier",
        "you said",
    )
    return any(marker in text for marker in followup_markers)


def build_task_description(state: AssistantState, branch: ThoughtBranch) -> str:
    """Compose the CrewAI task description from the user's message and the selected thought."""
    now = datetime.now(USER_TIMEZONE)
    parts = [
        f"Today's local date is {now:%Y-%m-%d} and the local time is {now:%I:%M %p} "
        "(America/New_York). Interpret relative dates like today/tomorrow using this local date."
    ]
    latest_request = latest_user_text(state)
    history = conversation_history_text(state)
    if history and _should_include_history_for_request(latest_request):
        parts.append(history)
        parts.append("Use prior turns only to resolve references (for example: 'that', 'it', 'same').")
    parts.append(f"User request (authoritative, execute this): {latest_request}")
    parts.append("Do not execute or summarize earlier requests unless the latest request explicitly refers to them.")
    
    # Branch handling varies by agent type
    if "thought" in branch:
        parts.append(f"Plan: {branch['thought']}")
    elif "target_agent" in branch:
        agent = branch["target_agent"]
        raw_user_text = latest_user_text(state)
        
        # Task guidance for secondary agents in multi-agent scenarios.
        # IMPORTANT: scope these agents to ONLY their own slice of the compound request.
        # Without this, a secondary agent (e.g. notes_knowledge) sees the full original
        # request and tries to address parts outside its domain (e.g. weather), producing
        # confusing "I couldn't find weather info" output from an agent with no weather tool.
        if agent == "weather_external":
            location_hint = _extract_location_hint(raw_user_text)
            parts.append(
                "Task: This is a SECONDARY task within a larger request. ONLY handle the weather-checking "
                "portion — ignore any scheduling/notes/saving instructions in the request, those are handled "
                "by other agents."
            )
            if location_hint:
                parts.append(f"Location to check: {location_hint}")
            else:
                parts.append(
                    "No specific location was named in the request. Do NOT guess or default to any city "
                    "(e.g. do not assume 'New York'). Instead, respond by asking the user which location "
                    "they'd like the weather for."
                )
        elif agent == "notes_knowledge":
            parts.append(
                "Task: This is a SECONDARY task within a larger request. ONLY save the note/preference "
                "portion of the request — ignore any scheduling or weather-checking instructions, those "
                "are handled by other agents. Do not attempt to look up or report on weather."
            )
    
    # For scheduler_todo agent: detect ambiguous date/time requests and ask for clarification
    if branch.get("target_agent") == "scheduler_todo":
        user_text = latest_user_text(state).lower()
        # For activity descriptors, check BOTH latest message and full conversation history
        full_context = (conversation_history_text(state) + " " + user_text).lower()
        ambiguities = []
        
        # Pre-calculate common day names to help agent with date conversion
        day_name_dates = {}
        for day_name in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
            if day_name in user_text:
                date_str = _get_next_weekday_date(day_name, now)
                if date_str:
                    day_name_dates[day_name] = date_str
        
        if day_name_dates:
            parts.append("\n📅 DATE MAPPING (for converting day names to actual dates):")
            for day, date in day_name_dates.items():
                parts.append(f"  - {day.capitalize()}: {date}")
            parts.append("Use these dates when converting the user's request to ISO datetime format.")
        
        # Check for vague date references
        if any(vague in user_text for vague in ["next week", "this week", "next month", "sometime", "soon", "later"]):
            if not any(specific in user_text for specific in ["september", "october", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th"]):
                ambiguities.append("- Specify the exact date (e.g., 'September 7' or 'next Monday')")
        
        # Check for vague times (morning, afternoon, evening WITHOUT specific hour)
        vague_times = ["morning", "afternoon", "evening", "night"]
        specific_times = ["am", "pm", "o'clock", ":", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        has_vague_time = any(vague in user_text for vague in vague_times)
        has_specific_time = any(specific in user_text for specific in specific_times)
        
        # Check if this is a FOLLOW-UP to a time clarification question
        # (i.e., assistant asked "what time?" and user is now responding with "morning/afternoon/evening")
        history_lower = (conversation_history_text(state) or "").lower()
        is_follow_up_to_time_question = (
            ("what time" in history_lower or "morning" in history_lower or "afternoon" in history_lower or "evening" in history_lower)
            and has_vague_time 
            and not has_specific_time
            and len(user_text) < 20  # Short response like just "morning" or "afternoon"
        )
        
        # Check if user is picking from previously offered available slots
        # (i.e., assistant offered "7 PM, 8 PM, 9 PM" and user said "7PM")
        is_picking_offered_slot = (
            has_specific_time
            and ("available slot" in history_lower or "available:" in history_lower or "available times" in history_lower)
            and not has_vague_time
            and len(user_text) < 20  # Short response like "7PM" or "9AM"
        )
        
        # Check if user is asking to delete/clean up/remove tasks
        # "clean up all tasks for Saturday", "delete all tasks", "remove all events", etc.
        # Uses the shared detector from supervisor.py so this agent-instruction check and
        # the real Guardrail 3 approval gate (supervisor.requires_human_approval) always
        # agree on what counts as destructive - previously they used separate, narrower
        # keyword lists that could disagree (e.g. "cancel my outdoor activity" wasn't
        # recognized as destructive by the approval gate, so it never got a real,
        # state-backed Approve/Reject prompt - just improvised free text from the agent).
        from src.agents.supervisor import is_destructive_action_request
        user_text_lower = user_text.lower()
        is_delete_action = is_destructive_action_request(user_text_lower)
        
        # Check if user is asking to list/show/get all tasks (NO scheduling needed)
        # "list all my saturday tasks", "show me my tasks", "what are my tasks", "get all tasks", etc.
        list_keywords = ["list", "show", "get all", "what are", "tell me", "all my", "which"]
        query_keywords = ["task", "tasks", "event", "events", "todo", "todos", "item", "items", "schedule", "scheduled"]
        is_list_query = (
            any(keyword in user_text.lower() for keyword in list_keywords)
            and any(keyword in user_text.lower() for keyword in query_keywords)
        )
        
        # Check if this is a GENUINE CONDITIONAL weather request
        # "IF the weather is nice, schedule a picnic, OTHERWISE cancel it" - needs clarification first,
        # since it's inherently ambiguous (what "nice" means, what to do on the other branch).
        # Plain phrasing like "schedule the picnic and check if the weather will be suitable" is NOT
        # a genuine conditional (only one outcome) and should proceed normally.
        from src.agents.planner import _is_genuine_conditional
        is_conditional_weather = (
            any(weather_kw in user_text.lower() for weather_kw in ["weather", "rain", "sunny", "nice", "forecast"])
            and ("schedule" in user_text.lower() or "activity" in user_text.lower() or "plan" in user_text.lower())
            and _is_genuine_conditional(user_text.lower())
        )
        
        if has_vague_time and not has_specific_time and not is_follow_up_to_time_question and not is_list_query:
            # Only flag as ambiguous if NOT a follow-up response and NOT a list query
            ambiguities.append("- Specify the exact time (e.g., 'Saturday morning at 10 AM' or 'Sunday 2 PM', not just 'morning')")
        elif not has_vague_time and not has_specific_time and not is_list_query:
            # Only ask for time if this is NOT a list query
            ambiguities.append("- Ask for the preferred time of day (e.g., '10 AM', '2:30 PM', or choose from morning/afternoon/evening)")
        
        # Check for missing event details (expanded to include travel/vacation keywords)
        # Accept EITHER formal event types OR descriptive mood/activity keywords
        # IMPORTANT: Check in FULL CONTEXT (history + current message) for descriptors
        formal_event_types = [
            "meeting", "call", "lunch", "dinner", "workout", "hike", "picnic", "task", "project", "work", "event", "appointment",
            "vacation", "trip", "travel", "getaway", "excursion", "visit", "tour", "retreat", "holiday",
            "swimming", "swim", "yoga", "run", "walk", "sport", "exercise", "activity", "plan", "reminder"
        ]
        activity_descriptors = [
            "fun", "productive", "relaxing", "relaxation", "chill", "active", "rest", "busy", "social", 
            "creative", "learning", "focused", "family", "friends", "alone", "outdoor", "indoor"
        ]
        has_formal_event = any(event in full_context for event in formal_event_types)
        has_descriptors = any(desc in full_context for desc in activity_descriptors)
        
        # Only flag as ambiguous if we have NEITHER formal events NOR descriptive keywords
        # EXCEPTION: Don't ask for event details on list queries
        if not has_formal_event and not has_descriptors and not is_list_query:
            ambiguities.append("- Ask what kind of event/task this is (for better context)")
        
        if is_delete_action:
            # User is asking to delete/clean up/remove tasks - handle as destructive action.
            # NOTE: Guardrail 3 (human_approval_node) already gates this action BEFORE dispatch
            # ever runs — the user only reaches this point after explicitly approving via the
            # interrupt() prompt. If we still tell the agent to "ask for confirmation" here too,
            # it re-asks a question the human already answered, which the QA critic then scores
            # as incomplete/wrong, causing pointless retries (up to max_retries) before escalating.
            already_approved = state.get("human_decision") == "approved"
            if already_approved:
                parts.append("\n🗑️ DESTRUCTIVE ACTION — ALREADY APPROVED BY USER (human_approval gate passed).")
                parts.append("\nDELETE WORKFLOW (confirmation already granted — DO NOT ask again):")
                parts.append("1. Call list_todos to fetch all tasks for the target day/period")
                parts.append("2. Call complete_todo (or delete_note) for EACH matching task to actually delete it")
                parts.append("3. CONFIRM completion: 'Deleted N task(s) for [day]: [list of what was deleted]'")
                parts.append("\n⚠️ CRITICAL RULES:")
                parts.append("- The user ALREADY confirmed this action. Do NOT ask 'are you sure?' again.")
                parts.append("- Actually CALL the delete/complete tool for each task — do not just describe deleting them.")
                parts.append("- If there are 0 matching tasks, say so plainly instead of asking a question.")
            else:
                parts.append("\n🗑️ DESTRUCTIVE ACTION DETECTED: User wants to clean up/delete tasks")
                parts.append("\nDELETE WORKFLOW (requires confirmation):")
                parts.append("1. FIRST: Call list_todos to fetch all tasks for the target day/period")
                parts.append("2. SHOW the user what will be deleted: List all tasks with their times")
                parts.append("3. ASK FOR CONFIRMATION: 'I found X tasks for Saturday. Are you sure you want to delete ALL of them?'")
                parts.append("4. WAIT for explicit confirmation from user (yes/confirm/delete/proceed)")
                parts.append("5. Only if confirmed: Call complete_todo or delete_note for each task")
                parts.append("6. CONFIRM completion: 'Successfully cleaned up all tasks for Saturday'")
                parts.append("\n⚠️ CRITICAL RULES FOR DELETE:")
                parts.append("- NEVER delete without showing what will be deleted")
                parts.append("- NEVER delete without explicit user confirmation")
                parts.append("- If user hesitates or asks questions, STOP and explain what will happen")
                parts.append("- Confirm EACH deletion action individually or use batch confirmation")
        elif is_conditional_weather:
            # Conditional weather request - inherently ambiguous and requires clarification
            parts.append("\n⚠️ CONDITIONAL WEATHER REQUEST - DO NOT USE TOOLS. Only ask clarifying questions.")
            parts.append("\nConditional scheduling based on weather is ambiguous. Ask the user to clarify:")
            parts.append("- What weather conditions count as 'nice'? (e.g., 'sunny and above 70°F', 'dry', 'warm')")
            parts.append("- What should happen if the weather is NOT nice? (cancel the picnic, reschedule, proceed anyway?)")
            parts.append("- When do you want to schedule this? (specific date, or whenever the weather is right?)")
            parts.append("\nCRITICAL RULES WHEN AMBIGUOUS:")
            parts.append("- DO NOT call add_todo, get_current_weather, or any other tool yet")
            parts.append("- ONLY respond with clarifying questions")
            parts.append("- Once user clarifies, THEN check weather and conditionally schedule")
        elif is_list_query:
            # User is asking to list/show/get tasks - NO scheduling needed, NO time preferences asked
            parts.append("\n📋 LIST QUERY DETECTED: User wants to see/list tasks")
            parts.append("\nLIST WORKFLOW (NO TIME PREFERENCES NEEDED):")
            parts.append("1. Call list_todos to fetch tasks")
            parts.append("2. If user mentioned a specific day (Saturday, today, etc.), filter for that day only")
            parts.append("3. Display results: List each task with its time and status")
            parts.append("4. Format: 'Here are your tasks for [day]: [task 1 at time], [task 2 at time], etc.'")
            parts.append("\n⚠️ CRITICAL RULES FOR LIST:")
            parts.append("- DO NOT ask for time preferences")
            parts.append("- DO NOT ask what kind of event/task this is")
            parts.append("- Just call list_todos and show the results")
            parts.append("- If no date specified, show all tasks (or ask 'which day?')")
        elif ambiguities:
            parts.append("\n⚠️ AMBIGUOUS REQUEST - DO NOT USE TOOLS. Only ask clarifying questions.")
            parts.append("\nAsk the user to clarify:")
            parts.extend(ambiguities)
            parts.append("\nCRITICAL RULES WHEN AMBIGUOUS:")
            parts.append("- DO NOT call add_todo, list_todos, or any other tool")
            parts.append("- DO NOT list existing tasks or ask what to do with them")
            parts.append("- ONLY respond with a natural language question asking for specifics")
            parts.append("- Wait for the user to provide the missing detail before doing anything else")
        elif is_picking_offered_slot:
            # Special case: user is picking a specific time from previously offered availability slots
            # Do NOT check availability again - user has already confirmed they want this time
            parts.append("\n✅ User picked a specific time from offered slots. Schedule immediately - NO MORE AVAILABILITY CHECKS.")
            parts.append("\nDIRECT SCHEDULING (USER CONFIRMED TIME):")
            parts.append("1. SKIP availability checking - user already saw available slots and picked one")
            parts.append("2. Extract time from user's message (e.g., '7PM' → 19:00)")
            parts.append("3. Convert day name to date (Saturday → 2026-09-05)")
            parts.append("4. Combine: 2026-09-05T19:00:00")
            parts.append("5. IMMEDIATELY call add_todo_tool with extracted time and description")
            parts.append("6. CONFIRM: 'Scheduled: Fun, productive, and relaxing activities for Saturday, September 5, 2026 at 7:00 PM'")
            parts.append("\n⚠️ CRITICAL: Do NOT call list_todos again. User has already seen and confirmed availability.")
        elif is_follow_up_to_time_question:
            # Special case: user provided follow-up time clarification (e.g., "morning" in response to "what time?")
            # Check availability in that time window and offer free slots
            parts.append("\n✅ User specified time window. CHECK AVAILABILITY and offer free slots.")
            parts.append("\nAVAILABILITY-AWARE SCHEDULING (MANDATORY STEPS):")
            parts.append("1. MUST CALL: list_todos tool to fetch all scheduled tasks for Saturday (2026-09-05)")
            parts.append("2. MUST INTERPRET the response EXACTLY AS RETURNED:")
            parts.append("   - If tool returns 'No to-do items found' → Entire day is free")
            parts.append("   - If tool returns tasks → Note their exact times from the response")
            parts.append("3. MUST IDENTIFY free slots in the requested window:")
            
            if "morning" in user_text:
                parts.append("   - User requested: MORNING (8 AM, 9 AM, 10 AM, 11 AM)")
                parts.append("   - Compare: Which of these times are NOT listed in list_todos output?")
                parts.append("   - Free slots = all times in morning window NOT mentioned by list_todos")
                parts.append("   - Tell user exactly which times are free and ask to pick one")
            elif "afternoon" in user_text:
                parts.append("   - User requested: AFTERNOON (12 PM, 1 PM, 2 PM, 3 PM, 4 PM, 5 PM)")
                parts.append("   - Compare: Which of these times are NOT listed in list_todos output?")
                parts.append("   - Free slots = all times in afternoon window NOT mentioned by list_todos")
                parts.append("   - Tell user exactly which times are free and ask to pick one")
            elif "evening" in user_text:
                parts.append("   - User requested: EVENING (6 PM, 7 PM, 8 PM, 9 PM)")
                parts.append("   - Compare: Which of these times are NOT listed in list_todos output?")
                parts.append("   - Free slots = all times in evening window NOT mentioned by list_todos")
                parts.append("   - Tell user exactly which times are free and ask to pick one")
            
            parts.append("4. MUST RESPOND WITH: 'I checked your schedule for Saturday (2026-09-05). [REPORT what list_todos returned]. Available slots: [specific times]. Which works best?'")
            parts.append("5. When user picks a time: MUST CALL add_todo with that exact time and date")
            parts.append("6. MUST CONFIRM: 'Scheduled: [description] for Saturday, September 5, 2026 at [specific time]'")
            parts.append("\n⚠️ CRITICAL CONSTRAINTS (VIOLATIONS WILL FAIL QA):")
            parts.append("   ❌ NEVER make up time ranges like 'busy from 12 PM to 11 AM'")
            parts.append("   ❌ NEVER say 'busy' without showing list_todos output confirming it")
            parts.append("   ❌ NEVER offer slots that ARE in list_todos output")
            parts.append("   ❌ NEVER say 'busy' and then 'available slots' in same response (contradictory)")
            parts.append("   ❌ DO NOT proceed without FIRST calling list_todos and showing results")
        else:
            # All details provided - give clear, concise scheduling instructions
            parts.append("\n✅ All required details are clear. Schedule immediately using add_todo_tool.")
            parts.append("\nQUICK INSTRUCTIONS:")
            parts.append("1. Convert day name to date using DATE MAPPING above (e.g., 'Saturday' → 2026-09-07)")
            parts.append("2. Extract time from user's latest message (e.g., '10 AM' → 10:00)")
            parts.append("3. Combine: YYYY-MM-DDTHH:MM:SS (e.g., 2026-09-07T10:00:00)")
            parts.append("4. Call add_todo_tool with this datetime")
            parts.append("5. If task already exists, just CONFIRM it: 'Confirmed: [description] at [date/time]'")
            parts.append("6. If new task created, state: 'Scheduled: [description] for [date/time]'")
            parts.append("7. NO more questions, NO clarifications - just execute the tool call")
            parts.append("8. Do NOT list unrelated tasks from prior turns.")
            
    # On retry after QA failure: attempt scheduling with best-effort interpretation instead of asking again
    # This is a FALLBACK: only used if initial clarification attempt failed QA review
    if state.get("qa_feedback") and state["retry_count"] > 0:
        parts.append(f"\n🔄 RETRY ATTEMPT {state['retry_count']}: Previous feedback indicated we need to proceed differently.")
        parts.append("OVERRIDE: Make best interpretation and schedule - no more clarifications allowed")
        parts.append("- Use DATE MAPPING for day names (e.g., Sunday → 2026-09-07)")
        parts.append("- For vague times, use: morning=09:00, afternoon=14:00, evening=18:00")
        parts.append("- Call add_todo_tool with ISO datetime immediately")
        parts.append("- Preserve the latest request intent. Do not switch from scheduling to listing.")
        parts.append("- This is a fallback only - prefer asking for clarification on first attempt")
        parts.append(f"Previous feedback: {state['qa_feedback'][:100]}")
    elif state.get("qa_feedback"):
        parts.append(f"Previous attempt feedback to address: {state['qa_feedback']}")
    return "\n".join(parts)


def result_to_message(result: SubAgentResult) -> AIMessage:
    """Wrap a sub-agent result as an assistant message for the conversation history."""
    return AIMessage(content=result["output"] or result.get("error") or "")


def find_branch(state: AssistantState, node_id: str) -> ThoughtBranch:
    """Look up a ThoughtBranch by id from the accumulated thoughts list."""
    for branch in state["thoughts"]:
        if branch["node_id"] == node_id:
            return branch
    raise KeyError(f"No thought branch found with node_id={node_id}")
