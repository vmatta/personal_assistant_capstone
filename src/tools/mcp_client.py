"""Scoped MCP-style tool layer (Guardrail 4): each domain agent only ever
receives the specific tools registered in its scope, never the full toolset.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from crewai.tools import tool

from src.tools import external, memory


USER_TIMEZONE = ZoneInfo("America/New_York")


def _format_due(due: str | None) -> str:
    if not due:
        return "no due time"
    try:
        due_dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
    except ValueError:
        return due
    if due_dt.tzinfo:
        due_dt = due_dt.astimezone(USER_TIMEZONE)
    return f"{due_dt.strftime('%Y-%m-%d %I:%M %p').lstrip('0')} ET"


def _extract_day_of_week(due: str | None) -> str | None:
    """Extract day of week name from ISO datetime string. Returns 'Monday', 'Tuesday', etc. or None."""
    if not due:
        return None
    try:
        due_dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
        return due_dt.strftime("%A")
    except ValueError:
        return None


def _check_preference_conflicts(description: str, due: str | None, day_of_week: str | None) -> dict | None:
    """Check if the scheduled item conflicts with stored user preferences.
    
    Only returns TRUE conflicts for VERY clear cases:
    - Preference must explicitly mention the target day
    - Preference must use restrictive language ("only", "just")
    - Scheduled activity must contradict the preference
    
    Example: If saved note is "On Sundays, I only relax at home" and user tries 
    to schedule "Go shopping" on Sunday, this returns a conflict.
    
    Returns dict with 'conflict' (bool), 'preference' (str), and 'clarification' (str),
    or None if no conflict detected.
    """
    if not day_of_week:
        return None
    
    # Search for preferences mentioning this day - use exact day name search
    day_lower = day_of_week.lower()
    preferences = memory.search_notes(f"on {day_lower}", k=3)
    
    if not preferences:
        return None
    
    # Check if any preference explicitly contradicts this scheduling request
    # Only flag if BOTH conditions are true:
    # 1. Preference explicitly mentions the day AND uses "only" or "just"
    # 2. Scheduled activity is NOT a relaxation activity
    
    relaxation_activities = {"sleep", "rest", "relax", "nap", "dream"}
    is_relaxing = any(word in description.lower() for word in relaxation_activities)
    
    for pref in preferences:
        pref_lower = pref.lower()
        
        # STRICT check: preference must contain day name AND a restriction word
        has_day = day_lower in pref_lower
        has_restriction = any(word in pref_lower for word in ["only", "just"])
        has_relaxation_constraint = any(word in pref_lower for word in ["relax", "rest", "sleep", "leisure"])
        
        # Only flag as conflict if ALL three conditions are met:
        # - Preference mentions the day
        # - Preference has restrictive language
        # - Preference is about relaxation
        # - User is trying to schedule a NON-relaxing activity
        if has_day and has_restriction and has_relaxation_constraint and not is_relaxing:
            # This is a genuine conflict - preference says "only relax" on that day
            return {
                "conflict": True,
                "preference": pref,
                "day": day_of_week,
                "clarification": (
                    f"I found a preference: \"{pref}\". \n"
                    f"You want to schedule: \"{description}\" on {day_of_week}. \n"
                    f"Does this align with your preference, or would you like to reschedule to a different day?"
                ),
            }
    
    return None


@tool("add_todo")
def add_todo_tool(description: str, due: str = "") -> str:
    """Add a to-do or scheduled item. `due` is an optional ISO date/time string.
    
    Checks for:
    1. Schedule conflicts (overlapping times)
    2. Preference conflicts (e.g., "relax on Sundays")
    
    Returns a clarification question if preference conflict detected.
    """
    # Check for time-based conflicts (existing to-dos at same time)
    conflicts = memory.find_open_todos_at_due(due or None)
    if conflicts:
        existing = "; ".join(f"{item['description']} (id: {item['id']}, due: {item['due']})" for item in conflicts)
        return f"Cannot add this item because you already have an open item at that time: {existing}."
    
    # Check for preference conflicts (e.g., "I relax on Sundays")
    day_of_week = _extract_day_of_week(due or None)
    preference_conflict = _check_preference_conflicts(description, due or None, day_of_week)
    
    if preference_conflict and preference_conflict.get("conflict"):
        # Return clarification question instead of adding
        return preference_conflict["clarification"]
    
    # No conflicts detected; proceed with adding
    item = memory.add_todo(description, due or None)
    return f"Added to-do {item['id']}: {item['description']}"


@tool("list_todos")
def list_todos_tool(include_done: bool = False) -> str:
    """List current to-do/scheduled items."""
    todos = memory.list_todos(include_done=include_done)
    if not todos:
        return "No to-do items found."
    return "\n".join(f"- [{t['id']}] {t['description']} (due: {_format_due(t.get('due'))})" for t in todos)


@tool("complete_todo")
def complete_todo_tool(todo_id: str) -> str:
    """Mark a to-do item as complete by id."""
    ok = memory.complete_todo(todo_id)
    return f"Marked {todo_id} complete." if ok else f"No to-do found with id {todo_id}."


@tool("save_note")
def save_note_tool(text: str) -> str:
    """Save a note to long-term memory for later semantic retrieval."""
    note_id = memory.save_note(text)
    if note_id is None:
        return "This is already in there."
    return f"Saved note {note_id}."


@tool("search_notes")
def search_notes_tool(query: str, k: int = 3) -> str:
    """Semantically search previously saved notes."""
    matches = memory.search_notes(query, k=k)
    return "\n".join(matches) if matches else "No matching notes found."


@tool("search_knowledge_base")
def search_knowledge_base_tool(query: str, k: int = 3) -> str:
    """Semantically search the general knowledge base for grounding context."""
    matches = memory.search_knowledge_base(query, k=k)
    return "\n".join(matches) if matches else "No matching knowledge base entries found."


@tool("delete_note")
def delete_note_tool(note_id: str) -> str:
    """Delete a saved note by its ID. Requires human approval."""
    ok = memory.delete_note(note_id)
    return f"Deleted note {note_id}." if ok else f"No note found with ID {note_id}."


@tool("get_current_weather")
def get_current_weather_tool(location: str) -> str:
    """Get current weather conditions for a named location."""
    try:
        data = external.get_current_weather(location)
    except external.AmbiguousLocationError as exc:
        if exc.suggestions:
            options = ", ".join(exc.suggestions)
            return f"{location} covers a large area. Which city did you mean? For example: {options}."
        return f"{location} is not specific enough. Could you name a city instead?"
    return (
        f"Weather in {data['location']}: {data['condition']}, {data['temperature_c']}C, "
        f"humidity {data['humidity_pct']}%, precipitation {data['precipitation_mm']}mm, "
        f"wind {data['windspeed_kmh']} km/h. Source: {data['source']}, as of {data['observed_at']}."
    )


# Guardrail 4: permission allowlist per CrewAI domain agent.
AGENT_TOOL_SCOPES = {
    "scheduler_todo": [add_todo_tool, list_todos_tool, complete_todo_tool],
    "notes_knowledge": [save_note_tool, search_notes_tool, search_knowledge_base_tool, delete_note_tool],
    "weather_external": [get_current_weather_tool],
}


def get_tools_for_agent(agent_name: str) -> list:
    """Return the permitted tool set for a given domain agent."""
    if agent_name not in AGENT_TOOL_SCOPES:
        raise ValueError(f"Unknown agent '{agent_name}' has no registered tool scope.")
    return AGENT_TOOL_SCOPES[agent_name]
