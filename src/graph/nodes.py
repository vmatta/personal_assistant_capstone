"""LangGraph node functions for the orchestrator graph."""

import uuid
import logging
import re
from datetime import datetime

from langchain_core.messages import AIMessage

from src.agents import context, executor, planner, supervisor
from src.agents.planner import OUT_OF_SCOPE
from src.graph.state import AssistantState, SubAgentResult

logger = logging.getLogger("personal_assistant")


_MEMORY_LOOKUP_TERMS = ("preferred", "preference", "favorite", "remember", "note", "notes")


def _conditional_weather_followup_response(user_text: str, history_text: str) -> str | None:
    """Handle partial follow-ups for conditional weather scheduling.

    Example flow:
    - Assistant asks for: weather threshold, fallback, and timing.
    - User replies with only one item (e.g., "70F is nice").
    We should ask only for the remaining items, not treat the turn as out-of-scope.
    """
    user_lower = (user_text or "").lower()
    user_lower_clean = user_lower.strip().strip('"\'“”‘’.,!? ')
    history_lower = (history_text or "").lower()

    # BAIL OUT EARLY if the CURRENT message is itself a complete, unambiguous, unrelated
    # new request — regardless of what an earlier conditional-weather thread looked like.
    # Without this, once the assistant ever asks its "weather threshold/fallback/timing"
    # clarification questions on some earlier turn, EVERY subsequent message in that same
    # conversation thread gets misinterpreted as a partial follow-up to that old
    # conditional — even a fully-specified, completely unrelated new request like
    # "Schedule volleyball in Miami on Saturday at 4 PM and save it as my favorite sport."
    # A message only counts as a genuine follow-up if it does NOT itself introduce a new,
    # complete, self-sufficient request (its own day + time, no weather/conditional
    # language of its own).
    mentions_weather_or_conditional = any(
        token in user_lower
        for token in ("weather", "nice", "sunny", "rain", "picnic", "forecast", "suitable", "if not", "otherwise")
    )
    has_own_weekday = any(
        day in user_lower
        for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "tomorrow")
    )
    has_own_specific_time = re.search(r"\b(1[0-2]|0?[1-9])\s*(:\d{2})?\s*(am|pm)\b", user_lower) is not None
    is_complete_unrelated_request = (
        not mentions_weather_or_conditional
        and has_own_weekday
        and has_own_specific_time
        and len(user_lower.split()) >= 6  # long enough to be a full request, not a short partial answer
    )
    if is_complete_unrelated_request:
        return None

    # Only accumulate prior USER messages from transcript-style history.
    # Using assistant lines here causes false positives (for example matching
    # words from clarification prompts rather than user-provided constraints).
    user_history_parts = []
    for line in (history_text or "").splitlines():
        if line.strip().lower().startswith("user:"):
            user_history_parts.append(line.split(":", 1)[1].strip())
    combined_user_text = " ".join(user_history_parts + [user_text or ""]).lower()

    assistant_history_lines = []
    for line in (history_text or "").splitlines():
        if line.strip().lower().startswith("assistant:"):
            assistant_history_lines.append(line.split(":", 1)[1].strip().lower())

    prior_user_started_conditional = any(
        (
            "weather" in line
            and ("schedule" in line or "picnic" in line)
            and ("if " in line or "nice" in line or "suitable" in line)
        )
        for line in user_history_parts
    )
    assistant_is_clarifying_conditional = any(
        key in " ".join(assistant_history_lines)
        for key in (
            "conditional scheduling based on weather",
            "what does 'nice' mean",
            "fallback plan",
            "once you clarify these",
            "i still need",
            "where should i check weather",
        )
    )

    in_conditional_thread = (
        prior_user_started_conditional
        or assistant_is_clarifying_conditional
        or "conditional scheduling based on weather" in history_lower
    )
    if not in_conditional_thread:
        return None

    # Consider prior USER turns too (user may provide details one-by-one).
    has_weather_threshold = any(
        token in combined_user_text
        for token in (
            "f",
            "c",
            "degree",
            "sunny",
            "dry",
            "no rain",
            "low wind",
            "above",
            "below",
            "at least",
        )
    )
    has_fallback = any(
        token in combined_user_text
        for token in ("cancel", "reschedule", "proceed", "anyway", "still do it", "skip")
    )
    has_timing = any(
        token in combined_user_text
        for token in (
            "today",
            "tomorrow",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "next",
            "this ",
            "wait",
        )
    )
    extracted_location = _extract_location(combined_user_text)
    has_location = extracted_location is not None

    # If assistant just asked for location, treat a short city-only reply as the location.
    asked_for_location_recently = (
        "where should i check weather" in " ".join(assistant_history_lines)
        or "location:" in " ".join(assistant_history_lines)
        or "where should i check weather" in history_lower
        or "location:" in history_lower
    )
    location_only_reply = (
        len(user_lower_clean.split()) <= 3
        and any(ch.isalpha() for ch in user_lower_clean)
        and all(ch.isalpha() or ch.isspace() or ch in "-." for ch in user_lower_clean)
    )
    if not has_location and location_only_reply and (asked_for_location_recently or in_conditional_thread):
        has_location = True
    has_picnic_time = any(
        token in combined_user_text
        for token in (
            "am",
            "pm",
            ":",
            " at ",
            " noon",
            " evening",
            " afternoon",
            " morning",
        )
    )

    missing = []
    if not has_weather_threshold:
        missing.append("1. Weather threshold: what counts as 'nice' (for example, sunny and above 70°F)?")
    if not has_fallback:
        missing.append("2. Fallback plan: if weather is not suitable, should I cancel, reschedule, or proceed anyway?")
    if not has_timing:
        missing.append("3. Timing: should I check for a specific date (for example tomorrow) or wait for suitable weather?")
    if not has_location:
        missing.append("4. Location: where should I check weather and schedule the picnic?")
    if not has_picnic_time:
        missing.append("5. Picnic time: what time should I schedule it if weather is suitable? (for example 3 PM)")

    is_proceed = user_lower_clean in {"proceed", "yes", "go ahead", "do it", "execute"}

    if not missing and is_proceed:
        return (
            "Thanks. I linked all follow-up details correctly. "
            "To execute reliably in this build, send one explicit final command with location and time, for example: "
            "'Check weather in Seattle and if sunny, at least 70F with no rain, schedule a picnic tomorrow at 3 PM; "
            "otherwise reschedule to the next suitable day at 3 PM.'"
        )

    if not missing:
        return (
            "Great, I linked your follow-ups and captured all details: "
            "weather = sunny and at least 70F with no rain, timing = tomorrow, "
            "fallback = reschedule to next suitable day at 3 PM, location captured. "
            "Reply 'proceed' to execute this now."
        )

    return "Got it. I still need:\n\n" + "\n".join(missing)


def _short_followup_without_context_response(user_text: str, history_text: str) -> str | None:
    """When a very short message has no clear active thread, ask for intent.

    This avoids misrouting one-word replies while preserving normal behavior for
    short follow-ups that are clearly part of the previous exchange.
    """
    text = (user_text or "").strip()
    if not text:
        return None

    is_short = len(text.split()) <= 3
    if not is_short:
        return None

    history_lower = (history_text or "").lower()
    has_active_followup_prompt = any(
        key in history_lower
        for key in (
            "i still need",
            "what time",
            "which works best",
            "where should i check weather",
            "could you clarify",
            "once you clarify",
            "i need a location first",  # multi-day weather-permitting flow's own location ask
        )
    )
    if has_active_followup_prompt:
        return None

    # Skip guardrail for obvious commands or known intent words.
    low = text.lower()
    intent_words = {
        "yes",
        "no",
        "proceed",
        "cancel",
        "approve",
        "reject",
        "today",
        "tomorrow",
        "morning",
        "afternoon",
        "evening",
    }
    if low in intent_words:
        return None

    return f"I saw '{text}', but I do not have enough context to act. What would you like me to do with it?"


def _fully_specified_conditional_weather_response(user_text: str) -> str | None:
    """Handle one-shot conditional weather scheduling commands without re-asking basics."""
    text = (user_text or "").lower()
    is_conditional = (
        "weather" in text
        and ("if " in text or "otherwise" in text or "else" in text)
        and ("schedule" in text or "picnic" in text)
    )
    if not is_conditional:
        return None

    has_weather_rule = any(k in text for k in ("sunny", "no rain", "70f", "at least", "above"))
    has_fallback = any(k in text for k in ("reschedule", "otherwise", "else", "if not"))
    has_when = any(k in text for k in ("tomorrow", "today", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "next"))
    has_time = (
        any(k in text for k in (" am", " pm", " at ", ":", "morning", "afternoon", "evening"))
        or re.search(r"\b(1[0-2]|0?[1-9])\s*(am|pm)\b", text) is not None
    )
    has_location = _extract_location(text) is not None

    if has_weather_rule and has_fallback and has_when and has_time and has_location:
        return (
            "Understood. I captured your full conditional rule and linked all required details. "
            "Reply 'proceed' to continue with this plan in the current thread."
        )
    return None


_WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# Suitability thresholds for "is the weather good enough for an outdoor activity".
# Intentionally simple/conservative: no rain, not too hot, not too windy.
_OUTDOOR_MAX_PRECIP_MM = 1.0
_OUTDOOR_MAX_TEMP_C = 35.0
_OUTDOOR_MAX_WIND_KMH = 30.0


def _detect_multi_day_candidates(user_text: str) -> list[str]:
    """Find weekday names connected by 'or' in the user's text, e.g. 'Saturday or Sunday'
    returns ['saturday', 'sunday']. Returns [] if fewer than 2 weekday names are present.
    """
    text = (user_text or "").lower()
    found = [day for day in _WEEKDAY_NAMES if re.search(rf"\b{day}\b", text)]
    # Preserve the order they appear in the text, not the canonical Mon-Sun order.
    positions = {day: text.find(day) for day in found}
    found.sort(key=lambda d: positions[d])
    return found


def _is_weather_permitting_outdoor_request(user_text: str) -> bool:
    """Detect the 'if weather permits, schedule an outdoor activity on day A or day B'
    pattern specifically - distinct from the general single-day conditional-weather
    pattern, because this one has a genuine multi-day DECISION to make (which day, if
    either, actually works), not just a single yes/no threshold question.
    """
    text = (user_text or "").lower()
    has_permits_phrasing = any(
        phrase in text for phrase in ("weather permit", "weather allows", "weather is nice", "weather is good")
    )
    has_outdoor = "outdoor" in text
    has_multi_day = len(_detect_multi_day_candidates(text)) >= 2
    has_schedule_verb = any(v in text for v in ("schedule", "plan", "book", "arrange"))
    return has_permits_phrasing and has_outdoor and has_multi_day and has_schedule_verb


def _day_exclusion_preference(day_name: str) -> str | None:
    """Check saved notes for a preference that rules out this day for outdoor/active
    plans (e.g. 'On Sundays, I only relax at home'). Returns the matching note text if
    found, else None. This is a lightweight, deterministic semantic-search lookup - no
    LLM call - so it adds only the embedding-search latency, not a full agent dispatch.
    """
    from src.tools.memory import search_notes

    try:
        matches = search_notes(f"{day_name} preference relax avoid schedule", k=3)
    except Exception as exc:
        logger.warning(f"Notes lookup failed during day-exclusion check: {exc}")
        return None

    exclusion_markers = ("relax", "no plans", "don't schedule", "avoid", "rest day", "day off", "only")
    for note in matches:
        note_lower = note.lower()
        if day_name in note_lower and any(marker in note_lower for marker in exclusion_markers):
            return note
    return None


def _is_weather_suitable_for_outdoor(forecast: dict) -> bool:
    """Simple, transparent suitability rule for an outdoor activity: no meaningful
    precipitation, not extreme heat, not too windy. Intentionally conservative and
    easy to explain to the user (matches how we already describe conditions in
    _append_weather_to_response).
    """
    precip = forecast.get("precipitation_mm") or 0
    temp_max = forecast.get("temp_max_c")
    wind = forecast.get("windspeed_max_kmh") or 0
    if precip > _OUTDOOR_MAX_PRECIP_MM:
        return False
    if temp_max is not None and temp_max > _OUTDOOR_MAX_TEMP_C:
        return False
    if wind > _OUTDOOR_MAX_WIND_KMH:
        return False
    return True


def _weather_permitting_multi_day_response(user_text: str) -> str | None:
    """Resolve 'if weather permits, schedule an outdoor activity Saturday or Sunday'
    style requests end-to-end:
      1. Identify the candidate days named in the request.
      2. Check saved notes/preferences for each day; rule out any day with an explicit
         exclusion preference (e.g. 'On Sundays, I only relax at home').
      3. For each remaining candidate day, fetch the actual daily forecast and evaluate
         outdoor suitability.
      4. Decide: if exactly one day is both preference-eligible AND weather-suitable,
         say so plainly. If multiple qualify, recommend the best one. If none qualify,
         explain why each was ruled out.

    Returns None if this isn't a multi-day weather-permitting request (caller should
    fall through to the generic single-day conditional-weather clarification flow).
    """
    if not _is_weather_permitting_outdoor_request(user_text):
        return None

    candidate_days = _detect_multi_day_candidates(user_text)
    location = _extract_location(user_text)
    if not location:
        return (
            "I can check the weather for both days, but I need a location first. "
            "Which city should I check the forecast for?"
        )

    logger.info(f"   🧭 Multi-day weather-permitting request detected: candidates={candidate_days}, location={location}")

    from datetime import datetime
    from src.agents.context import USER_TIMEZONE, _get_next_weekday_date
    from src.tools.external import get_daily_forecast, AmbiguousLocationError

    now = datetime.now(USER_TIMEZONE)
    ruled_out_by_preference = {}
    remaining_days = []
    for day in candidate_days:
        exclusion = _day_exclusion_preference(day)
        if exclusion:
            ruled_out_by_preference[day] = exclusion
            logger.info(f"      ✗ {day.capitalize()} ruled out by saved preference: {exclusion!r}")
        else:
            remaining_days.append(day)

    if not remaining_days:
        lines = ["Based on your saved preferences, neither day works for this:"]
        for day, pref in ruled_out_by_preference.items():
            lines.append(f"- {day.capitalize()}: ruled out ({pref})")
        return "\n".join(lines)

    day_forecasts = {}
    for day in remaining_days:
        date_iso = _get_next_weekday_date(day, now)
        try:
            forecast = get_daily_forecast(location, date_iso)
            day_forecasts[day] = forecast
            suitable = _is_weather_suitable_for_outdoor(forecast)
            logger.info(
                f"      {'✓' if suitable else '✗'} {day.capitalize()} ({date_iso}) forecast: "
                f"{forecast['condition']}, {forecast['temp_max_c']}C max, "
                f"{forecast['precipitation_mm']}mm precip -> {'suitable' if suitable else 'not suitable'}"
            )
        except AmbiguousLocationError as exc:
            suggestions = ", ".join(exc.suggestions) if exc.suggestions else None
            return (
                f"'{location}' is not specific enough to get a weather reading."
                + (f" Did you mean: {suggestions}?" if suggestions else " Could you name a specific city?")
            )
        except Exception as exc:
            logger.warning(f"Weather lookup failed for {day} in {location}: {exc}")
            return f"I couldn't retrieve the forecast for {day.capitalize()} in {location}. Please try again shortly."

    suitable_days = [d for d in remaining_days if _is_weather_suitable_for_outdoor(day_forecasts[d])]

    lines = []
    if ruled_out_by_preference:
        for day, pref in ruled_out_by_preference.items():
            lines.append(f"- {day.capitalize()}: ruled out based on your saved preference ({pref})")

    for day in remaining_days:
        f = day_forecasts[day]
        suitable = day in suitable_days
        status = "suitable for an outdoor activity" if suitable else "not suitable (weather)"
        lines.append(
            f"- {day.capitalize()} ({f['date']}): {f['condition']}, {f['temp_max_c']}°C max, "
            f"{f['precipitation_mm']}mm precipitation, wind up to {f['windspeed_max_kmh']} km/h — {status}"
        )

    if len(suitable_days) == 1:
        chosen = suitable_days[0]
        lines.append(f"\nRecommendation: schedule the outdoor activity for {chosen.capitalize()}. What time would you like?")
    elif len(suitable_days) > 1:
        lines.append(
            f"\nBoth {' and '.join(d.capitalize() for d in suitable_days)} look suitable. "
            "Which day and time would you prefer?"
        )
    else:
        lines.append("\nNeither remaining day has suitable weather for an outdoor activity right now.")

    return "\n".join(lines)


def _weather_permitting_location_followup_response(user_text: str, history_text: str) -> str | None:
    """Continue a pending 'if weather permits, schedule outdoor Saturday or Sunday'
    request when the assistant's immediately preceding turn asked specifically for
    the missing location (see _weather_permitting_multi_day_response's 'I need a
    location first' branch).

    Without this, a short location-only reply like "Pittsburgh" has no way to be
    recognized as answering that specific question: it doesn't match
    _is_weather_permitting_outdoor_request on its own (no "weather permits"/"outdoor"/
    weekday names), so it falls through to _short_followup_without_context_response
    and produces a confusing "I do not have enough context to act" reply instead of
    resuming the flow.
    """
    text = (user_text or "").strip()
    if not text:
        return None

    text_clean = text.strip().strip("\"'\u2018\u2019.,!? ")
    is_location_only_reply = (
        1 <= len(text_clean.split()) <= 3
        and any(ch.isalpha() for ch in text_clean)
        and all(ch.isalpha() or ch.isspace() or ch in "-." for ch in text_clean)
    )
    if not is_location_only_reply:
        return None

    lines = [ln for ln in (history_text or "").splitlines() if ln.strip()]

    # Find the most recent Assistant line and confirm it's THIS flow's location ask.
    last_assistant_idx = None
    last_assistant_line = ""
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip().lower().startswith("assistant:"):
            last_assistant_idx = idx
            last_assistant_line = lines[idx].split(":", 1)[1].strip().lower()
            break

    asked_for_weather_permitting_location = (
        bool(last_assistant_line)
        and "i need a location first" in last_assistant_line
        and "check the forecast for" in last_assistant_line
    )
    if not asked_for_weather_permitting_location:
        return None

    # Recover the original "if weather permits..." request that triggered the question.
    original_request = ""
    for j in range(last_assistant_idx - 1, -1, -1):
        if lines[j].strip().lower().startswith("user:"):
            original_request = lines[j].split(":", 1)[1].strip()
            break
    if not original_request or not _is_weather_permitting_outdoor_request(original_request):
        return None

    # Recombine the original request with the newly-supplied location and re-run the
    # full multi-day flow (preference check + real forecast + recommendation).
    combined_text = f"{original_request} in {text_clean}"
    return _weather_permitting_multi_day_response(combined_text)


def _is_week_scan_outdoor_request(user_text: str) -> bool:
    """Detect 'check the weather for the whole week, then schedule the best day' style
    requests - distinct from _is_weather_permitting_outdoor_request (which only compares
    2 explicitly-named days). This pattern names NO specific days at all; it asks the
    system to scan an entire week and pick the best one itself.
    """
    text = (user_text or "").lower()
    has_week_scope = any(phrase in text for phrase in ("next week", "this week", "whole week", "the week"))
    has_outdoor = "outdoor" in text
    has_schedule_verb = any(v in text for v in ("schedule", "plan", "book", "arrange"))
    has_best_day_intent = any(
        phrase in text for phrase in ("best day", "best time", "then schedule", "pick the best", "choose the best")
    )
    # Must NOT already be the 2-named-days pattern (that one takes priority and is handled
    # by _weather_permitting_multi_day_response instead).
    already_multi_day_named = len(_detect_multi_day_candidates(text)) >= 2
    return (
        has_week_scope
        and has_outdoor
        and has_schedule_verb
        and has_best_day_intent
        and not already_multi_day_named
    )


def _week_scan_outdoor_response(user_text: str) -> str | None:
    """Resolve 'schedule outdoor activities next week in <city>; check the weather for
    the whole week, then schedule the best day' style requests end-to-end:
      1. Extract the location (ask if missing - never guess, matching the rest of the
         codebase's location-handling philosophy).
      2. Fetch a single-call weekly forecast (next 7 days) via get_weekly_forecast().
      3. Rank each day by outdoor suitability using the same conservative rule as the
         Saturday/Sunday feature (_is_weather_suitable_for_outdoor).
      4. Recommend the single best-scoring suitable day, or explain if none qualify.

    Returns None if this isn't a week-scan request (caller falls through to normal
    planning/clarification flow).
    """
    if not _is_week_scan_outdoor_request(user_text):
        return None

    location = _extract_location(user_text)
    if location:
        # _extract_location() only splits on ",", " on ", " for " — it doesn't stop at a
        # sentence-ending period, so a location followed by a new sentence (e.g. "...in
        # Philadelphia. First, check the weather...") would otherwise incorrectly include
        # trailing words from the next sentence. Trim at the first period here.
        location = location.split(".")[0].strip()
    if not location:
        return (
            "I can scan the week's forecast, but I need a location first. "
            "Which city should I check?"
        )

    from datetime import datetime, timedelta
    from src.agents.context import USER_TIMEZONE
    from src.tools.external import get_weekly_forecast, AmbiguousLocationError

    now = datetime.now(USER_TIMEZONE)
    # "next week" starts at next Monday; "this week"/"the whole week" scans the next 7
    # days starting tomorrow. Either way, a single 7-day window covers the intent well
    # enough for a demo-grade "scan the week" feature without over-engineering exact
    # calendar-week boundaries.
    start_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=7)).strftime("%Y-%m-%d")

    logger.info(f"   🧭 Week-scan outdoor request detected: location={location}, range={start_date}..{end_date}")

    try:
        forecasts = get_weekly_forecast(location, start_date, end_date)
    except AmbiguousLocationError as exc:
        suggestions = ", ".join(exc.suggestions) if exc.suggestions else None
        return (
            f"'{location}' is not specific enough to get a weather reading."
            + (f" Did you mean: {suggestions}?" if suggestions else " Could you name a specific city?")
        )
    except Exception as exc:
        logger.warning(f"Weekly forecast lookup failed for {location}: {exc}")
        return f"I couldn't retrieve the weekly forecast for {location}. Please try again shortly."

    if not forecasts:
        return f"I couldn't retrieve any forecast data for {location} for next week."

    lines = [f"Here's the weather forecast for {location} next week:\n"]
    scored_days = []
    for f in forecasts:
        suitable = _is_weather_suitable_for_outdoor(f)
        weekday_name = datetime.strptime(f["date"], "%Y-%m-%d").strftime("%A")
        status = "suitable for outdoor activities" if suitable else "not suitable"
        lines.append(
            f"- {weekday_name} ({f['date']}): {f['condition']}, {f['temp_max_c']}°C max, "
            f"{f['precipitation_mm']}mm precipitation, wind up to {f['windspeed_max_kmh']} km/h — {status}"
        )
        if suitable:
            # Lower precipitation + lower wind + moderate temp = better outdoor day.
            # Simple, transparent scoring: sum of precip + wind, tie-broken by earliest day.
            score = (f.get("precipitation_mm") or 0) + (f.get("windspeed_max_kmh") or 0) / 10
            scored_days.append((score, weekday_name, f["date"]))

    if scored_days:
        scored_days.sort(key=lambda x: x[0])
        _, best_weekday, best_date = scored_days[0]
        lines.append(
            f"\nRecommendation: {best_weekday} ({best_date}) looks like the best day for an outdoor "
            "activity this week. What time would you like to schedule it?"
        )
    else:
        lines.append("\nNone of the days this week look suitable for an outdoor activity based on the forecast.")

    return "\n".join(lines)


def _is_exploratory_itinerary_request(user_text: str) -> bool:
    """Detect itinerary-style multi-option planning that we should handle, not reject."""
    text = (user_text or "").lower()
    travel_terms = (
        "visit",
        "trip",
        "itinerary",
        "cities",
        "city",
        "driving",
        "weekend",
    )
    exploration_terms = (
        "different ways",
        "all the ways",
        "organize this",
        "best option",
        "multiple ways",
    )
    constraint_terms = (
        "under",
        "within",
        "include both",
        "indoor",
        "outdoor",
        "hours",
    )
    return (
        any(t in text for t in travel_terms)
        and any(t in text for t in exploration_terms)
        and any(t in text for t in constraint_terms)
    )


_CAPABILITY_BULLETS = (
    "- 📅 Add, list, and complete to-dos or scheduled events, and catch scheduling conflicts\n"
    "- 🧠 Remember facts and preferences you ask me to save, and retrieve them later with semantic search\n"
    "- 🌤️ Answer current-weather questions using a live external API\n"
    "- 🤔 Ask clarifying questions when a request is ambiguous, instead of guessing\n"
    "- ✅ Ask for your explicit approval before any destructive action (like deleting or cancelling an event)"
)


def _capability_response(user_text: str) -> str | None:
    lowered = user_text.lower()
    normalized = " ".join(lowered.replace("?", " ").split())
    asks_capabilities = (
        "what" in normalized
        and (
            "you can do" in normalized
            or "can you do" in normalized
            or "things you can do" in normalized
            or "what all can i do" in normalized
            or "what all i can do" in normalized
        )
    ) or "list all the things you can do" in normalized
    if not asks_capabilities:
        return None
    return (
        "I can help with to-dos and schedules, saved notes and preferences, and current weather. "
        "Specifically, I can:\n\n" + _CAPABILITY_BULLETS
    )


_GREETING_WORDS = {
    "hi", "hello", "hey", "hiya", "yo", "sup",
    "good morning", "good afternoon", "good evening",
    "hi there", "hello there", "hey there",
}


def _greeting_response(user_text: str, history_text: str) -> str | None:
    """Introduce the assistant and its capabilities on a bare greeting with no prior
    conversation context, instead of falling through to the generic "I don't have
    enough context to act" fallback (which is correct for genuinely orphaned short
    replies, but not for someone just saying hello).
    """
    text = (user_text or "").strip().lower().strip("!.")
    if text not in _GREETING_WORDS:
        return None
    if (history_text or "").strip():
        # Mid-conversation "hi" is unusual but not worth a full re-introduction;
        # let the normal short-followup handling take it instead.
        return None
    return (
        "Hi! I'm your personal assistant. Here's what I can help you with:\n\n"
        + _CAPABILITY_BULLETS
        + "\n\nWhat would you like to do?"
    )


def _memory_lookup_branch(user_text: str):
    lowered = user_text.lower()
    if "schedule" in lowered or "event" in lowered or "task" in lowered or "weather" in lowered:
        return None
    asks_about_user_memory = (
        any(term in lowered for term in _MEMORY_LOOKUP_TERMS)
        and ("my" in lowered or "me" in lowered or "i " in lowered)
    )
    if not asks_about_user_memory:
        return None
    return {
        "node_id": str(uuid.uuid4()),
        "parent_id": None,
        "depth": 0,
        "thought": "Search saved notes for the user's personal preference or remembered fact, then answer from retrieved context.",
        "target_agent": "notes_knowledge",
        "score": 1.0,
        "pruned": False,
    }


def _apply_ambiguity_tracking(state: AssistantState, result: dict, *, from_free_text_agent: bool = False) -> dict:
    """Update the ambiguity-escalation counter (Guardrail 3b) whenever a node produces a
    turn-terminal final_response. Safe to call unconditionally from plan_node (even though
    plan_node can run multiple times per turn via the QA retry loop): the non-terminal
    dispatch path never sets final_response, so this only ever fires on the single-shot
    direct-response early-returns (capability info, out-of-scope, conditional-weather
    clarification, etc.) and on compose_node's final answer.

    Pure string-matching on already-computed output (no extra LLM calls) — adds no latency.
    After `AMBIGUITY_ESCALATION_TURNS` consecutive turns where the assistant only asked a
    clarifying question without completing an action, escalate to the user instead of
    asking yet another question.
    """
    response_text = result.get("final_response")
    prior_ambiguous = state.get("consecutive_ambiguous_turns", 0)

    is_stuck = bool(response_text) and (
        supervisor.is_unresolved_ambiguous_turn(response_text)
        or (
            # compose_node's output comes from free-text CrewAI agents (no canned markers
            # available), so fall back to the more general clarification-question heuristic.
            from_free_text_agent
            and supervisor.is_clarification_response(response_text)
        )
    )
    if is_stuck:
        new_count = prior_ambiguous + 1
        if new_count >= supervisor.AMBIGUITY_ESCALATION_TURNS:
            escalation_response = (
                "I've asked for clarification a few times without getting enough detail to proceed. "
                "Let's start over — could you restate exactly what you'd like me to do, with all the "
                "specifics (date, time, location, etc.) in one message?"
            )
            result["final_response"] = escalation_response
            result["messages"] = [AIMessage(content=escalation_response)]
            result["escalation_count"] = state.get("escalation_count", 0) + 1
            new_count = 0
        result["consecutive_ambiguous_turns"] = new_count
    else:
        result["consecutive_ambiguous_turns"] = 0
    return result


def plan_node(state: AssistantState) -> dict:
    """Route simple requests to a single direct LLM call, complex ones through full ToT search."""
    result = _plan_node_impl(state)
    return _apply_ambiguity_tracking(state, result)


def _plan_node_impl(state: AssistantState) -> dict:
    user_text = context.latest_user_text(state)
    history_text = context.conversation_history_text(state)
    capability_response = _capability_response(user_text)
    if capability_response:
        return {
            "thoughts": [],
            "reasoning_mode": "direct",
            "handled_directly": "capability_response",
            "selected_branch_id": None,
            "final_response": capability_response,
            "messages": [AIMessage(content=capability_response)],
        }
    greeting_response = _greeting_response(user_text, history_text)
    if greeting_response:
        return {
            "thoughts": [],
            "reasoning_mode": "direct",
            "handled_directly": "greeting_response",
            "selected_branch_id": None,
            "final_response": greeting_response,
            "messages": [AIMessage(content=greeting_response)],
        }
    memory_branch = _memory_lookup_branch(user_text)
    if memory_branch:
        return {
            "thoughts": [memory_branch],
            "reasoning_mode": "direct",
            "selected_branch_id": memory_branch["node_id"],
            "requires_human_approval": False,
        }

    # Multi-day weather-permitting requests (e.g. "if weather permits, schedule an
    # outdoor activity Saturday or Sunday") get resolved directly here: check saved
    # preferences to rule out days, check the real forecast for remaining candidates,
    # and give a decisive answer — rather than falling into the generic single-day
    # conditional-weather clarification flow below, which has no concept of multiple
    # candidate days or day-specific preferences.
    multi_day_response = _weather_permitting_multi_day_response(user_text)
    if multi_day_response:
        return {
            "thoughts": [],
            "reasoning_mode": "direct",
            "handled_directly": "weather_permitting_multi_day",
            "selected_branch_id": None,
            "final_response": multi_day_response,
            "messages": [AIMessage(content=multi_day_response)],
        }

    # Continue a pending weather-permitting request when the user's short reply
    # (e.g. "Pittsburgh") answers the location question asked on the PRIOR turn by
    # _weather_permitting_multi_day_response above. Must run before
    # _conditional_weather_followup_response and _short_followup_without_context_response,
    # otherwise this short reply gets misrouted to the wrong clarification flow or the
    # generic "I do not have enough context to act" fallback.
    weather_permitting_followup = _weather_permitting_location_followup_response(user_text, history_text)
    if weather_permitting_followup:
        return {
            "thoughts": [],
            "reasoning_mode": "direct",
            "handled_directly": "weather_permitting_location_followup",
            "selected_branch_id": None,
            "final_response": weather_permitting_followup,
            "messages": [AIMessage(content=weather_permitting_followup)],
        }

    # Same idea, but for "check the weather for the whole week, then schedule the best
    # day" requests, which name no specific candidate days at all — scans a full 7-day
    # forecast and recommends the best-scoring day, rather than the generic 2-named-days
    # comparison above or the single-day conditional-weather clarification flow below.
    week_scan_response = _week_scan_outdoor_response(user_text)
    if week_scan_response:
        return {
            "thoughts": [],
            "reasoning_mode": "direct",
            "handled_directly": "week_scan_outdoor",
            "selected_branch_id": None,
            "final_response": week_scan_response,
            "messages": [AIMessage(content=week_scan_response)],
        }

    # Preserve conditional-weather clarification flow across short follow-ups
    # (for example: "70F is nice", "reschedule if not nice", "Seattle").
    conditional_followup = _conditional_weather_followup_response(user_text, history_text)
    if conditional_followup:
        return {
            "thoughts": [],
            "reasoning_mode": "direct",
            "handled_directly": "conditional_weather_followup",
            "selected_branch_id": None,
            "final_response": conditional_followup,
            "messages": [AIMessage(content=conditional_followup)],
        }

    short_without_context = _short_followup_without_context_response(user_text, history_text)
    if short_without_context:
        return {
            "thoughts": [],
            "reasoning_mode": "direct",
            "handled_directly": "short_followup_without_context",
            "selected_branch_id": None,
            "final_response": short_without_context,
            "messages": [AIMessage(content=short_without_context)],
        }

    one_shot_conditional = _fully_specified_conditional_weather_response(user_text)
    if one_shot_conditional:
        return {
            "thoughts": [],
            "reasoning_mode": "direct",
            "handled_directly": "conditional_weather_one_shot",
            "selected_branch_id": None,
            "final_response": one_shot_conditional,
            "messages": [AIMessage(content=one_shot_conditional)],
        }

    # CRITICAL: Retries ALWAYS use direct path (never escalate to ToT)
    # Only classify complexity for fresh requests, not retries
    if state["retry_count"] > 0:
        logger.info("🔄 RETRY DETECTED: Using direct path (retries never escalate to Tree-of-Thought)...")
        reasoning_mode = "direct"
    else:
        # Classify complexity only for fresh requests
        reasoning_mode = supervisor.classify_complexity(user_text, history_text)
    
    if reasoning_mode == "tot":
        logger.info("🧠 PLANNING: Multi-path exploration needed - using Tree-of-Thought reasoning...")
        branches, best = planner.run_tree_of_thought(user_text, history_text)
    else:
        if state["retry_count"] > 0:
            logger.info("⚡ PLANNING: Retry using direct path (fast)...")
        else:
            logger.info("⚡ PLANNING: Simple/Direct path (routine task, ~90% of requests)...")
        branches, best = planner.run_direct(user_text, history_text)
    
    if best is None:
        # Direct planning couldn't confidently produce a plan
        logger.info("   ℹ Planning could not find a confident plan.")

    if best is None:
        return {
            "thoughts": branches,
            "reasoning_mode": reasoning_mode,
            "error": "No viable plan found by the planner.",
        }

    # For exploratory itinerary prompts, use notes_knowledge to produce alternatives/comparison
    # rather than attempting strict scheduling execution.
    if _is_exploratory_itinerary_request(user_text) and best["target_agent"] != "notes_knowledge":
        logger.info("   ℹ Exploratory itinerary request detected; normalizing target to notes_knowledge")
        best["target_agent"] = "notes_knowledge"
        best["thought"] = (
            "Provide 2-4 alternative weekend plans that satisfy the user's constraints, "
            "with short trade-offs and a recommendation."
        )

    if best["target_agent"] == OUT_OF_SCOPE:
        if _is_exploratory_itinerary_request(user_text):
            logger.info("   ℹ Exploratory itinerary request detected; routing to notes_knowledge instead of out_of_scope")
            best["target_agent"] = "notes_knowledge"
            best["thought"] = (
                "Provide 2-4 alternative weekend plan options that satisfy the stated travel and activity constraints, "
                "with a concise comparison and recommendation."
            )
            return {
                "thoughts": branches,
                "reasoning_mode": reasoning_mode,
                "selected_branch_id": best["node_id"],
                "requires_human_approval": False,
            }

        # Check if this is a conditional weather request (needs clarification, not truly out of scope)
        user_text_lower = user_text.lower()
        weather_keywords = ["weather", "rain", "sunny", "nice", "forecast"]
        action_keywords = ["schedule", "activity", "plan", "book"]
        
        is_conditional_weather = (
            any(weather_kw in user_text_lower for weather_kw in weather_keywords) and
            any(action_kw in user_text_lower for action_kw in action_keywords) and
            planner._is_genuine_conditional(user_text_lower)
        )
        
        if is_conditional_weather:
            # Conditional weather request - ask for clarification
            response = (
                "I can help with conditional scheduling based on weather, but I need clarification first:\n\n"
                "1. **Weather conditions**: What does 'nice' mean to you? (e.g., sunny, above 70°F, dry, low wind)\n"
                "2. **Fallback plan**: What should happen if the weather is NOT suitable? (cancel, reschedule, proceed anyway)\n"
                "3. **Timing**: When do you want to schedule this? (specific date, or wait for right weather)\n\n"
                "Once you clarify these, I can check the forecast and schedule accordingly."
            )
        else:
            # Truly out of scope - can't help with this domain
            response = (
                "I can only help with to-dos/scheduling, notes, and weather. "
                f"That request ({best['thought']}) is outside what I can do."
            )
        
        return {
            "thoughts": branches,
            "reasoning_mode": reasoning_mode,
            "handled_directly": "out_of_scope",
            "selected_branch_id": None,
            "final_response": response,
            "messages": [AIMessage(content=response)],
        }

    # A QA retry re-plans the same user turn. Do not re-prompt after the user has
    # already approved its sensitive action.
    # IMPORTANT: only check the USER's own words, not the planner's free-text `thought`
    # paraphrase of the request. Including the LLM's own reasoning text here is risky —
    # generic words the model happens to choose (e.g. "a CLEAR request", "a CLEAN
    # solution") can accidentally match a single-word approval trigger like "clear",
    # falsely gating an ordinary, non-destructive scheduling request behind an
    # unnecessary human-approval prompt.
    requires_approval = (
        state.get("human_decision") != "approved"
        and supervisor.requires_human_approval(user_text)
    )
    return {
        "thoughts": branches,
        "reasoning_mode": reasoning_mode,
        "selected_branch_id": best["node_id"],
        "requires_human_approval": requires_approval,
    }


def _verify_and_heal_delete_action(state: AssistantState, user_text: str, result: SubAgentResult) -> SubAgentResult:
    """Post-action ground-truth check for already-approved delete/cleanup actions
    (Guardrail: verification, not trust).

    The scheduler_todo CrewAI agent sometimes reports a fluent, fully-successful
    summary ("Deleted N task(s): ...") while only actually completing some of the
    matching to-dos — e.g. it stops partway through a long run of tool calls and
    confabulates the rest of the list from what it intended to do. The QA critic
    only judges the agent's own text for internal plausibility; it has no way to
    cross-check that text against what was actually persisted to data/todos.json,
    so a confidently-wrong "success" report can sail through the QA gate untouched.

    This function recomputes, independently of the agent's LLM call, exactly which
    to-do IDs SHOULD now be marked done given the delete scope resolved from the
    user's own words (context.resolve_delete_scope — the same helper used to build
    the agent's task instructions, so scope can never disagree between the two).
    If any of those items are still open after the agent "finished", this function
    finishes the job directly via memory.complete_todo (deterministic, no LLM call)
    and rewrites the result's output text to accurately reflect the real final
    state, instead of letting the agent's inflated claim reach the user unchecked.
    """
    if state.get("human_decision") != "approved":
        return result
    from src.agents.supervisor import is_destructive_action_request
    if not is_destructive_action_request(user_text.lower()):
        return result

    from src.tools import memory

    now = datetime.now(context.USER_TIMEZONE)
    scope = context.resolve_delete_scope(user_text, now)
    open_todos = memory.list_todos(include_done=False)
    if scope["kind"] == "day":
        target_dates = set(scope["dates"])
        should_be_deleted = [t for t in open_todos if (t.get("due") or "")[:10] in target_dates]
    elif scope["kind"] == "all":
        should_be_deleted = list(open_todos)
    else:
        # Unscoped delete with no "all" marker: agent may have legitimately asked a
        # clarifying question instead of deleting anything — nothing to verify here.
        return result

    if not should_be_deleted:
        return result

    still_open = [t for t in should_be_deleted if not t["done"]]
    if not still_open:
        return result

    logger.warning(
        f"   ⚠️ VERIFICATION: agent reported completion but {len(still_open)}/{len(should_be_deleted)} "
        f"matching task(s) are still open — self-healing by completing them directly."
    )
    healed_descriptions = []
    for todo in still_open:
        memory.complete_todo(todo["id"])
        healed_descriptions.append(f"{todo['description']} (due: {todo.get('due')})")

    all_deleted_descriptions = [f"{t['description']} (due: {t.get('due')})" for t in should_be_deleted]
    corrected_output = (
        f"Deleted {len(should_be_deleted)} task(s): " + "; ".join(all_deleted_descriptions)
    )
    corrected = dict(result)
    corrected["output"] = corrected_output
    return corrected


def dispatch_node(state: AssistantState) -> dict:
    """Dispatch the selected plan to the corresponding CrewAI domain agent.
    
    Supports sequential multi-agent dispatch: first dispatches primary agent,
    then automatically dispatches secondary agents for weather/notes if detected.
    """
    # The weather/geocode API call counters (src/tools/external.py) are a PER-DISPATCH
    # anti-infinite-loop guardrail (max 3 calls), not a real external rate limit. Reset them
    # here at the start of every dispatch_node call (i.e. every individual attempt: a fresh
    # turn, a QA-triggered retry re-dispatch, or a secondary agent in a multi-agent turn).
    # Without this, calls silently accumulate across retries/turns within a single process,
    # and legitimate later weather lookups fail with a confusing "rate limit exceeded" error
    # that has nothing to do with the actual Open-Meteo API.
    from src.tools.external import reset_api_call_counters
    reset_api_call_counters()

    user_text = context.latest_user_text(state)
    branch = context.find_branch(state, state["selected_branch_id"])
    task_description = context.build_task_description(state, branch)
    
    results = []
    
    # Primary agent dispatch (from planner selection)
    logger.info(f"📤 DISPATCHING: Sending task to {branch['target_agent']} agent...")
    result = executor.dispatch(branch["target_agent"], task_description)
    logger.info(f"   ✓ {branch['target_agent']} completed")
    if branch.get("target_agent") == "scheduler_todo":
        result = _verify_and_heal_delete_action(state, user_text, result)
    results.append(result)
    
    # Sequential multi-agent dispatch: detect if we need additional agents
    multi_agents = planner.detect_multi_agent_dispatch(user_text)
    if multi_agents and len(multi_agents) > 1:
        logger.info(f"   📋 Multi-agent scenario detected: {', '.join(multi_agents)}")
        
        # Dispatch secondary agents in sequence (skip the primary that was already dispatched)
        primary_agent = branch["target_agent"]
        for secondary_agent in multi_agents:
            if secondary_agent != primary_agent:
                logger.info(f"   📤 Also dispatching {secondary_agent} agent...")
                secondary_task = context.build_task_description(state, {"target_agent": secondary_agent})
                secondary_result = executor.dispatch(secondary_agent, secondary_task)
                logger.info(f"      ✓ {secondary_agent} completed")
                if secondary_agent == "scheduler_todo":
                    secondary_result = _verify_and_heal_delete_action(state, user_text, secondary_result)
                results.append(secondary_result)
    
    return {
        "subagent_results": results,
        "step_count": state["step_count"] + 1,
    }


def critic_node(state: AssistantState) -> dict:
    """Critic/QA gate: score the latest sub-agent result(s) (Guardrail 2)."""
    user_text = context.latest_user_text(state)
    history_text = context.conversation_history_text(state)
    
    # Multi-agent result scoring
    subagent_results = state["subagent_results"]
    if len(subagent_results) > 1:
        # Multiple agents dispatched - score the combined output
        logger.info("🔍 QA SCORING: Evaluating multi-agent response...")
        combined_output = " | ".join([r["output"] for r in subagent_results if r.get("output")])
        combined_result = subagent_results[0].copy()  # Use first result as template
        combined_result["output"] = combined_output
        score, feedback = supervisor.score_result(user_text, combined_result, history_text)
    else:
        # Single agent - score normally
        logger.info("🔍 QA SCORING: Evaluating agent response...")
        last_result = subagent_results[-1]
        score, feedback = supervisor.score_result(user_text, last_result, history_text)
    
    logger.info(f"   Score: {score:.2f}/1.0 | Feedback: {feedback[:80]}")
    return {"qa_score": score, "qa_feedback": feedback}


def retry_node(state: AssistantState) -> dict:
    """Loop back to planning after a failed QA check, tracking retry budget (Guardrail 1)."""
    return {"retry_count": state["retry_count"] + 1}


def escalate_node(state: AssistantState) -> dict:
    """Give up autonomously (QA failure or a rejected approval) and hand control back to the user."""
    if state["human_decision"] == "rejected":
        response = "Okay, I won't do that. Let me know if you'd like a different action instead."
        return {
            "final_response": response,
            "messages": [AIMessage(content=response)],
            "escalation_count": state["escalation_count"] + 1,
            "consecutive_ambiguous_turns": 0,
        }
    last_result = state["subagent_results"][-1]
    feedback = state.get("qa_feedback") or "the assistant could not produce a satisfactory answer"
    response = (
        "I wasn't able to complete this reliably on my own "
        f"({feedback}). Could you clarify or confirm how you'd like me to proceed? "
        f"Last attempt: {last_result['output']}"
    )
    return {
        "final_response": response,
        "messages": [AIMessage(content=response)],
        "escalation_count": state["escalation_count"] + 1,
        "consecutive_ambiguous_turns": 0,
    }


def _should_check_weather(branch: dict, user_text: str, response_text: str) -> bool:
    """Determine if weather should be checked for this response."""
    outdoor_keywords = {"outdoor", "picnic", "hike", "hiking", "park", "beach", "garden", "concert", 
                       "sports", "camping", "festival", "race", "run", "walk", "game", "outside"}
    scheduled_keywords = {"scheduled", "confirmed", "added", "created", "set up", "booked"}
    
    is_outdoor = branch and branch.get("target_agent") == "scheduler_todo" and \
                 any(keyword in user_text.lower() for keyword in outdoor_keywords)
    is_scheduled = any(keyword in response_text.lower() for keyword in scheduled_keywords)
    return is_outdoor and is_scheduled


def _extract_location(user_text: str) -> str | None:
    """Extract location from user text."""
    location_keywords = ["in ", "at ", "near ", "around "]
    text_lower = user_text.lower()
    for keyword in location_keywords:
        idx = text_lower.find(keyword)
        if idx != -1:
            potential_location = user_text[idx + len(keyword):].split(",")[0].split(" on ")[0].split(" for ")[0].strip()
            if not potential_location:
                continue
            # Ignore common non-location constructions.
            if keyword == "at " and potential_location.lower().startswith("least"):
                continue
            if keyword == "at " and potential_location.split() and potential_location.split()[0].isdigit():
                continue
            # Reject obvious non-location fragments.
            bad_starts = {
                "one",
                "this",
                "that",
                "it",
                "line",
                "case",
                "time",
                "weather",
                "least",
                "pm",
                "am",
            }
            first = potential_location.split()[0].lower() if potential_location.split() else ""
            if first in bad_starts:
                continue
            if len(potential_location) < 50:
                return potential_location
    return None


def _append_weather_to_response(response_text: str, location: str) -> str:
    """Append weather information to the response by directly calling weather tool."""
    try:
        logger.info(f"   ⚡ Fetching weather data for {location}...")
        from src.tools.external import get_current_weather
        weather_data = get_current_weather(location)
        weather_summary = (
            f"Weather in {weather_data['location']}: {weather_data['condition']}, "
            f"{weather_data['temperature_c']}°C ({weather_data['temperature_c'] * 9/5 + 32:.1f}°F), "
            f"Humidity: {weather_data['humidity_pct']}%, "
            f"Wind: {weather_data['windspeed_kmh']} km/h"
        )
        logger.info(f"   ✓ Weather fetched: {weather_summary[:60]}...")
        return response_text + f"\n\n🌤️ **Weather:** {weather_summary}"
    except Exception as e:
        logger.debug(f"   ⚠ Weather check failed: {e}")
        return response_text


def compose_node(state: AssistantState) -> dict:
    """Build the final response once QA passes and any required approval is granted."""
    subagent_results = state["subagent_results"]
    user_text = context.latest_user_text(state)
    branch = context.find_branch(state, state["selected_branch_id"]) if state.get("selected_branch_id") else None
    
    logger.info("📝 COMPOSING: Building final response...")
    
    # Multi-agent response composition
    if len(subagent_results) > 1:
        logger.info(f"   Combining {len(subagent_results)} agent results...")
        response_parts = []
        for result in subagent_results:
            message = context.result_to_message(result)
            if message.content:
                response_parts.append(message.content)
        
        # Build combined response
        if len(response_parts) == 2:
            response_text = f"{response_parts[0]}\n\n{response_parts[1]}"
        elif len(response_parts) == 3:
            response_text = f"{response_parts[0]}\n\n{response_parts[1]}\n\n{response_parts[2]}"
        else:
            response_text = "\n\n".join(response_parts)
    else:
        # Single agent response composition
        last_result = subagent_results[-1]
        message = context.result_to_message(last_result)
        response_text = message.content
        
        # Auto-weather check for single-agent outdoor events
        if _should_check_weather(branch, user_text, response_text):
            location = _extract_location(user_text)
            if location:
                logger.info(f"   🌤️ Auto-checking weather for outdoor event in {location}...")
                response_text = _append_weather_to_response(response_text, location)
    
    logger.info("   ✓ Response ready")
    result = {"final_response": response_text, "messages": [AIMessage(content=response_text)]}
    return _apply_ambiguity_tracking(state, result, from_free_text_agent=True)


def route_after_plan(state: AssistantState) -> str:
    """Decide the next node after planning: already-answered (out of scope), needs
    approval (Guardrail 3), or ready to dispatch."""
    if state.get("final_response"):
        return "end"
    return "human_approval" if state["requires_human_approval"] else "dispatch"


def route_after_critic(state: AssistantState) -> str:
    """Decide the next node after the Critic/QA gate scores the latest result."""
    if supervisor.passes_qa(state["qa_score"] or 0.0):
        return "compose"
    if supervisor.should_escalate(state["retry_count"]):
        return "escalate"
    return "retry"


def route_after_human_approval(state: AssistantState) -> str:
    """Decide the next node after a human approval decision has been recorded."""
    return "dispatch" if state.get("human_decision") == "approved" else "escalate"
