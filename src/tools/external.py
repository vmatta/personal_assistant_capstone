"""External, real-time data access for the Weather & External Info agent.

Uses Open-Meteo (no API key required) so the tool works out of the box for
grading/review without any secret provisioning.
"""

import httpx
import logging

logger = logging.getLogger("personal_assistant")

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Rate limiting: prevent infinite loops in tests
_API_CALL_COUNT = {"weather": 0, "geocode": 0}
_MAX_API_CALLS_PER_TEST = {"weather": 3, "geocode": 3}  # Limit weather API calls to 3 per test

# Open-Meteo feature_code prefixes that mean "region/country", not a specific city.
_NON_CITY_FEATURE_PREFIXES = ("ADM", "PCLI")

# US state/territory names, checked directly against the query. Open-Meteo's geocoding API
# does not reliably return the state itself as a candidate "place" (it only returns small
# towns that happen to share the state's name, often in OTHER states/countries), so relying
# on the returned candidates' admin1 field alone misses queries like "California", "Texas",
# or "Virginia" entirely — this direct lookup catches those cases regardless of what the API
# returns as candidates.
_US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "north carolina", "north dakota",
    "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
}

# WMO weather interpretation codes used by Open-Meteo (subset covering common conditions).
_WEATHER_CODE_DESCRIPTIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


class AmbiguousLocationError(ValueError):
    """Raised when a location resolves to a state/region/country instead of a city."""

    def __init__(self, location: str, suggestions: list[str]):
        self.location = location
        self.suggestions = suggestions
        super().__init__(f"'{location}' is not specific enough to get a single weather reading.")


def _geocode(location: str) -> tuple[float, float, str]:
    # Rate limit check: prevent infinite loops in tests
    _API_CALL_COUNT["geocode"] += 1
    if _API_CALL_COUNT["geocode"] > _MAX_API_CALLS_PER_TEST["geocode"]:
        logger.warning(f"⚠️ GEOCODE API rate limit exceeded ({_API_CALL_COUNT['geocode']}/{_MAX_API_CALLS_PER_TEST['geocode']} calls)")
        raise RuntimeError(f"Geocode API rate limit exceeded. Prevented infinite loop (> {_MAX_API_CALLS_PER_TEST['geocode']} calls/test).")
    
    resp = httpx.get(_GEOCODE_URL, params={"name": location, "count": 5}, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        raise ValueError(f"Could not resolve location: {location}")

    def _normalize(text: str) -> str:
        # Normalize punctuation so "St. Louis" and "St Louis" (or "st.louis") compare equal —
        # Open-Meteo's own results are inconsistent about periods/spacing in abbreviations.
        return "".join(ch for ch in text.strip().lower() if ch.isalnum() or ch.isspace()).strip()

    query_lower = _normalize(location)
    top = results[0]
    feature_code = top.get("feature_code", "")
    top_name_matches_query = _normalize(str(top.get("name", ""))) == query_lower

    # The query itself may directly name a US state (e.g. "California", "Texas") that
    # Open-Meteo's geocoding API does not reliably return as a candidate "place" at all —
    # it instead returns small, unrelated towns that happen to share the state's name. A
    # direct lookup against known state names catches this regardless of what the API
    # returns as candidates.
    named_a_us_state = query_lower in _US_STATE_NAMES

    # Separately: the query may name a state/country that the API DOES surface via a
    # candidate's own admin1/country field (e.g. "North Carolina" appearing as the admin1
    # of an unrelated park listing). This is a strong, reliable signal on its own.
    named_a_state_or_country = named_a_us_state or any(
        _normalize(str(r.get(field, ""))) == query_lower for r in results for field in ("admin1", "country")
    )

    # admin2 (COUNTY) matching the query is a much weaker/coincidental signal: many major
    # U.S. cities sit in a same-named county (Philadelphia PA in Philadelphia County,
    # Baltimore MD, St. Louis MO, etc.). Only treat this as ambiguous if the TOP-ranked
    # result's own name does NOT already match the query — if it does, the user named a
    # real, specific, correctly-resolved city, and the county-name coincidence is irrelevant.
    named_a_county_on_an_unrelated_top_hit = (
        not top_name_matches_query
        and any(_normalize(str(r.get("admin2", ""))) == query_lower for r in results)
    )

    named_a_region = named_a_state_or_country or named_a_county_on_an_unrelated_top_hit
    top_is_itself_a_region_entity = feature_code.startswith(_NON_CITY_FEATURE_PREFIXES)
    if named_a_region or top_is_itself_a_region_entity:
        # Only offer city suggestions when the TOP hit's own feature_code says it IS a
        # region/country entity (ADM1/PCLI) — in that case the API is explicitly telling us
        # "this result is the state/country itself," so its sibling results in the same
        # response are likely genuine, real cities within it, worth suggesting. Whenever
        # ambiguity is instead detected via an admin-field/state-name match (named_a_region),
        # we can't reliably enumerate cities within that region from a plain name search, so
        # suggesting its unrelated top text matches (parks, wildlife areas, coincidentally
        # named towns in other states, etc.) would just be misleading.
        suggestions = []
        if top_is_itself_a_region_entity:
            suggestions = [
                r["name"]
                for r in results
                if not r.get("feature_code", "").startswith(_NON_CITY_FEATURE_PREFIXES)
                and r["name"].strip().lower() != query_lower
            ]
        raise AmbiguousLocationError(location, suggestions[:5])

    return top["latitude"], top["longitude"], top.get("name", location)


def get_current_weather(location: str) -> dict:
    """Return current weather conditions for a named location."""
    # Rate limit check: prevent infinite loops in tests
    _API_CALL_COUNT["weather"] += 1
    if _API_CALL_COUNT["weather"] > _MAX_API_CALLS_PER_TEST["weather"]:
        logger.warning(f"⚠️ WEATHER API rate limit exceeded ({_API_CALL_COUNT['weather']}/{_MAX_API_CALLS_PER_TEST['weather']} calls)")
        raise RuntimeError(f"Weather API rate limit exceeded. Prevented infinite loop (> {_MAX_API_CALLS_PER_TEST['weather']} calls/test).")
    
    lat, lon, resolved_name = _geocode(location)
    resp = httpx.get(
        _FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
        },
        timeout=10,
    )
    resp.raise_for_status()
    current = resp.json().get("current", {})
    weather_code = current.get("weather_code")
    return {
        "location": resolved_name,
        "temperature_c": current.get("temperature_2m"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "precipitation_mm": current.get("precipitation"),
        "windspeed_kmh": current.get("wind_speed_10m"),
        "condition": _WEATHER_CODE_DESCRIPTIONS.get(weather_code, "unknown conditions"),
        "observed_at": current.get("time"),
        "source": "Open-Meteo",
    }


def get_daily_forecast(location: str, date_iso: str) -> dict:
    """Return the daily forecast summary (max/min temp, precipitation, wind) for a
    specific future date at a named location. Used for day-ahead decisions like
    "schedule outdoor Saturday or Sunday, whichever has better weather" — the plain
    get_current_weather() function only returns right-now conditions, which cannot
    answer questions about a future day.
    """
    _API_CALL_COUNT["weather"] += 1
    if _API_CALL_COUNT["weather"] > _MAX_API_CALLS_PER_TEST["weather"]:
        logger.warning(f"⚠️ WEATHER API rate limit exceeded ({_API_CALL_COUNT['weather']}/{_MAX_API_CALLS_PER_TEST['weather']} calls)")
        raise RuntimeError(f"Weather API rate limit exceeded. Prevented infinite loop (> {_MAX_API_CALLS_PER_TEST['weather']} calls/test).")

    lat, lon, resolved_name = _geocode(location)
    resp = httpx.get(
        _FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "start_date": date_iso,
            "end_date": date_iso,
            "timezone": "auto",
        },
        timeout=10,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})

    def _first(key):
        values = daily.get(key) or []
        return values[0] if values else None

    weather_code = _first("weather_code")
    return {
        "location": resolved_name,
        "date": date_iso,
        "temp_max_c": _first("temperature_2m_max"),
        "temp_min_c": _first("temperature_2m_min"),
        "precipitation_mm": _first("precipitation_sum"),
        "windspeed_max_kmh": _first("wind_speed_10m_max"),
        "condition": _WEATHER_CODE_DESCRIPTIONS.get(weather_code, "unknown conditions"),
        "source": "Open-Meteo",
    }


def get_weekly_forecast(location: str, start_date_iso: str, end_date_iso: str) -> list[dict]:
    """Return the daily forecast summary for EVERY day in a date range, in a single API
    call. Used for "check the weather for the whole week, then schedule the best day"
    style requests — get_daily_forecast() only answers about one specific date, which
    would require N separate API calls (and N geocode calls) to cover a week; this
    function covers the whole range in one HTTP request.
    """
    _API_CALL_COUNT["weather"] += 1
    if _API_CALL_COUNT["weather"] > _MAX_API_CALLS_PER_TEST["weather"]:
        logger.warning(f"⚠️ WEATHER API rate limit exceeded ({_API_CALL_COUNT['weather']}/{_MAX_API_CALLS_PER_TEST['weather']} calls)")
        raise RuntimeError(f"Weather API rate limit exceeded. Prevented infinite loop (> {_MAX_API_CALLS_PER_TEST['weather']} calls/test).")

    lat, lon, resolved_name = _geocode(location)
    resp = httpx.get(
        _FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "start_date": start_date_iso,
            "end_date": end_date_iso,
            "timezone": "auto",
        },
        timeout=10,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})

    dates = daily.get("time") or []
    weather_codes = daily.get("weather_code") or []
    temp_maxes = daily.get("temperature_2m_max") or []
    temp_mins = daily.get("temperature_2m_min") or []
    precips = daily.get("precipitation_sum") or []
    winds = daily.get("wind_speed_10m_max") or []

    results = []
    for i, date_iso in enumerate(dates):
        weather_code = weather_codes[i] if i < len(weather_codes) else None
        results.append({
            "location": resolved_name,
            "date": date_iso,
            "temp_max_c": temp_maxes[i] if i < len(temp_maxes) else None,
            "temp_min_c": temp_mins[i] if i < len(temp_mins) else None,
            "precipitation_mm": precips[i] if i < len(precips) else None,
            "windspeed_max_kmh": winds[i] if i < len(winds) else None,
            "condition": _WEATHER_CODE_DESCRIPTIONS.get(weather_code, "unknown conditions"),
            "source": "Open-Meteo",
        })
    return results


def reset_api_call_counters():
    """Reset API call counters (for testing between scenarios)."""
    global _API_CALL_COUNT
    _API_CALL_COUNT["weather"] = 0
    _API_CALL_COUNT["geocode"] = 0
    logger.info("🔄 API call counters reset")
