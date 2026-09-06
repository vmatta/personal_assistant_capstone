"""Tests for the tools layer: to-do persistence, scoped MCP tool registry,
and the weather tool (with the network call mocked out)."""

import pytest

from src.tools import external, mcp_client, memory


@pytest.fixture(autouse=True)
def isolated_todo_store(tmp_path, monkeypatch):
    """Redirect the JSON to-do store to a temp file so tests don't touch real data."""
    monkeypatch.setattr(memory, "_TODO_STORE_PATH", tmp_path / "todos.json")


def test_list_notes_returns_recent_notes(monkeypatch):
    class _FakeCollection:
        def get(self, include=None):
            return {
                "documents": ["older note", "newer note"],
                "metadatas": [{"created_at": 1}, {"created_at": 2}],
            }

    monkeypatch.setattr(memory, "_get_collection", lambda name: _FakeCollection())

    assert memory.list_notes(limit=1) == ["newer note"]


def test_save_note_duplicate_returns_none_and_tool_message_is_already_in_there(monkeypatch):
    class _FakeCollection:
        def __init__(self):
            self.documents = ["Remember that I prefer to relax on Sundays."]

        def get(self, include=None):
            return {"documents": self.documents}

        def add(self, documents=None, ids=None, metadatas=None):
            raise AssertionError("duplicate notes should not be saved again")

    monkeypatch.setattr(memory, "_get_collection", lambda name: _FakeCollection())

    assert memory.save_note("remember that i prefer to relax on sundays") is None
    assert mcp_client.save_note_tool.func("Remember that I prefer to relax on Sundays.") == "This is already in there."


def test_add_list_complete_todo_roundtrip():
    item = memory.add_todo("buy milk", due="2030-01-01")
    assert item["done"] is False

    todos = memory.list_todos()
    assert any(t["id"] == item["id"] for t in todos)

    assert memory.complete_todo(item["id"]) is True
    assert not any(t["id"] == item["id"] for t in memory.list_todos())
    assert any(t["id"] == item["id"] for t in memory.list_todos(include_done=True))


def test_list_todos_recovers_from_trailing_json_junk():
    memory._TODO_STORE_PATH.write_text(
        '[{"id": "1", "description": "buy milk", "due": null, "done": false}]junk',
        encoding="utf-8",
    )

    todos = memory.list_todos()

    assert todos[0]["description"] == "buy milk"
    assert memory._TODO_STORE_PATH.read_text(encoding="utf-8").strip().endswith("]")


def test_find_open_todos_at_due_matches_equivalent_timestamps():
    memory.add_todo("standup", due="2030-01-01T10:30:00-05:00")

    conflicts = memory.find_open_todos_at_due("2030-01-01T15:30:00Z")

    assert conflicts[0]["description"] == "standup"


def test_add_todo_tool_rejects_existing_open_time_conflict():
    item = memory.add_todo("standup", due="2030-01-01T10:30:00-05:00")

    result = mcp_client.add_todo_tool.func("another meeting", due="2030-01-01T15:30:00Z")

    assert "Cannot add" in result
    assert item["id"] in result
    assert len(memory.list_todos()) == 1


def test_list_todos_tool_formats_due_time():
    item = memory.add_todo("standup", due="2030-01-01T15:30:00Z")

    result = mcp_client.list_todos_tool.func()

    assert item["id"] in result
    assert "2030-01-01 10:30 AM ET" in result


def test_complete_todo_missing_id_returns_false():
    assert memory.complete_todo("does-not-exist") is False


def test_mcp_tool_scopes_are_disjoint_and_valid():
    all_scopes = mcp_client.AGENT_TOOL_SCOPES
    assert set(all_scopes) == {"scheduler_todo", "notes_knowledge", "weather_external"}
    for agent_name, tools in all_scopes.items():
        assert mcp_client.get_tools_for_agent(agent_name) == tools
        assert len(tools) > 0


def test_get_tools_for_unknown_agent_raises():
    with pytest.raises(ValueError):
        mcp_client.get_tools_for_agent("not_a_real_agent")


def test_get_current_weather_uses_geocode_then_forecast(monkeypatch):
    calls = []

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if "geocoding" in url:
            return _FakeResponse({"results": [{"latitude": 1.0, "longitude": 2.0, "name": "Testville"}]})
        return _FakeResponse(
            {
                "current": {
                    "temperature_2m": 21.5,
                    "relative_humidity_2m": 40,
                    "precipitation": 0.0,
                    "wind_speed_10m": 5.0,
                    "weather_code": 1,
                    "time": "2030-01-01T00:00",
                }
            }
        )

    monkeypatch.setattr(external.httpx, "get", fake_get)

    data = external.get_current_weather("Testville")
    assert data["location"] == "Testville"
    assert data["temperature_c"] == 21.5
    assert data["condition"] == "mainly clear"
    assert len(calls) == 2


def test_get_current_weather_unresolvable_location_raises(monkeypatch):
    class _EmptyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": []}

    monkeypatch.setattr(external.httpx, "get", lambda *a, **k: _EmptyResponse())

    with pytest.raises(ValueError):
        external.get_current_weather("Nowhereville")


def test_get_current_weather_ambiguous_region_raises_with_suggestions(monkeypatch):
    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(
            {
                "results": [
                    {"latitude": 1.0, "longitude": 2.0, "name": "Michigan", "feature_code": "ADM1"},
                    {"latitude": 3.0, "longitude": 4.0, "name": "Detroit", "feature_code": "PPL"},
                    {"latitude": 5.0, "longitude": 6.0, "name": "Lansing", "feature_code": "PPLA"},
                ]
            }
        )

    monkeypatch.setattr(external.httpx, "get", fake_get)

    with pytest.raises(external.AmbiguousLocationError) as exc_info:
        external.get_current_weather("Michigan")
    assert "Detroit" in exc_info.value.suggestions


def test_weather_tool_asks_clarifying_question_for_ambiguous_location(monkeypatch):
    def raise_ambiguous(location):
        raise external.AmbiguousLocationError(location, ["Detroit", "Lansing"])

    monkeypatch.setattr(external, "get_current_weather", raise_ambiguous)

    result = mcp_client.get_current_weather_tool.func("Michigan")
    assert "Detroit" in result
    assert "Lansing" in result
