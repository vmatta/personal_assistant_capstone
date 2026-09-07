"""Tests for the LangGraph routing logic and Critic/QA guardrail predicates.

These only exercise deterministic control flow (no live LLM calls), so they
stay fast and don't depend on API keys or network access.
"""

from src.agents import supervisor
from src.graph.nodes import route_after_critic, route_after_human_approval, route_after_plan
from src.graph.state import ThoughtBranch, new_state


def _state_with(**overrides):
    state = new_state("test request")
    state.update(overrides)
    return state


def test_new_state_shape():
    state = new_state("hello")
    assert state["messages"][0].content == "hello"
    assert state["thoughts"] == []
    assert state["retry_count"] == 0
    assert state["step_count"] == 0
    assert state["requires_human_approval"] is False
    assert state["final_response"] is None


def test_conditional_followup_does_not_block_new_schedule_listing():
    from src.graph import nodes

    history = """
    User: Schedule an outdoor picnic in Denver on Friday at 2 PM and check if the weather is suitable.
    Assistant: Got it. I still need:
    1. Fallback plan: if weather is not suitable, should I cancel, reschedule, or proceed anyway?
    """

    result = nodes._conditional_weather_followup_response("list all my scheduled events", history)

    assert result is None


def test_streamlit_log_handler_skips_ui_write_without_script_context(monkeypatch):
    import logging

    import streamlit_app

    class FakeContainer:
        def __init__(self):
            self.calls = 0

        def write(self, _message):
            self.calls += 1

    fake_container = FakeContainer()
    streamlit_app._log_container = fake_container
    streamlit_app._log_container_active = True
    monkeypatch.setattr(streamlit_app, "get_script_run_ctx", lambda: None, raising=False)

    handler = streamlit_app.StreamlitLogHandler()
    record = logging.LogRecord(name="test", level=logging.INFO, pathname=__file__, lineno=1, msg="hello", args=(), exc_info=None)
    handler.emit(record)

    assert fake_container.calls == 0


def test_route_after_plan_no_approval_needed():
    state = _state_with(requires_human_approval=False)
    assert route_after_plan(state) == "dispatch"


def test_route_after_plan_approval_needed():
    state = _state_with(requires_human_approval=True)
    assert route_after_plan(state) == "human_approval"


def test_plan_node_does_not_repeat_granted_approval_on_retry(monkeypatch):
    from src.agents import context
    from src.graph import nodes

    branch = ThoughtBranch(
        node_id="cancel-event",
        parent_id=None,
        depth=0,
        thought="cancel the event tomorrow",
        target_agent="scheduler_todo",
        score=1.0,
        pruned=False,
    )
    monkeypatch.setattr(context, "latest_user_text", lambda state: "Cancel my event tomorrow")
    monkeypatch.setattr(context, "conversation_history_text", lambda state: "")
    monkeypatch.setattr(nodes.supervisor, "classify_complexity", lambda user_text, history_text: "direct")
    monkeypatch.setattr(nodes.planner, "run_direct", lambda user_text, history_text: ([branch], branch))

    result = nodes.plan_node(_state_with(human_decision="approved", retry_count=0))

    assert result["requires_human_approval"] is False


def test_plan_node_routes_personal_preference_lookup_to_notes_agent():
    from src.graph import nodes

    result = nodes.plan_node(_state_with(messages=new_state("WHAT IS MY PREFERRED MEETING TIME?")["messages"]))

    assert result["reasoning_mode"] == "direct"
    assert result["requires_human_approval"] is False
    assert result["thoughts"][0]["target_agent"] == "notes_knowledge"


def test_plan_node_answers_capability_question_without_out_of_scope():
    from src.graph import nodes

    result = nodes.plan_node(_state_with(messages=new_state("LIST ALL THE THINGS YOU CAN DO")["messages"]))

    assert result["reasoning_mode"] == "direct"
    assert "to-dos and schedules" in result["final_response"]
    assert "outside what I can do" not in result["final_response"]


def test_plan_node_answers_what_all_can_i_do_capability_question():
    from src.graph import nodes

    result = nodes.plan_node(_state_with(messages=new_state("WHAT ALL CAN I DO")["messages"]))

    assert "to-dos and schedules" in result["final_response"]
    assert "outside what I can do" not in result["final_response"]


def test_plan_node_answers_what_all_i_can_do_capability_question():
    from src.graph import nodes

    result = nodes.plan_node(_state_with(messages=new_state("WHAT ALL I CAN DO ?")["messages"]))

    assert "to-dos and schedules" in result["final_response"]
    assert "outside what I can do" not in result["final_response"]


def test_route_after_plan_already_answered_ends():
    state = _state_with(requires_human_approval=False, final_response="I can't help with that.")
    assert route_after_plan(state) == "end"


def test_route_after_critic_pass():
    state = _state_with(qa_score=0.9, retry_count=0)
    assert route_after_critic(state) == "compose"


def test_route_after_critic_fail_retries_left():
    state = _state_with(qa_score=0.2, retry_count=1)
    assert route_after_critic(state) == "retry"


def test_route_after_critic_fail_retries_exhausted():
    state = _state_with(qa_score=0.2, retry_count=3)
    assert route_after_critic(state) == "escalate"


def test_route_after_human_approval_approved():
    state = _state_with(human_decision="approved")
    assert route_after_human_approval(state) == "dispatch"


def test_route_after_human_approval_rejected():
    state = _state_with(human_decision="rejected")
    assert route_after_human_approval(state) == "escalate"


def test_passes_qa_threshold():
    assert supervisor.passes_qa(0.85) is True
    assert supervisor.passes_qa(0.84) is False


def test_should_escalate_at_max_retries():
    assert supervisor.should_escalate(supervisor.MAX_RETRIES) is True
    assert supervisor.should_escalate(supervisor.MAX_RETRIES - 1) is False


def test_requires_human_approval_matches_configured_actions():
    assert supervisor.requires_human_approval("please cancel_event for tomorrow") is True
    assert supervisor.requires_human_approval("Cancel my event for tomorrow") is True
    assert supervisor.requires_human_approval("Can you delete the above event") is True
    assert supervisor.requires_human_approval("Remove my 10:30 event") is True
    assert supervisor.requires_human_approval("what is the weather today") is False


def test_find_branch_lookup():
    from src.agents.context import find_branch

    branch = ThoughtBranch(
        node_id="abc",
        parent_id=None,
        depth=0,
        thought="do the thing",
        target_agent="scheduler_todo",
        score=0.9,
        pruned=False,
    )
    state = _state_with(thoughts=[branch])
    assert find_branch(state, "abc")["thought"] == "do the thing"

    try:
        find_branch(state, "missing")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_build_task_description_uses_local_date_context():
    from src.agents.context import build_task_description

    branch = ThoughtBranch(
        node_id="abc",
        parent_id=None,
        depth=0,
        thought="list tomorrow's schedule",
        target_agent="scheduler_todo",
        score=0.9,
        pruned=False,
    )

    task_description = build_task_description(_state_with(), branch)

    assert "Today's local date" in task_description
    assert "America/New_York" in task_description
    assert "Interpret relative dates like today/tomorrow using this local date" in task_description


class _FakeResponse:
    def __init__(self, content):
        self.content = content


def test_classify_complexity_simple(monkeypatch):
    monkeypatch.setattr(
        "src.agents.supervisor.invoke_with_retry",
        lambda llm, messages: _FakeResponse('{"complexity": "simple"}'),
    )
    assert supervisor.classify_complexity("add a to-do to buy milk") == "direct"


def test_classify_complexity_complex(monkeypatch):
    monkeypatch.setattr(
        "src.agents.supervisor.invoke_with_retry",
        lambda llm, messages: _FakeResponse('{"complexity": "complex"}'),
    )
    assert supervisor.classify_complexity("plan my whole week around the weather") == "tree_of_thought"


def test_classify_complexity_fails_safe_to_tree_of_thought(monkeypatch):
    def _raise(llm, messages):
        raise RuntimeError("upstream error")

    monkeypatch.setattr("src.agents.supervisor.invoke_with_retry", _raise)
    assert supervisor.classify_complexity("anything") == "tree_of_thought"


def test_dispatch_node_delete_all_reports_full_pre_dispatch_scope(monkeypatch, tmp_path):
    """Regression: the agent completing some matching to-dos itself (but only mentioning
    a subset in its own text) must not shrink the reported scope. The verification must
    use a snapshot taken BEFORE dispatch, not a re-query of open items AFTER dispatch,
    otherwise items the agent already handled silently vanish from the final report."""
    from src.agents import executor
    from src.graph import nodes
    from src.tools import memory

    monkeypatch.setattr(memory, "_TODO_STORE_PATH", tmp_path / "todos.json")

    item_a = memory.add_todo("1-hour meeting", due="2026-09-08T11:00:00")
    item_b = memory.add_todo("Outdoor picnic in Denver", due="2026-09-11T14:00:00")
    item_c = memory.add_todo("Outdoor activity", due="2026-09-12T14:00:00")

    def fake_dispatch(agent_name, task_description):
        # Simulate the agent completing 2 of the 3 matching to-dos itself, but only
        # mentioning 1 of them in its own summary text (the exact bug reported).
        memory.complete_todo(item_a["id"])
        memory.complete_todo(item_c["id"])
        return {
            "agent": agent_name,
            "output": f"Deleted 1 task(s): {item_c['description']} (due: {item_c['due']})",
            "tool_calls": [],
            "success": True,
            "error": None,
        }

    monkeypatch.setattr(executor, "dispatch", fake_dispatch)
    monkeypatch.setattr(nodes.planner, "detect_multi_agent_dispatch", lambda user_text: [])

    branch = ThoughtBranch(
        node_id="cancel-all",
        parent_id=None,
        depth=0,
        thought="cancel all scheduled events",
        target_agent="scheduler_todo",
        score=1.0,
        pruned=False,
    )
    state = _state_with(
        messages=new_state("cancel all my scheduled events")["messages"],
        thoughts=[branch],
        selected_branch_id=branch["node_id"],
        human_decision="approved",
    )

    result = nodes.dispatch_node(state)

    output = result["subagent_results"][0]["output"]
    assert output.startswith("Deleted 3 task(s):")
    assert item_a["description"] in output
    assert item_b["description"] in output
    assert item_c["description"] in output
    assert memory.list_todos(include_done=False) == []


def test_score_result_includes_history_for_followup_qa(monkeypatch):
    captured = {}

    def fake_invoke(llm, messages):
        captured["prompt"] = messages[-1].content
        return _FakeResponse('{"score": 1.0, "feedback": "ok"}')

    monkeypatch.setattr("src.agents.supervisor.invoke_with_retry", fake_invoke)
    result = {
        "agent": "scheduler_todo",
        "output": "The event scheduled for tomorrow at 10:30 AM has been removed.",
        "tool_calls": [],
        "success": True,
        "error": None,
    }

    score, feedback = supervisor.score_result(
        "remove the above event",
        result,
        "Conversation so far:\nUser: add an event for tomorrow 10:30 AM",
    )

    assert score == 1.0
    assert feedback == "ok"
    assert "add an event for tomorrow 10:30 AM" in captured["prompt"]


def test_score_result_handles_provider_connection_error(monkeypatch):
    from openai import APIConnectionError

    def raise_connection_error(llm, messages):
        raise APIConnectionError(request=None)

    monkeypatch.setattr("src.agents.supervisor.invoke_with_retry", raise_connection_error)
    result = {
        "agent": "weather_external",
        "output": "Weather in Pittsburgh: clear sky, 24.5C.",
        "tool_calls": [],
        "success": True,
        "error": None,
    }

    score, feedback = supervisor.score_result("weather in Pittsburgh", result)

    assert score == 0.0
    assert "could not reach" in feedback


def test_run_direct_returns_single_trusted_branch(monkeypatch):
    from src.agents import planner

    monkeypatch.setattr(
        "src.agents.planner.invoke_with_retry",
        lambda llm, messages: _FakeResponse('{"thought": "add the to-do", "target_agent": "scheduler_todo"}'),
    )
    branches, best = planner.run_direct("add a to-do to buy milk")
    assert len(branches) == 1
    assert best["target_agent"] == "scheduler_todo"
    assert best["score"] == 1.0
    assert best["depth"] == 0


def test_run_direct_invalid_agent_returns_none(monkeypatch):
    from src.agents import planner

    monkeypatch.setattr(
        "src.agents.planner.invoke_with_retry",
        lambda llm, messages: _FakeResponse('{"thought": "?", "target_agent": "not_a_real_agent"}'),
    )
    branches, best = planner.run_direct("gibberish request")
    assert branches == []
    assert best is None


def test_run_direct_out_of_scope_is_accepted(monkeypatch):
    from src.agents import planner

    monkeypatch.setattr(
        "src.agents.planner.invoke_with_retry",
        lambda llm, messages: _FakeResponse(
            '{"thought": "this is a math question, not a to-do/notes/weather request", '
            '"target_agent": "out_of_scope"}'
        ),
    )
    branches, best = planner.run_direct("what is 2+2?")
    assert best["target_agent"] == planner.OUT_OF_SCOPE


def test_run_direct_overrides_misrouted_list_query_to_scheduler(monkeypatch):
    """Regression: 'list all scheduled events' was misrouted to notes_knowledge, which
    searches saved notes (finds nothing) instead of calling list_todos, producing
    "I couldn't find any scheduled events in your saved notes" even when open to-dos exist."""
    from src.agents import planner

    monkeypatch.setattr(
        "src.agents.planner.invoke_with_retry",
        lambda llm, messages: _FakeResponse(
            '{"thought": "look up scheduled events in notes", "target_agent": "notes_knowledge"}'
        ),
    )
    branches, best = planner.run_direct("list all scheduled events")
    assert best["target_agent"] == "scheduler_todo"


def test_plan_node_out_of_scope_skips_dispatch(monkeypatch):
    from src.agents import context
    from src.graph import nodes

    monkeypatch.setattr(context, "latest_user_text", lambda state: "what is 2+2?")
    monkeypatch.setattr(context, "conversation_history_text", lambda state: "")
    monkeypatch.setattr(
        nodes.planner,
        "run_direct",
        lambda user_text, history_text: (
            [],
            ThoughtBranch(
                node_id="x",
                parent_id=None,
                depth=0,
                thought="math is out of scope",
                target_agent="out_of_scope",
                score=1.0,
                pruned=False,
            ),
        ),
    )
    monkeypatch.setattr(nodes.supervisor, "classify_complexity", lambda user_text, history_text: "direct")

    state = _state_with()
    result = nodes.plan_node(state)
    assert "final_response" in result
    assert result["selected_branch_id"] is None
