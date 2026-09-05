"""LangGraph orchestrator workflow: wires the plan/dispatch/critic loop,
the human-in-the-loop approval gate, and retry/escalation edges together.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from src.graph.nodes import (
    compose_node,
    critic_node,
    dispatch_node,
    escalate_node,
    plan_node,
    retry_node,
    route_after_critic,
    route_after_human_approval,
    route_after_plan,
)
from src.graph.state import AssistantState


def human_approval_node(state: AssistantState) -> dict:
    """Pause the graph and surface the pending action for explicit approval (Guardrail 3)."""
    branch = state["thoughts"][-1] if state["thoughts"] else None
    pending_action = branch["thought"] if branch else "the planned action"
    decision = interrupt({"reason": "human_approval_required", "pending_action": pending_action})
    return {"human_decision": decision}


def build_workflow():
    graph = StateGraph(AssistantState)

    graph.add_node("plan", plan_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("critic", critic_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("retry", retry_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("compose", compose_node)

    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"human_approval": "human_approval", "dispatch": "dispatch", "end": END},
    )
    graph.add_edge("dispatch", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"compose": "compose", "retry": "retry", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "human_approval",
        route_after_human_approval,
        {"dispatch": "dispatch", "escalate": "escalate"},
    )
    graph.add_edge("retry", "plan")
    graph.add_edge("compose", END)
    graph.add_edge("escalate", END)

    # Conversation Memory Store: persists state across turns/interrupts, keyed by thread_id.
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


workflow = build_workflow()
