"""Shared LangGraph state schema for the personal assistant orchestrator.

The orchestrator (this graph) owns routing, Tree-of-Thought planning, the
Critic/QA gate, and the human-in-the-loop approval gate. The three domain
agents (Scheduler & To-Do, Notes & Knowledge, Weather & External) are CrewAI
crews invoked from a single graph node and their results are written back
into this state under `subagent_results`.
"""

from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

# Names of the CrewAI-backed domain agents, matching FINAL-DESIGN.png.
AgentName = Literal["scheduler_todo", "notes_knowledge", "weather_external"]

# What the planner may target: a real domain agent, or an explicit "doesn't fit any of them"
# signal so out-of-domain requests (math, trivia, small talk) get a graceful refusal instead
# of being forced onto a mismatched agent.
PlanTarget = Literal["scheduler_todo", "notes_knowledge", "weather_external", "out_of_scope"]

HumanDecision = Literal["approved", "rejected"]

# Guardrail/router decision (Checkpoint 4.1): whether a turn is handled by a single direct
# LLM call or the full Tree-of-Thought branch-and-prune search.
ReasoningMode = Literal["direct", "tree_of_thought"]


class ThoughtBranch(TypedDict):
    """A single Tree-of-Thought candidate: one node in the reasoning tree."""

    node_id: str
    parent_id: Optional[str]
    depth: int
    thought: str          # the candidate plan/interpretation at this node
    target_agent: PlanTarget
    score: Optional[float]  # evaluator score in [0, 1]; None until scored
    pruned: bool


class SubAgentResult(TypedDict):
    """Output of a CrewAI crew invocation for one domain agent."""

    agent: AgentName
    output: str
    tool_calls: list
    success: bool
    error: Optional[str]


class AssistantState(TypedDict):
    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # Intent routing
    intent: Optional[AgentName]

    # Tree-of-Thought planning (Checkpoint 4.1)
    reasoning_mode: Optional[ReasoningMode]
    # Per-turn planning output. This must replace prior-turn values to avoid
    # stale branches leaking into the next user request.
    thoughts: list[ThoughtBranch]
    selected_branch_id: Optional[str]

    # CrewAI dispatch + results (Checkpoint 5.1)
    # Per-turn dispatch results. This must replace prior-turn values; otherwise
    # QA may score combined outputs from previous turns and produce false failures.
    subagent_results: list[SubAgentResult]

    # Critic/QA gate (Guardrail 2)
    qa_score: Optional[float]
    qa_feedback: Optional[str]

    # Guardrail 1: harness retry/step accounting
    retry_count: int
    step_count: int

    # Guardrail 3: human-in-the-loop
    requires_human_approval: bool
    human_decision: Optional[HumanDecision]
    escalation_count: int

    # Guardrail 3b: ambiguity escalation. Counts consecutive turns where the assistant
    # only asked a clarifying question without resolving/dispatching anything. Persists
    # across turns on the same thread (via the checkpointer) so a user who keeps giving
    # partial/unclear answers gets escalated to a human-style "let's start over" response
    # after `ambiguity_escalation_turns` (config/settings.yaml) rounds, instead of looping
    # clarification questions forever. Pure state bookkeeping — no extra LLM calls, so this
    # adds zero latency.
    consecutive_ambiguous_turns: int

    # Final output
    final_response: Optional[str]
    error: Optional[str]


def new_state(user_text: str) -> AssistantState:
    """Build the initial state for a fresh turn."""
    return AssistantState(
        messages=[HumanMessage(content=user_text)],
        intent=None,
        reasoning_mode=None,
        thoughts=[],
        selected_branch_id=None,
        subagent_results=[],
        qa_score=None,
        qa_feedback=None,
        retry_count=0,
        step_count=0,
        requires_human_approval=False,
        human_decision=None,
        escalation_count=0,
        consecutive_ambiguous_turns=0,
        final_response=None,
        error=None,
    )
