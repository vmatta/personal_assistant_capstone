"""CrewAI executor: builds and runs the three domain agents (Scheduler &
To-Do, Notes & Knowledge, Weather & External) dispatched by the LangGraph
orchestrator. Each call is a single-agent, single-task crew so results map
cleanly back onto one `SubAgentResult` per orchestrator dispatch.
"""

from crewai import LLM, Agent, Crew, Process, Task

from config import (
    ACTIVE_PROVIDER,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SETTINGS,
)
from src.graph.state import AgentName, SubAgentResult
from src.tools.mcp_client import get_tools_for_agent

_AGENT_SPECS: dict[AgentName, dict] = {
    "scheduler_todo": {
        "role": "Smart Scheduler Agent",
        "goal": "Schedule tasks intelligently by checking availability and suggesting free time slots. Use add_todo, list_todos, or complete_todo tools to manage tasks.",
        "backstory": (
            "You are an intelligent scheduler.\n"
            "CRITICAL: READ YOUR TASK DESCRIPTION CAREFULLY. It will tell you which mode to use:\n"
            "  - If task says 'CHECK AVAILABILITY': Call list_todos FIRST, identify free slots, offer them\n"
            "  - If task says 'Do NOT call list_todos': Skip availability check, go straight to scheduling\n"
            "  - If task says 'Schedule immediately': User already picked time from offered slots, just add_todo\n"
            "\n"
            "AVAILABILITY CHECK MODE (when checking availability):\n"
            "1. ALWAYS call list_todos FIRST before offering any times\n"
            "2. Parse response: 'No to-do items found' = entire day free\n"
            "3. Identify free slots in the requested window (morning/afternoon/evening)\n"
            "4. Respond: 'I checked... [SHOW list_todos result]. Available slots: [times]. Which works best?'\n"
            "5. NEVER make contradictory statements like 'busy... available' or 'busy from 12 PM to 11 AM'\n"
            "\n"
            "DIRECT SCHEDULING MODE (when user already confirmed time):\n"
            "1. Do NOT call list_todos - user already saw availability and confirmed they want this time\n"
            "2. Just extract the time and date from the task description\n"
            "3. Call add_todo_tool with the specific time\n"
            "4. Confirm: 'Scheduled: [description] for [date/time]'\n"
            "\n"
            "KEY: Listen to your task description. It will be very specific about which mode to use."
        ),
    },
    "notes_knowledge": {
        "role": "Notes & Knowledge Agent",
        "goal": "Save user notes and retrieve grounded context from notes and the knowledge base before answering.",
        "backstory": (
            "You are a retrieval-augmented note-taking assistant. You always search "
            "existing notes/knowledge before writing a new answer, and you cite what you found."
        ),
    },
    "weather_external": {
        "role": "Weather & External Info Agent",
        "goal": "Fetch and summarize current weather or other real-time external information for the user.",
        "backstory": "You call external data sources and summarize results in plain language.",
    },
}

_OPENROUTER_FREE_FALLBACK_MODEL = (
    SETTINGS.get("llm", {})
    .get("openrouter", {})
    .get("fallback_model", "meta-llama/llama-3.1-8b-instruct:free")
)


def _build_llm() -> LLM:
    provider_prefix = "openai" if ACTIVE_PROVIDER == "openai" else "openrouter"
    return LLM(
        model=f"{provider_prefix}/{LLM_MODEL}",
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


def _build_openrouter_fallback_llm() -> LLM:
    return LLM(
        model=f"openrouter/{_OPENROUTER_FREE_FALLBACK_MODEL}",
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


# Lazy-load LLM to pick up config changes
_LLM = None

def _get_llm() -> LLM:
    """Get LLM instance for CrewAI agents, lazily initializing on first use."""
    global _LLM
    if _LLM is None:
        _LLM = _build_llm()
    return _LLM


def _build_agent(agent_name: AgentName) -> Agent:
    spec = _AGENT_SPECS[agent_name]
    return Agent(
        role=spec["role"],
        goal=spec["goal"],
        backstory=spec["backstory"],
        tools=get_tools_for_agent(agent_name),
        llm=_get_llm(),
        verbose=False,
        allow_delegation=False,
    )


def _build_agent_with_llm(agent_name: AgentName, llm: LLM) -> Agent:
    spec = _AGENT_SPECS[agent_name]
    return Agent(
        role=spec["role"],
        goal=spec["goal"],
        backstory=spec["backstory"],
        tools=get_tools_for_agent(agent_name),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def _kickoff(agent: Agent, task_description: str):
    task = Task(
        description=task_description,
        expected_output="A concise, direct answer or confirmation of the action taken.",
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
        tracing=False,
    )
    return crew.kickoff()


def dispatch(agent_name: AgentName, task_description: str) -> SubAgentResult:
    """Run one CrewAI domain agent on a single task and return its result."""
    try:
        agent = _build_agent(agent_name)
        result = _kickoff(agent, task_description)
        return SubAgentResult(
            agent=agent_name,
            output=str(result),
            tool_calls=[],
            success=True,
            error=None,
        )
    except Exception as exc:  # surfaced to the harness/critic, not swallowed
        # Fallback path for OpenRouter credit exhaustion: retry once with a free model.
        err = str(exc).lower()
        if ACTIVE_PROVIDER == "openrouter" and ("insufficient credits" in err or "error code: 402" in err):
            try:
                fallback_agent = _build_agent_with_llm(agent_name, _build_openrouter_fallback_llm())
                result = _kickoff(fallback_agent, task_description)
                return SubAgentResult(
                    agent=agent_name,
                    output=str(result),
                    tool_calls=[],
                    success=True,
                    error=None,
                )
            except Exception as fallback_exc:
                exc = fallback_exc

        return SubAgentResult(
            agent=agent_name,
            output="",
            tool_calls=[],
            success=False,
            error=str(exc),
        )
