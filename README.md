# Personal Assistant Capstone

An agentic personal assistant that plans with Tree-of-Thought reasoning,
delegates to specialized CrewAI agents, grounds answers with retrieval, and
gates risky actions behind a QA critic and human approval.

## Problem & user

Everyday productivity tasks (to-dos, notes, quick facts like the weather)
require juggling several small tools. This assistant gives a single
conversational entry point that plans, executes, checks its own work, and
asks for confirmation before anything irreversible, for an individual user
managing their own tasks and notes.

## Architecture

- **Orchestrator (LangGraph)** — owns the reasoning loop: Tree-of-Thought
  planning, routing, the Critic/QA gate, the human-in-the-loop approval gate,
  and response composition. See [`src/graph/`](src/graph/).
- **Domain agents (CrewAI)** — three single-responsibility agents dispatched
  as tool calls from the graph, not as separate LangGraph nodes:
  - Scheduler & To-Do (`scheduler_todo`)
  - Notes & Knowledge — retrieval-augmented (`notes_knowledge`)
  - Weather & External Info (`weather_external`) — calls the free
    [Open-Meteo](https://open-meteo.com/) geocoding + forecast APIs
    (no API key required). If a location resolves to a state/region/country
    rather than a specific city (e.g. "Michigan"), the tool asks a
    clarifying question with city suggestions instead of guessing.
    See [`src/tools/external.py`](src/tools/external.py).
  See [`src/agents/executor.py`](src/agents/executor.py).
- **Guardrails**:
  1. Harness retry/step/timeout limits ([`src/harness/`](src/harness/)).
  2. Critic/QA gate scoring every result before it's returned
     ([`src/agents/supervisor.py`](src/agents/supervisor.py)).
  3. Human-in-the-loop approval for sensitive actions, enforced **before**
     dispatch via `LangGraph.interrupt()`.
  4. Scoped MCP-style tool registry — each agent only gets its own allowlisted
     tools ([`src/tools/mcp_client.py`](src/tools/mcp_client.py)).
- **Memory** — Chroma vector store for notes/knowledge (RAG) plus a
  `MemorySaver` checkpointer for conversation state
  ([`src/tools/memory.py`](src/tools/memory.py), [`src/graph/workflow.py`](src/graph/workflow.py)).

Full diagram: [`../Assignment-overview/DESIGN/FINAL-DESIGN.png`](../Assignment-overview/DESIGN/FINAL-DESIGN.png).

```
User -> plan (ToT) -> [human_approval?] -> dispatch (CrewAI) -> critic (QA gate)
                                                                    |-- pass -> compose -> User
                                                                    |-- fail, retries left -> retry -> plan
                                                                    |-- fail, retries exhausted -> escalate -> User
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
Copy-Item .env.example .env   # then fill in your API key
```

Edit `config/settings.yaml` to choose `llm.provider` (`openai` or
`openrouter`) and set `.env` with the matching key (`OPENAI_API_KEY` or
`OPENROUTER_API_KEY`).

## Usage

```powershell
.\.venv\Scripts\python.exe main.py
```

```
You: Add a to-do to buy milk tomorrow
Assistant: To-do item added: buy milk for tomorrow.
```

Sensitive actions (e.g. "cancel my event") pause and prompt for
`[approve/reject]` before any CrewAI agent runs.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m src.harness.evaluate
```

Runs a fixed set of representative queries across all three domain agents
plus an approval-required action, and writes metrics (QA score, latency,
escalation rate) to `data/logs/evaluation_results.json`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
