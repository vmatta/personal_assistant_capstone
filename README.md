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

## Requirements

**Python 3.11.9** and **pip 26.1.1+** are required.

This project uses Python 3.11.9 features and pip 26.1.1+ dependency resolution. It has been tested exclusively on these versions.

### Check Your Versions

```powershell
python --version   # Should show 3.11.9
pip --version      # Should show 26.1.1 or higher
```

### Upgrade pip if needed

If you have an older pip version, upgrade it:

```powershell
python -m pip install --upgrade pip
```

### Python Version Setup

**Recommended setup method:** If you use `pyenv`, the `.python-version` file in the root directory will automatically select Python 3.11.9 when you enter the project folder.

```bash
# Using pyenv (recommended)
pyenv install 3.11.9  # if not already installed
cd personal_assistant_capstone  # .python-version auto-selects 3.11.9

# Or manually specify Python version
python3.11.9 -m venv .venv
```

### Environment Validation (Optional)

Before running the app, you can validate your environment:

```powershell
python scripts/check_env.py
```

This will verify Python 3.11.9+ and pip 26.1.1+ are installed.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install --upgrade pip  # Ensure pip 26.1.1+
.\.venv\Scripts\pip install -r requirements.txt
Copy-Item .env.example .env   # then fill in your API key
```

### LLM Provider Configuration

Choose your LLM provider and configure both `.env` and `config/settings.yaml`:

#### Option 1: OpenRouter (I have used this to test and demo)

1. **In `.env`:**
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
   ```

2. **In `config/settings.yaml`:**
   ```yaml
   llm:
     provider: "openrouter"
     openrouter:
       base_url: "https://openrouter.ai/api/v1"
       model: "openai/gpt-4o-mini"  # I have used this for testing and demo
   ```
#### Option 2: OpenAI (only if necessary)

1. **In `.env`:**
   ```
   OPENAI_API_KEY=sk-proj-your-actual-key-here
   ```

2. **In `config/settings.yaml`:**
   ```yaml
   llm:
     provider: "openai"
     openai:
       base_url: "https://api.openai.com/v1"
       model: "gpt-4o-mini"  # or another OpenAI model
   ```

## Usage

**Start the Streamlit UI** (opens at http://localhost:8501):
(Might take few seconds to load. Loads chromaDb in background for few seconds after)

```powershell
.\.venv\Scripts\streamlit run streamlit_app.py
```

**Or run via CLI** (without UI):

```powershell (not tested using powershell)
.\.venv\Scripts\python.exe main.py
```

CLI example output:
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

## Demo Scenarios: Prompt, Reasoning, Output

Live-run transcripts from `demo/*.py` (single-agent, multi-agent, Tree-of-Thought)
plus 5 additional scenarios exercising the guardrails not covered by those scripts.
`Assignment-overview/` was checked and contains only course assignment
instructions, not runnable prompts.

### A. Single-Agent Demos (`demo/single_agent_demo_presentation.py`)

| # | Prompt | Reasoning | Output |
|---|--------|-----------|--------|
| 1 | "Schedule a dentist appointment on Tuesday at 3 PM." | Direct/fast path -> routed to `scheduler_todo`. QA score 0.92. | "Scheduled: Dentist appointment for 2026-09-08T15:00:00." (4.2s) |
| 2 | "Show me all my scheduled events for today." | Direct path -> `scheduler_todo` list/retrieval. QA 0.95. | "Here are your tasks for today: None scheduled." (3.2s) |
| 3 | "Save that my favorite hobby is painting." | Direct path -> `notes_knowledge` save. QA 1.00. | "I have saved that your favorite hobby is painting." (3.2s) |
| 4 | "What is my favorite hobby?" (same thread as #3) | Direct path -> `notes_knowledge` retrieval, grounded in the note just saved. QA 1.00. | "Your favorite hobby is painting." (2.5s) |
| 5 | "What's the current weather in Pittsburgh?" | Direct path -> `weather_external` calls Open-Meteo. QA 0.90. | "The current weather in Pittsburgh is mainly clear with a temperature of 23.9°C..." (10.9s) |
| 6 | "What's the temperature in Miami right now?" | Direct path -> `weather_external`. QA 0.90. | "The current temperature in Miami is 28.5°C with overcast conditions..." (10.2s) |

### B. Multi-Agent Demos (`demo/multiagent_demo_presentation.py`)

| # | Prompt | Reasoning | Output |
|---|--------|-----------|--------|
| 1 | "Schedule an outdoor picnic in Denver on Friday at 2 PM, and check if the weather will be suitable for an outdoor event." | Primary agent `scheduler_todo` books the event; system detects a secondary weather need and sequentially dispatches `weather_external` (parallel dispatch was tried and deadlocked in LangGraph, so this is sequential by design). QA 0.92. | "Confirmed: Outdoor picnic in Denver at 2026-09-11T14:00:00.\n\nThe current weather in Denver is overcast, 33.7°C, humidity 14%... suitable for an outdoor event." (23.4s) |
| 2 | "Schedule volleyball in Miami on Saturday at 4 PM and save it as my favorite sport." | `scheduler_todo` books the event, then `notes_knowledge` is dispatched to save the stated preference, each scoped to only its own slice of the request. QA 0.92. | "Confirmed: Volleyball - favorite sport at 2026-09-12T16:00:00.\n\nThe note has been saved: 'Favorite sport: volleyball.'" (34.4s) |

### C. Tree-of-Thought Demos (`demo/tot_demo_presentation.py`)

| # | Prompt | Reasoning | Output |
|---|--------|-----------|--------|
| 1 | "I need to schedule 3 meetings: one that needs 2 hours, one that needs 1 hour, and one that needs 30 minutes. They can't overlap and I'm only available Tuesday-Thursday 9-5. Find the best way to schedule all three." | Complexity classifier routes to ToT (constraint satisfaction). Generates 2 candidate arrangements (Tuesday vs. Wednesday slotting), both survive at depth 0 with confidence 0.90; highest-confidence branch selected and executed. QA 0.90. | "Scheduled: 2-hour meeting for 2026-09-08T09:00:00. Confirmed: 1-hour meeting at 2026-09-08T11:30:00. Confirmed: 30-minute meeting at 2026-09-08T13:00:00." (41.3s) |
| 2 | "Plan my weekend: I want to visit 3 different cities, stay under 4 hours driving, and include both indoor and outdoor activities. Show me different ways to organize this." | ToT generates 2 candidate framings (concrete itinerary vs. "show multiple alternative plans"); the exploratory-alternatives branch is selected and routed to `notes_knowledge` for a structured comparison. QA 0.90. | Returns 4 named weekend plans (Hudson Valley, Coastal Getaway, Historic Cities, Nature & Culture), each with cities, drive times, indoor/outdoor activity mix, and trade-offs, plus a final recommendation. (6.7s) |

### D. Orchestration Logic Demo (`demo/orchestrationlogic/preference_aware_weather_scheduling_presentation.py`)

Shows 3 capabilities working together in one turn: multi-day candidate parsing, long-term preference lookup (RAG), and future-date weather forecasting.

| # | Prompt | Reasoning | Output |
|---|--------|-----------|--------|
| 1 | "If weather permits can you schedule an outdoor Saturday or Sunday" (no location) | System requires a location for weather lookups and refuses to guess one (design principle: never assume, ask). | "I can check the weather for both days, but I need a location first. Which city should I check the forecast for?" (~0s) |
| 2 | "If weather permits can you schedule an outdoor activity in New York City on Saturday or Sunday" | Parses "Saturday or Sunday" as two candidates. Searches saved notes, finds "On Sundays, I only relax at home," rules out Sunday. Fetches the actual future-date forecast for Saturday (not just "current weather"). | "- Sunday: ruled out based on your saved preference...\n- Saturday (2026-09-12): partly cloudy, 22.3°C max, 0.0mm precip — suitable...\n\nRecommendation: schedule the outdoor activity for Saturday. What time would you like?" (6.7s) |
| 3 | "Saturday at 2 PM" (follow-up, same thread) | Falls through to the normal `scheduler_todo` path and actually books the event. | "Scheduled: Outdoor activity for 2026-09-12 at 14:00." (18.5s) |

### E. Additional Scenarios (guardrails not covered above)

| # | Prompt | Reasoning | Output |
|---|--------|-----------|--------|
| 1 | "Delete all my tasks for this week." -> approve | Destructive-action detector flags this before dispatch; graph pauses via `interrupt()` and waits for human approval. On "approve," `scheduler_todo` proceeds. QA 0.95. | "Deleted 3 task(s) for this week: gym session, Dentist appointment, 30-minute meeting." (17.1s) |
| 2 | "Delete all my tasks for this week." -> reject | Same approval gate triggers; on "reject," the graph short-circuits and no agent is ever dispatched (0 agents called). | "Okay, I won't do that. Let me know if you'd like a different action instead." (1.6s) |
| 3 | "Schedule a dentist appointment on Wednesday at 3 PM." then "Schedule a team meeting on Wednesday at 3 PM." | Second request's `add_todo` tool checks existing open to-dos at the same due timestamp (`find_open_todos_at_due`) before creating a new item; finds the dentist appointment already there and returns a "Confirmed" acknowledgement instead of creating a duplicate/conflicting booking. | "Scheduled: Dentist appointment for 2026-09-09T15:00:00." then "Confirmed: Team meeting at 2026-09-09T15:00:00." — only 1 to-do exists at that slot afterward, not 2. |
| 4 | "Reschedule my networking event to sometime later, maybe Thursday, not totally sure." | Agent detects missing time-of-day and asks a clarifying question instead of guessing (self-correction / QA-guided clarification). QA 0.85 ("appropriately asked for clarification on an ambiguous request"). | "Could you please specify a preferred time of day for your networking event on Thursday? For example, would you like it in the morning, afternoon, or evening?" (2.6s) |
| 5 | "Schedule my Hawaii trip for Monday at 9 AM." then "Actually, reschedule my Hawaii trip to Wednesday at 2 PM instead." | Second turn is recognized as a reschedule of the existing item (same conversation thread) rather than a new/duplicate booking; the to-do's due time is updated. QA 0.92. | "Scheduled: Hawaii trip for 2026-09-07T09:00:00." then "Scheduled: Hawaii trip for 2026-09-09T14:00:00." |
