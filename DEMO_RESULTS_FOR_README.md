# Demo Execution Traces & Sample Outputs

This document contains comprehensive trace logs and sample outputs from all demo scenarios, showcasing the Personal Assistant Capstone system in action across different reasoning modes and agent combinations.

> **Note:** All timestamps, dates, and specific LLM phrasing will be regenerated on actual runs. These are reference transcripts from the demo presentation scripts.

---

## Single-Agent Demos

Demonstrates individual agent capabilities and fast-path reasoning for routine tasks.

### Demo 1: Scheduler Agent - Schedule an Appointment

**User Input:**
```
Schedule a dentist appointment on Tuesday at 3 PM.
```

**Execution Trace:**
```
PLANNING: Simple/Direct reasoning (routine task)...
   SELECTED: scheduler_todo
DISPATCHING: Sending task to scheduler_todo agent...
   ✓ scheduler_todo completed
QA SCORING: Evaluating agent response...
   Score: 0.92/1.0
COMPOSING: Building final response...
   ✓ Response ready
```

**Final Response:**
```
Confirmed: Dentist appointment at [date]T15:00:00.
```

**Metrics:**
- Elapsed Time: ~6-7 seconds
- QA Score: 0.92/1.0
- Agent Dispatched: `scheduler_todo`

---

### Demo 2: Scheduler Agent - List Today's Events

**User Input:**
```
Show me all my scheduled events for today.
```

**Execution Trace:**
```
PLANNING: Simple/Direct reasoning (routine task)...
   SELECTED: scheduler_todo
DISPATCHING: Sending task to scheduler_todo agent...
   ✓ scheduler_todo completed
QA SCORING: Evaluating agent response...
   Score: 0.95/1.0
COMPOSING: Building final response...
   ✓ Response ready
```

**Final Response:**
```
Here are your tasks for today, [date]:

- [task 1] at [time]
- [task 2] at [time]
...
```

**Metrics:**
- Elapsed Time: ~3 seconds
- QA Score: 0.95/1.0
- Agent Dispatched: `scheduler_todo`

---

### Demo 3: Notes Agent - Save a Preference

**User Input:**
```
Save that my favorite hobby is painting.
```

**Execution Trace:**
```
PLANNING: Simple/Direct reasoning (routine task)...
   SELECTED: notes_knowledge
DISPATCHING: Sending task to notes_knowledge agent...
   ✓ notes_knowledge completed (ChromaDB persistence)
QA SCORING: Evaluating agent response...
   Score: 1.0/1.0 (Perfect response)
COMPOSING: Building final response...
   ✓ Response ready
```

**Final Response:**
```
Your note has been saved: "My favorite hobby is painting."
```

**Metrics:**
- Elapsed Time: ~16 seconds (includes ChromaDB init)
- QA Score: 1.0/1.0
- Agent Dispatched: `notes_knowledge`

---

### Demo 4: Notes Agent - Query Saved Preference

**User Input:**
```
What is my favorite hobby?
```

**Execution Trace:**
```
PLANNING: Simple/Direct reasoning (routine task)...
   SELECTED: notes_knowledge
DISPATCHING: Sending task to notes_knowledge agent...
   ✓ notes_knowledge completed (retrieval from Demo 3 note)
QA SCORING: Evaluating agent response...
   Score: 1.0/1.0 (Retrieval confirmed)
COMPOSING: Building final response...
   ✓ Response ready
```

**Final Response:**
```
Your favorite hobby is painting.
```

**Metrics:**
- Elapsed Time: ~3 seconds
- QA Score: 1.0/1.0
- Agent Dispatched: `notes_knowledge`
- Thread Context: Same thread as Demo 3 (retrieval grounded in saved note)

---

### Demo 5: Weather Agent - Current Weather Query

**User Input:**
```
What's the current weather in Pittsburgh?
```

**Execution Trace:**
```
PLANNING: Simple/Direct reasoning (routine task)...
   SELECTED: weather_external
DISPATCHING: Sending task to weather_external agent...
   ✓ weather_external completed (external API call)
QA SCORING: Evaluating agent response...
   Score: 0.9/1.0
COMPOSING: Building final response...
   ✓ Response ready
```

**Final Response:**
```
The current weather in Pittsburgh is clear sky with a temperature of 32.7C. The
humidity is at 47%, there has been no precipitation, and the wind is blowing at
14.5 km/h.
```

**Metrics:**
- Elapsed Time: ~7 seconds
- QA Score: 0.9/1.0
- Agent Dispatched: `weather_external`

---

### Demo 6: Weather Agent - Temperature Query

**User Input:**
```
What's the temperature in Miami right now?
```

**Execution Trace:**
```
PLANNING: Simple/Direct reasoning (routine task)...
   SELECTED: weather_external
DISPATCHING: Sending task to weather_external agent...
   ✓ weather_external completed (external API call)
QA SCORING: Evaluating agent response...
   Score: 0.9/1.0
COMPOSING: Building final response...
   ✓ Response ready
```

**Final Response:**
```
The current temperature in Miami is 28.7C with overcast skies. The humidity is at
82%, and there is no precipitation. The wind is blowing at 4.9 km/h.
```

**Metrics:**
- Elapsed Time: ~7-8 seconds
- QA Score: 0.9/1.0
- Agent Dispatched: `weather_external`

---

## Multi-Agent Demos

Demonstrates automatic agent orchestration for complex requests requiring multiple capabilities.

### Demo 1: Schedule + Weather Check (Scheduler → Weather)

**User Input:**
```
Schedule an outdoor picnic in Denver on Friday at 2 PM, and check if the weather 
will be suitable for an outdoor event.
```

**Execution Trace:**
```
PLANNING: Simple/Direct reasoning (routine task)...
   SELECTED: scheduler_todo
DISPATCHING: Sending task to scheduler_todo agent...
   ✓ scheduler_todo completed
   📋 Multi-agent scenario detected: scheduler_todo, weather_external
   📤 Also dispatching weather_external agent...
      ✓ weather_external completed
QA SCORING: Evaluating multi-agent response...
   Score: 0.92/1.0 | Feedback: Scheduling confirmation matches requested weekday and time.
COMPOSING: Building final response...
   Combining 2 agent results...
   ✓ Response ready
```

**Agent Results:**

| Agent | Status | Output |
|-------|--------|--------|
| scheduler_todo | ✓ Success | Confirmed: Outdoor picnic in Denver at [date]T14:00:00. |
| weather_external | ✓ Success | The current weather in Denver, Colorado is overcast with a temperature of 31.8C. The humidity is at 19%, and there is no precipitation expected. Wind is blowing at 14.8 km/h. Overall, the weather seems suitable for an outdoor picnic. |

**Final Composed Response:**
```
Confirmed: Outdoor picnic in Denver at [date]T14:00:00.

The current weather in Denver, Colorado is overcast with a temperature of 31.8C. 
The humidity is at 19%, and there is no precipitation expected. Wind is blowing at 
14.8 km/h. Overall, the weather seems suitable for an outdoor picnic.
```

**Metrics:**
- Elapsed Time: ~10-18 seconds
- QA Score: 0.92/1.0
- Agents Dispatched: 2 (sequential)
  1. `scheduler_todo` (primary)
  2. `weather_external` (detected automatically)

---

### Demo 2: Schedule + Save Note (Scheduler → Notes)

**User Input:**
```
Schedule volleyball in Miami on Saturday at 4 PM and save it as my favorite sport.
```

**Execution Trace:**
```
PLANNING: Simple/Direct reasoning (routine task)...
   SELECTED: scheduler_todo
DISPATCHING: Sending task to scheduler_todo agent...
   ✓ scheduler_todo completed
   📋 Multi-agent scenario detected: scheduler_todo, notes_knowledge
   📤 Also dispatching notes_knowledge agent...
      ✓ notes_knowledge completed
QA SCORING: Evaluating multi-agent response...
   Score: 0.92/1.0 | Feedback: Scheduling confirmation matches requested weekday and time.
COMPOSING: Building final response...
   Combining 2 agent results...
   ✓ Response ready
```

**Agent Results:**

| Agent | Status | Output |
|-------|--------|--------|
| scheduler_todo | ✓ Success | Confirmed: Volleyball - favorite sport at [date]T16:00:00. |
| notes_knowledge | ✓ Success | The note has been saved: "Favorite sport: volleyball." |

**Final Composed Response:**
```
Confirmed: Volleyball - favorite sport at [date]T16:00:00.

The note has been saved: "Favorite sport: volleyball."
```

**Metrics:**
- Elapsed Time: ~5-18 seconds
- QA Score: 0.92/1.0
- Agents Dispatched: 2 (sequential)
  1. `scheduler_todo` (primary)
  2. `notes_knowledge` (detected automatically)

---

## Tree-of-Thought (ToT) Demos

Demonstrates advanced reasoning with multi-path exploration for complex constraint-satisfaction problems.

### Demo 1: Constraint-Satisfaction Scheduling

**User Input:**
```
I need to schedule 3 meetings: one that needs 2 hours, one that needs 1 hour, 
and one that needs 30 minutes. They can't overlap and I'm only available 
Tuesday-Thursday 9-5. Find the best way to schedule all three.
```

**Execution Trace:**
```
PLANNING: Multi-path exploration needed - using Tree-of-Thought reasoning...
PLANNING: Starting Tree-of-Thought reasoning...
   Depth 0: Generated 2 candidate(s)
     Candidate 1: scheduler_todo (confidence: 0.90)
       Thought: Schedule the 2-hour meeting first on Tuesday from 9 AM to 11 AM, 
       the 1-hour meeting on Tuesday from 11 AM to 12 PM, and the 30-minute 
       meeting on Tuesday from 12 PM to 12:30 PM. This utilizes the available 
       time on Tuesday efficiently.
     Candidate 2: scheduler_todo (confidence: 0.80)
       Thought: Schedule the 2-hour meeting on Tuesday from 9 AM to 11 AM, 
       the 1-hour meeting on Wednesday from 9 AM to 10 AM, and the 30-minute 
       meeting on Wednesday from 10 AM to 10:30 AM. This allows for meetings 
       on two different days without overlap.
   Confident solution found at depth 0
   SELECTED: scheduler_todo (confidence: 0.90)
DISPATCHING: Sending task to scheduler_todo agent...
   ✓ scheduler_todo completed
QA SCORING: Evaluating agent response...
   Score: 0.90/1.0 | Feedback: Exploratory optimization request answered with 
   a substantive multi-step response
COMPOSING: Building final response...
   ✓ Response ready
```

**Candidate Branches:**

| Branch | Confidence | Agent | Reasoning |
|--------|------------|-------|-----------|
| 1 (SELECTED) | 0.90 | scheduler_todo | Schedule 2h (Tue 9-11), 1h (Tue 11-12), 30m (Tue 12-12:30) - Efficient single-day packing |
| 2 | 0.80 | scheduler_todo | Schedule 2h (Tue 9-11), 1h (Wed 9-10), 30m (Wed 10-10:30) - Spread across two days |

**Final Response:**
```
Confirmed: 2-hour meeting at [date]T09:00:00
Confirmed: 1-hour meeting at [date]T11:00:00
Confirmed: 30-minute meeting at [date]T12:00:00
```

**Metrics:**
- Reasoning Mode: Tree-of-Thought
- Elapsed Time: ~9-10 seconds
- QA Score: 0.90/1.0
- Candidate Branches Generated: 2
- Agent Dispatched: `scheduler_todo`
- Depth: 0 (confident solution found immediately)

---

### Demo 2: Multi-Constraint Itinerary Exploration

**User Input:**
```
Plan my weekend: I want to visit 3 different cities, stay under 4 hours driving, 
and include both indoor and outdoor activities. Show me different ways to 
organize this.
```

**Execution Trace:**
```
PLANNING: Multi-path exploration needed - using Tree-of-Thought reasoning...
PLANNING: Starting Tree-of-Thought reasoning...
   Depth 0: Generated 2 candidate(s)
     Candidate 1: scheduler_todo (confidence: 0.90)
       Thought: Plan a weekend trip visiting three cities within a 4-hour driving 
       radius, including a mix of indoor and outdoor activities.
     Candidate 2: scheduler_todo (confidence: 0.90)
       Thought: Create a detailed itinerary for a weekend trip to three cities, 
       ensuring a balance of indoor and outdoor activities while adhering to the 
       driving limit.
   Confident solution found at depth 0
   SELECTED: scheduler_todo (confidence: 0.90)
   Exploratory itinerary request detected; normalizing target to notes_knowledge
DISPATCHING: Sending task to notes_knowledge agent...
   ✓ notes_knowledge completed
QA SCORING: Evaluating agent response...
   Score: 0.90/1.0 | Feedback: Exploratory itinerary response provided multiple 
   structured options as requested
COMPOSING: Building final response...
   ✓ Response ready
```

**Candidate Branches:**

| Branch | Confidence | Agent | Strategy |
|--------|------------|-------|----------|
| 1 | 0.90 | notes_knowledge | Provide 2-4 alternative weekend plans with trade-offs and recommendation |
| 2 | 0.90 | scheduler_todo | Create detailed itinerary for 3-city trip with indoor/outdoor balance |

**Final Response (Selected Branch 1):**
```
Here are four alternative weekend plans for visiting three different cities within 
a 4-hour driving radius, including both indoor and outdoor activities:

### Plan 1: Hudson Valley Adventure
- Cities: Beacon, Cold Spring, Poughkeepsie
- Activities: Dia:Beacon museum (indoor) + Mount Beacon hike (outdoor); waterfront 
  stroll (outdoor); Walkway Over the Hudson (outdoor) + FDR Presidential Library (indoor)
- Trade-offs: Great mix of art, history, and nature; some driving back and forth.

### Plan 2: Coastal Getaway
- Cities: Asbury Park, Long Branch, Red Bank
- Activities: beach/boardwalk (outdoor) + Stone Pony (indoor); Pier Village (outdoor); 
  Count Basie Center (indoor) + Navesink River walk (outdoor)
- Trade-offs: Great for beach lovers; may be crowded on weekends.

### Plan 3: Historic Cities Tour
- Cities: Philadelphia, Princeton, New Brunswick
- Activities: Liberty Bell/Independence Hall (indoor) + Fairmount Park (outdoor); 
  Princeton campus (outdoor) + art museum (indoor); State Theatre (indoor) + Raritan 
  River walk (outdoor)
- Trade-offs: Rich in history/culture; more urban driving.

### Plan 4: Nature and Culture Combo
- Cities: Catskill, Kingston, Woodstock
- Activities: Catskill Mountains hike (outdoor) + art galleries (indoor); historic 
  waterfront + maritime museum (indoor); outdoor scenery + Woodstock Artists museum (indoor)
- Trade-offs: Peaceful nature retreat; limited dining options.

### Recommendation
Plan 1: Hudson Valley Adventure is recommended for its balance of art, history, and 
outdoor activities, making it suitable for a diverse range of interests while keeping 
travel times manageable.
```

**Metrics:**
- Reasoning Mode: Tree-of-Thought
- Elapsed Time: ~7-11 seconds
- QA Score: 0.90/1.0
- Candidate Branches Generated: 2
- Agent Dispatched: `notes_knowledge` (auto-normalized from scheduler_todo)
- Depth: 0 (confident solution found immediately)

---

## Key Insights & Lessons Learned

### Single-Agent Performance
- **Fast Path:** Routine scheduling/query tasks complete in 3-7 seconds
- **Cold Start:** First ChromaDB/embedding init adds ~10-16 seconds (subsequent calls: ~3 seconds)
- **QA Scores:** Consistent 0.9-1.0 for well-defined requests
- **Reliability:** 100% success rate across all single-agent demos

### Multi-Agent Coordination
- **Sequential Dispatch:** Secondary agents dispatched after primary completes
- **Automatic Detection:** System correctly identifies multi-agent scenarios
- **Result Composition:** Natural integration of multiple agent outputs
- **Timing:** Total time ≈ primary agent + secondary agent (not purely sequential)

### Tree-of-Thought Reasoning
- **Branch Generation:** 2 candidate solutions generated at Depth 0
- **Quick Resolution:** Confident solutions found immediately (no deep exploration needed)
- **Confidence Scoring:** Branch confidence 0.80-0.90
- **Agent Normalization:** System adapts agent selection based on exploration results
- **Complex Queries:** Handles constraint-satisfaction and exploratory planning requests

### Rejected Candidate Patterns

**Single-Agent Concern:**
- ❌ "List all my tasks for this week" 
  - Issue: Word "list" alone causes misrouting (notes_knowledge vs scheduler_todo)
  - Fix: Use more specific language like "Show me all my scheduled events for today"

**Multi-Agent Concern:**
- ❌ "Save a note that my project deadline is next month, then schedule a reminder meeting for September 15th at 10 AM"
  - Issue: Date ambiguity conflict ("next month" vs concrete date) triggers QA retry loop
  - Result: Escalation after 35-50 seconds with low QA scores (~0.1)
  - Lesson: Avoid temporal contradictions in single prompt

### Best Practices for Demos
1. **Chain save + query on same thread** for notes_knowledge demos (thread_id carries context)
2. **Avoid ambiguous date references** when combining temporal constraints
3. **Use specific clarifying language** for task type disambiguation
4. **Start with single-agent** demos (faster, simpler) before multi-agent
5. **Highlight QA scores** to show quality validation in action
6. **Monitor elapsed time** to demonstrate performance characteristics

---

## Summary Statistics

| Category | Metric | Range | Note |
|----------|--------|-------|------|
| **Single-Agent** | Execution Time | 3-16s | Cold start ~16s, warm ~3s |
| **Single-Agent** | QA Score | 0.90-1.0 | Highly consistent |
| **Multi-Agent** | Execution Time | 5-18s | Sequential dispatch overhead |
| **Multi-Agent** | QA Score | 0.92/1.0 | Composition validated |
| **ToT** | Execution Time | 7-11s | Exploration at Depth 0 |
| **ToT** | QA Score | 0.90/1.0 | Substantive responses validated |
| **Overall** | Success Rate | 100% | All scenarios completed |

---

**Ready for README integration and live presentations.** ✅
