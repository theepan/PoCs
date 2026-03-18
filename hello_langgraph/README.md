# LangGraph ReAct Agent — PoC

A minimal **ReAct** (Reason + Act) agent built with [LangGraph](https://github.com/langchain-ai/langgraph) and OpenAI.
The agent decides whether to call a tool or answer directly, loops until it has a final answer, and returns it.

```
User query
    │
    ▼
┌─────────┐   tool calls?   ┌───────┐
│  agent  │ ──────────────► │ tools │
│  (LLM)  │ ◄────────────── │       │
└─────────┘   tool results  └───────┘
    │
    │ final answer
    ▼
  output
```

---

## Features

| Feature | Detail |
|---|---|
| LLM | OpenAI `gpt-4o-mini` (swappable) |
| Framework | LangGraph `StateGraph` |
| Pattern | ReAct loop (agent ↔ tools) |
| Tools | `multiply`, `lookup_hs_code` (stub) |

---

## Quick start

```bash
# 1. Clone
git clone <repo-url>
cd hello_langgraph

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Export your OpenAI key (if not already in your shell)
export OPENAI_API_KEY=sk-...

# 5. Run
python agent.py
```

**Expected output:**

```
============================================================
Q: What's 17.5 * 42.3?
============================================================
A: 17.5 multiplied by 42.3 equals 740.25.

============================================================
Q: What HS code would apply to copper wire?
============================================================
A: The HS code for copper wire is 7408.11 – Refined copper wire.

============================================================
Q: What's the HS code for a laptop, and multiply its heading number 8471 by 3?
============================================================
A: The HS code for a laptop is 8471.30. 8471 × 3 = 25413.
```

---

## Project structure

```
hello_langgraph/
├── agent.py          # Graph definition and CLI entry-point
├── tools.py          # Tool implementations
├── requirements.txt
├── .env.example      # Key template (copy → .env, never commit .env)
└── .gitignore
```

---

## How it works

1. **State** — a simple `TypedDict` holding the conversation message list.
2. **`agent` node** — calls the LLM with the current messages; the LLM either requests tool calls or produces a final answer.
3. **`tools` node** — executes each requested tool and appends `ToolMessage` results.
4. **Router** — `should_continue` checks whether the last message contains tool calls; if yes → `tools`, if no → `END`.

### Adding your own tool

```python
# tools.py
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """One-sentence docstring — the LLM reads this to decide when to use it."""
    return "result"

TOOLS = [..., my_tool]   # add it here and it's automatically wired in
```

---

## Configuration

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key (required) |

To switch models, change the `model` argument in `agent.py`:

```python
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(TOOLS)
```

---

## License

MIT
