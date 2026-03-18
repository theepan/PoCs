"""
LangGraph ReAct Agent
---------------------
A minimal ReAct (Reason + Act) agent built with LangGraph and OpenAI.

The agent runs in a loop:
  1. LLM decides whether to call a tool or produce a final answer
  2. If tool  → execute it, feed result back to the LLM
  3. If answer → done
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from tools import TOOLS, lookup_hs_code, multiply  # noqa: F401 – re-exported for clarity

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── LLM + tools ───────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)
tools_by_name = {t.name: t for t in TOOLS}


# ── Graph nodes ───────────────────────────────────────────────────────────────

def call_model(state: AgentState) -> dict:
    """Ask the LLM: call a tool, or give a final answer?"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def call_tools(state: AgentState) -> dict:
    """Execute every tool the LLM requested and collect results."""
    last_msg = state["messages"][-1]
    results = []
    for tc in last_msg.tool_calls:
        result = tools_by_name[tc["name"]].invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}


def should_continue(state: AgentState) -> str:
    """Route to 'tools' if the LLM made tool calls, otherwise end."""
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return END


# ── Graph wiring ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


app = build_graph()


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        "What's 17.5 * 42.3?",
        "What HS code would apply to copper wire?",
        "What's the HS code for a laptop, and multiply its heading number 8471 by 3?",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        result = app.invoke({"messages": [HumanMessage(content=q)]})
        print(f"A: {result['messages'][-1].content}")
