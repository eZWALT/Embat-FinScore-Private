"""Minimal tool-calling loop on LangChain messages. No framework state; the caller keeps history."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from poc.agent.tools import PlotRegistry, use_registry
from poc.llm import get_llm


@dataclass
class TurnResult:
    answer: str
    messages: list[BaseMessage]
    plots: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)  # [{name, args, result_preview}]


def run_turn(system: str, history: list[BaseMessage], user_text: str, tools, max_steps: int = 8) -> TurnResult:
    """history: prior Human/AI messages (no system). Returns the assistant answer and the new messages to append."""
    llm = get_llm().bind_tools(tools)
    by_name = {t.name: t for t in tools}
    reg = PlotRegistry()
    token = use_registry(reg)
    new: list[BaseMessage] = [HumanMessage(content=user_text)]
    calls: list[dict] = []
    try:
        for _ in range(max_steps):
            ai: AIMessage = llm.invoke([SystemMessage(content=system), *history, *new])
            new.append(ai)
            if not ai.tool_calls:
                break
            for tc in ai.tool_calls:
                t = by_name.get(tc["name"])
                try:
                    out = t.invoke(tc["args"]) if t else f'{{"error":"unknown tool {tc["name"]}"}}'
                except Exception as e:  # tool bug surfaces to the model, not the user
                    out = f'{{"error":"{type(e).__name__}: {str(e)[:200]}"}}'
                out = str(out)
                calls.append({"name": tc["name"], "args": tc["args"], "result_preview": out[:300]})
                new.append(ToolMessage(content=out, tool_call_id=tc["id"]))
        else:
            new.append(AIMessage(content="I stopped after too many tool calls. Ask a narrower question."))
    finally:
        try:
            from poc.agent.tools import _registry
            _registry.reset(token)
        except Exception:
            pass
    answer = next((m.content for m in reversed(new) if isinstance(m, AIMessage) and m.content), "")
    if isinstance(answer, list):  # content blocks
        answer = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in answer)
    return TurnResult(answer=answer, messages=new, plots=reg.plots, tool_calls=calls)
