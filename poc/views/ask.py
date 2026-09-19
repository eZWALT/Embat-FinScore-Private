"""Ask: chat with tools over the bundle and the cleaned DuckDB, scoped to a selected company or group."""

from __future__ import annotations

import json

import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage

from poc import plots
from poc.agent.context import system_prompt
from poc.agent.runner import run_turn
from poc.agent.tools import CHAT_TOOLS
from poc.bundle import Bundle, load
from poc.llm import model_name

HISTORY_CAP = 12  # LangChain messages kept for the model

SUGGESTED = {
    "company": [
        "¿Por qué tiene este score?",
        "¿Qué ha cambiado en los últimos 3 meses?",
        "¿Es un bache o un deterioro?",
        "¿Qué clientes tienen facturas vencidas?",
        "¿Cómo se compara con su grupo?",
    ],
    "group": [
        "¿Qué empresas del grupo necesitan atención?",
        "¿Cómo ha evolucionado el grupo?",
        "¿Qué alertas hay en el grupo?",
    ],
    "none": [
        "¿Qué empresas tienen alertas act este mes?",
        "¿Qué significa el score?",
    ],
}


# ----------------------------------------------------------------------------- state


def _init_state() -> None:
    st.session_state.setdefault("ask_display", [])  # [{role, content, plots, tool_calls, error}]
    st.session_state.setdefault("ask_history", [])  # LangChain messages for the model


def _reset_chat() -> None:
    st.session_state.ask_display = []
    st.session_state.ask_history = []


def trim_history(msgs: list[BaseMessage], cap: int = HISTORY_CAP) -> list[BaseMessage]:
    """Drop the oldest Human..AI block (with its tool messages) until at most `cap` messages remain."""
    while len(msgs) > cap:
        nxt = next((i for i, m in enumerate(msgs) if i > 0 and isinstance(m, HumanMessage)), None)
        if nxt is None:
            break
        del msgs[:nxt]
    return msgs


# ----------------------------------------------------------------------------- sidebar / context


def _company_label(c: dict) -> str:
    score = "–" if c.get("score") is None else f"{c['score']:.0f}"
    label = f"{c['company_id']} · score {score} · {c.get('trajectory') or 'n/a'}"
    if c.get("guard"):
        label += f" · {c['guard']}"
    return label


def _sidebar(b: Bundle) -> tuple[str, str | None, str | None]:
    """Returns (kind, entity_id, extra prompt)."""
    with st.sidebar:
        st.subheader("Contexto")
        kind = st.radio("Entidad", ["Company", "Group", "None"], horizontal=True, key="ask_ctx_kind")
        entity_id, extra = None, None
        if kind == "Company":
            rows = {c["company_id"]: c for c in b.companies}
            entity_id = st.selectbox("Company", list(rows), format_func=lambda x: _company_label(rows[x]), key="ask_ctx_company")
            extra = (
                f"Session context: the user is looking at company {entity_id} (group {rows[entity_id]['group_id']}). "
                "Use it as the default entity when none is named."
            )
        elif kind == "Group":
            rows = {g["group_id"]: g for g in b.groups}
            entity_id = st.selectbox(
                "Group", list(rows), format_func=lambda x: f"{x} · {rows[x]['n_companies']} companies", key="ask_ctx_group"
            )
            extra = (
                f"Session context: the user is looking at group {entity_id} ({rows[entity_id]['n_companies']} companies). "
                "Use it as the default entity when none is named."
            )
        else:
            extra = "Session context: no company or group selected. Ask which one when the question needs an entity."
        st.button("Nueva conversación", on_click=_reset_chat, width="stretch")
    return kind.lower(), entity_id, extra


# ----------------------------------------------------------------------------- rendering


def _render_tool_calls(calls: list[dict]) -> None:
    if not calls:
        return
    with st.expander(f"Herramientas usadas ({len(calls)})"):
        for tc in calls:
            args = json.dumps(tc.get("args", {}), ensure_ascii=False, default=str)
            st.code(f"{tc['name']}({args})", language="text")
            preview = (tc.get("result_preview") or "")[:300]
            if preview:
                st.caption(preview)


def _render_entry(entry: dict) -> None:
    with st.chat_message(entry["role"]):
        if entry.get("error"):
            st.error(entry["error"])
        elif entry.get("content"):
            st.markdown(entry["content"])
        elif entry["role"] == "assistant":
            st.write("Sin respuesta")
        for spec in entry.get("plots") or []:
            try:
                plots.render(spec)
            except Exception as e:  # a bad spec must not break the page
                st.warning(f"No se pudo dibujar «{spec.get('title', '')}»: {e}")
        _render_tool_calls(entry.get("tool_calls") or [])


def _suggestions(kind: str) -> str | None:
    """Entity-aware suggested questions; returns the clicked one (submitted as if typed)."""
    qs = SUGGESTED.get(kind, SUGGESTED["none"])
    st.caption("Preguntas sugeridas")
    cols = st.columns(len(qs))
    clicked = None
    for col, q in zip(cols, qs):
        if col.button(q, key=f"ask_sugg_{q}", width="stretch"):
            clicked = q
    return clicked


# ----------------------------------------------------------------------------- turn


def _run(b: Bundle, extra: str | None, text: str) -> None:
    display = st.session_state.ask_display
    display.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"):
        with st.spinner("Consultando herramientas…"):
            try:
                history = trim_history(st.session_state.ask_history)
                result = run_turn(system_prompt("chat", b, extra), history, text, CHAT_TOOLS)
            except Exception as e:
                display.append({"role": "assistant", "error": f"{type(e).__name__}: {e}", "plots": [], "tool_calls": []})
                st.rerun()
    history.extend(result.messages)
    st.session_state.ask_history = trim_history(history)
    display.append(
        {"role": "assistant", "content": result.answer, "plots": result.plots, "tool_calls": result.tool_calls}
    )
    st.rerun()


# ----------------------------------------------------------------------------- page


def render() -> None:
    _init_state()
    try:
        b = load()
    except FileNotFoundError as e:
        st.title("Ask")
        st.error(str(e))
        return

    kind, entity_id, extra = _sidebar(b)

    st.title("Ask")
    st.caption(
        f"Modelo: `{model_name()}` · bundle as-of {b.manifest['as_of_month']} · "
        + (f"contexto: {entity_id}" if entity_id else "sin entidad seleccionada")
    )

    for entry in st.session_state.ask_display:
        _render_entry(entry)

    suggested = _suggestions(kind) if not st.session_state.ask_display else None

    typed = st.chat_input("Pregunta sobre tu empresa o grupo")
    text = typed or suggested
    if text:
        _run(b, extra, text)
