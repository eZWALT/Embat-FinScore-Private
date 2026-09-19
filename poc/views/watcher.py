"""Sentinel watcher page: push-style digest over a chosen watch set. Novelty engine in poc/watcher_state.py."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from poc import plots
from poc import watcher_state as ws
from poc.agent.context import system_prompt
from poc.agent.runner import run_turn
from poc.agent.tools import SENTINEL_TOOLS
from poc.bundle import load

SEV_LABEL = {"act": "Actuar", "watch": "Vigilar", "opportunity": "Oportunidades", "info": "Info"}


# ----------------------------------------------------------------------------- check


def _user_text(nov: dict) -> str:
    return f"Novelties since the last digest (JSON):\n```json\n{json.dumps(nov, ensure_ascii=False)}\n```\nWrite the digest."


def _fallback_plots(b, nov: dict) -> list[dict]:
    """Score history with alert months marked for the companies behind act/watch alerts (no LLM needed)."""
    ids, out = [], []
    for a in nov["new_alerts"]:
        eid = a["entity"]["id"]
        if a["entity"]["type"] == "company" and a["severity"] in ("act", "watch") and eid not in ids:
            ids.append(eid)
    for cid in ids[:3]:
        detail = b.company(cid)
        detail = {**detail, "months": [m for m in detail["months"] if m["month"] <= nov["as_of"]]}
        out.append(plots.score_history_spec(detail, [a for a in b.alerts_for(cid) if a["month"] <= nov["as_of"]]))
    return out


def run_check(b, cids: list[str], gids: list[str], as_of: str) -> dict:
    key = ws.watch_key(cids, gids)
    state = ws.load_state(key)
    nov = ws.compute_novelties(b, cids, gids, as_of, state)
    res = {"time": datetime.now(), "as_of": as_of, "key": key, "cids": cids, "gids": gids, "nov": nov,
           "answer": None, "plots": [], "error": None, "pending": True}
    if ws.has_news(nov):
        try:
            with st.spinner("Escribiendo el aviso..."):
                r = run_turn(system_prompt("sentinel", b), [], _user_text(nov), SENTINEL_TOOLS)
            if not r.answer.strip():
                raise RuntimeError("el modelo devolvió una respuesta vacía")
            res["answer"], res["plots"] = r.answer, r.plots
        except Exception as e:  # show the error, keep the deterministic section
            res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        if not res["plots"]:
            res["plots"] = _fallback_plots(b, nov)
    return res


# ----------------------------------------------------------------------------- deterministic rendering


def _alert_card(a: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{a['entity']['id']}** · {a['month']} · {a['title']}")
        st.write(a["summary"])
        st.markdown(f"Responsable: **{a['owner_label']}** · Acción: {a['action']}")
        if a["reasons"]:
            st.markdown("\n".join(
                f"- {r['label']}: {ws.fmt_eur(r.get('eur'))}" + (f" ({r['points']:+.1f} pts)" if r.get("points") else "")
                for r in a["reasons"]))
        ev = a.get("evidence") or {}
        if ev.get("members_moving_most"):
            st.caption(f"Miembros que más se mueven: {ev['members_moving_most']}")
        p = a.get("persistence") or {}
        st.caption(f"Persistencia: {p.get('rule', '—')} · {p.get('months_flagged', '—')} meses marcados")


def _bucket(a: dict) -> str:
    return "opportunity" if a.get("direction") == "opportunity" else a["severity"]


def render_deterministic(nov: dict) -> None:
    alerts = nov["new_alerts"]
    for sev in ("act", "watch", "opportunity"):
        group = [a for a in alerts if _bucket(a) == sev]
        if group:
            st.markdown(f"#### {SEV_LABEL[sev]} ({len(group)})")
            for a in group:
                _alert_card(a)
    info = [a for a in alerts if _bucket(a) == "info"]
    if info:
        with st.expander(f"Info ({len(info)}): registradas, no empujadas"):
            for a in info:
                st.markdown(f"- **{a['entity']['id']}** · {a['month']} · {a['title']} — {a['summary']}")

    changes = []
    for m in nov["score_moves"]:
        changes.append(f"- **{m['entity_id']}**: score {m['score_from']:.0f} → {m['score_to']:.0f} ({m['change']:+.1f}) "
                       f"entre {m['from_month']} y {m['to_month']}" + (f" · guard {m['guard']}" if m.get("guard") else ""))
    for t in nov["trajectory_changes"]:
        changes.append(f"- **{t['entity_id']}**: trayectoria {t['from']} → {t['to']} ({t['month']})")
    for g in nov["guard_changes"]:
        changes.append(f"- **{g['entity_id']}**: guard {g['from'] or 'ninguno'} → {g['to'] or 'ninguno'} ({g['month']}, "
                       f"score {g['score']:.0f}, sin tope {g['score_pre_cap']:.0f})")
    if changes:
        st.markdown("#### Cambios frente al último aviso")
        st.markdown("\n".join(changes))

    s = nov["stats"]
    st.caption(
        f"Alertas por caída del score: seguidas de un resultado adverso en ~{s['false_alarm_rate']:.0%} de los casos, "
        f"frente a ~{s['false_alarm_rate_at_chance']:.0%} en un mes al azar; anticipación mediana "
        f"{s['median_lead_time_months']:.0f} meses. Cliente principal en silencio: ~{s['top_customer_precision']:.0%} "
        f"pierden el cliente frente a una base del {s['top_customer_base_rate']:.0%}. Medido en empresas de entrenamiento."
    )

    comp = nov["watched"]["companies"]
    if comp:
        st.markdown(f"#### Empresas vigiladas (as-of {nov['as_of']})")
        df = pd.DataFrame(comp)[["company_id", "group_id", "month", "score", "trajectory", "guard", "confidence", "delta_3m"]]
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"score": st.column_config.NumberColumn("score", format="%.0f"),
                                    "delta_3m": st.column_config.NumberColumn("delta_3m", format="%+.1f")})
    grp = nov["watched"]["groups"]
    if grp:
        st.markdown("Grupos: " + " · ".join(f"{g['group_id']} media {g['mean_score']:.0f} ({g['n_companies']} empresas)" for g in grp))
    if nov["quiet"]:
        st.caption("Sin novedades: " + ", ".join(nov["quiet"]))


def render_result(res: dict) -> None:
    nov = res["nov"]
    st.caption(f"Comprobado {res['time']:%Y-%m-%d %H:%M} · as-of {nov['as_of']}"
               + (" · primera ejecución para este conjunto" if nov["first_run"] else ""))
    if not ws.has_news(nov):
        st.write(f"Sin novedades desde el último aviso (as-of {nov['as_of']}).")
        return
    if res["error"]:
        st.warning(f"El modelo no pudo escribir el aviso ({res['error']}). Se muestra la sección determinista.")
    if res["answer"]:
        st.markdown(res["answer"])
    for spec in res["plots"]:
        plots.render(spec)
    st.divider()
    render_deterministic(nov)


# ----------------------------------------------------------------------------- page


def _advance() -> None:
    b = load()
    months = b.months
    cur = st.session_state["w_as_of"]
    nxt = months[min(months.index(cur) + 1, len(months) - 1)]
    st.session_state["w_as_of"] = min(nxt, b.manifest["as_of_month"])


def render() -> None:
    st.title("Watcher")
    b = load()
    as_of_bundle = b.manifest["as_of_month"]
    ss = st.session_state
    ss.setdefault("w_as_of", as_of_bundle)
    ss.setdefault("w_history", [])
    ss.setdefault("w_tick", 0)

    rows = {c["company_id"]: c for c in b.companies}
    groups = {g["group_id"]: g for g in b.groups}
    with st.sidebar:
        st.markdown("**Conjunto vigilado**")
        cids = st.multiselect("Empresas", list(rows), key="w_cids",
                              format_func=lambda i: f"{i} · {rows[i]['score']:.0f} · {rows[i]['trajectory']}")
        gids = st.multiselect("Grupos", list(groups), key="w_gids",
                              format_func=lambda i: f"{i} · {groups[i]['n_companies']} companies · mean {groups[i]['latest_mean_score']:.0f}")
        st.selectbox("Mes as-of", [m for m in b.months if m <= as_of_bundle], key="w_as_of")
        c1, c2 = st.columns(2)
        c1.button("Avanzar un mes", on_click=_advance, disabled=ss["w_as_of"] >= as_of_bundle)
        check = c2.button("Comprobar ahora", type="primary", disabled=not (cids or gids))
        auto = st.toggle("Auto-check", value=False)
        tick_fired = False
        if auto:
            count = st_autorefresh(interval=ws.INTERVAL_SEC * 1000, key="w_autorefresh") or 0
            tick_fired = count > ss["w_tick"]
            ss["w_tick"] = count
        nxt = (ss.get("w_last_check_time") or datetime.now()) + timedelta(seconds=ws.INTERVAL_SEC)
        st.caption(f"Intervalo {ws.INTERVAL_SEC // 60} min (`POC_WATCH_INTERVAL_SEC`). "
                   f"Próxima comprobación: {nxt:%H:%M}" if auto else
                   f"Intervalo {ws.INTERVAL_SEC // 60} min (`POC_WATCH_INTERVAL_SEC`). Próxima comprobación: manual.")
        if cids or gids:
            st.caption(f"Estado: `poc/.state/{ws.watch_key(cids, gids)}.json`")
            if st.button("Olvidar lo entregado"):
                ws.reset_state(ws.watch_key(cids, gids))
                ss.pop("w_last", None)

    if not (cids or gids):
        st.write("Elige empresas o grupos en la barra lateral y pulsa Comprobar ahora.")
        return

    if check or tick_fired:
        res = run_check(b, cids, gids, ss["w_as_of"])
        ss["w_last"] = res
        ss["w_last_check_time"] = res["time"]

    res = ss.get("w_last")
    if res and (res["cids"], res["gids"]) == (cids, gids):
        render_result(res)
        if res["pending"]:  # mark delivered only after the digest (or the fallback) rendered
            state = ws.load_state(res["key"])
            ws.save_state(res["key"], ws.mark_delivered(b, state, res["nov"], cids, gids))
            res["pending"] = False
            if ws.has_news(res["nov"]):
                ss["w_history"].insert(0, {"time": res["time"], "as_of": res["as_of"], "answer": res["answer"],
                                           "n_new": len(res["nov"]["new_alerts"]), "error": res["error"]})
                del ss["w_history"][5:]
    elif res:
        st.write("El conjunto vigilado ha cambiado. Pulsa Comprobar ahora.")
    else:
        st.write("Pulsa Comprobar ahora para buscar novedades.")

    if ss["w_history"]:
        with st.expander(f"Avisos anteriores ({len(ss['w_history'])})"):
            for h in ss["w_history"]:
                st.markdown(f"**{h['time']:%Y-%m-%d %H:%M} · as-of {h['as_of']} · {h['n_new']} alertas nuevas**")
                st.markdown(h["answer"] or f"_Sin texto del modelo ({h['error']})._")
                st.divider()
