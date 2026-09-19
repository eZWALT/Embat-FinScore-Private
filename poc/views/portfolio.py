"""Portfolio: group -> members -> company, from the export bundle (schema 1.1.0)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from poc import plots
from poc.bundle import load

OWNER = {"treasurer": "Tesorero", "cfo": "CFO", "collections": "Cobros"}


def _eur(x: float | None) -> str:
    if x is None:
        return ""
    a = abs(x)
    if a >= 1e6:
        return f"€{x/1e6:.1f}M"
    if a >= 1e3:
        return f"€{x/1e3:.0f}k"
    return f"€{x:.0f}"


def render() -> None:
    st.title("Portfolio")
    try:
        b = load()
    except FileNotFoundError as e:
        st.write(str(e))
        return
    m = b.manifest
    st.caption(f"Bundle {m['scorecard_version']} · as-of {m['as_of_month']} · {m['counts']['companies']} companies, "
               f"{m['counts']['groups']} groups" + (" · SAMPLE" if m.get("is_sample") else ""))

    groups = sorted(b.groups, key=lambda g: (-g["n_companies"], g["group_id"]))
    gid = st.selectbox(
        "Group", [g["group_id"] for g in groups],
        format_func=lambda x: next(f"{x} · {g['n_companies']} companies · mean {g['latest_mean_score']:.0f}"
                                   if g["latest_mean_score"] is not None else f"{x} · {g['n_companies']} companies"
                                   for g in groups if g["group_id"] == x),
    )
    g = b.group(gid)
    rows = {c["company_id"]: c for c in b.companies}
    members = [rows[c] for c in g["company_ids"] if c in rows]
    table = pd.DataFrame(
        [
            {
                "company_id": c["company_id"], "score": c["score"], "delta_3m": c["delta_3m"],
                "trajectory": c["trajectory"], "confidence": c["confidence"], "guard": c["guard"],
                "n_alerts": c["n_alerts"], "max_alert_severity": c["max_alert_severity"], "top_reason": c["top_reason"],
            }
            for c in members
        ]
    ).sort_values("score")

    # group mean series
    hist = [(mo, s) for mo, s in zip(b.months, g["mean_scores"]) if s is not None]
    if hist:
        plots.render({"title": f"{gid} · mean score", "x": [h[0] for h in hist],
                      "series": {"mean score": [h[1] for h in hist]}, "kind": "line", "markers": [], "band": None,
                      "y_label": "0-100"}, height=180)
    if not g["limits_available"]:
        st.caption("Small group: mean only, no funnel limits, no group alerts.")

    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "company_id": "Company", "score": st.column_config.NumberColumn("Score", format="%.0f"),
            "delta_3m": st.column_config.NumberColumn("Δ 3m", format="%+.1f"), "trajectory": "Trajectory",
            "confidence": "Confidence", "guard": "Guard", "n_alerts": "Alerts", "max_alert_severity": "Max severity",
            "top_reason": "Top reason",
        },
    )

    cid = st.selectbox("Company", table["company_id"].tolist())
    d = b.company(cid)
    latest = d["months"][-1]
    alerts = b.alerts_for(cid)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{latest['score']:.0f}", None if latest["guard"] is None else f"cap {latest['guard']}")
    c2.metric("Trajectory", latest["trajectory"])
    c3.metric("Confidence", latest["confidence"])
    c4.metric("Alerts (12m)", len(alerts))
    if latest.get("confidence_note"):
        st.caption(latest["confidence_note"])

    plots.render(plots.score_history_spec(d, alerts))

    left, right = st.columns(2)
    with left:
        st.subheader("Why not higher")
        for r in latest.get("reasons") or []:
            st.write(f"- **{r['label']}** ({r['points']:+.1f} pts, {_eur(r['eur'])}): {r['sentence']}")
        cats = latest["categories"]
        st.dataframe(
            pd.DataFrame([{"category": m["spec"]["categories"][k]["label"], "score": v["score"], "contribution": v["contribution"]}
                          for k, v in cats.items()]),
            hide_index=True, width="stretch",
            column_config={"score": st.column_config.NumberColumn(format="%.0f"),
                           "contribution": st.column_config.NumberColumn(format="%.1f")},
        )
    with right:
        st.subheader("Alerts")
        if not alerts:
            st.write("None in the feed window.")
        for a in alerts:
            st.write(f"- **{a['month']} · {a['title']}** [{a['severity']}] → {OWNER.get(a['owner'], a['owner'])}: {a['action']}")
        ch = next((c for c in d.get("control") or [] if c["comparison"] == "own_history" and c["metric"] == "score"), None)
        if ch:
            plots.render(plots.control_chart_spec(ch, f"{cid} · score vs own history"), height=220)

    st.caption(m["disclaimer"])
