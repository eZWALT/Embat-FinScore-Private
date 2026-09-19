"""Render plot specs (from poc.agent.tools.plot_series or built by a view) with Altair in Streamlit.

spec = {"title", "x": [months], "series": {name: [values]}, "kind": "line"|"bar",
        "markers": [months], "band": {"lower": [...], "upper": [...]} | None, "y_label": str | None}
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


def _frame(spec: dict) -> pd.DataFrame:
    rows = []
    for name, vals in spec["series"].items():
        for x, v in zip(spec["x"], vals):
            rows.append({"month": x, "series": name, "value": v})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["month"] + "-01")
    return df


def chart(spec: dict, height: int = 260) -> alt.Chart:
    df = _frame(spec)
    y_title = spec.get("y_label") or ""
    base = alt.Chart(df).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y-%m")))
    mark = base.mark_bar() if spec.get("kind") == "bar" else base.mark_line(point=True)
    main = mark.encode(
        y=alt.Y("value:Q", title=y_title),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top")),
        tooltip=["month:N", "series:N", alt.Tooltip("value:Q", format=".1f")],
    )
    layers = [main]
    band = spec.get("band")
    if band and band.get("lower") and band.get("upper"):
        bdf = pd.DataFrame({"month": spec["x"], "lower": band["lower"], "upper": band["upper"]})
        bdf["date"] = pd.to_datetime(bdf["month"] + "-01")
        layers.insert(0, alt.Chart(bdf).mark_area(opacity=0.15).encode(x="date:T", y="lower:Q", y2="upper:Q"))
    markers = [m for m in (spec.get("markers") or []) if m in spec["x"]]
    if markers:
        mdf = pd.DataFrame({"date": pd.to_datetime([m + "-01" for m in markers])})
        layers.append(alt.Chart(mdf).mark_rule(strokeDash=[4, 4], color="#c0392b").encode(x="date:T"))
    return alt.layer(*layers).properties(title=spec.get("title", ""), height=height).interactive()


def render(spec: dict, height: int = 260) -> None:
    st.altair_chart(chart(spec, height), use_container_width=True)


def score_history_spec(company: dict, alerts: list[dict] | None = None) -> dict:
    months = [m["month"] for m in company["months"]]
    return {
        "title": f"{company['company_id']} · score",
        "x": months,
        "series": {"score": [m["score"] for m in company["months"]]},
        "kind": "line",
        "markers": [a["month"] for a in (alerts or [])],
        "band": None,
        "y_label": "0-100",
    }


def control_chart_spec(ch: dict, title: str) -> dict:
    return {
        "title": title,
        "x": ch["months"],
        "series": {"value": ch["values"], "center": ch["center"], **({"ewma": ch["ewma"]} if ch.get("ewma") else {})},
        "kind": "line",
        "markers": [m for m, s in zip(ch["months"], ch["signal"]) if s != "none"],
        "band": {"lower": ch["lower"], "upper": ch["upper"]},
        "y_label": ch["metric"],
    }
