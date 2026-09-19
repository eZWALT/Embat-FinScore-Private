import pandas as pd
import streamlit as st

from poc import data


def render() -> None:
    st.title("Portfolio")

    if not data.store_available():
        st.write("Feature store not built. Run `python -m analysis.features.build_feature_store`.")
        return

    panel = data.load_panel()
    scores = data.load_scores()
    g = data.groups(panel)

    group_id = st.selectbox(
        "Group",
        g["group_id"],
        format_func=lambda x: f"{x} · {int(g.loc[g.group_id == x, 'companies'].iloc[0])} companies",
    )

    table = data.group_table(panel, group_id, scores)
    if data.scores_available():
        st.caption("v0 dummy score (`score_3m`). Sorted worst first. Not a hidden-test claim.")
    else:
        st.caption("Score file missing. Run `PYTHONPATH=. python -m product.score`. Sorted by runway.")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "company_id": "Company",
            "last_period": "Last month",
            "score_3m": st.column_config.NumberColumn("Score (3m)", format="%.0f"),
            "state": "Trajectory",
            "confidence_band": "Confidence",
            "runway": st.column_config.NumberColumn("Runway (months)", format="%.1f"),
            "runway_3m_change": st.column_config.NumberColumn("Δ runway, 3m", format="%+.1f"),
            "inflows": st.column_config.NumberColumn("Inflows", format="%.0f"),
        },
    )

    company_id = st.selectbox("Company", table["company_id"])
    series = data.company_series(panel, company_id, scores)
    reasons = data.latest_reasons(company_id, scores)

    if "Health score (3m)" in series.columns:
        st.line_chart(series[["Health score (3m)"]])
    if reasons is not None and pd.notna(reasons.get("reason_1_es")):
        st.write(reasons.get("reason_1_es") or reasons.get("reason_1_en"))
    st.line_chart(series[["Runway (months)"]])
    st.line_chart(series[["Operating inflows", "Operating outflows"]])
