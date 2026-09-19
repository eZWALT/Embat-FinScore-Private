"""Read-only access to the local feature store and the v0 dummy score.

Store: `data/feature_store/monthly.parquet`
(built by `python -m analysis.features.build_feature_store`, gitignored).
Score: `product/score/outputs/monthly_scores.parquet`
(`PYTHONPATH=. python -m product.score`, gitignored).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "data" / "feature_store" / "monthly.parquet"
SCORES = REPO / "product" / "score" / "outputs" / "monthly_scores.parquet"
COMPANIES = REPO / "data" / "companies.csv"
SCORE_COLS = ("score_3m", "score", "state", "confidence_band", "reason_1_es", "reason_1_en")

COLUMNS = {
    "b_runway": "Runway (months)",
    "b_liq": "Cash (reconstructed)",
    "a_op_in": "Operating inflows",
    "a_op_out": "Operating outflows",
}


def store_available() -> bool:
    return STORE.exists()


def scores_available() -> bool:
    return SCORES.exists()


@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    cols = ", ".join(["company_id", "group_id", "period", *COLUMNS])
    df = duckdb.sql(f"select {cols} from '{STORE.as_posix()}'").df()
    df["period"] = pd.to_datetime(df["period"].astype(str)).dt.to_period("M").dt.to_timestamp()
    return df.sort_values(["company_id", "period"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    if not SCORES.exists():
        return pd.DataFrame()
    keep = ["company_id", "period", *SCORE_COLS]
    df = pd.read_parquet(SCORES, columns=keep)
    df["period"] = pd.to_datetime(df["period"].astype(str)).dt.to_period("M").dt.to_timestamp()
    return df.sort_values(["company_id", "period"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def groups(panel: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby("group_id")["company_id"].nunique().rename("companies").reset_index()
    return g.sort_values(["companies", "group_id"], ascending=[False, True]).reset_index(drop=True)


def group_table(panel: pd.DataFrame, group_id: str, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per company: latest dummy score, runway, 3-month change, inflows."""
    sub = panel[panel["group_id"] == group_id]
    scored = scores if scores is not None else load_scores()
    rows = []
    for cid, c in sub.groupby("company_id"):
        c = c.dropna(subset=["b_runway"])
        if c.empty:
            rows.append(
                {
                    "company_id": cid,
                    "last_period": None,
                    "score_3m": None,
                    "state": None,
                    "confidence_band": None,
                    "runway": None,
                    "runway_3m_change": None,
                    "inflows": None,
                }
            )
            continue
        last = c.iloc[-1]
        prev = c.iloc[-4] if len(c) >= 4 else None
        row = {
            "company_id": cid,
            "last_period": last["period"].strftime("%Y-%m"),
            "score_3m": None,
            "state": None,
            "confidence_band": None,
            "runway": last["b_runway"],
            "runway_3m_change": (last["b_runway"] - prev["b_runway"]) if prev is not None else None,
            "inflows": last["a_op_in"],
        }
        if not scored.empty:
            s = scored.loc[scored["company_id"] == cid]
            if not s.empty:
                sl = s.iloc[-1]
                row["score_3m"] = sl["score_3m"]
                row["state"] = sl["state"]
                row["confidence_band"] = sl["confidence_band"]
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_col = "score_3m" if out["score_3m"].notna().any() else "runway"
    return out.sort_values(sort_col, na_position="last").reset_index(drop=True)


def company_series(panel: pd.DataFrame, company_id: str, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    c = panel[panel["company_id"] == company_id].set_index("period")
    out = c[list(COLUMNS)].rename(columns=COLUMNS)
    scored = scores if scores is not None else load_scores()
    if scored.empty:
        return out
    s = scored.loc[scored["company_id"] == company_id].set_index("period")
    if s.empty:
        return out
    return out.join(s[["score_3m"]].rename(columns={"score_3m": "Health score (3m)"}), how="left")


def latest_reasons(company_id: str, scores: pd.DataFrame | None = None) -> pd.Series | None:
    scored = scores if scores is not None else load_scores()
    if scored.empty:
        return None
    s = scored.loc[scored["company_id"] == company_id]
    if s.empty:
        return None
    return s.iloc[-1]
