"""Trailing-window item values per company-month, computed from the feature store. As-of only.

Every window looks back from the month; nothing uses a later row. The store has one contiguous row
per company from its first month, so a row shift is a month shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import spec

KEYS = ["company_id", "period"]
STORE_COLS = [
    "e_delay_paid", "e_delay_coll", "e_ap_overdue_30", "e_ar_overdue_30", "e_ap_open", "e_ar_open",
    "e_credit_note_ratio", "b_liq", "a_out6", "a_out3", "a_in3", "a_op_in", "a_op_out",
    "f_ds_r", "f_fc_r", "f_debt_service", "f_fin_cost", "c_zero_in_share_6",
    "c_recency_days", "d_cust_hhi", "d_cust_top1",
]


def _g(df: pd.DataFrame, col: str):
    return df.groupby("company_id", sort=False)[col]


def _roll(df, col, w, how="mean", min_periods=1):
    return _g(df, col).transform(lambda s: getattr(s.rolling(w, min_periods=min_periods), how)())


def compute_items(store: pd.DataFrame) -> pd.DataFrame:
    """Item values (spec.ITEMS names), guard inputs and the raw numbers behind each reason."""
    missing = [c for c in STORE_COLS if c not in store.columns]
    if missing:
        raise KeyError(f"feature store lacks columns {missing}")
    df = store[KEYS + STORE_COLS].copy()
    df["period"] = pd.to_datetime(df["period"])
    df = df.sort_values(KEYS).reset_index(drop=True)
    gap = _g(df, "period").diff().dropna()
    if not gap.dt.days.between(28, 31).all():
        raise ValueError("feature store months are not contiguous per company")
    out = df[KEYS].copy()
    out["trail_months"] = _g(df, "period").cumcount() + 1

    # payment history
    for item, col in (("delay_paid", "e_delay_paid"), ("delay_coll", "e_delay_coll"),
                      ("ap_overdue30", "e_ap_overdue_30"), ("ar_overdue30", "e_ar_overdue_30")):
        out[item] = _roll(df, col, spec.WINDOW)

    # amounts owed
    liq = df["b_liq"].round(2)  # store flags compare unrounded cash to 0 and flip on 1e-10 noise
    df["_liq"] = liq
    out["liq"] = liq
    denom = np.where(df["a_out6"].notna(), df["a_out6"] / 6.0, df["a_out3"] / 3.0)
    out["out_month"] = denom
    df["_runway"] = (liq / np.maximum(denom, 1.0)).clip(*spec.RUNWAY_CLIP)
    out["runway"] = _roll(df, "_runway", spec.WINDOW)
    df["_neg"] = (liq < 0).astype(float).where(liq.notna())
    out["neg_liq"] = _roll(df, "_neg", spec.WINDOW, min_periods=2)
    prev_neg = _g(df, "_neg").shift(1)
    df["_onset"] = ((df["_neg"] == 1) & (prev_neg == 0)).astype(float).where(df["_neg"].notna())
    out["neg_episodes"] = _roll(df, "_onset", spec.EPISODE_WINDOW, "sum", min_periods=3)
    out["min_liq3"] = _roll(df, "_liq", spec.WINDOW, "min")
    out["ds_ratio"] = _roll(df, "f_ds_r", spec.WINDOW)
    out["fc_ratio"] = _roll(df, "f_fc_r", spec.WINDOW)
    # the ratios above average 3 overlapping 3-month windows, i.e. cover 5 months: the euro amounts use the same 5
    out["ds5"] = _roll(df, "f_debt_service", spec.WINDOW + 2, "sum")
    out["fc5"] = _roll(df, "f_fin_cost", spec.WINDOW + 2, "sum")
    out["in5"] = _roll(df, "a_op_in", spec.WINDOW + 2, "sum")
    out["in3"] = _roll(df, "a_op_in", spec.WINDOW, "sum")

    # stability
    out["months_observed"] = np.minimum(out["trail_months"], spec.LENGTH_CAP_MONTHS) / spec.LENGTH_CAP_MONTHS * 100.0
    out["active_share"] = (1.0 - df["c_zero_in_share_6"]) * 100.0
    sd6 = _roll(df, "a_op_out", spec.EPISODE_WINDOW, "std", min_periods=spec.EPISODE_WINDOW)
    mean6 = _roll(df, "a_op_out", spec.EPISODE_WINDOW, "mean", min_periods=spec.EPISODE_WINDOW)
    out["out_sd6"] = sd6
    out["out_vol"] = (sd6 / np.maximum(mean6, 1.0)).clip(upper=spec.OUT_VOL_CLIP)

    # new credit: rise against the same window 6 months earlier
    for item, col in (("ds_increase", "ds_ratio"), ("fc_increase", "fc_ratio")):
        prior = out.groupby("company_id", sort=False)[col].shift(spec.PRIOR_LAG)
        out[item] = (out[col] - prior).clip(lower=0.0).where(out[col].notna() & prior.notna())
        out[f"{item}_prior"] = prior

    # mix
    hhi = _roll(df, "d_cust_hhi", spec.WINDOW)
    out["cust_tail"] = (hhi - spec.HHI_TAIL).clip(lower=0.0).where(hhi.notna())
    out["cust_hhi"] = hhi
    out["cust_top1"] = _roll(df, "d_cust_top1", spec.WINDOW)
    out["credit_note"] = _roll(df, "e_credit_note_ratio", spec.WINDOW)

    # points for the reasons' euro amounts (values at month-end, as in the store)
    out["ar_od30_eur"] = df["e_ar_open"] * df["e_ar_overdue_30"]
    out["ap_od30_eur"] = df["e_ap_open"] * df["e_ap_overdue_30"]

    # going-dark guard inputs
    out["recency_days"] = df["c_recency_days"]
    recent = _roll(df, "a_op_in", spec.WINDOW, "mean")
    older = _g(df, "a_op_in").transform(lambda s: s.shift(spec.WINDOW).rolling(6, min_periods=6).mean())
    out["inflow_recent"], out["inflow_older"] = recent, older
    # dark: no bank movement at all. (Three months without incoming money while paying out is not dark: 99% of
    # those rows keep booking transactions, so it is left to the ratios.) fading: inflows collapsed vs own history.
    out["dark_level"] = np.select(
        [df["c_recency_days"] >= spec.DARK_NO_TX_DAYS,
         (older > 0) & (recent < spec.FADING_INFLOW_RATIO * older)],
        [2, 1], default=0)
    return out
