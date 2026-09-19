"""Synthetic-panel checks for the v0 dummy card. No feature store."""
from __future__ import annotations

import numpy as np
import pandas as pd

from product.score.card import DARK_DAYS_CAP, DARK_LONG_CAP
from product.score.score import derive_columns, fit_ref, score_panel


def _panel() -> pd.DataFrame:
    months = pd.date_range("2025-01-01", periods=8, freq="MS")
    rows = []
    for i, m in enumerate(months):
        rows.append(
            {
                "company_id": "TRAIN_OK",
                "period": m,
                "first_month": months[0],
                "group_id": "G1",
                "b_runway": 3.0 + 0.1 * i,
                "f_ds_r": 0.10,
                "f_fc_r": 0.02,
                "e_ar_overdue_30": 0.05,
                "e_ap_overdue_30": 0.04,
                "e_delay_coll": 2.0,
                "e_delay_paid": 1.0,
                "d_cust_top1": 0.40,
                "c_n_days_with_tx": 12,
                "c_recency_days": 1,
            }
        )
        rows.append(
            {
                "company_id": "TRAIN_STRESS",
                "period": m,
                "first_month": months[0],
                "group_id": "G1",
                "b_runway": -1.0,
                "f_ds_r": 0.80,
                "f_fc_r": 0.20,
                "e_ar_overdue_30": 0.60,
                "e_ap_overdue_30": 0.50,
                "e_delay_coll": 40.0,
                "e_delay_paid": 35.0,
                "d_cust_top1": 0.99,
                "c_n_days_with_tx": 8,
                "c_recency_days": 2,
            }
        )
        rows.append(
            {
                "company_id": "HOLD",
                "period": m,
                "first_month": months[0],
                "group_id": "G2",
                "b_runway": 50.0,
                "f_ds_r": 0.01,
                "f_fc_r": 0.00,
                "e_ar_overdue_30": 0.00,
                "e_ap_overdue_30": 0.00,
                "e_delay_coll": 0.0,
                "e_delay_paid": 0.0,
                "d_cust_top1": 0.20,
                "c_n_days_with_tx": 15,
                "c_recency_days": 1,
            }
        )
        dark_month = i == len(months) - 1
        rows.append(
            {
                "company_id": "TRAIN_DARK",
                "period": m,
                "first_month": months[0],
                "group_id": "G1",
                "b_runway": 8.0,
                "f_ds_r": 0.05,
                "f_fc_r": 0.01,
                "e_ar_overdue_30": np.nan,
                "e_ap_overdue_30": np.nan,
                "e_delay_coll": np.nan,
                "e_delay_paid": np.nan,
                "d_cust_top1": np.nan,
                "c_n_days_with_tx": 0 if dark_month else 10,
                "c_recency_days": 80 if dark_month else 1,
            }
        )
    return pd.DataFrame(rows)


def test_holdout_not_in_ref_and_healthy_beats_stress():
    panel = derive_columns(_panel())
    holdout = {"HOLD"}
    ref = fit_ref(panel, holdout)
    assert "HOLD" not in set(panel.loc[panel.company_id.isin(["TRAIN_OK"]), "company_id"])
    assert max(ref["signals"]["b_runway"]["values"]) < 50.0

    scored = score_panel(panel, ref)
    last = scored.sort_values("period").groupby("company_id").tail(1).set_index("company_id")
    assert last.loc["TRAIN_OK", "score"] > last.loc["TRAIN_STRESS", "score"]
    assert last.loc["TRAIN_DARK", "score"] <= DARK_LONG_CAP
    assert last.loc["TRAIN_DARK", "score_pre_cap"] > last.loc["TRAIN_DARK", "score"]
    assert last.loc["TRAIN_DARK", "no_invoices"]
    assert last.loc["TRAIN_DARK", "confidence_band"] in {"low", "medium"}


def test_dark_month_cannot_raise_score():
    panel = derive_columns(_panel())
    ref = fit_ref(panel, {"HOLD"})
    scored = score_panel(panel, ref)
    dark = scored.loc[scored.company_id == "TRAIN_DARK"].sort_values("period").iloc[-1]
    assert dark.score <= DARK_DAYS_CAP
    assert dark.going_dark_long
    assert dark.reason_1_code == "going_dark_long"


def test_missing_invoices_reweights():
    panel = derive_columns(_panel())
    ref = fit_ref(panel, {"HOLD"})
    scored = score_panel(panel, ref)
    dark = scored.loc[scored.company_id == "TRAIN_DARK"].sort_values("period").iloc[-1]
    assert dark.n_payment_history == 0
    assert dark.n_mix == 0
    assert dark.w_payment_history == 0
    assert dark.cov_w == 55.0  # amounts 40 + stability 15
