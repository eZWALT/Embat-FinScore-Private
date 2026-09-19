"""Shrink the global Y7 engine to a small SHAP card (brief Q4 dip vs fall).

Global LightGBM on y7_top1_lost only (not the inflow-clause sibling).
Families never D. Never d_cust_top1 / d_cust_hhi. Never static meta.
Holdout 72 companies never enter a fit, early-stop, pick, or SHAP sample.
Quote train group-fold CV (5 folds, seed 20260918). Holdout is a check only.

A) SHAP card from shap_y7.md: DSO (clipped 24m), lag1 issued (not euro
   level), credit-note ratio ± lag1, collection delay, DSO lag1, f_fc_r_lag3.
B) A minus e_dso_proxy_lag1 (SHAP + vs univ − 0.506).
C) A plus log1p(e_ar_issued) if the dropped euro stem needs a volume proxy.

Published 278-col claim is CV 0.663 (trees=50 in explain_y7; gbm_y7y8
LGB_BASE is 400 + early stop). Same honesty as Y3: early-stop on tiny X
can collapse to 1 tree — do not KEEP that number; quote the 50 / depth-3
diagnostic separately.

Decision (this wave): CLOSE. B_shallow 0.712 / n_x=6 holds vs 0.663 / 278
but 400+ES collapses. Next idea: drop DSO, add issued-lag CV
(TURNOVER 0.720 / n_x=5, fold 4 0.680). See analysis/outputs/y7_core.md.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.models.gbm_y7_core
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate.protocol import (
    FOLD_SEED,
    auroc,
    assert_no_holdout,
    group_folds,
    leakage_check,
    load_holdout,
    pr_auc,
    train_companies,
)
from analysis.features.common import connect, train_mask
from analysis.models.gbm_y7y8 import (
    LAGS,
    LGB_BASE,
    N_FOLDS,
    _assert_allowed,
    _family_of,
    _fit_lgb,
    _gain_table,
    _keys,
    _scale_pos_weight,
    add_lags,
    append_registry,
    load_store,
    load_y,
)

Y_COL = "y7_top1_lost"
ALLOWED = ("a", "b", "c", "e", "f", "g", "h")
FORBIDDEN = ("d",)
AGENT = "410a183a"
WAVE = 4
ROUND = "R4"
PUBLISHED_CV = 0.663
PUBLISHED_N_X = 278
KEEP_EPS = 0.002
PARK_DELTA = 0.02
CLIP_DSO = 24.0
CLIP_ISSUED_CV = 2.0  # constant bound, not a fit — same honesty as DSO clip
CONC_COPY_CUT = 0.80
SIZE_CUT = 0.85
COLLAPSE_TREES = 5
BANNED_META = ("n_banking", "group_size", "h_group_size")
BANNED_EURO = ("e_ar_issued",)
BANNED_D = ("d_cust_top1", "d_cust_hhi")

# Spec A: 7 columns. Lags only where SHAP said lead time. No euro issued level.
CORE_A = (
    "e_dso_proxy",
    "e_ar_issued_lag1",
    "e_credit_note_ratio",
    "e_credit_note_ratio_lag1",
    "e_delay_coll",
    "e_dso_proxy_lag1",
    "f_fc_r_lag3",
)
CORE_B = tuple(c for c in CORE_A if c != "e_dso_proxy_lag1")
CORE_C = CORE_A + ("e_ar_issued_log1p",)
CORE_D = tuple(c for c in CORE_A if c != "f_fc_r_lag3")
CORE_E = tuple(c for c in CORE_A if c != "e_credit_note_ratio")
CORE_F = CORE_A + ("e_ar_overdue",)
CORE_G = CORE_A + ("e_delay_coll_lag1",)
CORE_MIN = ("e_dso_proxy", "e_ar_issued_lag1")
# Q5 contemporaneous invoice book only (no finance lag, no sign-flip DSO lag).
CORE_Q5 = (
    "e_dso_proxy",
    "e_credit_note_ratio",
    "e_delay_coll",
)
# Q6 lead-time names only (SHAP lead mass).
CORE_Q6 = (
    "e_ar_issued_lag1",
    "e_credit_note_ratio_lag1",
    "e_dso_proxy_lag1",
    "f_fc_r_lag3",
)
# B minus finance lag: 5-col invoice card.
CORE_B5 = tuple(c for c in CORE_B if c != "f_fc_r_lag3")
# Unclipped DSO (raw stem + its lag1). Clip is otherwise the default.
CORE_NOCLIP = (
    "e_dso_proxy_raw",
    "e_ar_issued_lag1",
    "e_credit_note_ratio",
    "e_credit_note_ratio_lag1",
    "e_delay_coll",
    "e_dso_proxy_raw_lag1",
    "f_fc_r_lag3",
)
CORE_C12 = (
    "e_dso_proxy_c12",
    "e_ar_issued_lag1",
    "e_credit_note_ratio",
    "e_credit_note_ratio_lag1",
    "e_delay_coll",
    "f_fc_r_lag3",
)
CORE_IX = CORE_B + ("e_dso_x_cn",)
CORE_SLOPE = CORE_B + ("e_ar_issued_lag1_over_lag3",)
CORE_VOL = CORE_B + ("e_ar_issued_lag_cv",)
# No DSO: test whether fold-4 short-DSO churn is a turnover card, not stretched AR.
CORE_TURNOVER = (
    "e_ar_issued_lag1",
    "e_ar_issued_lag_cv",
    "e_credit_note_ratio",
    "e_credit_note_ratio_lag1",
    "f_fc_r_lag3",
)

# Stems that must exist so selected lags can be built (past-only shift).
LAG_PARENTS = (
    "e_dso_proxy",
    "e_ar_issued",
    "e_credit_note_ratio",
    "e_delay_coll",
    "e_ar_overdue",
    "f_fc_r",
    "f_ds_r",
)

LGB_SHALLOW = dict(
    objective="binary",
    n_estimators=50,
    max_depth=3,
    num_leaves=8,
    learning_rate=0.05,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=FOLD_SEED,
    n_jobs=1,
    verbosity=-1,
)

LGB_FIXED50 = dict(
    objective="binary",
    n_estimators=50,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=FOLD_SEED,
    n_jobs=2,
    verbosity=-1,
)

SPECS = (
    {
        "id": "A",
        "model": "lgbm_y7_core_shap",
        "cols": CORE_A,
        "mode": "early_stop",
        "claimable": True,
        "notes": (
            "SHAP card: clipped DSO, lag1 issued (not euro level), credit-note "
            "±lag1, delay_coll, DSO lag1, f_fc_r_lag3; never D / meta"
        ),
    },
    {
        "id": "A_shallow",
        "model": "lgbm_y7_core_shap_d3_n50",
        "cols": CORE_A,
        "mode": "shallow",
        "claimable": False,
        "notes": (
            "same X as A; 50 trees, max_depth=3; diagnostic — not used for KEEP "
            "(Y3 lesson: early-stop on tiny X can collapse)"
        ),
    },
    {
        "id": "A_n50",
        "model": "lgbm_y7_core_shap_n50",
        "cols": CORE_A,
        "mode": "fixed50",
        "claimable": False,
        "notes": (
            "same X as A; 50 trees no early stop (explain_y7 final-fit trees); "
            "diagnostic if 400+ES collapses"
        ),
    },
    {
        "id": "B",
        "model": "lgbm_y7_core_nodsolag",
        "cols": CORE_B,
        "mode": "early_stop",
        "claimable": True,
        "notes": (
            "A minus e_dso_proxy_lag1 (SHAP + vs univ − 0.506 sign flip)"
        ),
    },
    {
        "id": "B_shallow",
        "model": "lgbm_y7_core_nodsolag_d3_n50",
        "cols": CORE_B,
        "mode": "shallow",
        "claimable": False,
        "explain": True,
        "notes": "same X as B; 50 / depth-3 diagnostic",
    },
    {
        "id": "C",
        "model": "lgbm_y7_core_logissued",
        "cols": CORE_C,
        "mode": "early_stop",
        "claimable": True,
        "notes": "A plus log1p(e_ar_issued) volume proxy; raw euro level still out",
    },
    {
        "id": "C_shallow",
        "model": "lgbm_y7_core_logissued_d3_n50",
        "cols": CORE_C,
        "mode": "shallow",
        "claimable": False,
        "notes": "same X as C; 50 / depth-3 diagnostic",
    },
    {
        "id": "D",
        "model": "lgbm_y7_core_nofc",
        "cols": CORE_D,
        "mode": "early_stop",
        "claimable": True,
        "notes": "A minus f_fc_r_lag3 (Q6 finance-cost lag ablation)",
    },
    {
        "id": "D_shallow",
        "model": "lgbm_y7_core_nofc_d3_n50",
        "cols": CORE_D,
        "mode": "shallow",
        "claimable": False,
        "notes": "same X as D; 50 / depth-3 diagnostic",
    },
    {
        "id": "E",
        "model": "lgbm_y7_core_nocn",
        "cols": CORE_E,
        "mode": "early_stop",
        "claimable": True,
        "notes": "A minus contemporaneous e_credit_note_ratio (keep lag1)",
    },
    {
        "id": "E_shallow",
        "model": "lgbm_y7_core_nocn_d3_n50",
        "cols": CORE_E,
        "mode": "shallow",
        "claimable": False,
        "notes": "same X as E; 50 / depth-3 diagnostic",
    },
    {
        "id": "F",
        "model": "lgbm_y7_core_overdue",
        "cols": CORE_F,
        "mode": "early_stop",
        "claimable": True,
        "notes": "A plus e_ar_overdue (optional 8th; ratio, not euro)",
    },
    {
        "id": "F_shallow",
        "model": "lgbm_y7_core_overdue_d3_n50",
        "cols": CORE_F,
        "mode": "shallow",
        "claimable": False,
        "notes": "same X as F; 50 / depth-3 diagnostic",
    },
    {
        "id": "G",
        "model": "lgbm_y7_core_delaylag",
        "cols": CORE_G,
        "mode": "early_stop",
        "claimable": True,
        "notes": "A plus e_delay_coll_lag1 (optional 8th lead)",
    },
    {
        "id": "G_shallow",
        "model": "lgbm_y7_core_delaylag_d3_n50",
        "cols": CORE_G,
        "mode": "shallow",
        "claimable": False,
        "notes": "same X as G; 50 / depth-3 diagnostic",
    },
    {
        "id": "MIN",
        "model": "lgbm_y7_core_min2",
        "cols": CORE_MIN,
        "mode": "early_stop",
        "claimable": True,
        "notes": "two-col floor: clipped DSO + lag1 issued",
    },
    {
        "id": "MIN_shallow",
        "model": "lgbm_y7_core_min2_d3_n50",
        "cols": CORE_MIN,
        "mode": "shallow",
        "claimable": False,
        "notes": "same X as MIN; 50 / depth-3 diagnostic",
    },
    {
        "id": "B_n50",
        "model": "lgbm_y7_core_nodsolag_n50",
        "cols": CORE_B,
        "mode": "fixed50",
        "claimable": False,
        "notes": "same X as B; 50 trees no early stop",
    },
    {
        "id": "B5_shallow",
        "model": "lgbm_y7_core_b5_d3_n50",
        "cols": CORE_B5,
        "mode": "shallow",
        "claimable": False,
        "notes": "B minus f_fc_r_lag3; 5-col invoice card; 50 / depth-3",
    },
    {
        "id": "Q5_shallow",
        "model": "lgbm_y7_core_q5_d3_n50",
        "cols": CORE_Q5,
        "mode": "shallow",
        "claimable": False,
        "notes": "Q5 contemporaneous only (DSO, credit-note, delay); 50 / depth-3",
    },
    {
        "id": "Q6_shallow",
        "model": "lgbm_y7_core_q6_d3_n50",
        "cols": CORE_Q6,
        "mode": "shallow",
        "claimable": False,
        "notes": "Q6 lead-time only (issued lag1, CN lag1, DSO lag1, fc_r lag3)",
    },
    {
        "id": "NOCLIP_shallow",
        "model": "lgbm_y7_core_noclip_d3_n50",
        "cols": CORE_NOCLIP,
        "mode": "shallow",
        "claimable": False,
        "notes": "same as A but unclipped DSO (raw max is millions of months)",
    },
    {
        "id": "C12_shallow",
        "model": "lgbm_y7_core_c12_d3_n50",
        "cols": CORE_C12,
        "mode": "shallow",
        "claimable": False,
        "notes": "B card with DSO clipped at 12 months instead of 24",
    },
    {
        "id": "IX_shallow",
        "model": "lgbm_y7_core_ix_d3_n50",
        "cols": CORE_IX,
        "mode": "shallow",
        "claimable": False,
        "notes": "B plus DSO × credit-note (Q5 compound, not a fit)",
    },
    {
        "id": "SLOPE_shallow",
        "model": "lgbm_y7_core_slope_d3_n50",
        "cols": CORE_SLOPE,
        "mode": "shallow",
        "claimable": False,
        "notes": "B plus issued_lag1/(issued_lag3+1) lead slope",
    },
    {
        "id": "VOL",
        "model": "lgbm_y7_core_vol",
        "cols": CORE_VOL,
        "mode": "early_stop",
        "claimable": False,
        "notes": "B plus issued-lag CV; diagnostic (not the SHAP card KEEP gate)",
    },
    {
        "id": "VOL_shallow",
        "model": "lgbm_y7_core_vol_d3_n50",
        "cols": CORE_VOL,
        "mode": "shallow",
        "claimable": False,
        "explain": True,
        "notes": "B plus cv of issued lags 1/2/3 (fold-4 short-DSO churn hypothesis)",
    },
    {
        "id": "TURNOVER_shallow",
        "model": "lgbm_y7_core_turnover_d3_n50",
        "cols": CORE_TURNOVER,
        "mode": "shallow",
        "claimable": False,
        "explain": True,
        "notes": "no DSO; issued lag1 + issued-lag CV + credit-note ±lag1 + fc_r lag3",
    },
    {
        "id": "TURNOVER_n50",
        "model": "lgbm_y7_core_turnover_n50",
        "cols": CORE_TURNOVER,
        "mode": "fixed50",
        "claimable": False,
        "notes": "same X as TURNOVER; 50 trees no depth cap",
    },
    {
        "id": "TURNOVER",
        "model": "lgbm_y7_core_turnover",
        "cols": CORE_TURNOVER,
        "mode": "early_stop",
        "claimable": False,
        "notes": "same X as TURNOVER_shallow; 400+ES diagnostic",
    },
    {
        "id": "TURNDELAY_shallow",
        "model": "lgbm_y7_core_turndelay_d3_n50",
        "cols": CORE_TURNOVER + ("e_delay_coll",),
        "mode": "shallow",
        "claimable": False,
        "explain": True,
        "notes": "TURNOVER plus collection delay (Q5 why without DSO)",
    },
    {
        "id": "TURN2_shallow",
        "model": "lgbm_y7_core_turn2_d3_n50",
        "cols": ("e_ar_issued_lag1", "e_ar_issued_lag_cv"),
        "mode": "shallow",
        "claimable": False,
        "notes": "two-col turnover floor: issued lag1 + issued-lag CV",
    },
    {
        "id": "TURN_NOFC_shallow",
        "model": "lgbm_y7_core_turn_nofc_d3_n50",
        "cols": tuple(c for c in CORE_TURNOVER if c != "f_fc_r_lag3"),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER minus f_fc_r_lag3",
    },
    {
        "id": "TURN_NOCN_shallow",
        "model": "lgbm_y7_core_turn_nocn_d3_n50",
        "cols": tuple(c for c in CORE_TURNOVER if c != "e_credit_note_ratio"),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER minus contemporaneous credit-note (keep lag1)",
    },
    {
        "id": "TURNCV_shallow",
        "model": "lgbm_y7_core_turncv_d3_n50",
        "cols": ("e_ar_issued_lag_cv",),
        "mode": "shallow",
        "claimable": False,
        "notes": "single issued-lag CV (fold-4 lever, not a size clone)",
    },
    {
        "id": "VOL13_shallow",
        "model": "lgbm_y7_core_vol13_d3_n50",
        "cols": CORE_B + ("e_ar_issued_lag13_cv",),
        "mode": "shallow",
        "claimable": False,
        "notes": "B plus CV of issued lag1 and lag3 only (no lag2)",
    },
    {
        "id": "TURNLOG_shallow",
        "model": "lgbm_y7_core_turnlog_d3_n50",
        "cols": (
            "e_ar_issued_lag1",
            "e_ar_issued_lag_cv_log",
            "e_credit_note_ratio",
            "e_credit_note_ratio_lag1",
            "f_fc_r_lag3",
        ),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER with log1p issued-lag CV instead of raw euro CV",
    },
    {
        "id": "TURNOV_shallow",
        "model": "lgbm_y7_core_turnov_d3_n50",
        "cols": CORE_TURNOVER + ("e_ar_overdue",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus e_ar_overdue (Q5 ratio, no DSO)",
    },
    {
        "id": "TURNFULL_shallow",
        "model": "lgbm_y7_core_turnfull_d3_n50",
        "cols": CORE_TURNOVER + ("e_delay_coll", "e_ar_overdue"),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus delay and overdue; still no DSO",
    },
    {
        "id": "TURNCLIP_shallow",
        "model": "lgbm_y7_core_turnclip_d3_n50",
        "cols": (
            "e_ar_issued_lag1",
            "e_ar_issued_lag_cv_clip",
            "e_credit_note_ratio",
            "e_credit_note_ratio_lag1",
            "f_fc_r_lag3",
        ),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER with issued-lag CV clipped at 2 (constant, not a fit)",
    },
    {
        "id": "TURNSLOPE_shallow",
        "model": "lgbm_y7_core_turnslope_d3_n50",
        "cols": CORE_TURNOVER + ("e_ar_issued_lag1_over_lag3",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus issued lag1/lag3 slope (Q6 thinning)",
    },
    {
        "id": "TURNZERO_shallow",
        "model": "lgbm_y7_core_turnzero_d3_n50",
        "cols": CORE_TURNOVER + ("e_ar_issued_lag_nzero",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus count of zero-issuance months in t-1..t-3",
    },
    {
        "id": "TURNFC1_shallow",
        "model": "lgbm_y7_core_turnfc1_d3_n50",
        "cols": CORE_TURNOVER + ("f_fc_r_lag1",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus f_fc_r_lag1 (closer finance-cost lead)",
    },
    {
        "id": "TURNFC0_shallow",
        "model": "lgbm_y7_core_turnfc0_d3_n50",
        "cols": CORE_TURNOVER + ("f_fc_r",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus contemporaneous f_fc_r",
    },
    {
        "id": "TURNMEAN_shallow",
        "model": "lgbm_y7_core_turnmean_d3_n50",
        "cols": (
            "e_ar_issued_lag_mean",
            "e_ar_issued_lag_cv",
            "e_credit_note_ratio",
            "e_credit_note_ratio_lag1",
            "f_fc_r_lag3",
        ),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER with 3-month issued mean instead of last-month lag1",
    },
    {
        "id": "TURNL1LOG_shallow",
        "model": "lgbm_y7_core_turnl1log_d3_n50",
        "cols": (
            "e_ar_issued_lag1_log1p",
            "e_ar_issued_lag_cv",
            "e_credit_note_ratio",
            "e_credit_note_ratio_lag1",
            "f_fc_r_lag3",
        ),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER with log1p(issued_lag1) instead of raw euros",
    },
    {
        "id": "TURNIX_shallow",
        "model": "lgbm_y7_core_turnix_d3_n50",
        "cols": CORE_TURNOVER + ("e_cn_x_issued_cv",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus credit-note × issued-lag CV",
    },
    {
        "id": "TURN3_shallow",
        "model": "lgbm_y7_core_turn3_d3_n50",
        "cols": tuple(c for c in CORE_TURNOVER if c != "e_ar_issued_lag1"),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER minus last-month issued (is lag1 needed?)",
    },
    {
        "id": "TURNL13_shallow",
        "model": "lgbm_y7_core_turnl13_d3_n50",
        "cols": CORE_TURNOVER + ("e_ar_issued_lag3",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus issued lag3 as a separate stem",
    },
    {
        "id": "TURNPEND_shallow",
        "model": "lgbm_y7_core_turnpend_d3_n50",
        "cols": CORE_TURNOVER + ("e_pending_amt_share",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus pending amount share (Q5 book, not euro level)",
    },
    {
        "id": "TURNFX_shallow",
        "model": "lgbm_y7_core_turnfx_d3_n50",
        "cols": CORE_TURNOVER + ("e_fx_share",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus FX invoice share",
    },
    {
        "id": "TURNDPO_shallow",
        "model": "lgbm_y7_core_turndpo_d3_n50",
        "cols": CORE_TURNOVER + ("e_dpo_proxy",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus DPO (AP-side; is fold 4 a payables story?)",
    },
    {
        "id": "TURNDS_shallow",
        "model": "lgbm_y7_core_turnds_d3_n50",
        "cols": CORE_TURNOVER + ("f_ds_r_lag3",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus debt-service / inflow at t-3",
    },
    {
        "id": "TURNOV30_shallow",
        "model": "lgbm_y7_core_turnov30_d3_n50",
        "cols": CORE_TURNOVER + ("e_ar_overdue_30",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus AR overdue 30d share/count",
    },
    {
        "id": "TURNPAID_shallow",
        "model": "lgbm_y7_core_turnpaid_d3_n50",
        "cols": CORE_TURNOVER + ("e_delay_paid",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus AP payment delay",
    },
    {
        "id": "TURNDSSWAP_shallow",
        "model": "lgbm_y7_core_turndsswap_d3_n50",
        "cols": (
            "e_ar_issued_lag1",
            "e_ar_issued_lag_cv",
            "e_credit_note_ratio",
            "e_credit_note_ratio_lag1",
            "f_ds_r_lag3",
        ),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER with f_ds_r_lag3 instead of f_fc_r_lag3",
    },
    {
        "id": "TURNCNCV_shallow",
        "model": "lgbm_y7_core_turncncv_d3_n50",
        "cols": CORE_TURNOVER + ("e_credit_note_ratio_lag_cv",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus credit-note lag1/lag3 CV",
    },
    {
        "id": "BHI_shallow",
        "model": "lgbm_y7_core_bhi_d3_n50",
        "cols": (
            "e_dso_proxy_hi",
            "e_ar_issued_lag1",
            "e_credit_note_ratio",
            "e_credit_note_ratio_lag1",
            "e_delay_coll",
            "f_fc_r_lag3",
        ),
        "mode": "shallow",
        "claimable": False,
        "notes": "B with DSO gated at 1 month (short DSO zeroed; constant, not a fit)",
    },
    {
        "id": "TURNHI_shallow",
        "model": "lgbm_y7_core_turnhi_d3_n50",
        "cols": CORE_TURNOVER + ("e_dso_proxy_hi",),
        "mode": "shallow",
        "claimable": False,
        "notes": "TURNOVER plus gated DSO (can stretched-only DSO help fold 4?)",
    },
)


def _clip_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed 24-month clip. Constant, not a fit — holdout never used."""
    out = df.copy()
    if "e_dso_proxy" in out.columns:
        s = pd.to_numeric(out["e_dso_proxy"], errors="coerce")
        out["e_dso_proxy"] = s.clip(lower=0.0, upper=CLIP_DSO)
    return out


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "e_ar_issued" in out.columns:
        issued = pd.to_numeric(out["e_ar_issued"], errors="coerce").clip(lower=0.0)
        out["e_ar_issued_log1p"] = np.log1p(issued)
    dso = pd.to_numeric(out.get("e_dso_proxy"), errors="coerce") if "e_dso_proxy" in out.columns else None
    raw = (
        pd.to_numeric(out.get("e_dso_proxy_raw"), errors="coerce")
        if "e_dso_proxy_raw" in out.columns
        else None
    )
    if raw is not None:
        out["e_dso_proxy_c12"] = raw.clip(lower=0.0, upper=12.0)
    cn = pd.to_numeric(out.get("e_credit_note_ratio"), errors="coerce") if "e_credit_note_ratio" in out.columns else None
    if dso is not None:
        # Constant gate: only stretched DSO counts. Not a fit.
        out["e_dso_proxy_hi"] = dso.where(dso >= 1.0, 0.0)
    if dso is not None and cn is not None:
        out["e_dso_x_cn"] = dso * cn
    if "e_ar_issued_lag1" in out.columns and "e_ar_issued_lag3" in out.columns:
        a = pd.to_numeric(out["e_ar_issued_lag1"], errors="coerce").clip(lower=0.0)
        b = pd.to_numeric(out["e_ar_issued_lag3"], errors="coerce").clip(lower=0.0)
        out["e_ar_issued_lag1_over_lag3"] = a / (b + 1.0)
    lag_names = [c for c in ("e_ar_issued_lag1", "e_ar_issued_lag2", "e_ar_issued_lag3") if c in out.columns]
    if len(lag_names) >= 2:
        mat = out[lag_names].apply(pd.to_numeric, errors="coerce").clip(lower=0.0)
        mu = mat.mean(axis=1)
        sd = mat.std(axis=1, ddof=0)
        out["e_ar_issued_lag_cv"] = sd / (mu.abs() + 1.0)
        out["e_ar_issued_lag_cv_clip"] = out["e_ar_issued_lag_cv"].clip(upper=CLIP_ISSUED_CV)
        out["e_ar_issued_lag_nzero"] = (mat <= 0).sum(axis=1).astype(float)
        out["e_ar_issued_lag_mean"] = mu
        out["e_ar_issued_lag1_log1p"] = np.log1p(pd.to_numeric(out["e_ar_issued_lag1"], errors="coerce").clip(lower=0.0))
        cn = pd.to_numeric(out.get("e_credit_note_ratio"), errors="coerce") if "e_credit_note_ratio" in out.columns else None
        if cn is not None:
            out["e_cn_x_issued_cv"] = cn * out["e_ar_issued_lag_cv"]
        logm = np.log1p(mat)
        out["e_ar_issued_lag_cv_log"] = logm.std(axis=1, ddof=0) / (logm.mean(axis=1).abs() + 1e-6)
    if "e_ar_issued_lag1" in out.columns and "e_ar_issued_lag3" in out.columns:
        pair = out[["e_ar_issued_lag1", "e_ar_issued_lag3"]].apply(pd.to_numeric, errors="coerce").clip(lower=0.0)
        mu13 = pair.mean(axis=1)
        sd13 = pair.std(axis=1, ddof=0)
        out["e_ar_issued_lag13_cv"] = sd13 / (mu13.abs() + 1.0)
    cn_lags = [c for c in ("e_credit_note_ratio_lag1", "e_credit_note_ratio_lag3") if c in out.columns]
    if len(cn_lags) == 2:
        cnm = out[cn_lags].apply(pd.to_numeric, errors="coerce")
        cmu = cnm.mean(axis=1)
        csd = cnm.std(axis=1, ddof=0)
        out["e_credit_note_ratio_lag_cv"] = csd / (cmu.abs() + 1e-6)
    return out


def _spearman(a: pd.Series, b: pd.Series) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    )
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 30 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def _assert_specs_clean() -> None:
    """Fail fast if a spec smuggles D, meta, Y, or euro issued level."""
    for spec in SPECS:
        cols = list(spec["cols"])
        _assert_core(cols)
        if spec["mode"] not in {"early_stop", "shallow", "fixed50"}:
            raise RuntimeError(f"bad mode {spec['mode']} on {spec['id']}")


def _assert_core(cols: list[str]) -> None:
    leak = leakage_check(cols, Y_COL, FORBIDDEN)
    if not leak["ok"]:
        raise RuntimeError(f"leakage: {leak['issues']}")
    _assert_allowed(cols, Y_COL, FORBIDDEN)
    bad_d = [c for c in cols if str(c).startswith("d_") or c in BANNED_D]
    if bad_d:
        raise RuntimeError(f"family D leaked into X: {bad_d}")
    bad_meta = [c for c in cols if c in BANNED_META]
    if bad_meta:
        raise RuntimeError(f"static meta in X: {bad_meta}")
    bad_euro = [c for c in cols if c in BANNED_EURO]
    if bad_euro:
        raise RuntimeError(f"contemporaneous euro issued in X: {bad_euro}")
    yish = [c for c in cols if c == Y_COL or str(c).startswith("y")]
    if yish:
        raise RuntimeError(f"y columns in X: {yish}")
    if Y_COL.startswith("y7_") and any(str(c).startswith("d_") for c in cols):
        raise RuntimeError("D leaked into X for y7")


def _resolve_cols(panel: pd.DataFrame, cols: tuple[str, ...]) -> list[str]:
    have = [c for c in cols if c in panel.columns]
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        print(f"WARN missing columns (skipped): {missing}")
    if not have:
        raise RuntimeError("no requested columns present on the panel")
    _assert_core(have)
    return have


def _fit_shallow(Xtr, ytr) -> lgb.LGBMClassifier:
    params = dict(LGB_SHALLOW)
    params["scale_pos_weight"] = _scale_pos_weight(pd.Series(ytr))
    clf = lgb.LGBMClassifier(**params)
    clf.fit(Xtr, ytr)
    return clf


def _fit_fixed50(Xtr, ytr) -> lgb.LGBMClassifier:
    params = dict(LGB_FIXED50)
    params["scale_pos_weight"] = _scale_pos_weight(pd.Series(ytr))
    clf = lgb.LGBMClassifier(**params)
    clf.fit(Xtr, ytr)
    return clf


def _fit_mode(mode: str, Xtr, ytr, Xva=None, yva=None):
    if mode == "shallow":
        clf = _fit_shallow(Xtr, ytr)
        trees = int(LGB_SHALLOW["n_estimators"])
        return clf, trees
    if mode == "fixed50":
        clf = _fit_fixed50(Xtr, ytr)
        trees = int(LGB_FIXED50["n_estimators"])
        return clf, trees
    clf = _fit_lgb(Xtr, ytr, Xva, yva)
    trees = int(getattr(clf, "best_iteration_", LGB_BASE["n_estimators"]) or LGB_BASE["n_estimators"])
    return clf, trees


def _best_single_cv(panel: pd.DataFrame, feat_cols: list[str], train_lab: pd.Series) -> dict:
    """Per-fold train pick among *this* X, val score. Holdout already out."""
    rows = []
    pick_counts: dict[str, int] = {}
    for k in range(N_FOLDS):
        tr = train_lab & (panel["fold"] != k)
        va = train_lab & (panel["fold"] == k)
        assert_no_holdout(panel.loc[tr, "company_id"])
        assert_no_holdout(panel.loc[va, "company_id"])
        ytr = panel.loc[tr, Y_COL].astype(float)
        yva = panel.loc[va, Y_COL].astype(float)
        n_tr = int(ytr.notna().sum())
        min_n = max(200, int(0.25 * n_tr)) if n_tr else 200
        best = None
        for col in feat_cols:
            s = pd.to_numeric(panel[col], errors="coerce")
            trd = pd.DataFrame({"y": ytr, "s": s[tr]}).dropna()
            if len(trd) < min_n or trd["y"].nunique() < 2 or trd["s"].nunique() < 2:
                continue
            auc_p = auroc(trd["y"], trd["s"])
            auc_n = auroc(trd["y"], -trd["s"])
            if not np.isfinite(auc_p) and not np.isfinite(auc_n):
                continue
            sign = -1 if (np.isfinite(auc_n) and (not np.isfinite(auc_p) or auc_n > auc_p)) else 1
            train_auc = auc_n if sign < 0 else auc_p
            if best is None or train_auc > best["train_auroc"]:
                best = {"feature": col, "sign": sign, "train_auroc": float(train_auc)}
        if best is None or yva.nunique() < 2:
            rows.append({"fold": k, "auroc": float("nan"), "feature": None})
            continue
        sva = best["sign"] * pd.to_numeric(panel.loc[va, best["feature"]], errors="coerce")
        vad = pd.DataFrame({"y": yva, "s": sva}).dropna()
        auc = auroc(vad["y"], vad["s"]) if len(vad) and vad["y"].nunique() == 2 else float("nan")
        rows.append(
            {
                "fold": k,
                "auroc": auc,
                "feature": best["feature"],
                "sign": best["sign"],
                "train_auroc": best["train_auroc"],
            }
        )
        pick_counts[best["feature"]] = pick_counts.get(best["feature"], 0) + 1
    aucs = np.asarray([r["auroc"] for r in rows], dtype=float)
    ok = aucs[np.isfinite(aucs)]
    top = max(pick_counts, key=pick_counts.get) if pick_counts else None
    return {
        "cv_auroc": float(ok.mean()) if ok.size else float("nan"),
        "folds": rows,
        "picked_most": top,
        "pick_counts": pick_counts,
    }


def _collapsed(cv_rows: list[dict]) -> bool:
    trees = [int(r["best_iteration"]) for r in cv_rows if r.get("best_iteration") is not None]
    if not trees:
        return False
    return int(np.median(trees)) <= COLLAPSE_TREES or min(trees) <= COLLAPSE_TREES


def leak_size_screen(panel: pd.DataFrame, feat_cols: list[str], train_lab: pd.Series) -> dict:
    """Train-labeled only. Fail D-copy |ρ|≥0.80 or size |ρ|>0.85 vs a_in3."""
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    rows = []
    fail_d = []
    fail_size = []
    for col in feat_cols:
        x = panel.loc[train_lab, col]
        rec = {
            "feature": col,
            "rho_d_top1": (
                _spearman(x, panel.loc[train_lab, "d_cust_top1"])
                if "d_cust_top1" in panel.columns
                else float("nan")
            ),
            "rho_d_hhi": (
                _spearman(x, panel.loc[train_lab, "d_cust_hhi"])
                if "d_cust_hhi" in panel.columns
                else float("nan")
            ),
            "rho_a_in3": (
                _spearman(x, panel.loc[train_lab, "a_in3"])
                if "a_in3" in panel.columns
                else float("nan")
            ),
            "rho_log1p_a_in3": (
                _spearman(x, np.log1p(pd.to_numeric(panel.loc[train_lab, "a_in3"], errors="coerce").abs()))
                if "a_in3" in panel.columns
                else float("nan")
            ),
        }
        if np.isfinite(rec["rho_d_top1"]) and abs(rec["rho_d_top1"]) >= CONC_COPY_CUT:
            fail_d.append((col, "d_cust_top1", rec["rho_d_top1"]))
        if np.isfinite(rec["rho_d_hhi"]) and abs(rec["rho_d_hhi"]) >= CONC_COPY_CUT:
            fail_d.append((col, "d_cust_hhi", rec["rho_d_hhi"]))
        if np.isfinite(rec["rho_a_in3"]) and abs(rec["rho_a_in3"]) > SIZE_CUT:
            fail_size.append((col, "a_in3", rec["rho_a_in3"]))
        rows.append(rec)
    tab = pd.DataFrame(rows)
    worst_d = None
    if len(tab):
        mag = tab[["rho_d_top1", "rho_d_hhi"]].abs().max(axis=1)
        i = int(mag.idxmax()) if mag.notna().any() else None
        if i is not None and np.isfinite(mag.loc[i]):
            worst_d = {
                "feature": tab.loc[i, "feature"],
                "rho_d_top1": float(tab.loc[i, "rho_d_top1"]),
                "rho_d_hhi": float(tab.loc[i, "rho_d_hhi"]),
                "abs": float(mag.loc[i]),
            }
    worst_size = None
    if len(tab) and tab["rho_a_in3"].notna().any():
        i = int(tab["rho_a_in3"].abs().idxmax())
        worst_size = {
            "feature": tab.loc[i, "feature"],
            "rho_a_in3": float(tab.loc[i, "rho_a_in3"]),
            "rho_log1p_a_in3": float(tab.loc[i, "rho_log1p_a_in3"]),
        }
    ytr = panel.loc[train_lab, Y_COL].astype(float)
    size_auc = (
        auroc(ytr, np.log1p(pd.to_numeric(panel.loc[train_lab, "a_in3"], errors="coerce").abs()))
        if "a_in3" in panel.columns
        else float("nan")
    )
    ok = (not fail_d) and (not fail_size)
    print("LEAK/SIZE screen (train labeled; D not in X)")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Y vs log1p(a_in3) AUROC={size_auc:.3f} (published 0.465; gate <0.60)")
    if worst_d:
        print(
            f"worst |ρ| vs D: {worst_d['feature']} top1={worst_d['rho_d_top1']:.3f} "
            f"hhi={worst_d['rho_d_hhi']:.3f} (fail≥{CONC_COPY_CUT:.2f})"
        )
    if worst_size:
        print(
            f"worst |ρ| vs a_in3: {worst_size['feature']} ρ={worst_size['rho_a_in3']:.3f} "
            f"(fail>{SIZE_CUT:.2f})"
        )
    if fail_d:
        print(f"screen FAIL D-concentration |ρ|≥{CONC_COPY_CUT:.2f}: {fail_d}")
    elif fail_size:
        print(f"screen FAIL SIZE |ρ|>{SIZE_CUT:.2f} vs a_in3: {fail_size}")
    else:
        print("screen PASS")
    return {
        "ok": ok,
        "rows": rows,
        "worst_d": worst_d,
        "worst_size": worst_size,
        "size_auc_y": float(size_auc),
        "fail_d": fail_d,
        "fail_size": fail_size,
    }


def prepare_panel(store: pd.DataFrame, y_panel: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    panel = store.merge(y_panel[["company_id", "period", Y_COL]], on=["company_id", "period"], how="left")
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    if "e_dso_proxy" in panel.columns:
        panel["e_dso_proxy_raw"] = pd.to_numeric(panel["e_dso_proxy"], errors="coerce")
    panel = _clip_proxies(panel)
    parents = [c for c in LAG_PARENTS if c in panel.columns]
    if "e_dso_proxy_raw" in panel.columns:
        parents = parents + ["e_dso_proxy_raw"]
    panel = add_lags(panel, parents, LAGS)
    n0 = len(panel)
    if "e_ar_issued" in panel.columns and "e_ar_issued_lag2" not in panel.columns:
        # lag2 only for the issued-volatility diagnostic (past-only shift).
        extra = add_lags(panel[["company_id", "period", "e_ar_issued"]].copy(), ["e_ar_issued"], (2,))
        extra = extra[["company_id", "period", "e_ar_issued_lag2"]].drop_duplicates(
            ["company_id", "period"]
        )
        panel = panel.merge(extra, on=["company_id", "period"], how="left")
        if len(panel) != n0:
            raise RuntimeError(f"lag2 merge changed row count {n0} -> {len(panel)}")
    panel = _add_derived(panel)
    if panel.duplicated(["company_id", "period"]).any():
        raise RuntimeError("duplicate company_id+period after prepare_panel")
    return panel


def _univ_table(panel: pd.DataFrame, cols: list[str], train_lab: pd.Series) -> pd.DataFrame:
    rows = []
    y = panel.loc[train_lab, Y_COL].astype(float)
    for col in cols:
        s = pd.to_numeric(panel.loc[train_lab, col], errors="coerce")
        d = pd.DataFrame({"y": y, "s": s}).dropna()
        if len(d) < 30 or d["y"].nunique() < 2 or d["s"].nunique() < 2:
            rows.append({"feature": col, "n": int(len(d)), "auroc": float("nan"), "sign": 0})
            continue
        auc_p = auroc(d["y"], d["s"])
        auc_n = auroc(d["y"], -d["s"])
        sign = -1 if (np.isfinite(auc_n) and (not np.isfinite(auc_p) or auc_n > auc_p)) else 1
        rows.append(
            {
                "feature": col,
                "n": int(len(d)),
                "auroc": float(auc_n if sign < 0 else auc_p),
                "sign": sign,
            }
        )
    tab = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    print("univariate train AUROC on card union (sign from train)")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return tab


def _fold_coverage(panel: pd.DataFrame, train_lab: pd.Series) -> None:
    print("fold coverage / DSO / issued (train labeled)")
    for k in range(N_FOLDS):
        sl = train_lab & (panel["fold"] == k)
        assert_no_holdout(panel.loc[sl, "company_id"])
        dso = pd.to_numeric(panel.loc[sl, "e_dso_proxy"], errors="coerce")
        raw = (
            pd.to_numeric(panel.loc[sl, "e_dso_proxy_raw"], errors="coerce")
            if "e_dso_proxy_raw" in panel.columns
            else dso
        )
        iss = pd.to_numeric(panel.loc[sl, "e_ar_issued"], errors="coerce")
        print(
            f"  fold {k}: dso_cov={dso.notna().mean():.3f} dso_med={dso.median():.3f} "
            f"raw_p99={raw.quantile(0.99):.1f} issued_cov={iss.notna().mean():.3f} "
            f"issued_med={iss.median():.0f}"
        )


def _shap_train(clf: lgb.LGBMClassifier, panel: pd.DataFrame, feat_cols: list[str], train_lab: pd.Series) -> dict:
    """Train-only TreeSHAP. Holdout never sampled."""
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    idx = panel.index[train_lab].to_numpy()
    if len(idx) > 4000:
        rng = np.random.default_rng(FOLD_SEED)
        idx = np.sort(rng.choice(idx, size=4000, replace=False))
    assert_no_holdout(panel.loc[idx, "company_id"])
    X = panel.loc[idx, feat_cols]
    try:
        import shap
    except Exception as exc:
        print(f"SHAP import failed: {exc}")
        return {"ok": False, "error": str(exc)}
    explainer = shap.TreeExplainer(clf)
    raw = explainer.shap_values(X)
    arr = np.asarray(raw[1] if isinstance(raw, list) and len(raw) > 1 else raw)
    if arr.ndim == 3:
        arr = arr[:, :, -1]
    mean_abs = np.nanmean(np.abs(arr), axis=0)
    rows = []
    for j, c in enumerate(feat_cols):
        rho = _spearman(X[c], pd.Series(arr[:, j], index=X.index))
        sign = 1 if np.isfinite(rho) and rho > 0 else (-1 if np.isfinite(rho) and rho < 0 else 0)
        rows.append({"feature": c, "mean_abs_shap": float(mean_abs[j]), "sign": sign, "rho_shap": float(rho)})
    tab = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    print(f"train SHAP sample={len(X)} (holdout excluded)")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    return {"ok": True, "n": int(len(X)), "rows": tab.to_dict(orient="records")}


def _fold4_groups(panel: pd.DataFrame, oof: pd.DataFrame) -> list[dict]:
    """Train fold 4 only. Which groups make the weak 0.55 fold?"""
    sl = oof[oof["fold"] == 4]
    if sl.empty or "group_id" not in panel.columns:
        print("fold4 group dump skipped")
        return []
    g = sl.groupby("group_id", dropna=False)
    rows = []
    for gid, d in g:
        y = d["y"].to_numpy(dtype=float)
        s = d["s"].to_numpy(dtype=float)
        auc = auroc(y, s) if (y == 0).any() and (y == 1).any() else float("nan")
        rows.append(
            {
                "group_id": str(gid),
                "n": int(len(d)),
                "n_pos": int((y == 1).sum()),
                "rate": float(y.mean()) if len(d) else float("nan"),
                "auroc": auc,
                "mean_pred": float(np.nanmean(s)),
            }
        )
    tab = pd.DataFrame(rows).sort_values(["n", "n_pos"], ascending=False)
    print("fold 4 groups (train OOF; largest first)")
    print(tab.head(12).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return tab.head(12).to_dict(orient="records")


def _perm_fold4(panel: pd.DataFrame, feat_cols: list[str], train_lab: pd.Series) -> None:
    """Train on folds 0–3, permute each col on fold 4. Holdout already out."""
    tr = train_lab & (panel["fold"] != 4)
    va = train_lab & (panel["fold"] == 4)
    assert_no_holdout(panel.loc[tr, "company_id"])
    assert_no_holdout(panel.loc[va, "company_id"])
    ytr = panel.loc[tr, Y_COL].astype(float)
    yva = panel.loc[va, Y_COL].astype(float)
    if ytr.nunique() < 2 or yva.nunique() < 2:
        return
    clf = _fit_shallow(panel.loc[tr, feat_cols], ytr)
    base = auroc(yva, clf.predict_proba(panel.loc[va, feat_cols])[:, 1])
    rng = np.random.default_rng(FOLD_SEED)
    print(f"fold-4 permutation ΔAUROC (base={base:.4f}; train folds 0–3 only)")
    rows = []
    for c in feat_cols:
        X = panel.loc[va, feat_cols].copy()
        X[c] = rng.permutation(pd.to_numeric(X[c], errors="coerce").to_numpy())
        auc = auroc(yva, clf.predict_proba(X)[:, 1])
        delta = float(base - auc)
        rows.append((c, delta, auc))
        print(f"  {c}: Δ={delta:+.4f} -> {auc:.4f}")
    rows.sort(key=lambda t: t[1], reverse=True)
    if rows:
        print(f"  worst drop: {rows[0][0]} {rows[0][1]:+.4f}")


def _dso_quintile_auroc(panel: pd.DataFrame, oof: pd.DataFrame, train_lab: pd.Series) -> None:
    """Train OOF AUROC by clipped-DSO quintile. Holdout already out of OOF."""
    assert_no_holdout(oof["company_id"])
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    if "e_dso_proxy" not in oof.columns:
        print("OOF AUROC by DSO quintile skipped (no DSO on OOF)")
        return
    o = oof.copy()
    dso = pd.to_numeric(o["e_dso_proxy"], errors="coerce")
    ok = dso.notna()
    if int(ok.sum()) < 100:
        print("OOF AUROC by DSO quintile skipped (too few DSO values)")
        return
    q = pd.qcut(dso[ok], 5, duplicates="drop")
    print("OOF AUROC by DSO quintile (train folds only; NaN DSO dropped)")
    for name, d in o.loc[ok].groupby(q, observed=False):
        auc = auroc(d["y"], d["s"]) if d["y"].nunique() == 2 else float("nan")
        print(
            f"  {name}: n={len(d)} pos={int((d['y'] == 1).sum())} "
            f"dso_med={float(pd.to_numeric(d['e_dso_proxy'], errors='coerce').median()):.3f} "
            f"rate={float(d['y'].mean()):.3f} auroc={auc:.3f}"
        )
    rho = _spearman(o.loc[ok, "s"], dso[ok])
    print(f"OOF score vs clipped DSO Spearman ρ={rho:.3f} (B-card should be high; TURNOVER near 0)")


def _group_contrast(panel: pd.DataFrame, train_lab: pd.Series, gids: tuple[str, ...]) -> None:
    """Train-labeled DSO / issued / Y rate for the fold-4 problem groups vs rest."""
    if "group_id" not in panel.columns:
        return
    sl = panel.loc[train_lab].copy()
    assert_no_holdout(sl["company_id"])
    sl["gid"] = sl["group_id"].astype(str)
    sl["hit"] = sl["gid"].isin(set(gids))
    print("group contrast (train labeled; fold-4 problem groups vs rest)")
    for name, d in (("problem", sl[sl["hit"]]), ("rest", sl[~sl["hit"]])):
        y = d[Y_COL].astype(float)
        dso = pd.to_numeric(d.get("e_dso_proxy"), errors="coerce")
        iss = pd.to_numeric(d.get("e_ar_issued_lag1"), errors="coerce") if "e_ar_issued_lag1" in d.columns else None
        cn = pd.to_numeric(d.get("e_credit_note_ratio"), errors="coerce")
        vol = pd.to_numeric(d.get("e_ar_issued_lag_cv"), errors="coerce") if "e_ar_issued_lag_cv" in d.columns else None
        print(
            f"  {name}: n={len(d)} pos={int((y == 1).sum())} rate={float(y.mean()):.3f} "
            f"dso_med={float(dso.median()):.3f} dso_cov={float(dso.notna().mean()):.3f} "
            f"issued_lag1_med={float(iss.median()) if iss is not None else float('nan'):.0f} "
            f"cn_med={float(cn.median()):.3f} "
            f"issued_cv_med={float(vol.median()) if vol is not None else float('nan'):.3f}"
        )


def run_spec(spec: dict, panel: pd.DataFrame, hold_ids: set[str], write_registry: bool) -> dict:
    print("\n" + "=" * 72)
    print(
        f"SPEC {spec['id']} model={spec['model']} mode={spec['mode']} "
        f"n_req={len(spec['cols'])}"
    )
    print("=" * 72)

    feat_cols = _resolve_cols(panel, spec["cols"])
    fams = sorted({_family_of(c) for c in feat_cols if c not in BANNED_META})
    print(f"n_x={len(feat_cols)} cols={feat_cols} families={fams}")

    is_train = train_mask(panel["company_id"])
    is_hold = panel["company_id"].isin(hold_ids)
    labeled = panel[Y_COL].notna()
    train_lab = is_train & labeled
    ho_lab = is_hold & labeled
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    hold_in_train = set(panel.loc[train_lab, "company_id"].astype(str)) & hold_ids
    if hold_in_train:
        raise RuntimeError("holdout leaked into train labeled rows")

    n_lab = int(train_lab.sum())
    n_pos = int((panel.loc[train_lab, Y_COL] == 1).sum())
    rate = float(panel.loc[train_lab, Y_COL].mean()) if n_lab else float("nan")
    n_train_cm = int(is_train.sum())
    coverage = (n_lab / n_train_cm) if n_train_cm else float("nan")
    print(
        f"train_cm={n_train_cm} labeled={n_lab} pos={n_pos} "
        f"rate={rate:.4f} coverage={coverage:.4f} hold_labeled={int(ho_lab.sum())}"
    )

    screen = leak_size_screen(panel, feat_cols, train_lab)

    cv_rows = []
    best_iters = []
    oof_chunks = []
    for k in range(N_FOLDS):
        tr = train_lab & (panel["fold"] != k)
        va = train_lab & (panel["fold"] == k)
        assert_no_holdout(panel.loc[tr, "company_id"])
        assert_no_holdout(panel.loc[va, "company_id"])
        ytr = panel.loc[tr, Y_COL].astype(float)
        yva = panel.loc[va, Y_COL].astype(float)
        if ytr.nunique() < 2 or yva.nunique() < 2:
            print(f"fold {k}: skip (one class empty)")
            cv_rows.append(
                {
                    "fold": k,
                    "auroc": float("nan"),
                    "pr_auc": float("nan"),
                    "n_val": int(va.sum()),
                    "n_pos": 0,
                    "best_iteration": None,
                }
            )
            continue
        clf, trees = _fit_mode(
            spec["mode"],
            panel.loc[tr, feat_cols],
            ytr,
            panel.loc[va, feat_cols],
            yva,
        )
        pred = clf.predict_proba(panel.loc[va, feat_cols])[:, 1]
        oof_chunks.append(
            pd.DataFrame(
                {
                    "company_id": panel.loc[va, "company_id"].astype(str).to_numpy(),
                    "group_id": (
                        panel.loc[va, "group_id"].astype(str).to_numpy()
                        if "group_id" in panel.columns
                        else np.array([""] * int(va.sum()))
                    ),
                    "period": pd.to_datetime(panel.loc[va, "period"]).to_numpy(),
                    "e_dso_proxy": pd.to_numeric(panel.loc[va, "e_dso_proxy"], errors="coerce").to_numpy()
                    if "e_dso_proxy" in panel.columns
                    else np.full(int(va.sum()), np.nan),
                    "fold": k,
                    "y": yva.to_numpy(dtype=float),
                    "s": pred,
                }
            )
        )
        row = {
            "fold": k,
            "auroc": auroc(yva, pred),
            "pr_auc": pr_auc(yva, pred),
            "n_val": int(va.sum()),
            "n_pos": int((yva == 1).sum()),
            "best_iteration": trees,
        }
        best_iters.append(trees)
        cv_rows.append(row)
        print(
            f"fold {k}: auroc={row['auroc']:.4f} pr_auc={row['pr_auc']:.4f} "
            f"n={row['n_val']} pos={row['n_pos']} trees={trees}"
        )

    aucs = np.asarray([r["auroc"] for r in cv_rows], dtype=float)
    ok = aucs[np.isfinite(aucs)]
    cv_auroc = float(ok.mean()) if ok.size else float("nan")
    cv_sd = float(ok.std(ddof=1)) if ok.size >= 2 else float("nan")
    prs = np.asarray([r.get("pr_auc", float("nan")) for r in cv_rows], dtype=float)
    pr_ok = prs[np.isfinite(prs)]
    cv_pr = float(pr_ok.mean()) if pr_ok.size else float("nan")
    collapsed = spec["mode"] == "early_stop" and _collapsed(cv_rows)
    print(
        f"CV mean AUROC={cv_auroc:.4f} sd={cv_sd:.4f} PR-AUC={cv_pr:.4f} "
        f"n_x={len(feat_cols)} vs_published={PUBLISHED_CV:.3f} collapsed={collapsed}"
    )

    single = _best_single_cv(panel, feat_cols, train_lab)
    print(
        f"best-single CV AUROC={single['cv_auroc']:.4f} "
        f"picked_most={single['picked_most']} counts={single['pick_counts']}"
    )
    dummy_cv = 0.5
    beats_dummy = np.isfinite(cv_auroc) and cv_auroc > dummy_cv
    beats_single = (
        np.isfinite(cv_auroc)
        and np.isfinite(single["cv_auroc"])
        and cv_auroc > single["cv_auroc"]
    )
    print(f"beats dummy={beats_dummy} beats_single={beats_single}")

    assert_no_holdout(panel.loc[train_lab, "company_id"])
    ytr_all = panel.loc[train_lab, Y_COL].astype(float)
    if spec["mode"] == "shallow":
        final = _fit_shallow(panel.loc[train_lab, feat_cols], ytr_all)
        n_trees = int(LGB_SHALLOW["n_estimators"])
    elif spec["mode"] == "fixed50":
        final = _fit_fixed50(panel.loc[train_lab, feat_cols], ytr_all)
        n_trees = int(LGB_FIXED50["n_estimators"])
    else:
        n_trees = int(np.median(best_iters)) if best_iters else int(LGB_BASE["n_estimators"])
        n_trees = max(50, n_trees)
        final = _fit_lgb(panel.loc[train_lab, feat_cols], ytr_all, n_estimators=n_trees)
    imp = _gain_table(final, feat_cols, 12)
    print("top gain (train fit, not a claim)")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    hold_auroc = float("nan")
    hold_pr = float("nan")
    n_hold_pos = 0
    n_hold_lab = int(ho_lab.sum())
    if n_hold_lab and ho_lab.any():
        ho_pred = final.predict_proba(panel.loc[ho_lab, feat_cols])[:, 1]
        ho_y = panel.loc[ho_lab, Y_COL].to_numpy(dtype=float)
        ho_ok = np.isfinite(ho_y) & np.isfinite(ho_pred)
        hold_auroc = auroc(ho_y[ho_ok], ho_pred[ho_ok])
        hold_pr = pr_auc(ho_y[ho_ok], ho_pred[ho_ok])
        n_hold_pos = int((np.asarray(ho_y) == 1).sum())
        print(
            f"HOLDOUT check (not the claim) AUROC={hold_auroc:.4f} "
            f"PR-AUC={hold_pr:.4f} n={int(ho_ok.sum())} pos={n_hold_pos}"
        )

    shap_card = None
    fold4 = []
    if spec.get("explain"):
        shap_card = _shap_train(final, panel, feat_cols, train_lab)
        if oof_chunks:
            oof = pd.concat(oof_chunks, ignore_index=True)
            fold4 = _fold4_groups(panel, oof)
            hard = tuple(
                str(r["group_id"])
                for r in fold4
                if r.get("n", 0) >= 100 and r.get("rate", 0) >= 0.5
            )
            if hard:
                _group_contrast(panel, train_lab, hard)
            _dso_quintile_auroc(panel, oof, train_lab)
            _perm_fold4(panel, feat_cols, train_lab)

    out = {
        "id": spec["id"],
        "model": spec["model"],
        "mode": spec["mode"],
        "claimable": bool(spec.get("claimable", False)),
        "cols": feat_cols,
        "n_x": len(feat_cols),
        "families": fams,
        "cv_auroc": cv_auroc,
        "cv_auroc_sd": cv_sd,
        "cv_pr_auc": cv_pr,
        "cv_folds": cv_rows,
        "collapsed": collapsed,
        "n_trees_median": n_trees,
        "train_labeled": n_lab,
        "train_pos": n_pos,
        "train_base_rate": rate,
        "coverage": coverage,
        "dummy_cv": dummy_cv,
        "single": single,
        "beats_dummy": beats_dummy,
        "beats_single": beats_single,
        "holdout_auroc": hold_auroc,
        "holdout_pr_auc": hold_pr,
        "n_hold_labeled": n_hold_lab,
        "n_hold_pos": n_hold_pos,
        "top12": imp.to_dict(orient="records"),
        "screen": screen,
        "shap": shap_card,
        "fold4_groups": fold4,
        "notes": spec["notes"],
    }
    if write_registry:
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
        append_registry(registry_rows(ts, [out], {"decision": "ITER", "winner": None, "reason": "per-spec"}))
        print("appended registry rows for this spec")
    return out


def _verdict(results: dict[str, dict]) -> dict:
    claim = []
    for r in results.values():
        if r.get("mode") != "early_stop":
            continue
        if r.get("collapsed"):
            continue
        if not r.get("claimable"):
            continue
        a = r["cv_auroc"]
        if (r.get("screen") or {}).get("ok") is False:
            continue
        if np.isfinite(a) and a >= PUBLISHED_CV - KEEP_EPS and r["n_x"] < PUBLISHED_N_X:
            claim.append((r["id"], a, r["n_x"]))
    if claim:
        claim.sort(key=lambda t: (t[2], -t[1]))
        winner = claim[0][0]
        return {
            "decision": "KEEP",
            "winner": winner,
            "reason": (
                f"{winner} CV {results[winner]['cv_auroc']:.3f} >= {PUBLISHED_CV:.3f} "
                f"(eps {KEEP_EPS}) with n_x={results[winner]['n_x']} < {PUBLISHED_N_X}"
            ),
        }
    close = []
    park_ids = []
    for r in results.values():
        if r.get("mode") != "early_stop" or not r.get("claimable"):
            continue
        a = r["cv_auroc"]
        if not np.isfinite(a) or a < PUBLISHED_CV - PARK_DELTA:
            park_ids.append(r["id"])
        else:
            close.append(r["id"])
    early = [r for r in results.values() if r.get("mode") == "early_stop" and r.get("claimable")]
    if early and close:
        names = ", ".join(f"{i}={results[i]['cv_auroc']:.3f}" for i in close)
        return {
            "decision": "CLOSE",
            "winner": None,
            "reason": (
                f"no KEEP (or early-stop collapsed); within {PARK_DELTA:.2f} of "
                f"{PUBLISHED_CV:.3f}: {names}"
            ),
        }
    if early and park_ids and not close:
        names = ", ".join(f"{i}={results[i]['cv_auroc']:.3f}" for i in park_ids)
        return {
            "decision": "PARK",
            "winner": None,
            "reason": (
                f"claimable early-stop specs lose by >{PARK_DELTA:.2f} vs "
                f"{PUBLISHED_CV:.3f}: {names}"
            ),
        }
    return {
        "decision": "CLOSE",
        "winner": None,
        "reason": "no claimable early-stop number yet; diagnostics only so far",
    }


def registry_rows(ts: str, results: list[dict], verdict: dict) -> list[dict]:
    out = []
    for r in results:
        folds = ",".join(
            f"{x['fold']}:{x['auroc']:.4f}" if np.isfinite(x.get("auroc", float("nan"))) else f"{x['fold']}:nan"
            for x in r["cv_folds"]
        )
        trees_f = ",".join(
            f"{x['fold']}:{x.get('best_iteration')}" for x in r["cv_folds"]
        )
        fam = "+".join(s.upper() for s in r["families"])
        cov = r["coverage"]
        worst_d = (r.get("screen") or {}).get("worst_d") or {}
        worst_s = (r.get("screen") or {}).get("worst_size") or {}
        notes = (
            f"{verdict['decision']}; {r['notes']}; vs_published={PUBLISHED_CV:.3f}/n_x={PUBLISHED_N_X}; "
            f"cv={r['cv_auroc']:.4f}±{r['cv_auroc_sd']:.4f}; n_x={r['n_x']}; "
            f"mode={r['mode']}; collapsed={r['collapsed']}; trees={r['n_trees_median']}; "
            f"vs_dummy={r['cv_auroc']:.3f}/{r['dummy_cv']:.3f}; "
            f"vs_single={r['single'].get('picked_most')} {r['single']['cv_auroc']}; "
            f"beats_single={r['beats_single']}; folds={folds}; trees_folds={trees_f}; "
            f"hold_check={r.get('holdout_auroc')}; "
            f"train={r['train_labeled']}/{r['train_pos']}/{r['train_base_rate']:.4f}; "
            f"worst_d={worst_d.get('feature')} {worst_d.get('abs')}; "
            f"worst_size={worst_s.get('feature')} {worst_s.get('rho_a_in3')}; "
            f"quote=cv5_group_not_holdout; {verdict['reason']}"
        )
        cov_s = f"{cov:.4f}" if np.isfinite(cov) else ""
        val = f"{r['cv_auroc']:.6g}" if np.isfinite(r["cv_auroc"]) else ""
        out.append(
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": fam,
                "y": Y_COL,
                "model": r["model"],
                "split": "cv5_group",
                "metric": "auroc",
                "value": val,
                "coverage": cov_s,
                "notes": notes,
            }
        )
        if np.isfinite(r["cv_auroc_sd"]):
            out.append(
                {
                    "ts": ts,
                    "round": ROUND,
                    "wave": WAVE,
                    "agent": AGENT,
                    "x_families": fam,
                    "y": Y_COL,
                    "model": r["model"],
                    "split": "cv5_group",
                    "metric": "auroc_sd",
                    "value": f"{r['cv_auroc_sd']:.6g}",
                    "coverage": cov_s,
                    "notes": notes,
                }
            )
        if r["single"].get("picked_most") and r.get("mode") == "early_stop":
            out.append(
                {
                    "ts": ts,
                    "round": ROUND,
                    "wave": WAVE,
                    "agent": AGENT,
                    "x_families": _family_of(str(r["single"]["picked_most"])).upper(),
                    "y": Y_COL,
                    "model": f"single_{r['single']['picked_most']}",
                    "split": "cv5_group",
                    "metric": "auroc",
                    "value": (
                        f"{r['single']['cv_auroc']:.6g}"
                        if np.isfinite(r["single"]["cv_auroc"])
                        else ""
                    ),
                    "coverage": cov_s,
                    "notes": (
                        f"train-fold pick among {r['id']} cols; "
                        f"picked_most={r['single']['picked_most']}; "
                        f"counts={r['single']['pick_counts']}"
                    ),
                }
            )
    return out


def _parse_specs(wanted: str | None) -> tuple[dict, ...]:
    if not wanted or wanted.strip() in {"all", "*"}:
        return SPECS
    ids = [x.strip() for x in wanted.split(",") if x.strip()]
    by = {s["id"]: s for s in SPECS}
    missing = [i for i in ids if i not in by]
    if missing:
        raise SystemExit(f"unknown spec ids {missing}; known={list(by)}")
    return tuple(by[i] for i in ids)


def run(wanted: str | None = None, write_registry: bool = True) -> dict:
    _assert_specs_clean()
    specs = _parse_specs(wanted)
    hold_ids = load_holdout()
    con = connect()
    store, x_source = load_store(con)
    store = _keys(store)
    train_cos = train_companies(con)
    assert_no_holdout(train_cos)
    if set(train_cos["company_id"].astype(str)) & hold_ids:
        raise RuntimeError("holdout companies in train_companies()")
    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    grid = store[["company_id", "period"]].drop_duplicates()
    y7, _y8, y_source = load_y(con, grid)
    con.close()
    if Y_COL not in y7.columns:
        raise RuntimeError(f"{Y_COL} missing from Y source={y_source}")
    y_panel = _keys(y7[["company_id", "period", Y_COL]])
    panel = prepare_panel(store, y_panel, folds)
    print(
        f"store={store.shape} source={x_source} y_source={y_source} "
        f"train_cos={train_cos['company_id'].nunique()} hold_cos={len(hold_ids)} "
        f"folds={folds['fold'].nunique()} panel={panel.shape}"
    )
    print("Y7 dip-vs-fall (Q4). Never family D. Never e_ar_issued level. Quote CV not holdout.")
    is_train = train_mask(panel["company_id"])
    labeled = panel[Y_COL].notna()
    train_lab0 = is_train & labeled
    assert_no_holdout(panel.loc[train_lab0, "company_id"])
    print("fold base rates (train labeled only)")
    for k in range(N_FOLDS):
        sl = train_lab0 & (panel["fold"] == k)
        assert_no_holdout(panel.loc[sl, "company_id"])
        yk = panel.loc[sl, Y_COL]
        print(
            f"  fold {k}: n={int(sl.sum())} pos={int((yk == 1).sum())} "
            f"rate={float(yk.mean()) if sl.any() else float('nan'):.4f}"
        )
    _fold_coverage(panel, train_lab0)
    union_cols = []
    for spec in specs:
        for c in spec["cols"]:
            if c in panel.columns and c not in union_cols:
                union_cols.append(c)
    if union_cols:
        _univ_table(panel, union_cols, train_lab0)

    results = []
    for spec in specs:
        results.append(run_spec(spec, panel, hold_ids, write_registry=False))
    by_id = {r["id"]: r for r in results}
    verdict = _verdict(by_id)
    print("\nVERDICT")
    print(json.dumps(verdict, indent=2))

    if write_registry:
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
        append_registry(registry_rows(ts, results, verdict))
        print(f"appended {len(results)} spec block(s) to registry")

    slim = {
        r["id"]: {
            "cv_auroc": r["cv_auroc"],
            "cv_auroc_sd": r["cv_auroc_sd"],
            "n_x": r["n_x"],
            "cols": r["cols"],
            "collapsed": r["collapsed"],
            "mode": r["mode"],
            "dummy_cv": r["dummy_cv"],
            "single_cv": r["single"]["cv_auroc"],
            "single": r["single"].get("picked_most"),
            "beats_dummy": r["beats_dummy"],
            "beats_single": r["beats_single"],
            "train_labeled": r["train_labeled"],
            "train_pos": r["train_pos"],
            "holdout_check": r["holdout_auroc"],
            "folds": [
                {
                    "fold": x["fold"],
                    "auroc": x["auroc"],
                    "trees": x.get("best_iteration"),
                }
                for x in r["cv_folds"]
            ],
        }
        for r in results
    }
    slim["verdict"] = verdict
    slim["published"] = {"cv": PUBLISHED_CV, "n_x": PUBLISHED_N_X, "quote": "cv5_group_0.663_not_holdout_0.680"}
    print("\nTABLE (train group-fold CV; holdout is a check only)")
    print(f"{'spec':<14} {'mode':<11} {'n_x':>3} {'CV':>7} {'sd':>6} {'vs663':>7} {'single':>7} {'coll':>5} {'screen':>6}")
    for r in results:
        delta = r["cv_auroc"] - PUBLISHED_CV if np.isfinite(r["cv_auroc"]) else float("nan")
        print(
            f"{r['id']:<14} {r['mode']:<11} {r['n_x']:>3} {r['cv_auroc']:7.4f} "
            f"{r['cv_auroc_sd']:6.3f} {delta:+7.3f} {r['single']['cv_auroc']:7.4f} "
            f"{str(r['collapsed']):>5} {str((r.get('screen') or {}).get('ok')):>6}"
        )
    print("\nQUOTE (train group-fold CV; holdout is a check only)")
    print(json.dumps(slim, indent=2, default=str))
    return {"results": by_id, "verdict": verdict, "slim": slim, "panel_n": int(len(panel))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--specs",
        default="A,A_shallow,A_n50,B,B_shallow",
        help="comma ids or all",
    )
    p.add_argument("--no-registry", action="store_true")
    p.add_argument("--self-check", action="store_true", help="validate spec X only and exit")
    args = p.parse_args()
    if args.self_check:
        _assert_specs_clean()
        print(f"ok {len(SPECS)} specs; never D / meta / euro issued level")
        return
    run(wanted=args.specs, write_registry=not args.no_registry)


if __name__ == "__main__":
    main()
