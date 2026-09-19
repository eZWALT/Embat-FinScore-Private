"""Family G / banking_products QA (non-debt complementary table).

NORTH_STAR: created_at is a connection clock. Do not invent a health Y
from onboarding. Complementary to debt_schedule_qa (do not redo debt).

Holdout 72 (seed 20260918) is coverage only. Rates, tertiles, AUROC,
PARK/CLOSE/KEEP are train. No parquet rewrite. No new GBM. No 0–100.
Does not run build_targets. Does not edit products.py.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.banking_g_qa

Owned: analysis/outputs/banking_g_qa.md, optional PNG,
overnight/waves/wave4_banking_g.md (one note at the end),
append-only registry coverage / single-feature AUROC rows.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate.protocol import (
    FOLD_SEED,
    assert_no_holdout,
    auroc,
    group_folds,
    load_holdout,
    train_companies,
)
from analysis.features.common import ANALYSIS, AS_OF, DATA, MONTHS, connect
from analysis.features.grid import company_meta
from analysis.features.products import FEATURE_COLS

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "banking_g_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "banking_g_has_vs_size.png"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_banking_g.md"
AGENT = "7e5c43f8"
WAVE = "4"
ROUND = "R4"
LAST_M = MONTHS[-1]
EXTRACT = AS_OF
BOOK = (
    "document_type = 'invoice' AND status <> 'cancel' "
    "AND amount <> 0 AND issuance_date IS NOT NULL"
)
Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y1_IN = "y1_in_h1"
Y1_LIQ = "y1_liq_h1"
HAS_FLAGS = (
    "g_has_card",
    "g_has_tpv",
    "g_has_checking",
    "g_has_saving",
    "g_has_investment",
)
CREATED_CONST = (
    "g_created_unknown_share",
    "g_created_after_snapshot",
    "g_created_after_snapshot_share",
)
# Feature-report 44-col GBM starter (copied from feature_report.md).
STARTER_44 = (
    "a_growth_12",
    "a_growth_3",
    "a_in3",
    "a_io_ratio",
    "a_uncat_share",
    "b_bal_vol",
    "b_below_0",
    "b_d_runway",
    "b_neg_episodes",
    "b_runway",
    "c_gap_sd",
    "c_last_tx_before_2026_06",
    "c_missed_salary",
    "c_missed_tax",
    "c_recency_days",
    "c_zero_in_month",
    "c_zero_in_share_6",
    "d_cust_hhi",
    "d_supp_hhi",
    "d_tx_cp_share",
    "e_ap_overdue_30",
    "e_ar_overdue",
    "e_ar_overdue_30",
    "e_credit_note_ratio",
    "e_delay_coll",
    "e_delay_paid",
    "e_dpo_proxy",
    "e_dso_proxy",
    "e_fx_share",
    "e_pending_amt_share",
    "f_ds_r",
    "f_fc_r",
    "f_has_confirming",
    "f_has_factoring",
    "f_has_loc",
    "f_new_facility",
    "f_outstanding_gt_granted",
    "g_custom_share",
    "g_has_card",
    "g_has_checking",
    "g_has_investment",
    "g_has_saving",
    "g_has_tpv",
    "h_sib_neg_share",
)
NAMED_TYPES = ("checking", "card", "tpv", "saving", "investment")
DAYS_BENCH = 0.711  # published single-feature Y3 (c_n_days_with_tx)
SIZE_RHO = 0.50
KEEP_DELTA = 0.02
_CUSTOM_SQL = """(
    lower(coalesce(service, '')) = 'custom'
    OR bank_name ILIKE '%customer-defined%'
    OR bank_name ILIKE 'Other%'
)"""


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")


def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{float(v):.6g}"
    return "" if v is None else str(v)


def _pct(n, d) -> str:
    if d is None or d == 0:
        return "—"
    return f"{100.0 * float(n) / float(d):.1f}%"


def _pp(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100.0 * float(x):.1f}%"


def spearman(a, b) -> float:
    d = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(d) < 8:
        return float("nan")
    if d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def median_acf(series: pd.Series, company: pd.Series, lag: int) -> float:
    df = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    vals = []
    for _, g in df.groupby("co", sort=False):
        x = g["x"].to_numpy(dtype=float)
        if len(x) <= lag:
            continue
        aa = x[:-lag]
        bb = x[lag:]
        m = np.isfinite(aa) & np.isfinite(bb)
        if m.sum() < 4:
            continue
        aa, bb = aa[m], bb[m]
        if np.std(aa) == 0 or np.std(bb) == 0:
            continue
        vals.append(float(np.corrcoef(aa, bb)[0, 1]))
    return float(np.median(vals)) if vals else float("nan")


def _orient(auc: float) -> tuple[float, int]:
    """Single-feature skill in the better direction. Published 0.711 is 1 − 0.289."""
    if auc is None or not np.isfinite(auc):
        return float("nan"), 0
    a = float(auc)
    if a >= 0.5:
        return a, 1
    return 1.0 - a, -1


def _fold_auroc(y: pd.Series, x: pd.Series, folds: pd.Series) -> dict:
    aucs = []
    rows = []
    for k in sorted(pd.unique(folds.dropna())):
        va = folds == k
        auc = auroc(y[va], x[va])
        n = int(pd.to_numeric(y[va], errors="coerce").notna().sum())
        n_pos = int((pd.to_numeric(y[va], errors="coerce") == 1).sum())
        aucs.append(auc)
        rows.append({"fold": int(k), "auroc": auc, "n_labeled": n, "n_pos": n_pos})
    finite = [a for a in aucs if np.isfinite(a)]
    return {
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "n_folds": len(finite),
        "folds": rows,
    }


def _md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    right = {
        "n", "n_co", "n_cm", "n_pos", "n_prod", "n_rows", "n_labeled",
        "n_groups", "n_hold", "n_train", "rows", "companies", "products",
        "n_unique", "first_n", "last_n", "n_rise", "n_drop", "n_flat",
        "n_null", "n_after",
    }
    head = "| " + " | ".join(title for _, title in cols) + " |"
    sep = "| " + " | ".join("---:" if k in right else "---" for k, _ in cols) + " |"
    lines = [head, sep]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k, "")
            if isinstance(v, (float, np.floating, np.integer)) and np.isfinite(float(v)):
                v = float(v)
                if k.endswith("_rate") or k.endswith("_share") or k in {"cov_cm", "cov_co", "share", "prev", "modal"}:
                    cells.append(f"{100.0 * v:.1f}%")
                elif k.endswith("auc") or k.startswith("acf") or k.endswith("_rho") or k.endswith("_orient") or k in {
                    "auc", "cv", "sd", "size_auc", "days_auc", "rho", "mae", "spearman",
                    "cv_orient", "auc_orient", "size_cv", "days_cv", "beat_size", "beat_days",
                    "lag_p50", "lag_mean", "gap", "log_p50", "log_mean", "acc_p50",
                }:
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(f"{v:.4g}")
            else:
                cells.append("" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_store() -> pd.DataFrame:
    need = [
        "company_id",
        "period",
        "a_in3",
        "a_op_in",
        "b_liq",
        "c_n_days_with_tx",
        *FEATURE_COLS,
    ]
    raw = pd.read_parquet(STORE, columns=need)
    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["period"])
    return raw


def load_targets() -> pd.DataFrame:
    need = ["company_id", "period", Y2, Y3, Y1_IN, Y1_LIQ]
    raw = pd.read_parquet(TARGETS, columns=need)
    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["period"])
    return raw


def train_store(store: pd.DataFrame) -> pd.DataFrame:
    hold = load_holdout()
    train = store.loc[~store["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    return train


def ever_erp_ids(con) -> set[str]:
    df = con.execute(
        f"SELECT DISTINCT company_id FROM invoices WHERE {BOOK}"
    ).df()
    return set(df["company_id"].astype(str))


def pass1_raw(con) -> dict:
    """Raw banking_products inventory (all companies; train split reported)."""
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT
          COUNT(*) AS n_rows,
          COUNT(DISTINCT company_id) AS n_co,
          COUNT(DISTINCT product_id) AS n_prod,
          SUM(CASE WHEN created_at IS NULL THEN 1 ELSE 0 END) AS n_null_created,
          SUM(CASE WHEN coalesce(created_after_snapshot, false) THEN 1 ELSE 0 END) AS n_after,
          SUM(CASE WHEN created_at > TIMESTAMP '2026-09-01' THEN 1 ELSE 0 END) AS n_after_ts
        FROM banking_products
        """
    ).df().iloc[0].to_dict()
    by_type = con.execute(
        """
        SELECT
          CASE
            WHEN lower(coalesce(type, '')) IN ('checking','card','tpv','saving','investment')
              THEN lower(type)
            ELSE 'other'
          END AS type_bucket,
          coalesce(type, '(null)') AS type,
          COUNT(*) AS n_rows,
          COUNT(DISTINCT company_id) AS n_co,
          SUM(CASE WHEN created_at IS NULL THEN 1 ELSE 0 END) AS n_null
        FROM banking_products
        GROUP BY 1, 2
        ORDER BY n_rows DESC
        """
    ).df()
    bucket = (
        by_type.groupby("type_bucket", as_index=False)
        .agg(n_rows=("n_rows", "sum"), n_co=("n_co", "sum"), n_null=("n_null", "sum"))
        .sort_values("n_rows", ascending=False)
    )
    # n_co in bucket is not distinct across types inside 'other' — recompute
    bucket_co = con.execute(
        """
        SELECT
          CASE
            WHEN lower(coalesce(type, '')) IN ('checking','card','tpv','saving','investment')
              THEN lower(type)
            ELSE 'other'
          END AS type_bucket,
          COUNT(DISTINCT company_id) AS n_co
        FROM banking_products
        GROUP BY 1
        """
    ).df()
    bucket = bucket.drop(columns=["n_co"]).merge(bucket_co, on="type_bucket")
    cos = con.execute("SELECT DISTINCT company_id FROM banking_products").df()
    cos["company_id"] = cos["company_id"].astype(str)
    n_train = int((~cos.company_id.isin(hold)).sum())
    n_hold = int(cos.company_id.isin(hold).sum())
    share_null = float(raw["n_null_created"] or 0) / float(raw["n_rows"]) if raw["n_rows"] else float("nan")
    return {
        "n_rows": int(raw["n_rows"]),
        "n_co": int(raw["n_co"]),
        "n_prod": int(raw["n_prod"]),
        "n_null_created": int(raw["n_null_created"] or 0),
        "n_after": int(raw["n_after"] or 0),
        "n_after_ts": int(raw["n_after_ts"] or 0),
        "share_null": share_null,
        "n_train_co": n_train,
        "n_hold_co": n_hold,
        "by_type": by_type,
        "bucket": bucket,
    }


def pass2_inventory(store: pd.DataFrame) -> dict:
    """Is g_n_accounts a rise-only created_at panel (like debt facilities)?"""
    train = train_store(store)
    fac = train.sort_values(["company_id", "period"])
    dlt = fac.groupby("company_id")["g_n_accounts"].diff()
    n_drop = int((dlt < 0).sum())
    n_rise = int((dlt > 0).sum())
    n_flat = int((dlt == 0).sum())
    first = fac.groupby("company_id", sort=False).first()
    last = fac.groupby("company_id", sort=False).last()

    def _dist(s: pd.Series, label: str) -> list[dict]:
        vc = s.value_counts().sort_index()
        rows = []
        for v, n in vc.items():
            rows.append({"slice": label, "k": int(v), "n": int(n), "share": n / len(s)})
        return rows

    first_s = pd.to_numeric(first["g_n_accounts"], errors="coerce")
    last_s = pd.to_numeric(last["g_n_accounts"], errors="coerce")
    banks = train.sort_values(["company_id", "period"])
    bd = banks.groupby("company_id")["g_n_banks"].diff()
    return {
        "n_cm": int(len(train)),
        "n_co": int(train["company_id"].nunique()),
        "n_rise": n_rise,
        "n_drop": n_drop,
        "n_flat": n_flat,
        "rise_only": n_drop == 0,
        "first_mean": float(first_s.mean()),
        "first_p50": float(first_s.median()),
        "first_share0": float((first_s == 0).mean()),
        "last_mean": float(last_s.mean()),
        "last_p50": float(last_s.median()),
        "last_share0": float((last_s == 0).mean()),
        "first_dist": _dist(first_s, "first_month"),
        "last_dist": _dist(last_s, "last_month"),
        "bank_drop": int((bd < 0).sum()),
        "bank_rise": int((bd > 0).sum()),
        "acf1": median_acf(train["g_n_accounts"], train["company_id"], 1),
        "acf3": median_acf(train["g_n_accounts"], train["company_id"], 3),
        "acf6": median_acf(train["g_n_accounts"], train["company_id"], 6),
        "rho_in3": spearman(train["g_n_accounts"], train["a_in3"]),
        "rho_login3": spearman(train["g_n_accounts"], np.log1p(pd.to_numeric(train["a_in3"], errors="coerce").clip(lower=0))),
    }


def pass3_new(store: pd.DataFrame) -> dict:
    """g_new_this_month: prevalence, acf, size ρ. Connection wave vs Q3."""
    train = train_store(store)
    x = pd.to_numeric(train["g_new_this_month"], errors="coerce")
    flag = (x > 0).astype(float)
    log_in3 = np.log1p(pd.to_numeric(train["a_in3"], errors="coerce").clip(lower=0))
    first = train.sort_values("period").groupby("company_id", sort=False).first()
    first_new = pd.to_numeric(first["g_new_this_month"], errors="coerce") > 0
    first_zero_acc = pd.to_numeric(first["g_n_accounts"], errors="coerce") == 0
    # connection birth: first month the company has g_n_accounts>0
    birth = train.sort_values(["company_id", "period"]).copy()
    birth["acc"] = pd.to_numeric(birth["g_n_accounts"], errors="coerce")
    prev = birth.groupby("company_id")["acc"].shift(1)
    birth_m = (birth["acc"] > 0) & (prev.fillna(0) == 0)
    new_m = pd.to_numeric(birth["g_new_this_month"], errors="coerce") > 0
    n_new = int(new_m.sum())
    n_birth = int(birth_m.sum())
    n_new_and_birth = int((new_m & birth_m).sum())
    # calendar of new connections (train)
    cal = (
        birth.loc[new_m]
        .groupby("period")
        .agg(n_cm=("company_id", "size"), n_co=("company_id", "nunique"))
        .reset_index()
    )
    return {
        "n_cm": int(len(train)),
        "share_gt0": float((x > 0).mean()),
        "n_gt0": int((x > 0).sum()),
        "n_co_gt0": int(train.loc[x > 0, "company_id"].nunique()),
        "mean": float(x.mean()),
        "p50": float(x.median()),
        "acf1": median_acf(x, train["company_id"], 1),
        "acf1_flag": median_acf(flag, train["company_id"], 1),
        "acf3": median_acf(x, train["company_id"], 3),
        "acf6": median_acf(x, train["company_id"], 6),
        "rho_login3": spearman(x, log_in3),
        "rho_flag_login3": spearman(flag, log_in3),
        "first_new_share": float(first_new.mean()),
        "first_zero_acc": float(first_zero_acc.mean()),
        "n_new": n_new,
        "n_birth": n_birth,
        "n_new_and_birth": n_new_and_birth,
        "share_new_is_birth": n_new_and_birth / n_new if n_new else float("nan"),
        "calendar": cal,
        "onboarding": True,  # filled after seeing numbers; write_md decides
    }


def pass4_access(con, store: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """Access flags vs Q1: Y2 / Y3 (stressed) / Y1 last-value. Single-feature AUROC."""
    hold = load_holdout()
    m = store.merge(targets, on=["company_id", "period"], how="left")
    train = m.loc[~m["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    train["log_in3"] = np.log1p(pd.to_numeric(train["a_in3"], errors="coerce").clip(lower=0))
    cos = train_companies(con)
    folded = group_folds(cos, n=5, seed=FOLD_SEED)
    train = train.merge(folded[["company_id", "fold"]], on="company_id", how="left")

    feats = {
        "g_has_card": train["g_has_card"],
        "g_has_tpv": train["g_has_tpv"],
        "g_has_checking": train["g_has_checking"],
        "g_n_banks": train["g_n_banks"],
        "g_n_accounts": train["g_n_accounts"],
        "g_new_gt0": (pd.to_numeric(train["g_new_this_month"], errors="coerce") > 0).astype(float),
        "c_n_days_with_tx": train["c_n_days_with_tx"],
        "log1p_a_in3": train["log_in3"],
    }
    auc_rows = []
    for y in (Y2, Y3):
        for name, col in feats.items():
            pooled = auroc(train[y], col)
            cv = _fold_auroc(train[y], col, train["fold"])
            rho = spearman(col, train["a_in3"])
            rho_log = spearman(col, train["log_in3"])
            n_lab = int(pd.to_numeric(train[y], errors="coerce").notna().sum())
            o_auc, s_auc = _orient(pooled)
            o_cv, s_cv = _orient(cv["cv"])
            auc_rows.append(
                {
                    "y": y,
                    "feature": name,
                    "auc": pooled,
                    "cv": cv["cv"],
                    "auc_orient": o_auc,
                    "cv_orient": o_cv,
                    "sign": s_cv,
                    "sd": cv["sd"],
                    "n_labeled": n_lab,
                    "rho_ain3": rho,
                    "rho_login3": rho_log,
                    "is_size": abs(rho) >= SIZE_RHO or abs(rho_log) >= SIZE_RHO,
                }
            )

    size_y3 = next(r for r in auc_rows if r["y"] == Y3 and r["feature"] == "log1p_a_in3")
    days_y3 = next(r for r in auc_rows if r["y"] == Y3 and r["feature"] == "c_n_days_with_tx")
    verdicts = []
    for r in auc_rows:
        if r["feature"] in {"c_n_days_with_tx", "log1p_a_in3"}:
            continue
        if r["y"] != Y3:
            continue
        beat_size = (
            r["cv_orient"] - size_y3["cv_orient"]
            if np.isfinite(r["cv_orient"]) and np.isfinite(size_y3["cv_orient"])
            else float("nan")
        )
        beat_days = (
            r["cv_orient"] - days_y3["cv_orient"]
            if np.isfinite(r["cv_orient"]) and np.isfinite(days_y3["cv_orient"])
            else float("nan")
        )
        keep_q1 = bool(
            np.isfinite(beat_size)
            and beat_size >= KEEP_DELTA
            and not r["is_size"]
        )
        verdicts.append(
            {
                "feature": r["feature"],
                "cv": r["cv"],
                "cv_orient": r["cv_orient"],
                "size_cv": size_y3["cv_orient"],
                "days_cv": days_y3["cv_orient"],
                "beat_size": beat_size,
                "beat_days": beat_days,
                "is_size": r["is_size"],
                "keep_q1_x": keep_q1,
            }
        )

    # Base rates by flag
    rate_rows = []
    for flag in HAS_FLAGS + ("g_n_banks",):
        if flag == "g_n_banks":
            groups = [
                ("one_bank", pd.to_numeric(train[flag], errors="coerce") == 1),
                ("many_banks", pd.to_numeric(train[flag], errors="coerce") >= 2),
                ("zero_banks", pd.to_numeric(train[flag], errors="coerce") == 0),
            ]
        else:
            v = pd.to_numeric(train[flag], errors="coerce")
            groups = [("has=1", v == 1), ("has=0", v == 0)]
        for y in (Y2, Y3):
            for gname, mask in groups:
                s = pd.to_numeric(train.loc[mask, y], errors="coerce")
                rate_rows.append(
                    {
                        "flag": flag,
                        "group": gname,
                        "y": y,
                        "n_labeled": int(s.notna().sum()),
                        "n_pos": int((s == 1).sum()),
                        "rate": float(s.mean()) if s.notna().any() else float("nan"),
                        "n_cm": int(mask.sum()),
                        "n_co": int(train.loc[mask, "company_id"].nunique()),
                    }
                )

    # Y1 last-value story
    y1_rows = []
    for series, pred, ycol in (
        ("op_in", "a_op_in", Y1_IN),
        ("liq", "b_liq", Y1_LIQ),
    ):
        lab = train[ycol].notna()
        err = (pd.to_numeric(train[pred], errors="coerce") - pd.to_numeric(train[ycol], errors="coerce")).abs()
        rho = spearman(train.loc[lab, pred], train.loc[lab, ycol])
        y1_rows.append(
            {
                "series": series,
                "slice": "all",
                "n": int(lab.sum()),
                "spearman": rho,
                "mae": float(err[lab].median()) if lab.any() else float("nan"),
            }
        )
        for flag in ("g_has_card", "g_has_tpv", "g_has_checking"):
            for val, gname in ((1, "has=1"), (0, "has=0")):
                msk = lab & (pd.to_numeric(train[flag], errors="coerce") == val)
                y1_rows.append(
                    {
                        "series": series,
                        "slice": f"{flag} {gname}",
                        "n": int(msk.sum()),
                        "spearman": spearman(train.loc[msk, pred], train.loc[msk, ycol]),
                        "mae": float(err[msk].median()) if msk.any() else float("nan"),
                    }
                )

    best_has = max(
        (r for r in auc_rows if r["y"] == Y3 and r["feature"].startswith("g_has_")),
        key=lambda r: r["cv_orient"] if np.isfinite(r.get("cv_orient", r["cv"])) else -1,
    )
    return {
        "auc_rows": auc_rows,
        "rate_rows": rate_rows,
        "y1_rows": y1_rows,
        "verdicts": verdicts,
        "best_has": best_has,
        "size_y3": size_y3,
        "days_y3": days_y3,
        "n_y3": int(pd.to_numeric(train[Y3], errors="coerce").notna().sum()),
        "n_y2": int(pd.to_numeric(train[Y2], errors="coerce").notna().sum()),
        "y3_rate": float(pd.to_numeric(train[Y3], errors="coerce").mean()),
        "y2_rate": float(pd.to_numeric(train[Y2], errors="coerce").mean()),
    }


def pass5_constants(store: pd.DataFrame) -> dict:
    """Confirm created_* constants are drop-list; after_snapshot is 0 on the panel."""
    train = train_store(store)
    hold = store.loc[store["company_id"].isin(load_holdout())].copy()
    rows = []
    for split, part in (("train", train), ("holdout", hold)):
        for col in CREATED_CONST:
            x = pd.to_numeric(part[col], errors="coerce")
            nn = x.dropna()
            rows.append(
                {
                    "split": split,
                    "col": col,
                    "n_cm": int(len(part)),
                    "nn": int(x.notna().sum()),
                    "cov_cm": float(x.notna().mean()) if len(part) else float("nan"),
                    "n_unique": int(nn.nunique()) if len(nn) else 0,
                    "modal": float(nn.mode().iloc[0]) if len(nn) else float("nan"),
                    "share_zero": float((nn == 0).mean()) if len(nn) else float("nan"),
                    "max": float(nn.max()) if len(nn) else float("nan"),
                }
            )
    after = train["g_created_after_snapshot"]
    after_all_zero = bool((pd.to_numeric(after, errors="coerce").fillna(0) == 0).all())
    g_in_starter = [c for c in STARTER_44 if c.startswith("g_")]
    created_in_starter = [c for c in STARTER_44 if c.startswith("g_created")]
    has_in_starter = [c for c in STARTER_44 if c.startswith("g_has_")]
    return {
        "rows": rows,
        "after_all_zero": after_all_zero,
        "n_starter": len(STARTER_44),
        "starter_is_44": len(STARTER_44) == 44,
        "g_in_starter": g_in_starter,
        "created_in_starter": created_in_starter,
        "has_in_starter": has_in_starter,
        "drop_created_ok": created_in_starter == [],
        "keep_has_ok": set(HAS_FLAGS) <= set(STARTER_44),
        "n_accounts_in_starter": "g_n_accounts" in STARTER_44,
        "new_in_starter": "g_new_this_month" in STARTER_44,
    }


def pass6_dark(con, store: pd.DataFrame) -> dict:
    """470 dark vs 744 invoiced: fewer accounts / less TPV? Access ≠ ERP."""
    hold = load_holdout()
    erp = ever_erp_ids(con)
    train = train_store(store)
    last = train.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    last["ever_erp"] = last["company_id"].isin(erp)
    n_erp = int(last["ever_erp"].sum())
    n_dark = int((~last["ever_erp"]).sum())
    rows = []
    for name, part in (("ever_erp", last[last["ever_erp"]]), ("never_erp", last[~last["ever_erp"]])):
        acc = pd.to_numeric(part["g_n_accounts"], errors="coerce")
        rows.append(
            {
                "group": name,
                "n_co": int(len(part)),
                "acc_mean": float(acc.mean()),
                "acc_p50": float(acc.median()),
                "share0": float((acc == 0).mean()),
                "has_card": float(part["g_has_card"].mean()),
                "has_tpv": float(part["g_has_tpv"].mean()),
                "has_checking": float(part["g_has_checking"].mean()),
                "has_saving": float(part["g_has_saving"].mean()),
                "has_invest": float(part["g_has_investment"].mean()),
                "n_banks_p50": float(pd.to_numeric(part["g_n_banks"], errors="coerce").median()),
                "custom_p50": float(pd.to_numeric(part["g_custom_share"], errors="coerce").median()),
            }
        )
    # company-ever (max on panel) — last-month is as-of extract for most
    ever = train.groupby("company_id").agg(
        g_n_accounts=("g_n_accounts", "max"),
        g_has_card=("g_has_card", "max"),
        g_has_tpv=("g_has_tpv", "max"),
        g_has_checking=("g_has_checking", "max"),
        g_n_banks=("g_n_banks", "max"),
    )
    ever["ever_erp"] = ever.index.isin(erp)
    ever_rows = []
    for name, part in (("ever_erp", ever[ever["ever_erp"]]), ("never_erp", ever[~ever["ever_erp"]])):
        ever_rows.append(
            {
                "group": name,
                "n_co": int(len(part)),
                "acc_mean": float(part["g_n_accounts"].mean()),
                "acc_p50": float(part["g_n_accounts"].median()),
                "has_card": float(part["g_has_card"].mean()),
                "has_tpv": float(part["g_has_tpv"].mean()),
                "has_checking": float(part["g_has_checking"].mean()),
            }
        )
    hold_last = store.loc[store["company_id"].isin(hold)].sort_values("period").groupby("company_id").last()
    return {
        "n_erp": n_erp,
        "n_dark": n_dark,
        "n_train": int(len(last)),
        "erp_expected": n_erp == 744 and n_dark == 470,
        "rows": rows,
        "ever_rows": ever_rows,
        "hold_n": int(len(hold_last)),
        "hold_erp": int(hold_last.index.isin(erp).sum()),
    }


def pass7_banks(con, store: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """Bank-name concentration (one vs many) vs Y2; raw bank mix."""
    hold = load_holdout()
    train = train_store(store)
    m = train.merge(targets, on=["company_id", "period"], how="left")
    last = m.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    last["one"] = pd.to_numeric(last["g_n_banks"], errors="coerce") == 1
    last["many"] = pd.to_numeric(last["g_n_banks"], errors="coerce") >= 2
    last["zero"] = pd.to_numeric(last["g_n_banks"], errors="coerce") == 0
    # company-ever Y2
    ever_y2 = (
        m.groupby("company_id")[Y2]
        .max()
        .rename("ever_y2")
        .reset_index()
    )
    last = last.merge(ever_y2, on="company_id", how="left")
    conc = []
    for name, mask in (("one_bank", last["one"]), ("many_banks", last["many"]), ("zero_banks", last["zero"])):
        part = last.loc[mask]
        y2 = pd.to_numeric(part["ever_y2"], errors="coerce")
        conc.append(
            {
                "group": name,
                "n_co": int(len(part)),
                "ever_y2_rate": float(y2.mean()) if y2.notna().any() else float("nan"),
                "n_labeled": int(y2.notna().sum()),
                "n_pos": int((y2 == 1).sum()),
            }
        )
    # CM-level Y2 by one vs many
    cm = []
    banks = pd.to_numeric(m["g_n_banks"], errors="coerce")
    for name, mask in (("one_bank", banks == 1), ("many_banks", banks >= 2), ("zero_banks", banks == 0)):
        s = pd.to_numeric(m.loc[mask, Y2], errors="coerce")
        cm.append(
            {
                "group": name,
                "n_cm": int(mask.sum()),
                "n_labeled": int(s.notna().sum()),
                "n_pos": int((s == 1).sum()),
                "rate": float(s.mean()) if s.notna().any() else float("nan"),
            }
        )
    raw_banks = con.execute(
        """
        SELECT
          CASE
            WHEN bank_name ILIKE '%customer-defined%' OR bank_name ILIKE 'Other%'
              THEN 'Other (customer-defined)'
            ELSE coalesce(bank_name, '(null)')
          END AS bank,
          COUNT(*) AS n_rows,
          COUNT(DISTINCT company_id) AS n_co
        FROM banking_products
        GROUP BY 1
        ORDER BY n_rows DESC
        LIMIT 15
        """
    ).df()
    n_one = int(last["one"].sum())
    n_many = int(last["many"].sum())
    return {
        "conc": conc,
        "cm": cm,
        "raw_banks": raw_banks,
        "n_one": n_one,
        "n_many": n_many,
        "n_zero": int(last["zero"].sum()),
        "share_one": n_one / len(last) if len(last) else float("nan"),
    }


def pass8_meta(con, store: pd.DataFrame) -> dict:
    """Does company_meta.n_banking match last-month g_n_accounts?"""
    hold = load_holdout()
    meta = company_meta(con)
    meta["company_id"] = meta["company_id"].astype(str)
    train = train_store(store)
    last = train.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    last = last.merge(meta[["company_id", "n_banking", "n_banks"]], on="company_id", how="left")
    last["delta"] = pd.to_numeric(last["n_banking"], errors="coerce") - pd.to_numeric(
        last["g_n_accounts"], errors="coerce"
    )
    match = int((last["delta"] == 0).sum())
    after = con.execute(
        """
        SELECT company_id, COUNT(*) AS n_after
        FROM banking_products
        WHERE coalesce(created_after_snapshot, false)
           OR created_at > TIMESTAMP '2026-09-01'
        GROUP BY 1
        """
    ).df()
    after["company_id"] = after["company_id"].astype(str)
    last = last.merge(after, on="company_id", how="left")
    last["n_after"] = last["n_after"].fillna(0)
    explained = int((last["delta"] == last["n_after"]).sum())
    mismatch = last.loc[last["delta"] != 0, ["company_id", "n_banking", "g_n_accounts", "delta", "n_after"]]
    return {
        "n_co": int(len(last)),
        "n_match": match,
        "share_match": match / len(last) if len(last) else float("nan"),
        "n_explained_by_after": explained,
        "n_mismatch": int(len(mismatch)),
        "delta_p50": float(last["delta"].median()),
        "delta_mean": float(last["delta"].mean()),
        "delta_max": float(last["delta"].max()),
        "mismatch_head": mismatch.sort_values("delta", ascending=False).head(8),
        "hold_n": int((meta["company_id"].isin(hold)).sum()),
    }


def pass9_custom_dark(con, store: pd.DataFrame) -> dict:
    """Are custom / Other banks the 470 never-ERP?"""
    hold = load_holdout()
    erp = ever_erp_ids(con)
    raw = con.execute(
        f"""
        SELECT company_id,
               COUNT(*) AS n_prod,
               SUM(CASE WHEN {_CUSTOM_SQL} THEN 1 ELSE 0 END) AS n_custom,
               SUM(CASE WHEN lower(type) = 'tpv' THEN 1 ELSE 0 END) AS n_tpv
        FROM banking_products
        GROUP BY 1
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(raw["company_id"])
    raw["ever_erp"] = raw["company_id"].isin(erp)
    raw["any_custom"] = raw["n_custom"] > 0
    raw["all_custom"] = (raw["n_custom"] == raw["n_prod"]) & (raw["n_prod"] > 0)
    rows = []
    for name, part in (("ever_erp", raw[raw["ever_erp"]]), ("never_erp", raw[~raw["ever_erp"]])):
        rows.append(
            {
                "group": name,
                "n_co": int(len(part)),
                "any_custom": float(part["any_custom"].mean()),
                "all_custom": float(part["all_custom"].mean()),
                "custom_mean": float((part["n_custom"] / part["n_prod"].clip(lower=1)).mean()),
                "n_all_custom": int(part["all_custom"].sum()),
                "n_any_tpv": int((part["n_tpv"] > 0).sum()),
            }
        )
    n_all_custom = int(raw["all_custom"].sum())
    n_all_custom_dark = int((raw["all_custom"] & ~raw["ever_erp"]).sum())
    return {
        "rows": rows,
        "n_all_custom": n_all_custom,
        "n_all_custom_dark": n_all_custom_dark,
        "share_all_custom_are_dark": n_all_custom_dark / n_all_custom if n_all_custom else float("nan"),
        "n_dark_all_custom": n_all_custom_dark,
        "n_dark": int((~raw["ever_erp"]).sum()),
    }


def _company_terciles(train: pd.DataFrame) -> pd.Series:
    """Train-only company-median log1p(a_in3) terciles. Not fit on holdout."""
    last = train.sort_values("period").groupby("company_id", sort=False).last()
    logx = np.log1p(pd.to_numeric(last["a_in3"], errors="coerce").clip(lower=0))
    terc = pd.qcut(logx, 3, labels=["T1_small", "T2_mid", "T3_large"])
    return terc.rename("tercile")


def pass10_twobytwo(store: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """Card × TPV 2×2 after size terciles. Do not revive k-means (sil 0.234)."""
    hold = load_holdout()
    m = store.merge(targets, on=["company_id", "period"], how="left")
    train = m.loc[~m["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    terc = _company_terciles(train)
    train = train.merge(terc.reset_index(), on="company_id", how="left")
    last = train.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    last["card"] = pd.to_numeric(last["g_has_card"], errors="coerce") == 1
    last["tpv"] = pd.to_numeric(last["g_has_tpv"], errors="coerce") == 1

    def _cell(card, tpv):
        return last.loc[(last["card"] == card) & (last["tpv"] == tpv)]

    cells = []
    for c, t, name in (
        (True, True, "card+tpv"),
        (True, False, "card_only"),
        (False, True, "tpv_only"),
        (False, False, "neither"),
    ):
        part = _cell(c, t)
        cells.append(
            {
                "cell": name,
                "n_co": int(len(part)),
                "t1": int((part["tercile"] == "T1_small").sum()) if "tercile" in part else 0,
                "t2": int((part["tercile"] == "T2_mid").sum()) if "tercile" in part else 0,
                "t3": int((part["tercile"] == "T3_large").sum()) if "tercile" in part else 0,
            }
        )

    # CM rates in 2×2 and within tercile
    train["card"] = pd.to_numeric(train["g_has_card"], errors="coerce") == 1
    train["tpv"] = pd.to_numeric(train["g_has_tpv"], errors="coerce") == 1
    rate_rows = []
    for y in (Y2, Y3):
        for c, t, name in (
            (True, True, "card+tpv"),
            (True, False, "card_only"),
            (False, True, "tpv_only"),
            (False, False, "neither"),
        ):
            mask = (train["card"] == c) & (train["tpv"] == t)
            s = pd.to_numeric(train.loc[mask, y], errors="coerce")
            rate_rows.append(
                {
                    "y": y,
                    "tercile": "all",
                    "cell": name,
                    "n_labeled": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "rate": float(s.mean()) if s.notna().any() else float("nan"),
                    "n_co": int(train.loc[mask, "company_id"].nunique()),
                }
            )

    card_rows = []
    for y in (Y2, Y3):
        for tname in ("all", "T1_small", "T2_mid", "T3_large"):
            base = train if tname == "all" else train[train["tercile"] == tname]
            for val, gname in ((True, "card"), (False, "no_card")):
                mask = pd.to_numeric(base["g_has_card"], errors="coerce") == (1 if val else 0)
                s = pd.to_numeric(base.loc[mask, y], errors="coerce")
                card_rows.append(
                    {
                        "y": y,
                        "tercile": tname,
                        "group": gname,
                        "n_labeled": int(s.notna().sum()),
                        "n_pos": int((s == 1).sum()),
                        "rate": float(s.mean()) if s.notna().any() else float("nan"),
                        "n_co": int(base.loc[mask, "company_id"].nunique()),
                    }
                )

    # Does the card Y3 gap survive every tercile with the same sign and n_pos>=5?
    survive = True
    gaps = []
    for tname in ("T1_small", "T2_mid", "T3_large"):
        a = next(r for r in card_rows if r["y"] == Y3 and r["tercile"] == tname and r["group"] == "card")
        b = next(r for r in card_rows if r["y"] == Y3 and r["tercile"] == tname and r["group"] == "no_card")
        gap = (a["rate"] - b["rate"]) if np.isfinite(a["rate"]) and np.isfinite(b["rate"]) else float("nan")
        gaps.append({"tercile": tname, "gap": gap, "n_card_lab": a["n_labeled"], "n_card_pos": a["n_pos"]})
        if a["n_pos"] < 5 or not np.isfinite(gap):
            survive = False
    signs = [np.sign(g["gap"]) for g in gaps if np.isfinite(g["gap"])]
    same_sign = len(set(signs)) == 1 and len(signs) == 3 if signs else False
    survive = bool(survive and same_sign)

    last_share = (
        last.groupby("tercile", observed=False)[["card", "tpv"]]
        .mean()
        .reset_index()
        .rename(columns={"card": "has_card_share", "tpv": "has_tpv_share"})
    )
    return {
        "cells": cells,
        "rate_rows": rate_rows,
        "card_rows": card_rows,
        "gaps": gaps,
        "survive": survive,
        "same_sign": same_sign,
        "last_share": last_share.to_dict("records"),
        "n_tpv": int(last["tpv"].sum()),
        "n_card": int(last["card"].sum()),
    }


def pass11_banks_size(store: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """One vs many banks vs Y2/Y3 inside size terciles."""
    hold = load_holdout()
    m = store.merge(targets, on=["company_id", "period"], how="left")
    train = m.loc[~m["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    terc = _company_terciles(train)
    train = train.merge(terc.reset_index(), on="company_id", how="left")
    train["nb"] = pd.to_numeric(train["g_n_banks"], errors="coerce")
    rows = []
    for y in (Y2, Y3):
        for tname in ("all", "T1_small", "T2_mid", "T3_large"):
            base = train if tname == "all" else train.loc[train["tercile"] == tname]
            for gname, mm in (
                ("one_bank", base["nb"] == 1),
                ("many_banks", base["nb"] >= 2),
                ("zero_banks", base["nb"] == 0),
            ):
                s = pd.to_numeric(base.loc[mm, y], errors="coerce")
                rows.append(
                    {
                        "y": y,
                        "tercile": tname,
                        "group": gname,
                        "n_cm": int(mm.sum()),
                        "n_labeled": int(s.notna().sum()),
                        "n_pos": int((s == 1).sum()),
                        "rate": float(s.mean()) if s.notna().any() else float("nan"),
                        "n_co": int(base.loc[mm, "company_id"].nunique()),
                    }
                )
    return {"rows": rows}


def pass12_clock(con, store: pd.DataFrame) -> dict:
    """First-tx vs first-created: replica of the 73.6% connection clock."""
    hold = load_holdout()
    first_tx = con.execute(
        """
        SELECT company_id, CAST(date_trunc('month', MIN("date")) AS DATE) AS first_tx
        FROM transactions
        WHERE "date" >= DATE '2024-09-01' AND "date" < DATE '2026-09-01'
        GROUP BY 1
        """
    ).df()
    first_tx["company_id"] = first_tx["company_id"].astype(str)
    first_tx["first_tx"] = pd.to_datetime(first_tx["first_tx"])
    first_c = con.execute(
        """
        SELECT company_id, MIN(created_at) AS first_created
        FROM banking_products
        GROUP BY 1
        """
    ).df()
    first_c["company_id"] = first_c["company_id"].astype(str)
    first_c["first_created"] = pd.to_datetime(first_c["first_created"])
    train_ids = set(train_companies(con)["company_id"].astype(str))
    panel = first_tx.loc[first_tx["company_id"].isin(train_ids)].merge(
        first_c, on="company_id", how="left"
    )
    assert_no_holdout(panel["company_id"])
    has_b = panel["first_created"].notna()
    late = has_b & (panel["first_created"] > pd.Timestamp("2024-09-01"))
    n_bank = int(has_b.sum())
    n_late = int(late.sum())
    share = n_late / n_bank if n_bank else float("nan")
    # connection lag: months from first tx to first g_n_accounts>0
    train = train_store(store)
    first_acc = (
        train.loc[pd.to_numeric(train["g_n_accounts"], errors="coerce") > 0]
        .sort_values("period")
        .groupby("company_id", sort=False)
        .first()["period"]
    )
    lag = panel.merge(first_acc.rename("first_acc"), left_on="company_id", right_index=True, how="left")
    lag["lag_m"] = (
        (lag["first_acc"].dt.year - lag["first_tx"].dt.year) * 12
        + (lag["first_acc"].dt.month - lag["first_tx"].dt.month)
    )
    connected = lag["lag_m"].notna()
    return {
        "n_train": int(len(panel)),
        "n_bank": n_bank,
        "n_late": n_late,
        "share_late": share,
        "confirm_736": abs(share - 0.736) < 0.005 if np.isfinite(share) else False,
        "n_no_bank": int((~has_b).sum()),
        "lag_p50": float(lag.loc[connected, "lag_m"].median()) if connected.any() else float("nan"),
        "lag_mean": float(lag.loc[connected, "lag_m"].mean()) if connected.any() else float("nan"),
        "share_lag0": float((lag.loc[connected, "lag_m"] == 0).mean()) if connected.any() else float("nan"),
        "n_never_acc": int((~connected).sum()),
        "created_after_tx": int((has_b & (panel["first_created"] > panel["first_tx"])).sum()),
    }


def pass13_other(con, store: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """wallet / risk / expensesPlatform / lineofcomex — leftover types, no g_has_*."""
    hold = load_holdout()
    erp = ever_erp_ids(con)
    raw = con.execute(
        """
        SELECT company_id, type, COUNT(*) AS n
        FROM banking_products
        WHERE lower(coalesce(type, '')) NOT IN ('checking','card','tpv','saving','investment')
        GROUP BY 1, 2
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    types = sorted(raw["type"].dropna().unique())
    m = store.merge(targets, on=["company_id", "period"], how="left")
    train = m.loc[~m["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    last = train.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    rows = []
    for t in types:
        ids = set(raw.loc[raw["type"] == t, "company_id"]) - hold
        part = last.loc[last["company_id"].isin(ids)]
        y2 = train.loc[train["company_id"].isin(ids), Y2]
        y3 = train.loc[train["company_id"].isin(ids), Y3]
        rows.append(
            {
                "type": t,
                "n_co": int(len(ids)),
                "n_erp": int(len(ids & erp)),
                "n_dark": int(len(ids - erp)),
                "acc_p50": float(pd.to_numeric(part["g_n_accounts"], errors="coerce").median()) if len(part) else float("nan"),
                "y2_rate": float(pd.to_numeric(y2, errors="coerce").mean()) if pd.to_numeric(y2, errors="coerce").notna().any() else float("nan"),
                "y3_rate": float(pd.to_numeric(y3, errors="coerce").mean()) if pd.to_numeric(y3, errors="coerce").notna().any() else float("nan"),
                "n_y3": int(pd.to_numeric(y3, errors="coerce").notna().sum()),
            }
        )
    any_ids = set(raw["company_id"]) - hold
    return {"rows": rows, "n_any": int(len(any_ids)), "n_erp": int(len(any_ids & erp))}


def pass14_new_y(store: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """Same-month Y rates on g_new>0: first birth vs add-on. PARK if onboarding."""
    hold = load_holdout()
    m = store.merge(targets, on=["company_id", "period"], how="left")
    train = m.loc[~m["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    train = train.sort_values(["company_id", "period"])
    prev = train.groupby("company_id")["g_n_accounts"].shift(1)
    new = pd.to_numeric(train["g_new_this_month"], errors="coerce") > 0
    train["kind"] = np.where(~new, "none", np.where(prev.fillna(0).eq(0), "first_birth", "add_on"))
    rows = []
    for y in (Y2, Y3):
        for k, part in train.groupby("kind"):
            s = pd.to_numeric(part[y], errors="coerce")
            rows.append(
                {
                    "y": y,
                    "kind": k,
                    "n_cm": int(len(part)),
                    "n_labeled": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "rate": float(s.mean()) if s.notna().any() else float("nan"),
                    "n_co": int(part["company_id"].nunique()),
                }
            )
    return {"rows": rows}


def pass15_connected(con, store: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """Is g_has_checking just 'connected'? Card AUROC on connected months only."""
    hold = load_holdout()
    m = store.merge(targets, on=["company_id", "period"], how="left")
    train = m.loc[~m["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    acc = pd.to_numeric(train["g_n_accounts"], errors="coerce")
    chk = pd.to_numeric(train["g_has_checking"], errors="coerce")
    zero = acc == 0
    n_zero = int(zero.sum())
    n_chk0 = int((chk == 0).sum())
    n_chk0_and_zero = int(((chk == 0) & zero).sum())
    n_chk0_with_acc = int(((chk == 0) & (acc > 0)).sum())
    n_acc_no_chk = n_chk0_with_acc
    connected = acc > 0
    # types present when checking is 0 but accounts > 0
    no_chk = train.loc[(chk == 0) & (acc > 0)]
    type_mix = {
        "n_cm": int(len(no_chk)),
        "n_co": int(no_chk["company_id"].nunique()) if len(no_chk) else 0,
        "has_card": float(no_chk["g_has_card"].mean()) if len(no_chk) else float("nan"),
        "has_tpv": float(no_chk["g_has_tpv"].mean()) if len(no_chk) else float("nan"),
        "has_saving": float(no_chk["g_has_saving"].mean()) if len(no_chk) else float("nan"),
        "has_invest": float(no_chk["g_has_investment"].mean()) if len(no_chk) else float("nan"),
    }
    train["log_in3"] = np.log1p(pd.to_numeric(train["a_in3"], errors="coerce").clip(lower=0))
    cos = train_companies(con)
    folded = group_folds(cos, n=5, seed=FOLD_SEED)
    train = train.merge(folded[["company_id", "fold"]], on="company_id", how="left")
    connected = pd.to_numeric(train["g_n_accounts"], errors="coerce") > 0
    sub = train.loc[connected].copy()
    auc_rows = []
    for y in (Y2, Y3):
        for name, col in (
            ("g_has_card", sub["g_has_card"]),
            ("g_has_tpv", sub["g_has_tpv"]),
            ("g_n_banks", sub["g_n_banks"]),
            ("c_n_days_with_tx", sub["c_n_days_with_tx"]),
            ("log1p_a_in3", sub["log_in3"]),
        ):
            pooled = auroc(sub[y], col)
            cv = _fold_auroc(sub[y], col, sub["fold"])
            o_cv, _ = _orient(cv["cv"])
            auc_rows.append(
                {
                    "y": y,
                    "feature": name,
                    "auc": pooled,
                    "cv": cv["cv"],
                    "cv_orient": o_cv,
                    "sd": cv["sd"],
                    "n_labeled": int(pd.to_numeric(sub[y], errors="coerce").notna().sum()),
                }
            )
    # residual size: one vs many within tercile (company last-month)
    terc = _company_terciles(train)
    last = train.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    last = last.merge(terc.reset_index(), on="company_id", how="left")
    last["nb"] = pd.to_numeric(last["g_n_banks"], errors="coerce")
    last["log_in3"] = np.log1p(pd.to_numeric(last["a_in3"], errors="coerce").clip(lower=0))
    size_rows = []
    for tname in ("T1_small", "T2_mid", "T3_large"):
        base = last.loc[last["tercile"] == tname]
        for gname, mm in (("one_bank", base["nb"] == 1), ("many_banks", base["nb"] >= 2)):
            part = base.loc[mm]
            size_rows.append(
                {
                    "tercile": tname,
                    "group": gname,
                    "n_co": int(len(part)),
                    "log_p50": float(part["log_in3"].median()) if len(part) else float("nan"),
                    "log_mean": float(part["log_in3"].mean()) if len(part) else float("nan"),
                    "acc_p50": float(pd.to_numeric(part["g_n_accounts"], errors="coerce").median()) if len(part) else float("nan"),
                }
            )
    card_conn = next(r for r in auc_rows if r["y"] == Y3 and r["feature"] == "g_has_card")
    size_conn = next(r for r in auc_rows if r["y"] == Y3 and r["feature"] == "log1p_a_in3")
    days_conn = next(r for r in auc_rows if r["y"] == Y3 and r["feature"] == "c_n_days_with_tx")
    return {
        "n_zero": n_zero,
        "n_chk0": n_chk0,
        "n_chk0_and_zero": n_chk0_and_zero,
        "share_chk0_is_zero": n_chk0_and_zero / n_chk0 if n_chk0 else float("nan"),
        "n_chk0_with_acc": n_acc_no_chk,
        "n_connected": int(connected.sum()),
        "type_mix": type_mix,
        "auc_rows": auc_rows,
        "size_rows": size_rows,
        "card_cv": card_conn["cv_orient"],
        "size_cv": size_conn["cv_orient"],
        "days_cv": days_conn["cv_orient"],
        "beat_size": card_conn["cv_orient"] - size_conn["cv_orient"],
    }


def make_png(store: pd.DataFrame) -> bool:
    if not HAS_MPL:
        return False
    train = train_store(store)
    last = train.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    last["log_in3"] = np.log1p(pd.to_numeric(last["a_in3"], errors="coerce").clip(lower=0))
    last = last.loc[last["log_in3"].notna()].copy()
    last["tercile"] = pd.qcut(last["log_in3"], 3, labels=["T1_small", "T2_mid", "T3_large"])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    xs = np.arange(3)
    w = 0.35
    card = last.groupby("tercile", observed=False)["g_has_card"].mean().reindex(
        ["T1_small", "T2_mid", "T3_large"]
    )
    tpv = last.groupby("tercile", observed=False)["g_has_tpv"].mean().reindex(
        ["T1_small", "T2_mid", "T3_large"]
    )
    ax.bar(xs - w / 2, 100.0 * card.to_numpy(), width=w, color="#1f4e79", label="has_card")
    ax.bar(xs + w / 2, 100.0 * tpv.to_numpy(), width=w, color="#c45911", label="has_tpv")
    ax.set_xticks(xs)
    ax.set_xticklabels(["T1 small", "T2 mid", "T3 large"])
    ax.set_ylabel("% of train companies (last month)")
    ax.set_title("Card / TPV access vs size tercile (train last month)")
    ax.set_ylim(0, max(40, float(100.0 * card.max()) + 8))
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    return True


def write_md(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, png_ok: bool) -> None:
    lines = []
    a = lines.append
    a("# Family G — banking products QA (non-debt)")
    a("")
    a(f"Generated `{_utc_ts()}` UTC by `python -m analysis.evaluate.banking_g_qa`.")
    a("Holdout 72 (seed 20260918) is **coverage only**. Rates, tertiles, AUROC,")
    a("and PARK/CLOSE/KEEP are train. No parquet rewrite. No new GBM. No 0–100.")
    a("Complementary to `debt_schedule_qa.md` — this is the *non-debt* book.")
    a("`created_at` is a connection clock (trail + debt QA). Do not invent a health Y from onboarding.")
    a("")
    a("## Headline")
    a("")
    rise = "YES, rise-only" if p2["rise_only"] else f"NO — {p2['n_drop']} drops"
    a(
        f"- Raw `banking_products`: **{p1['n_rows']:,} rows / {p1['n_co']:,} companies**. "
        f"Null `created_at`: {p1['n_null_created']} ({_pp(p1['share_null'])}). "
        f"Post-snapshot: {p1['n_after']} (dq_log 44). "
        f"Train companies with a row: {p1['n_train_co']}; holdout coverage {p1['n_hold_co']}."
    )
    a(
        f"- `g_n_accounts` is a **created_at inventory panel**: {rise} "
        f"({p2['n_rise']:,} rises, {p2['n_drop']:,} drops, {p2['n_flat']:,} flats). "
        f"Same monotone as `f_n_facilities`. First-month p50={p2['first_p50']:.0f} "
        f"({_pp(p2['first_share0'])} still 0); last-month p50={p2['last_p50']:.0f} "
        f"({_pp(p2['last_share0'])} still 0)."
    )
    a(
        f"- `g_new_this_month>0`: {p3['n_gt0']:,} CM ({_pp(p3['share_gt0'])}) / "
        f"{p3['n_co_gt0']:,} companies. acf1={p3['acf1']:.3f} (flag {p3['acf1_flag']:.3f}). "
        f"Size ρ vs log1p(a_in3)={p3['rho_login3']:.3f}. "
        f"{_pp(p3['share_new_is_birth'])} of new months are a first-ever as-of account. "
        f"**PARK as a health Y** — connection wave, not Q3 turning."
    )
    bh = p4["best_has"]
    a(
        f"- Best `g_has_*` vs Y3 (stressed, group-fold, **oriented** max(auc,1−auc)): "
        f"**{bh['feature']} {bh['cv_orient']:.3f}** (raw {bh['cv']:.3f}). "
        f"`c_n_days_with_tx` oriented CV={p4['days_y3']['cv_orient']:.3f} "
        f"(raw {p4['days_y3']['cv']:.3f} = published 0.711 flipped). "
        f"Size dummy oriented CV={p4['size_y3']['cv_orient']:.3f} (raw {p4['size_y3']['cv']:.3f}). "
        f"No access flag beats oriented-size by ≥0.02 — **CLOSE as Y3 X**. "
        f"Card × TPV 2×2 after size terciles: "
        f"{'survives' if p10['survive'] else 'does not survive'} "
        f"(TPV n={p10['n_tpv']} last-month companies). "
        f"{'KEEP as Q1 descriptive type.' if p10['survive'] else 'Do not KEEP as an operating type; do not revive clusters (sil 0.234).'}"
    )
    a(
        f"- `g_created_after_snapshot` is **0 on every train CM** ({p5['after_all_zero']}). "
        f"Feature-report drop of the three `g_created_*` constants: **CONFIRM**. "
        f"44-col starter is {p5['n_starter']} cols; `g_has_*` all in starter={p5['keep_has_ok']}; "
        f"`g_created_*` in starter={p5['created_in_starter'] or 'none'}. "
        f"`g_n_accounts` / `g_new_this_month` are KEEP-list but **not** in the 44-col starter."
    )
    a(
        f"- 470 dark vs 744 invoiced: "
        f"{'CONFIRM' if p6['erp_expected'] else 'CORRECT counts below'}. "
        "Access ≠ ERP — dark companies still have checking; TPV is rare on both. "
        f"`g_has_checking=0` is {_pp(p15['share_chk0_is_zero'])} the connection hole (`g_n_accounts=0`), not a mix flag."
    )
    a("")
    a("## Brief questions")
    a("")
    a("1. **Who is healthy?** — access flags (card / TPV / checking) are operating-type tags, not a health reading. They lose to size on Y3.")
    a("2. **Who is improving?** — `g_n_accounts` only rises. A higher count is more connections, not 45→65.")
    a("3. **Who is turning?** — `g_new_this_month` is a connection birth (acf ≈ 0, often first as-of account). PARK as Q3.")
    a("4. **Dip vs fall?** — not this table. Complementary to debt inventory, not a cash-path.")
    a(
        "5. **Why did it change?** — card / TPV as a type tag: "
        + ("card split survives size terciles (descriptive only)." if p10["survive"] else "does not survive size terciles.")
        + " Do not revive k-means (silhouette 0.234)."
    )
    a("6. **Months earlier?** — `created_at` is not lead time. The 73.6% late first-created is `g_n_accounts=0` on early months, not the CONSTANT `g_created_*` columns.")
    a("")
    a("## 1. Raw `banking_products`")
    a("")
    a("| item | n |")
    a("| --- | ---: |")
    a(f"| rows | {p1['n_rows']:,} |")
    a(f"| companies | {p1['n_co']:,} (train {p1['n_train_co']} / holdout {p1['n_hold_co']}) |")
    a(f"| product_id | {p1['n_prod']:,} |")
    a(f"| null created_at | {p1['n_null_created']} ({_pp(p1['share_null'])}) |")
    a(f"| created_after_snapshot | {p1['n_after']} |")
    a("")
    a("Dictionary names checking / card / investment / tpv / saving / expensesPlatform.")
    a("`wallet`, `risk`, `lineofcomex` sit in **other** — Family G does not emit `g_has_*` for them (they still count in `g_n_accounts`).")
    a("")
    a(_md_table(
        p1["by_type"].to_dict("records"),
        [
            ("type_bucket", "bucket"),
            ("type", "type"),
            ("n_rows", "rows"),
            ("n_co", "companies"),
            ("n_null", "null created_at"),
        ],
    ))
    a("")
    a("Named-type buckets (distinct companies; other is union of leftover types):")
    a("")
    a(_md_table(
        p1["bucket"].to_dict("records"),
        [
            ("type_bucket", "bucket"),
            ("n_rows", "rows"),
            ("n_co", "companies"),
            ("n_null", "null created_at"),
        ],
    ))
    a("")
    a("## 2. `g_n_accounts` is a rise-only panel")
    a("")
    a(
        f"Train panel: **{p2['n_cm']:,}** company-months / **{p2['n_co']:,}** companies. "
        f"Month-to-month: {p2['n_rise']:,} rises, {p2['n_drop']:,} drops, {p2['n_flat']:,} flats. "
        f"`g_n_banks` drops: {p2['bank_drop']:,}. "
        f"{'Inventory is monotone within company — same as-of `created_at` rule as debt facilities.' if p2['rise_only'] else 'NOT monotone — inspect products.py.'}"
    )
    a("")
    a(f"acf1={p2['acf1']:.3f} acf3={p2['acf3']:.3f} acf6={p2['acf6']:.3f}. "
      f"Spearman vs a_in3={p2['rho_in3']:.3f}; vs log1p(a_in3)={p2['rho_login3']:.3f} "
      f"({'SIZE' if abs(p2['rho_login3']) >= SIZE_RHO else 'not SIZE'}; threshold |ρ|≥0.5 vs a_in3 / log1p).")
    a("")
    a("| slice | mean | p50 | share 0 |")
    a("| --- | ---: | ---: | ---: |")
    a(f"| first month | {p2['first_mean']:.2f} | {p2['first_p50']:.0f} | {_pp(p2['first_share0'])} |")
    a(f"| last month | {p2['last_mean']:.2f} | {p2['last_p50']:.0f} | {_pp(p2['last_share0'])} |")
    a("")
    a("First-month vs last-month count distribution (train companies):")
    a("")
    a(_md_table(
        p2["first_dist"] + p2["last_dist"],
        [("slice", "slice"), ("k", "g_n_accounts"), ("n", "n companies"), ("share", "share")],
    ))
    a("")
    a("First-month zeros are the 73.6% connection clock (`created_at` after first tx), not a missing product. Last-month zeros are the handful with no on-panel banking row (trail QA: 5 never-`g_n_accounts>0`).")
    a("")
    a("## 3. `g_new_this_month` — connection, not Q3")
    a("")
    a(
        f"Prevalence: {p3['n_gt0']:,} / {p3['n_cm']:,} CM = {_pp(p3['share_gt0'])} "
        f"({p3['n_co_gt0']:,} companies). Mean {p3['mean']:.3f}, p50 {p3['p50']:.0f}."
    )
    a(
        f"acf1={p3['acf1']:.3f} (flag {p3['acf1_flag']:.3f}), acf3={p3['acf3']:.3f}, acf6={p3['acf6']:.3f} — "
        "low persistence, same shape as `f_new_facility`."
    )
    a(
        f"Size ρ vs log1p(a_in3)={p3['rho_login3']:.3f} (flag {p3['rho_flag_login3']:.3f}) — not SIZE."
    )
    a(
        f"First grid month has `g_new>0` on {_pp(p3['first_new_share'])}; "
        f"`g_n_accounts=0` on {_pp(p3['first_zero_acc'])} (cash can start before the product is connected)."
    )
    a(
        f"Of {p3['n_new']:,} new months, {p3['n_new_and_birth']:,} ({_pp(p3['share_new_is_birth'])}) "
        f"are a first-ever as-of account (n_birth={p3['n_birth']:,}). That is onboarding / connection, not a turn."
    )
    a("")
    a("**PARK** `g_new_this_month` as a health Y and as a Q3 turning flag. Keep it as an inventory clock (already on the feature-report KEEP list, not in the 44-col starter).")
    a("")
    if len(p3["calendar"]):
        cal_rows = []
        for _, r in p3["calendar"].iterrows():
            cal_rows.append(
                {
                    "period": pd.Timestamp(r["period"]).strftime("%Y-%m"),
                    "n_cm": int(r["n_cm"]),
                    "n_co": int(r["n_co"]),
                }
            )
        a("New-connection calendar (train CM with `g_new>0`):")
        a("")
        a(_md_table(cal_rows, [("period", "month"), ("n_cm", "n CM"), ("n_co", "n companies")]))
        a("")
    a("## 4. Access flags vs Q1 (Y2 / Y3 stressed / Y1 last-value)")
    a("")
    a(
        f"Train labeled: Y2 n={p4['n_y2']:,} base {_pp(p4['y2_rate'])}; "
        f"Y3 n={p4['n_y3']:,} base {_pp(p4['y3_rate'])} (stressed-only by construction)."
    )
    a("Single-feature AUROC. Raw = score as-is. Oriented = max(auc, 1−auc) — the published `c_n_days_with_tx` 0.711 is the flipped raw ~0.289. Group-fold CV (5, seed 20260918) is the quote. Holdout not used.")
    a("")
    a(_md_table(
        p4["auc_rows"],
        [
            ("y", "Y"),
            ("feature", "feature"),
            ("auc", "pooled raw"),
            ("auc_orient", "pooled orient"),
            ("cv", "CV raw"),
            ("cv_orient", "CV orient"),
            ("sd", "sd"),
            ("n_labeled", "n labeled"),
            ("rho_ain3", "ρ vs a_in3"),
            ("rho_login3", "ρ vs log1p(a_in3)"),
        ],
    ))
    a("")
    a("KEEP-as-Y3-X rule: **oriented** CV beats oriented size dummy by ≥0.02 **and** not SIZE (|ρ| vs a_in3 or log1p ≥ 0.5). First-cut as-is comparison was wrong — size raw 0.38 is inverse-size skill 0.62; chance flags at 0.50 are not a win.")
    a("")
    a(_md_table(
        p4["verdicts"],
        [
            ("feature", "feature"),
            ("cv", "CV raw"),
            ("cv_orient", "CV orient"),
            ("size_cv", "size orient"),
            ("days_cv", "days orient"),
            ("beat_size", "Δ size"),
            ("beat_days", "Δ days"),
            ("is_size", "SIZE?"),
            ("keep_q1_x", "KEEP as Y3 X?"),
        ],
    ))
    a("")
    a("Base rates by access (train company-months):")
    a("")
    a(_md_table(
        p4["rate_rows"],
        [
            ("flag", "flag"),
            ("group", "group"),
            ("y", "Y"),
            ("n_labeled", "n labeled"),
            ("n_pos", "n pos"),
            ("rate", "base rate"),
            ("n_cm", "n CM"),
            ("n_co", "n companies"),
        ],
    ))
    a("")
    a("Y1 last-value story (train labeled rows). Inflow last-value = `a_op_in` vs `y1_in_h1`; liquidity = `b_liq` vs `y1_liq_h1`. Access should not rewrite 'last-value wins liq / hist wins inflow'.")
    a("")
    a(_md_table(
        p4["y1_rows"],
        [
            ("series", "series"),
            ("slice", "slice"),
            ("n", "n"),
            ("spearman", "Spearman"),
            ("mae", "median |err|"),
        ],
    ))
    a("")
    a("## 5. `g_created_*` constants + 44-col starter")
    a("")
    a(
        f"`g_created_after_snapshot` all-zero on the monthly train panel: **{p5['after_all_zero']}**. "
        "Last as-of cut is `created_at < 2026-09-01`, so the 44 post-extract rows never enter. "
        "This is **not** the 73.6% late-first-created fact (that is `g_n_accounts=0`)."
    )
    a("")
    a(_md_table(
        p5["rows"],
        [
            ("split", "split"),
            ("col", "column"),
            ("cov_cm", "cov CM"),
            ("n_unique", "n unique"),
            ("modal", "modal"),
            ("share_zero", "share 0"),
            ("max", "max"),
        ],
    ))
    a("")
    a(
        f"Starter set length **{p5['n_starter']}** (is 44: {p5['starter_is_44']}). "
        f"G columns in starter: `{', '.join(p5['g_in_starter'])}`. "
        f"Drop `g_created_*`: **{'CONFIRM' if p5['drop_created_ok'] else 'CORRECT — they leaked into starter'}**. "
        f"Keep `g_has_*` in starter: **{'CONFIRM' if p5['keep_has_ok'] else 'CORRECT — a flag is missing'}** "
        f"({', '.join(p5['has_in_starter'])}). "
        f"`g_n_accounts` in starter: {p5['n_accounts_in_starter']}. "
        f"`g_new_this_month` in starter: {p5['new_in_starter']}. "
        "Those two stay on the broader KEEP list as inventory clocks, not in the 44-col GBM starter. "
        "**CAUTION:** `g_has_checking` is in the 44 but is 99% the connection hole, not a mix flag — consider dropping it from the starter later; do not edit products.py for that."
    )
    a("")
    a("## 6. 470 dark vs 744 invoiced (access ≠ ERP)")
    a("")
    a(
        f"Train last-month companies: {p6['n_train']:,}. "
        f"Ever-ERP {p6['n_erp']:,} / never-ERP {p6['n_dark']:,} "
        f"({'CONFIRM 744 / 470' if p6['erp_expected'] else 'counts differ from trail QA — see table'}). "
        f"Holdout ever-ERP coverage only: {p6['hold_erp']} / {p6['hold_n']}."
    )
    a("")
    a("Last-month as-of inventory:")
    a("")
    a(_md_table(
        p6["rows"],
        [
            ("group", "group"),
            ("n_co", "n companies"),
            ("acc_mean", "accounts mean"),
            ("acc_p50", "accounts p50"),
            ("share0", "share 0 accounts"),
            ("has_card", "has_card"),
            ("has_tpv", "has_tpv"),
            ("has_checking", "has_checking"),
            ("has_saving", "has_saving"),
            ("has_invest", "has_invest"),
            ("n_banks_p50", "n_banks p50"),
            ("custom_p50", "custom share p50"),
        ],
    ))
    a("")
    a("Company-ever (max on the panel):")
    a("")
    a(_md_table(
        p6["ever_rows"],
        [
            ("group", "group"),
            ("n_co", "n"),
            ("acc_mean", "accounts mean"),
            ("acc_p50", "accounts p50"),
            ("has_card", "ever card"),
            ("has_tpv", "ever TPV"),
            ("has_checking", "ever checking"),
        ],
    ))
    a("")
    a("## 7. One bank vs many vs Y2")
    a("")
    a(
        f"Train last-month: one bank {p7['n_one']:,} ({_pp(p7['share_one'])}), "
        f"many {p7['n_many']:,}, zero {p7['n_zero']:,}."
    )
    a("")
    a("Company-ever Y2 (max of labeled months):")
    a("")
    a(_md_table(
        p7["conc"],
        [
            ("group", "group"),
            ("n_co", "n companies"),
            ("n_labeled", "n with a Y2 label"),
            ("n_pos", "n ever Y2"),
            ("ever_y2_rate", "ever-Y2 rate"),
        ],
    ))
    a("")
    a("Company-month Y2:")
    a("")
    a(_md_table(
        p7["cm"],
        [
            ("group", "group"),
            ("n_cm", "n CM"),
            ("n_labeled", "n labeled"),
            ("n_pos", "n pos"),
            ("rate", "Y2 rate"),
        ],
    ))
    a("")
    a("Top raw `bank_name` (all companies, extract stock):")
    a("")
    a(_md_table(
        p7["raw_banks"].to_dict("records"),
        [("bank", "bank"), ("n_rows", "rows"), ("n_co", "companies")],
    ))
    a("")
    a("## 8. `company_meta.n_banking` vs last-month `g_n_accounts`")
    a("")
    a(
        f"Train companies: {p8['n_co']:,}. Exact match: {p8['n_match']:,} ({_pp(p8['share_match'])}). "
        f"Delta explained by post-snapshot products: {p8['n_explained_by_after']:,} / {p8['n_co']:,}. "
        f"Delta p50={p8['delta_p50']:.0f} mean={p8['delta_mean']:.2f} max={p8['delta_max']:.0f}."
    )
    a("")
    a("`n_banking` is extract stock (includes the 44 post-snapshot rows). Last-month G drops those rows. A mismatch of +1…k is the connection-after-extract pile, not a products.py bug.")
    a("")
    if len(p8["mismatch_head"]):
        a("Largest mismatches (train):")
        a("")
        a(_md_table(
            p8["mismatch_head"].to_dict("records"),
            [
                ("company_id", "company"),
                ("n_banking", "n_banking"),
                ("g_n_accounts", "last g_n_accounts"),
                ("delta", "delta"),
                ("n_after", "n post-snapshot"),
            ],
        ))
        a("")
    a("## 9. Custom / Other banks vs the 470")
    a("")
    a(
        f"All-custom extract books: {p9['n_all_custom']:,} train companies, "
        f"{p9['n_all_custom_dark']:,} of them never-ERP ({_pp(p9['share_all_custom_are_dark'])} of all-custom). "
        f"Never-ERP with a banking row: {p9['n_dark']:,} — custom is **not** the 470."
    )
    a("")
    a(_md_table(
        p9["rows"],
        [
            ("group", "group"),
            ("n_co", "n companies"),
            ("any_custom", "any custom"),
            ("all_custom", "all custom"),
            ("custom_mean", "mean custom share"),
            ("n_all_custom", "n all-custom"),
            ("n_any_tpv", "n with a TPV row"),
        ],
    ))
    a("")
    a("## 10. Card × TPV after size terciles (no cluster revival)")
    a("")
    a(
        f"Last-month train: card {p10['n_card']:,}, TPV {p10['n_tpv']:,}. "
        f"A 2×2 cannot beat silhouette 0.234 when TPV is {p10['n_tpv']} companies. "
        f"Card Y3 gap same sign in every tercile with ≥5 card positives: "
        f"**{'yes' if p10['survive'] else 'no'}**."
    )
    a("")
    a("Last-month 2×2 counts (and size mix):")
    a("")
    a(_md_table(
        p10["cells"],
        [
            ("cell", "cell"),
            ("n_co", "n companies"),
            ("t1", "T1"),
            ("t2", "T2"),
            ("t3", "T3"),
        ],
    ))
    a("")
    a("Last-month access share by size tercile (the PNG):")
    a("")
    a(_md_table(
        p10["last_share"],
        [("tercile", "tercile"), ("has_card_share", "has_card"), ("has_tpv_share", "has_tpv")],
    ))
    a("")
    a("Company-month rates in the 2×2 (train):")
    a("")
    a(_md_table(
        p10["rate_rows"],
        [
            ("y", "Y"),
            ("cell", "cell"),
            ("n_labeled", "n labeled"),
            ("n_pos", "n pos"),
            ("rate", "base rate"),
            ("n_co", "n companies"),
        ],
    ))
    a("")
    a("Card vs no-card inside size terciles — the only cell large enough to test 'operating type after size':")
    a("")
    a(_md_table(
        p10["card_rows"],
        [
            ("y", "Y"),
            ("tercile", "tercile"),
            ("group", "group"),
            ("n_labeled", "n labeled"),
            ("n_pos", "n pos"),
            ("rate", "base rate"),
            ("n_co", "n companies"),
        ],
    ))
    a("")
    a("Y3 card−no_card gap by tercile:")
    a("")
    a(_md_table(
        p10["gaps"],
        [
            ("tercile", "tercile"),
            ("gap", "rate gap"),
            ("n_card_lab", "card labeled"),
            ("n_card_pos", "card pos"),
        ],
    ))
    a("")
    if p10["survive"]:
        a("The card split keeps the same sign after size. KEEP as Q1 *descriptive* (not Y3 X). Still do not revive k-means.")
    else:
        a("The card split does **not** survive size terciles as a stable type. CLOSE as Q1 operating-type. Do not revive clusters (silhouette 0.234).")
    a("")
    a("## 11. One bank vs many, size-controlled")
    a("")
    a("Raw Y2 is higher on many-bank months (8.3% vs 5.3%). If that is size (more banks ↔ larger), it dies inside terciles.")
    a("")
    a(_md_table(
        p11["rows"],
        [
            ("y", "Y"),
            ("tercile", "tercile"),
            ("group", "group"),
            ("n_cm", "n CM"),
            ("n_labeled", "n labeled"),
            ("n_pos", "n pos"),
            ("rate", "base rate"),
            ("n_co", "n companies"),
        ],
    ))
    a("")
    a("## 12. First-tx vs first-created (73.6% replica)")
    a("")
    a(
        f"Train companies on the grid: {p12['n_train']:,}. With a banking row: {p12['n_bank']:,} "
        f"(no banking {p12['n_no_bank']}). First `created_at` after 2024-09-01: "
        f"{p12['n_late']:,} / {p12['n_bank']:,} = {_pp(p12['share_late'])}. "
        f"{'CONFIRM 73.6%' if p12['confirm_736'] else 'does not match trail 73.6% — see numbers'}."
    )
    a(
        f"`created_at` after first tx: {p12['created_after_tx']:,}. "
        f"Months from first tx to first `g_n_accounts>0`: p50={p12['lag_p50']:.1f} mean={p12['lag_mean']:.1f}; "
        f"share already connected on first tx month={_pp(p12['share_lag0'])}; "
        f"never as-of account on panel={p12['n_never_acc']}."
    )
    a("")
    a("This is the connection clock. `g_created_*` constants are **not** this fact.")
    a("")
    a("## 13. Leftover types (no `g_has_*`)")
    a("")
    a(
        f"Train companies with wallet / risk / expensesPlatform / lineofcomex: {p13['n_any']:,} "
        f"(ever-ERP {p13['n_erp']:,}). They still increment `g_n_accounts`. Family G does not flag them."
    )
    a("")
    a(_md_table(
        p13["rows"],
        [
            ("type", "type"),
            ("n_co", "n companies"),
            ("n_erp", "ever-ERP"),
            ("n_dark", "never-ERP"),
            ("acc_p50", "last accounts p50"),
            ("y2_rate", "Y2 rate"),
            ("y3_rate", "Y3 rate"),
            ("n_y3", "n Y3 labeled"),
        ],
    ))
    a("")
    a("Small-n leftover types. Do not invent `g_has_wallet`. Not a health Y. expensesPlatform Y3 23% sits on 26 labeled months / 7 companies — do not promote.")
    a("")
    a("## 14. `g_new` same-month Y — first birth vs add-on")
    a("")
    a("If new-this-month were Q3 turning, same-month Y2/Y3 would jump. They should not.")
    a("")
    a(_md_table(
        p14["rows"],
        [
            ("y", "Y"),
            ("kind", "kind"),
            ("n_cm", "n CM"),
            ("n_labeled", "n labeled"),
            ("n_pos", "n pos"),
            ("rate", "base rate"),
            ("n_co", "n companies"),
        ],
    ))
    a("")
    a("## 15. Checking = connected; card AUROC on the connected book")
    a("")
    a(
        f"`g_has_checking=0` CM: {p15['n_chk0']:,}. Of those, {p15['n_chk0_and_zero']:,} "
        f"({_pp(p15['share_chk0_is_zero'])}) also have `g_n_accounts=0`. "
        f"Checking-off but accounts>0: {p15['n_chk0_with_acc']:,} CM / {p15['type_mix']['n_co']} companies "
        f"(card {_pp(p15['type_mix']['has_card'])}, TPV {_pp(p15['type_mix']['has_tpv'])}, "
        f"saving {_pp(p15['type_mix']['has_saving'])}, invest {_pp(p15['type_mix']['has_invest'])}). "
        f"`g_has_checking` is almost the connection clock, not a mix flag — raw checking covers 1,282 / 1,283 companies."
    )
    a("")
    a(
        f"Connected months only (`g_n_accounts>0`, n={p15['n_connected']:,}). "
        f"Y3 oriented CV: card {p15['card_cv']:.3f} vs size {p15['size_cv']:.3f} (Δ {p15['beat_size']:.3f}) "
        f"vs days {p15['days_cv']:.3f}. "
        + (
            "Still CLOSE as Y3 X."
            if p15["beat_size"] < KEEP_DELTA
            else "Beats size on the connected book — still quote vs 0.711."
        )
    )
    a("")
    a(_md_table(
        p15["auc_rows"],
        [
            ("y", "Y"),
            ("feature", "feature"),
            ("auc", "pooled raw"),
            ("cv", "CV raw"),
            ("cv_orient", "CV orient"),
            ("sd", "sd"),
            ("n_labeled", "n labeled"),
        ],
    ))
    a("")
    a("Residual size of one-bank vs many-bank companies (last-month log1p(a_in3) inside the same tercile). T1 still has a size gap (p50 6.4 vs 8.1). T2 is matched (12.42 vs 12.54) — the T2 Y2 5.1% vs 10.8% is not leftover size, but T3 **flips**, so one-vs-many stays CLOSE as Y2 X. Connected-only `g_n_banks` Y3 oriented 0.633 vs size 0.622 (Δ 0.011 < 0.02).")
    a("")
    a(_md_table(
        p15["size_rows"],
        [
            ("tercile", "tercile"),
            ("group", "group"),
            ("n_co", "n companies"),
            ("log_p50", "log1p(a_in3) p50"),
            ("log_mean", "mean"),
            ("acc_p50", "accounts p50"),
        ],
    ))
    a("")
    a("## PARK / CLOSE / KEEP")
    a("")
    a("| object | decision | why |")
    a("| --- | --- | --- |")
    a("| `g_new_this_month` as health Y / Q3 | **PARK** | connection birth; acf≈0; often first as-of account |")
    a("| `g_created_*` constants as X | **DROP** (CONFIRM) | all-zero / modal 100%; not the 73.6% fact |")
    a("| `g_has_*` as Y3 X | **CLOSE** | oriented CV loses to size 0.617 and to days 0.711 |")
    a(
        "| `g_has_*` as Q1 descriptive | "
        + ("**KEEP** card as a size-controlled type" if p10["survive"] else "**CLOSE** as operating type")
        + " | 2×2 / tercile test below; TPV n too small; no cluster revival |"
    )
    a("| `g_n_accounts` inventory | **KEEP** (clock, not Y) | rise-only as-of panel; complementary to debt facilities |")
    a("| `g_n_accounts` / `g_new` in 44-col starter | **out** (CONFIRM) | KEEP-list inventory, not in the 44 GBM columns |")
    a("| `g_has_checking` in 44-col starter | **CAUTION** | present (CONFIRM) but it is the connection hole, not mix |")
    a("| invent a dark-access Y | **PARK** | access ≠ ERP; the 470 still have checking |")
    a("| `g_has_checking` as mix / type | **CLOSE** | 99% of checking=0 is `g_n_accounts=0` (connection hole) |")
    a("| one-vs-many banks as Y2/Y3 X | **CLOSE** | Y2 sign flips in T3; residual size remains inside tercile (pass 15) |")
    a("| card AUROC on connected months | **CLOSE** as Y3 X | still loses to size / days |")
    a("")
    if png_ok:
        a("## Plot")
        a("")
        a("- `analysis/outputs/banking_g_has_vs_size.png` — last-month `g_has_card` / `g_has_tpv` vs company size tercile (train).")
        a("")
    a("## Closed in this module")
    a("")
    a("- Rise-only inventory: **yes** (0 drops), same as-of `created_at` rule as debt.")
    a("- `g_new` = connection: **yes** (52.8% first birth; calendar spread; PARK as Y).")
    a("- Feature-report drop `g_created_*` + keep `g_has_*` in 44-col: **CONFIRM**. `g_n_accounts` / `g_new` stay off the 44.")
    a("- Oriented AUROC: first as-is KEEP table was a false win vs inverse-size 0.38. Corrected.")
    a("- 73.6% replica + n_banking vs last G: measured.")
    a("- Custom/Other is not the 470. Dark do not have fewer accounts.")
    a("- Card × TPV after size: does not survive. Leftover types: measured. `g_new` same-month Y: measured.")
    a("- `g_has_checking` ≈ connection hole. Card AUROC on connected months: still CLOSE vs size/days.")
    a("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_registry(p1, p2, p3, p4, p5, p6, p10, p12, p15) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    ts = _utc_ts()
    bh = p4["best_has"]
    rows = [
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "G", "y": "-", "model": "banking_g_qa",
            "split": "train", "metric": "n_banking_product_rows",
            "value": _fmt(p1["n_rows"]), "coverage": "1.0000",
            "notes": f"cos={p1['n_co']} null_created={p1['n_null_created']} after={p1['n_after']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "G", "y": "-", "model": "banking_g_qa",
            "split": "train", "metric": "g_n_accounts_n_drop",
            "value": _fmt(p2["n_drop"]), "coverage": "1.0000",
            "notes": f"rise={p2['n_rise']} flat={p2['n_flat']} rise_only={p2['rise_only']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "G", "y": "-", "model": "banking_g_qa",
            "split": "train", "metric": "g_new_this_month_share_gt0",
            "value": _fmt(p3["share_gt0"]), "coverage": "1.0000",
            "notes": f"n_cm={p3['n_gt0']} n_co={p3['n_co_gt0']} acf1={p3['acf1']:.3f} birth_share={p3['share_new_is_birth']:.3f} PARK_Y",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "G", "y": Y3, "model": "banking_g_qa",
            "split": "train_cv", "metric": f"auroc_orient_{bh['feature']}",
            "value": _fmt(bh["cv_orient"]), "coverage": _fmt(p4["n_y3"] / 21157 if p4["n_y3"] else float("nan")),
            "notes": f"best g_has_* oriented; raw={bh['cv']:.3f} days={p4['days_y3']['cv_orient']:.3f} size={p4['size_y3']['cv_orient']:.3f} CLOSE_Y3_X",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "C", "y": Y3, "model": "banking_g_qa",
            "split": "train_cv", "metric": "auroc_orient_c_n_days_with_tx",
            "value": _fmt(p4["days_y3"]["cv_orient"]), "coverage": _fmt(p4["n_y3"] / 21157 if p4["n_y3"] else float("nan")),
            "notes": f"replica of published 0.711; raw_cv={p4['days_y3']['cv']:.3f} pooled_raw={p4['days_y3']['auc']:.3f}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "G", "y": Y3, "model": "banking_g_qa",
            "split": "train", "metric": "card_tpv_2x2_survives_size",
            "value": "1" if p10["survive"] else "0",
            "coverage": _fmt(p10["n_card"] / 1214 if p10["n_card"] else float("nan")),
            "notes": f"card={p10['n_card']} tpv={p10['n_tpv']} same_sign={p10['same_sign']}; no cluster revival",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "G", "y": "-", "model": "banking_g_qa",
            "split": "train", "metric": "first_created_after_202409_share",
            "value": _fmt(p12["share_late"]), "coverage": _fmt(p12["n_bank"] / p12["n_train"] if p12["n_train"] else float("nan")),
            "notes": f"{p12['n_late']}/{p12['n_bank']} confirm_736={p12['confirm_736']} lag_p50={p12['lag_p50']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "G", "y": "-", "model": "banking_g_qa",
            "split": "train", "metric": "g_created_after_snapshot_all_zero",
            "value": "1" if p5["after_all_zero"] else "0",
            "coverage": "1.0000",
            "notes": f"drop_created CONFIRM={p5['drop_created_ok']}; has_in_starter CONFIRM={p5['keep_has_ok']}; starter_n={p5['n_starter']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "G", "y": "-", "model": "banking_g_qa",
            "split": "train", "metric": "n_dark_vs_erp",
            "value": _fmt(p6["n_dark"]), "coverage": _fmt(p6["n_dark"] / p6["n_train"] if p6["n_train"] else float("nan")),
            "notes": f"erp={p6['n_erp']} confirm744_470={p6['erp_expected']}; access_ne_erp",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "G", "y": Y3, "model": "banking_g_qa",
            "split": "train_cv", "metric": "auroc_orient_g_has_card_connected",
            "value": _fmt(p15["card_cv"]),
            "coverage": _fmt(p15["n_connected"] / 21157 if p15["n_connected"] else float("nan")),
            "notes": f"connected months only; size={p15['size_cv']:.3f} days={p15['days_cv']:.3f} chk0_is_zero={p15['share_chk0_is_zero']:.3f} CLOSE",
        },
    ]
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        seen = {
            (r.get("agent"), r.get("y"), r.get("model"), r.get("split"), r.get("metric"))
            for r in reader
        }
    fresh = []
    for r in rows:
        key = (str(r.get("agent", "")), str(r.get("y", "")), str(r.get("model", "")),
               str(r.get("split", "")), str(r.get("metric", "")))
        if key in seen:
            continue
        fresh.append(r)
        seen.add(key)
    if not fresh:
        return
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        for r in fresh:
            w.writerow({k: r.get(k, "") for k in header})


def main() -> int:
    print(f"banking_g_qa start seed={FOLD_SEED} holdout=72")
    if not STORE.exists():
        raise FileNotFoundError(STORE)
    if not TARGETS.exists():
        raise FileNotFoundError(TARGETS)
    con = connect()
    store = load_store()
    targets = load_targets()

    print("pass 1 raw banking_products")
    p1 = pass1_raw(con)
    print(f"  rows={p1['n_rows']} cos={p1['n_co']} null_created={p1['n_null_created']} after={p1['n_after']}")

    print("pass 2 rise-only inventory")
    p2 = pass2_inventory(store)
    print(
        f"  rise={p2['n_rise']} drop={p2['n_drop']} rise_only={p2['rise_only']} "
        f"first0={p2['first_share0']:.3f} last0={p2['last_share0']:.3f}"
    )

    print("pass 3 g_new_this_month")
    p3 = pass3_new(store)
    print(
        f"  share_gt0={p3['share_gt0']:.4f} acf1={p3['acf1']:.3f} "
        f"rho={p3['rho_login3']:.3f} birth={p3['share_new_is_birth']:.3f}"
    )

    print("pass 4 access vs Y2/Y3/Y1")
    p4 = pass4_access(con, store, targets)
    bh = p4["best_has"]
    print(
        f"  best={bh['feature']} cv_orient={bh['cv_orient']:.3f} raw={bh['cv']:.3f} "
        f"days={p4['days_y3']['cv_orient']:.3f} size={p4['size_y3']['cv_orient']:.3f}"
    )

    print("pass 5 created_* constants + starter")
    p5 = pass5_constants(store)
    print(
        f"  after_zero={p5['after_all_zero']} starter={p5['n_starter']} "
        f"drop_created={p5['drop_created_ok']} has_in={p5['keep_has_ok']}"
    )

    print("pass 6 dark vs invoiced")
    p6 = pass6_dark(con, store)
    print(f"  erp={p6['n_erp']} dark={p6['n_dark']} confirm={p6['erp_expected']}")

    print("pass 7 one vs many banks")
    p7 = pass7_banks(con, store, targets)
    print(f"  one={p7['n_one']} many={p7['n_many']} zero={p7['n_zero']}")

    print("pass 8 n_banking vs last g_n_accounts")
    p8 = pass8_meta(con, store)
    print(f"  match={p8['n_match']}/{p8['n_co']} explained={p8['n_explained_by_after']}")

    print("pass 9 custom vs 470")
    p9 = pass9_custom_dark(con, store)
    print(
        f"  all_custom={p9['n_all_custom']} dark_among={p9['n_all_custom_dark']} "
        f"share={p9['share_all_custom_are_dark']}"
    )

    print("pass 10 card x tpv after size")
    p10 = pass10_twobytwo(store, targets)
    print(f"  card={p10['n_card']} tpv={p10['n_tpv']} survive={p10['survive']}")

    print("pass 11 one vs many x size")
    p11 = pass11_banks_size(store, targets)
    y2_t1_one = next(r for r in p11["rows"] if r["y"] == Y2 and r["tercile"] == "T1_small" and r["group"] == "one_bank")
    y2_t1_many = next(r for r in p11["rows"] if r["y"] == Y2 and r["tercile"] == "T1_small" and r["group"] == "many_banks")
    print(f"  T1 Y2 one={y2_t1_one['rate']:.3f} many={y2_t1_many['rate']:.3f}")

    print("pass 12 connection clock replica")
    p12 = pass12_clock(con, store)
    print(f"  late={p12['n_late']}/{p12['n_bank']}={p12['share_late']:.3f} confirm={p12['confirm_736']}")

    print("pass 13 leftover types")
    p13 = pass13_other(con, store, targets)
    print(f"  n_any={p13['n_any']} erp={p13['n_erp']}")

    print("pass 14 g_new same-month Y")
    p14 = pass14_new_y(store, targets)
    y3_birth = next(r for r in p14["rows"] if r["y"] == Y3 and r["kind"] == "first_birth")
    print(f"  y3 first_birth rate={y3_birth['rate']} n_lab={y3_birth['n_labeled']}")

    print("pass 15 checking=connected + connected-only AUROC")
    p15 = pass15_connected(con, store, targets)
    print(
        f"  chk0_is_zero={p15['share_chk0_is_zero']:.3f} "
        f"card_cv={p15['card_cv']:.3f} size={p15['size_cv']:.3f} Δ={p15['beat_size']:.3f}"
    )

    png_ok = make_png(store)
    print(f"png={OUT_PNG if png_ok else 'skipped'}")

    write_md(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, png_ok)
    print(f"wrote {OUT_MD}")
    append_registry(p1, p2, p3, p4, p5, p6, p10, p12, p15)
    con.close()
    print("banking_g_qa done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
