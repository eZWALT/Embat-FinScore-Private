"""Debt schedule + utilisation snapshot QA.

NORTH_STAR said schedule and utilisation are last-month thin (~1.7%).
This module confirms or corrects that on the monthly store. Holdout 72
(seed 20260918) is counted for coverage only. Rates, tertiles, AUROC,
and PARK/CLOSE cuts are train-only.

Does not rewrite parquet / duckdb / family modules / models / Ys.
Does not revive a utilisation label. Does not run build_targets.
Does not write a 0–100. Does not touch product/.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.debt_schedule_qa

Owned: analysis/outputs/debt_schedule_qa.md, optional PNG,
overnight/waves/wave4_debt_schedule.md (one note at the end),
append-only registry coverage rows.

Iteration (same module):
1. Raw schedule rows / join / month coverage / last-month share
2. Inventory panel vs snapshot amounts; flow overlap; Y base rates
3. OGTG prevalence; Q6 next_payment_date / total_periods honesty
4. Product-type mix; groups sharing a schedule; 40 / 87 check
5. f_new_facility as Q3 turning flag (counts + size + acf only)
6. Settlement join; granted=0; 39-vs-38; created_at = connection
7. f_sched_vs_obs clip; snapshot rate vs observed f_fc_r
8. Group-fold single-feature AUROC (protocol.py); flow vs book
9. Fair Y4 split (schedule vs repayment-no-schedule); odd-row inventory
10. New-facility first-birth vs add-on; variable vs fixed rate
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
OUT_MD = ANALYSIS / "outputs" / "debt_schedule_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "debt_schedule_who_when.png"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_debt_schedule.md"
AGENT = "1882a607"
WAVE = "4"
ROUND = "R4"
LAST_M = MONTHS[-1]
EXTRACT = AS_OF
Y_COLS = (
    "y3_recover_cash_6m",
    "y4_ds_r_double",
    "y9_fee_r_ownp80",
    "y2_neg_2of3",
)
SNAP_COLS = (
    "f_w_rate",
    "f_util_snapshot",
    "f_months_to_next_pay",
    "f_sched_vs_obs",
)
INV_COLS = ("f_n_facilities", "f_new_facility", "f_has_loc")
FLOW_CATS = ("debt_repayment", "interest_charge", "fee")


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


def _md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(title for _, title in cols) + " |"
    sep = "| " + " | ".join("---" if not k.endswith("_n") and "n_" not in k and k not in {
        "n", "n_co", "n_cm", "n_pos", "n_prod", "n_rows", "n_labeled", "n_groups"
    } else "---:" for k, _ in cols) + " |"
    # numeric-looking keys right-aligned
    right = {
        "n", "n_co", "n_cm", "n_pos", "n_prod", "n_rows", "n_labeled",
        "n_groups", "n_hold", "n_train", "rows", "companies", "products",
        "last_n", "nn", "last_share",
    }
    sep = "| " + " | ".join("---:" if k in right else "---" for k, _ in cols) + " |"
    lines = [head, sep]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k, "")
            if isinstance(v, float) and np.isfinite(v):
                if k.endswith("_rate") or k.endswith("_share") or k in {
                    "cov_cm", "cov_co", "last_share", "auc", "acf1", "acf3", "acf6",
                    "size_auc",
                }:
                    cells.append(f"{v:.3f}" if abs(v) < 1 or k.endswith("auc") or k.startswith("acf") or k.endswith("share") or k.startswith("cov") else f"{v:.4f}")
                    if k.endswith("_rate") or k in {"cov_cm", "cov_co", "last_share"}:
                        cells[-1] = f"{100.0 * v:.1f}%"
                    elif k.endswith("auc") or k.startswith("acf") or k == "size_auc":
                        cells[-1] = f"{v:.3f}"
                else:
                    cells.append(f"{v:.4g}")
            else:
                cells.append("" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def median_acf(series: pd.Series, company: pd.Series, lag: int) -> float:
    """Median company-wise Pearson autocorr. Train caller must already filter."""
    df = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    vals = []
    for _, g in df.groupby("co", sort=False):
        x = g["x"].to_numpy(dtype=float)
        if len(x) <= lag:
            continue
        a = x[:-lag]
        b = x[lag:]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 4:
            continue
        a, b = a[m], b[m]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        vals.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.median(vals)) if vals else float("nan")


def load_store() -> pd.DataFrame:
    need = [
        "company_id",
        "period",
        "a_in3",
        *SNAP_COLS,
        *INV_COLS,
        "f_n_types",
        "f_has_factoring",
        "f_has_confirming",
        "f_outstanding_gt_granted",
        "f_ds_r",
        "f_fc_r",
        "f_debt_service",
    ]
    raw = pd.read_parquet(STORE, columns=need)
    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["period"])
    return raw


def load_targets() -> pd.DataFrame:
    raw = pd.read_parquet(TARGETS, columns=["company_id", "period", *Y_COLS])
    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["period"])
    return raw


def pass1_raw(con) -> dict:
    """Raw debt_schedule_config rows, companies, products; join to debt_products."""
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM debt_schedule_config) AS n_rows,
          (SELECT COUNT(DISTINCT company_id) FROM debt_schedule_config) AS n_co,
          (SELECT COUNT(DISTINCT product_id) FROM debt_schedule_config) AS n_prod,
          (SELECT COUNT(*) FROM debt_products) AS n_dp,
          (SELECT COUNT(DISTINCT company_id) FROM debt_products) AS n_dp_co,
          (SELECT COUNT(DISTINCT product_id) FROM debt_products) AS n_dp_prod
        """
    ).df().iloc[0].to_dict()
    join = con.execute(
        """
        SELECT
          SUM(CASE WHEN d.product_id IS NULL THEN 1 ELSE 0 END) AS sched_no_dp,
          SUM(CASE WHEN d.product_id IS NOT NULL THEN 1 ELSE 0 END) AS joined,
          SUM(CASE WHEN d.company_id IS NOT NULL AND s.company_id <> d.company_id
                   THEN 1 ELSE 0 END) AS company_mismatch,
          SUM(CASE WHEN coalesce(d.created_after_snapshot, false) THEN 1 ELSE 0 END)
            AS created_after_snapshot
        FROM debt_schedule_config s
        LEFT JOIN debt_products d ON s.product_id = d.product_id
        """
    ).df().iloc[0].to_dict()
    by_type = con.execute(
        """
        SELECT coalesce(d.type, '(unmatched)') AS type,
               COUNT(*) AS n_rows,
               COUNT(DISTINCT s.company_id) AS n_co,
               COUNT(DISTINCT s.product_id) AS n_prod
        FROM debt_schedule_config s
        LEFT JOIN debt_products d ON s.product_id = d.product_id
        GROUP BY 1
        ORDER BY n_rows DESC
        """
    ).df()
    cos = con.execute("SELECT DISTINCT company_id FROM debt_schedule_config").df()
    cos["company_id"] = cos["company_id"].astype(str)
    n_train = int((~cos.company_id.isin(hold)).sum())
    n_hold = int(cos.company_id.isin(hold).sum())
    hold_ids = sorted(cos.loc[cos.company_id.isin(hold), "company_id"])
    still_40_87 = int(raw["n_rows"]) == 87 and int(raw["n_co"]) == 40 and int(raw["n_prod"]) == 87
    return {
        "n_rows": int(raw["n_rows"]),
        "n_co": int(raw["n_co"]),
        "n_prod": int(raw["n_prod"]),
        "n_dp": int(raw["n_dp"]),
        "n_dp_co": int(raw["n_dp_co"]),
        "n_dp_prod": int(raw["n_dp_prod"]),
        "sched_no_dp": int(join["sched_no_dp"] or 0),
        "joined": int(join["joined"] or 0),
        "company_mismatch": int(join["company_mismatch"] or 0),
        "created_after_snapshot": int(join["created_after_snapshot"] or 0),
        "n_train_co": n_train,
        "n_hold_co": n_hold,
        "hold_ids": hold_ids,
        "still_40_87": still_40_87,
        "by_type": by_type,
    }


def pass2_month_coverage(store: pd.DataFrame) -> dict:
    """Which calendar months have non-null schedule / util fields."""
    hold = load_holdout()
    df = store.copy()
    df["is_train"] = ~df["company_id"].isin(hold)
    train = df.loc[df["is_train"]].copy()
    hold_df = df.loc[~df["is_train"]].copy()
    assert_no_holdout(train["company_id"])

    rows = []
    for split_name, part in (("train", train), ("holdout", hold_df)):
        n_cm = len(part)
        n_co = part["company_id"].nunique()
        last = part["period"].max() if n_cm else pd.NaT
        for col in (*SNAP_COLS, "f_outstanding_gt_granted", *INV_COLS):
            nn = part[col].notna()
            n = int(nn.sum())
            n_last = int((nn & (part["period"] == last)).sum()) if pd.notna(last) else 0
            n_cos = int(part.loc[nn, "company_id"].nunique())
            rows.append(
                {
                    "split": split_name,
                    "col": col,
                    "n_cm": n_cm,
                    "n_co_panel": n_co,
                    "nn": n,
                    "cov_cm": n / n_cm if n_cm else float("nan"),
                    "cov_co": n_cos / n_co if n_co else float("nan"),
                    "n_co": n_cos,
                    "last_n": n_last,
                    "last_share": n_last / n if n else float("nan"),
                    "last_month_only": bool(n > 0 and n_last == n),
                }
            )

    by_month = []
    for p, g in train.groupby("period"):
        rec = {"period": pd.Timestamp(p), "n_co": int(g["company_id"].nunique())}
        for col in SNAP_COLS:
            rec[col] = float(g[col].notna().mean())
        rec["fac_gt0"] = float((pd.to_numeric(g["f_n_facilities"], errors="coerce") > 0).mean())
        rec["new_gt0"] = float((pd.to_numeric(g["f_new_facility"], errors="coerce") > 0).mean())
        rec["ogtg_nn"] = float(g["f_outstanding_gt_granted"].notna().mean())
        by_month.append(rec)

    train_snap = {r["col"]: r for r in rows if r["split"] == "train"}
    # Headline correction
    util_last = bool(train_snap["f_util_snapshot"]["last_month_only"])
    rate_last = bool(train_snap["f_w_rate"]["last_month_only"])
    n_sched_cm = train_snap["f_w_rate"]["nn"]
    n_sched_co = train_snap["f_w_rate"]["n_co"]
    return {
        "rows": rows,
        "by_month": by_month,
        "train_n_cm": len(train),
        "train_n_co": int(train["company_id"].nunique()),
        "hold_n_cm": len(hold_df),
        "hold_n_co": int(hold_df["company_id"].nunique()),
        "util_last_month_only": util_last,
        "ogtg_last_month_only": bool(train_snap["f_outstanding_gt_granted"]["last_month_only"]),
        "rate_last_month_only": rate_last,
        "rate_cov_cm": train_snap["f_w_rate"]["cov_cm"],
        "util_cov_cm": train_snap["f_util_snapshot"]["cov_cm"],
        "n_sched_cm": n_sched_cm,
        "n_sched_co": n_sched_co,
        "rate_last_share": train_snap["f_w_rate"]["last_share"],
        "verdict": (
            "CORRECT last-month-only for f_util_snapshot / f_outstanding_gt_granted; "
            "CORRECT ~1.7% coverage for f_w_rate / f_months_to_next_pay / f_sched_vs_obs; "
            "WRONG to call those three last-month-only — they are a thin growing panel "
            f"({n_sched_cm} train CM / {n_sched_co} train companies; "
            f"{100.0 * train_snap['f_w_rate']['last_share']:.1f}% of non-nulls are 2026-08)."
        ),
    }


def pass3_inventory(con, store: pd.DataFrame) -> dict:
    """created_at inventory is a real series; amounts/rates are the extract still."""
    hold = load_holdout()
    train = store.loc[~store["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])

    inv_stats = []
    for col in INV_COLS:
        x = pd.to_numeric(train[col], errors="coerce")
        inv_stats.append(
            {
                "col": col,
                "cov_cm": float(x.notna().mean()),
                "share_gt0": float((x > 0).mean()),
                "n_co_gt0": int(train.loc[x > 0, "company_id"].nunique()),
                "acf1": median_acf(x, train["company_id"], 1),
                "acf3": median_acf(x, train["company_id"], 3),
                "acf6": median_acf(x, train["company_id"], 6),
                "mean": float(x.mean()),
                "p50": float(x.median()),
            }
        )

    # created_at month of debt_products vs schedule products
    created = con.execute(
        """
        SELECT 'debt_products' AS src,
               date_trunc('month', created_at) AS m,
               COUNT(*) AS n,
               COUNT(DISTINCT company_id) AS n_co
        FROM debt_products
        WHERE NOT coalesce(created_after_snapshot, false)
        GROUP BY 1, 2
        UNION ALL
        SELECT 'schedule' AS src,
               date_trunc('month', d.created_at) AS m,
               COUNT(*) AS n,
               COUNT(DISTINCT d.company_id) AS n_co
        FROM debt_schedule_config s
        JOIN debt_products d ON s.product_id = d.product_id
        WHERE NOT coalesce(d.created_after_snapshot, false)
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).df()
    created["m"] = pd.to_datetime(created["m"])

    # Is f_n_facilities non-decreasing within company? (as-of created_at should be)
    fac = train.sort_values(["company_id", "period"])
    dlt = fac.groupby("company_id")["f_n_facilities"].diff()
    n_drop = int((dlt < 0).sum())
    n_rise = int((dlt > 0).sum())
    n_flat = int((dlt == 0).sum())

    # Rate uniqueness: snapshot copied forward
    w = train.loc[train["f_w_rate"].notna()]
    nuniq_rate = w.groupby("company_id")["f_w_rate"].nunique() if len(w) else pd.Series(dtype=float)
    nuniq_next = (
        train.loc[train["f_months_to_next_pay"].notna()]
        .groupby("company_id")["f_months_to_next_pay"]
        .nunique()
        if train["f_months_to_next_pay"].notna().any()
        else pd.Series(dtype=float)
    )

    months_per_co = w.groupby("company_id").size() if len(w) else pd.Series(dtype=float)
    return {
        "inv_stats": inv_stats,
        "created": created,
        "n_fac_drop": n_drop,
        "n_fac_rise": n_rise,
        "n_fac_flat": n_flat,
        "inventory_is_panel": n_rise > 0 and train["f_n_facilities"].notna().mean() > 0.99,
        "rate_nuniq_1": int((nuniq_rate == 1).sum()) if len(nuniq_rate) else 0,
        "rate_nuniq_gt1": int((nuniq_rate > 1).sum()) if len(nuniq_rate) else 0,
        "next_nuniq_median": float(nuniq_next.median()) if len(nuniq_next) else float("nan"),
        "months_per_sched_co_p50": float(months_per_co.median()) if len(months_per_co) else float("nan"),
        "months_per_sched_co_max": int(months_per_co.max()) if len(months_per_co) else 0,
        "n_sched_co_train_store": int(w["company_id"].nunique()),
    }


def pass4_flow_overlap(con) -> dict:
    """Share of train companies with a schedule vs a debt/fee/interest flow."""
    hold = load_holdout()
    cos = con.execute("SELECT company_id, group_id FROM companies").df()
    cos["company_id"] = cos["company_id"].astype(str)
    cos["group_id"] = cos["group_id"].astype(str)
    train_cos = set(cos.loc[~cos.company_id.isin(hold), "company_id"])
    n_train = len(train_cos)

    sched = con.execute("SELECT DISTINCT company_id FROM debt_schedule_config").df()
    sched["company_id"] = sched["company_id"].astype(str)
    sched_train = set(sched.company_id) & train_cos

    sets = {"schedule": sched_train}
    for cat in FLOW_CATS:
        d = con.execute(
            """
            SELECT DISTINCT company_id FROM transactions
            WHERE category = ? AND NOT coalesce(is_extreme, false)
            """,
            [cat],
        ).df()
        d["company_id"] = d["company_id"].astype(str)
        sets[cat] = set(d.company_id) & train_cos

    any_flow = set.union(*(sets[c] for c in FLOW_CATS))
    sets["any_fin_flow"] = any_flow
    dp = con.execute(
        """
        SELECT DISTINCT company_id FROM debt_products
        WHERE NOT coalesce(created_after_snapshot, false)
        """
    ).df()
    dp["company_id"] = dp["company_id"].astype(str)
    sets["debt_product"] = set(dp.company_id) & train_cos

    rows = []
    for name, s in sets.items():
        rows.append(
            {
                "set": name,
                "n_co": len(s),
                "share": len(s) / n_train if n_train else float("nan"),
                "n_and_sched": len(s & sched_train),
                "sched_share": (len(s & sched_train) / len(s)) if s else float("nan"),
            }
        )
    return {
        "n_train": n_train,
        "n_sched_train": len(sched_train),
        "sched_share": len(sched_train) / n_train if n_train else float("nan"),
        "n_flow": len(any_flow),
        "sched_and_rep": len(sched_train & sets["debt_repayment"]),
        "sched_no_rep": len(sched_train - sets["debt_repayment"]),
        "sched_and_fee": len(sched_train & (sets["fee"] | sets["interest_charge"])),
        "rows": rows,
        "sched_train": sched_train,
        "train_cos": train_cos,
    }


def _tertile(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    try:
        return pd.qcut(x, 3, labels=["T1_small", "T2_mid", "T3_large"], duplicates="drop")
    except ValueError:
        return pd.Series(index=s.index, dtype="object")


def pass5_y_rates(store: pd.DataFrame, targets: pd.DataFrame, sched_train: set[str]) -> dict:
    """Accepted-Y base rates on schedule vs no-schedule. Size control log1p(a_in3)."""
    hold = load_holdout()
    m = store.merge(targets, on=["company_id", "period"], how="left")
    train = m.loc[~m["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    train["ever_sched"] = train["company_id"].isin(sched_train)
    train["has_sched_cm"] = train["f_w_rate"].notna()
    train["log_in3"] = np.log1p(pd.to_numeric(train["a_in3"], errors="coerce").clip(lower=0))

    cm_rows = []
    for y in Y_COLS:
        for lab, mask in (("schedule", train["ever_sched"]), ("no_schedule", ~train["ever_sched"])):
            s = pd.to_numeric(train.loc[mask, y], errors="coerce")
            nn = int(s.notna().sum())
            cm_rows.append(
                {
                    "y": y,
                    "group": lab,
                    "n_labeled": nn,
                    "n_pos": int((s == 1).sum()),
                    "rate": float(s.mean()) if nn else float("nan"),
                    "n_co": int(train.loc[mask & s.notna(), "company_id"].nunique()),
                }
            )
        auc_ever = auroc(train[y], train["ever_sched"].astype(float))
        auc_cm = auroc(train[y], train["has_sched_cm"].astype(float))
        auc_size = auroc(train[y], train["log_in3"])
        cm_rows.append(
            {
                "y": y,
                "group": "auroc_ever_sched",
                "n_labeled": int(pd.to_numeric(train[y], errors="coerce").notna().sum()),
                "n_pos": int((pd.to_numeric(train[y], errors="coerce") == 1).sum()),
                "rate": float(auc_ever),
                "n_co": int(train.loc[pd.to_numeric(train[y], errors="coerce").notna(), "company_id"].nunique()),
                "auc_ever": auc_ever,
                "auc_cm": auc_cm,
                "auc_size": auc_size,
            }
        )

    aucs = []
    for y in Y_COLS:
        aucs.append(
            {
                "y": y,
                "auc_ever": auroc(train[y], train["ever_sched"].astype(float)),
                "auc_cm": auroc(train[y], train["has_sched_cm"].astype(float)),
                "auc_size": auroc(train[y], train["log_in3"]),
            }
        )

    # company-median size tertile (train), then CM rates inside tertile
    med = (
        train.groupby("company_id", sort=False)["log_in3"]
        .median()
        .rename("med_log_in3")
        .reset_index()
    )
    med["tertile"] = _tertile(med["med_log_in3"])
    train = train.merge(med[["company_id", "tertile"]], on="company_id", how="left")
    tert_rows = []
    for tert in ("T1_small", "T2_mid", "T3_large"):
        part = train.loc[train["tertile"].astype(str) == tert]
        for y in Y_COLS:
            for lab, mask in (("schedule", part["ever_sched"]), ("no_schedule", ~part["ever_sched"])):
                s = pd.to_numeric(part.loc[mask, y], errors="coerce")
                nn = int(s.notna().sum())
                tert_rows.append(
                    {
                        "tertile": tert,
                        "y": y,
                        "group": lab,
                        "n_labeled": nn,
                        "n_pos": int((s == 1).sum()),
                        "rate": float(s.mean()) if nn else float("nan"),
                        "n_co": int(part.loc[mask, "company_id"].nunique()),
                    }
                )

    # company-ever Y
    ever = train.groupby("company_id", sort=False).agg(
        ever_sched=("ever_sched", "max"),
        **{y: (y, "max") for y in Y_COLS},
    )
    ever_rows = []
    for y in Y_COLS:
        for lab, mask in (("schedule", ever["ever_sched"] == 1), ("no_schedule", ever["ever_sched"] == 0)):
            s = pd.to_numeric(ever.loc[mask, y], errors="coerce")
            ever_rows.append(
                {
                    "y": y,
                    "group": lab,
                    "n_co": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "rate": float(s.mean()) if s.notna().any() else float("nan"),
                }
            )

    return {
        "cm_rows": cm_rows,
        "aucs": aucs,
        "tert_rows": tert_rows,
        "ever_rows": ever_rows,
        "n_train_cm": len(train),
        "n_sched_cm": int(train["ever_sched"].sum()),
        "n_sched_labeled": {
            y: int(pd.to_numeric(train.loc[train["ever_sched"], y], errors="coerce").notna().sum())
            for y in Y_COLS
        },
    }


def pass6_ogtg(con, store: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """outstanding_balance > granted_balance: prevalence and Y4/Y9 overlap."""
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT
          SUM(CASE WHEN outstanding_gt_granted THEN 1 ELSE 0 END) AS n_true,
          SUM(CASE WHEN NOT outstanding_gt_granted THEN 1 ELSE 0 END) AS n_false,
          SUM(CASE WHEN outstanding_gt_granted IS NULL THEN 1 ELSE 0 END) AS n_null,
          COUNT(DISTINCT CASE WHEN outstanding_gt_granted THEN company_id END) AS n_co_true
        FROM debt_products
        """
    ).df().iloc[0].to_dict()
    sched = con.execute(
        """
        SELECT
          SUM(CASE WHEN outstanding_gt_granted THEN 1 ELSE 0 END) AS n_true,
          COUNT(DISTINCT CASE WHEN outstanding_gt_granted THEN company_id END) AS n_co_true,
          SUM(CASE WHEN outstanding_balance > granted_balance THEN 1 ELSE 0 END) AS n_gt
        FROM debt_schedule_config
        """
    ).df().iloc[0].to_dict()

    train = store.loc[~store["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    last = train["period"].max()
    lm = train.loc[train["period"] == last]
    n_ogtg_lm = int((pd.to_numeric(lm["f_outstanding_gt_granted"], errors="coerce") == 1).sum())
    ogtg_cos = set(lm.loc[pd.to_numeric(lm["f_outstanding_gt_granted"], errors="coerce") == 1, "company_id"])

    m = train.merge(targets, on=["company_id", "period"], how="left")
    ever = m.groupby("company_id", sort=False).agg(**{y: (y, "max") for y in Y_COLS})
    ever["ogtg"] = ever.index.isin(ogtg_cos)
    y_rows = []
    for y in Y_COLS:
        for lab, mask in (("ogtg", ever["ogtg"]), ("not_ogtg", ~ever["ogtg"])):
            s = pd.to_numeric(ever.loc[mask, y], errors="coerce")
            y_rows.append(
                {
                    "y": y,
                    "group": lab,
                    "n_co": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "rate": float(s.mean()) if s.notna().any() else float("nan"),
                }
            )

    # last-month Ys are typically NaN (horizon). Quote that.
    lm_y = lm.merge(targets, on=["company_id", "period"], how="left")
    last_y_nn = {y: int(pd.to_numeric(lm_y[y], errors="coerce").notna().sum()) for y in Y_COLS}

    return {
        "dp_true": int(raw["n_true"] or 0),
        "dp_false": int(raw["n_false"] or 0),
        "dp_null": int(raw["n_null"] or 0),
        "dp_co": int(raw["n_co_true"] or 0),
        "sched_true": int(sched["n_true"] or 0),
        "sched_co": int(sched["n_co_true"] or 0),
        "n_ogtg_lm_train": n_ogtg_lm,
        "n_lm_train": len(lm),
        "ogtg_share_lm": n_ogtg_lm / len(lm) if len(lm) else float("nan"),
        "y_rows": y_rows,
        "last_y_nn": last_y_nn,
        "last_period": str(pd.Timestamp(last).date()),
    }


def pass7_q6(con, store: pd.DataFrame) -> dict:
    """Can next_payment_date or total_periods be used as lead time?"""
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT
          COUNT(*) AS n,
          SUM(CASE WHEN next_payment_date < DATE '2026-09-01' THEN 1 ELSE 0 END) AS next_before_asof,
          SUM(CASE WHEN next_payment_date >= DATE '2026-09-01' THEN 1 ELSE 0 END) AS next_on_or_after,
          SUM(CASE WHEN next_payment_date > DATE '2026-09-01' THEN 1 ELSE 0 END) AS next_after,
          SUM(CASE WHEN last_payment_date > DATE '2026-09-01' THEN 1 ELSE 0 END) AS last_after_asof,
          SUM(CASE WHEN last_payment_date > next_payment_date THEN 1 ELSE 0 END) AS last_after_next,
          MIN(next_payment_date) AS min_next,
          MAX(next_payment_date) AS max_next,
          MIN(last_payment_date) AS min_last,
          MAX(last_payment_date) AS max_last,
          MIN(total_periods) AS min_tp,
          MAX(total_periods) AS max_tp,
          AVG(total_periods) AS mean_tp,
          SUM(CASE WHEN granted_balance > 0 THEN 1 ELSE 0 END) AS g_pos,
          SUM(CASE WHEN granted_balance = 0 THEN 1 ELSE 0 END) AS g_zero,
          SUM(CASE WHEN granted_balance < 0 THEN 1 ELSE 0 END) AS g_neg
        FROM debt_schedule_config
        """
    ).df().iloc[0].to_dict()
    freq = con.execute(
        """
        SELECT amortising_frequency, COUNT(*) AS n
        FROM debt_schedule_config GROUP BY 1 ORDER BY n DESC
        """
    ).df()
    itype = con.execute(
        """
        SELECT interest_type, COUNT(*) AS n
        FROM debt_schedule_config GROUP BY 1 ORDER BY n DESC
        """
    ).df()

    train = store.loc[~store["company_id"].isin(hold)].copy()
    nxt = pd.to_numeric(train["f_months_to_next_pay"], errors="coerce")
    nn = nxt.dropna()
    share_neg = float((nn < 0).mean()) if len(nn) else float("nan")
    share_pos = float((nn > 0).mean()) if len(nn) else float("nan")

    # Within-company, does months_to_next_pay drop by ~1 each month? (frozen date)
    diffs = []
    for _, g in train.loc[nxt.notna()].groupby("company_id"):
        s = pd.to_numeric(g.sort_values("period")["f_months_to_next_pay"], errors="coerce")
        d = s.diff().dropna()
        if len(d):
            diffs.extend(d.tolist())
    med_step = float(np.median(diffs)) if diffs else float("nan")

    usable_lead = False  # honesty: no
    reason = (
        "next_payment_date is a static extract field (81/87 already before 2026-09-01). "
        "f_months_to_next_pay is a countdown to that frozen date and is usually already "
        "negative. total_periods is the original installment count, not remaining tenor. "
        "Neither is a living lead-time clock (Q6)."
    )
    return {
        "n": int(raw["n"]),
        "next_before_asof": int(raw["next_before_asof"] or 0),
        "next_on_or_after": int(raw["next_on_or_after"] or 0),
        "next_after": int(raw["next_after"] or 0),
        "last_after_asof": int(raw["last_after_asof"] or 0),
        "last_after_next": int(raw["last_after_next"] or 0),
        "min_next": str(raw["min_next"]),
        "max_next": str(raw["max_next"]),
        "min_last": str(raw["min_last"]),
        "max_last": str(raw["max_last"]),
        "min_tp": int(raw["min_tp"]),
        "max_tp": int(raw["max_tp"]),
        "mean_tp": float(raw["mean_tp"]),
        "g_pos": int(raw["g_pos"] or 0),
        "g_zero": int(raw["g_zero"] or 0),
        "g_neg": int(raw["g_neg"] or 0),
        "freq": freq,
        "itype": itype,
        "months_mean": float(nn.mean()) if len(nn) else float("nan"),
        "months_p50": float(nn.median()) if len(nn) else float("nan"),
        "months_min": float(nn.min()) if len(nn) else float("nan"),
        "months_max": float(nn.max()) if len(nn) else float("nan"),
        "share_neg": share_neg,
        "share_pos": share_pos,
        "med_step": med_step,
        "usable_lead": usable_lead,
        "reason": reason,
    }


def pass8_groups(con, sched_train: set[str]) -> dict:
    """Do groups share a schedule? Product mix already in pass 1."""
    hold = load_holdout()
    cos = con.execute("SELECT company_id, group_id FROM companies").df()
    cos["company_id"] = cos["company_id"].astype(str)
    cos["group_id"] = cos["group_id"].astype(str)
    sched = con.execute("SELECT DISTINCT company_id FROM debt_schedule_config").df()
    sched["company_id"] = sched["company_id"].astype(str)
    g = sched.merge(cos, on="company_id", how="left")
    sizes = g.groupby("group_id").size()
    train_g = g.loc[~g.company_id.isin(hold)]
    train_sizes = train_g.groupby("group_id").size()
    # siblings in those groups
    sib = cos.loc[cos.group_id.isin(set(g.group_id))]
    n_sib = len(sib)
    n_sib_sched = int(sib.company_id.isin(set(sched.company_id)).sum())
    share_multi = float((sizes >= 2).mean()) if len(sizes) else float("nan")
    return {
        "n_groups": int(g.group_id.nunique()),
        "n_groups_train": int(train_g.group_id.nunique()),
        "n_solo": int((sizes == 1).sum()),
        "n_multi": int((sizes >= 2).sum()),
        "max_in_group": int(sizes.max()) if len(sizes) else 0,
        "share_multi": share_multi,
        "n_sib": n_sib,
        "n_sib_sched": n_sib_sched,
        "sizes": sizes.value_counts().sort_index().to_dict(),
        "train_sizes": train_sizes.value_counts().sort_index().to_dict(),
        "n_sched_train": len(sched_train),
    }


def pass9_new_facility(store: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """f_new_facility as a Q3 turning flag: counts + size + acf. No tree."""
    hold = load_holdout()
    m = store.merge(targets, on=["company_id", "period"], how="left")
    train = m.loc[~m["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    x = pd.to_numeric(train["f_new_facility"], errors="coerce")
    flag = (x > 0).astype(float)
    train["log_in3"] = np.log1p(pd.to_numeric(train["a_in3"], errors="coerce").clip(lower=0))

    by_m = (
        train.assign(new_gt0=flag)
        .groupby("period", sort=True)
        .agg(n_co=("company_id", "nunique"), n_new=("new_gt0", "sum"), share=("new_gt0", "mean"))
        .reset_index()
    )
    aucs = []
    for y in Y_COLS:
        aucs.append(
            {
                "y": y,
                "auc_new": auroc(train[y], flag),
                "auc_size": auroc(train[y], train["log_in3"]),
                "n_labeled": int(pd.to_numeric(train[y], errors="coerce").notna().sum()),
            }
        )
    # coincidence: new this month and Y this month (descriptive)
    coin = []
    for y in Y_COLS:
        yy = pd.to_numeric(train[y], errors="coerce")
        both = (flag == 1) & (yy == 1)
        coin.append(
            {
                "y": y,
                "n_new_and_y": int(both.sum()),
                "rate_y_given_new": float(yy[flag == 1].mean()) if (flag == 1).any() else float("nan"),
                "rate_y_given_none": float(yy[flag == 0].mean()) if (flag == 0).any() else float("nan"),
            }
        )
    return {
        "n_cm_gt0": int((x > 0).sum()),
        "share_gt0": float((x > 0).mean()),
        "n_co_gt0": int(train.loc[x > 0, "company_id"].nunique()),
        "acf1": median_acf(x, train["company_id"], 1),
        "acf3": median_acf(x, train["company_id"], 3),
        "acf6": median_acf(x, train["company_id"], 6),
        "acf1_flag": median_acf(flag, train["company_id"], 1),
        "size_auc_vs_new": auroc(flag, train["log_in3"]),
        "by_month": by_m,
        "aucs": aucs,
        "coin": coin,
        "value_counts": x.value_counts().sort_index().to_dict(),
    }


def _spearman(a: pd.Series, b: pd.Series) -> float:
    d = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(d) < 20 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def pass10_settlement_clock(con, store: pd.DataFrame, sched_train: set[str]) -> dict:
    """Settlement join, granted holes, 39-vs-38, created_at = connection."""
    hold = load_holdout()
    settle = con.execute(
        """
        SELECT
          SUM(CASE WHEN b.product_id IS NOT NULL THEN 1 ELSE 0 END) AS hit_bank,
          SUM(CASE WHEN b.product_id IS NULL THEN 1 ELSE 0 END) AS miss_bank,
          SUM(CASE WHEN b.product_id IS NOT NULL AND b.company_id = s.company_id
                   THEN 1 ELSE 0 END) AS bank_same_co,
          SUM(CASE WHEN b.product_id IS NULL AND d2.product_id IS NOT NULL THEN 1 ELSE 0 END)
            AS miss_to_debt,
          SUM(CASE WHEN b.product_id IS NULL AND d2.product_id IS NULL THEN 1 ELSE 0 END)
            AS orphan
        FROM debt_schedule_config s
        LEFT JOIN banking_products b ON s.settlement_product_id = b.product_id
        LEFT JOIN debt_products d2 ON s.settlement_product_id = d2.product_id
        """
    ).df().iloc[0].to_dict()
    dest_type = con.execute(
        """
        SELECT coalesce(d2.type, '(orphan)') AS dest_type, COUNT(*) AS n
        FROM debt_schedule_config s
        LEFT JOIN banking_products b ON s.settlement_product_id = b.product_id
        LEFT JOIN debt_products d2 ON s.settlement_product_id = d2.product_id
        WHERE b.product_id IS NULL
        GROUP BY 1 ORDER BY n DESC
        """
    ).df()
    granted = con.execute(
        """
        SELECT
          COUNT(*) AS n,
          SUM(CASE WHEN abs(abs(d.granted) - s.granted_balance) < 1 THEN 1 ELSE 0 END) AS n_abs_match,
          SUM(CASE WHEN s.granted_balance = 0 THEN 1 ELSE 0 END) AS n_zero,
          SUM(CASE WHEN s.granted_balance > 0 AND d.granted < 0 THEN 1 ELSE 0 END) AS sched_pos_dp_neg,
          SUM(CASE WHEN s.next_payment_date < d.created_at THEN 1 ELSE 0 END) AS next_before_created,
          SUM(CASE WHEN CAST(s.last_payment_date AS DATE) = CAST(s.next_payment_date AS DATE)
                    AND s.last_payment_date > s.next_payment_date THEN 1 ELSE 0 END) AS last_gt_next_sameday
        FROM debt_schedule_config s
        JOIN debt_products d ON s.product_id = d.product_id
        """
    ).df().iloc[0].to_dict()
    post = con.execute(
        """
        SELECT s.company_id
        FROM debt_schedule_config s
        JOIN debt_products d ON s.product_id = d.product_id
        WHERE coalesce(d.created_after_snapshot, false)
        """
    ).df()
    post_cos = set(post["company_id"].astype(str))
    train = store.loc[~store["company_id"].isin(hold)].copy()
    store_sched = set(train.loc[train["f_w_rate"].notna(), "company_id"])
    raw_not_store = sorted(sched_train - store_sched)

    # created_at vs first repayment (schedule companies with a repayment)
    first_rep = con.execute(
        """
        SELECT company_id, MIN(CAST(date AS DATE)) AS first_rep
        FROM transactions
        WHERE category = 'debt_repayment' AND NOT coalesce(is_extreme, false)
        GROUP BY 1
        """
    ).df()
    first_rep["company_id"] = first_rep["company_id"].astype(str)
    first_sch = con.execute(
        """
        SELECT d.company_id, MIN(d.created_at) AS first_created
        FROM debt_schedule_config s
        JOIN debt_products d ON s.product_id = d.product_id
        GROUP BY 1
        """
    ).df()
    first_sch["company_id"] = first_sch["company_id"].astype(str)
    js = first_sch.merge(first_rep, on="company_id", how="left")
    js = js.loc[~js.company_id.isin(hold)].copy()
    js["lag"] = (pd.to_datetime(js["first_rep"]) - pd.to_datetime(js["first_created"])).dt.days
    n_js = int(js["lag"].notna().sum())
    n_before = int((js["lag"] < 0).sum())

    first_prod = con.execute(
        """
        SELECT company_id, MIN(created_at) AS first_prod
        FROM debt_products
        WHERE NOT coalesce(created_after_snapshot, false)
        GROUP BY 1
        """
    ).df()
    first_prod["company_id"] = first_prod["company_id"].astype(str)
    jp = first_prod.merge(first_rep, on="company_id", how="inner")
    jp = jp.loc[~jp.company_id.isin(hold)].copy()
    jp["lag"] = (pd.to_datetime(jp["first_rep"]) - pd.to_datetime(jp["first_prod"])).dt.days

    # first-ever facility among new_facility months
    t = train.sort_values(["company_id", "period"])
    prev = t.groupby("company_id")["f_n_facilities"].shift(1)
    new = pd.to_numeric(t["f_new_facility"], errors="coerce") > 0
    first_birth = new & prev.fillna(0).eq(0)

    # unbalanced panel
    n_by_m = t.groupby("period")["company_id"].nunique()
    return {
        "hit_bank": int(settle["hit_bank"] or 0),
        "miss_bank": int(settle["miss_bank"] or 0),
        "bank_same_co": int(settle["bank_same_co"] or 0),
        "miss_to_debt": int(settle["miss_to_debt"] or 0),
        "orphan": int(settle["orphan"] or 0),
        "dest_type": dest_type,
        "n_abs_match": int(granted["n_abs_match"] or 0),
        "n_zero_granted": int(granted["n_zero"] or 0),
        "sched_pos_dp_neg": int(granted["sched_pos_dp_neg"] or 0),
        "next_before_created": int(granted["next_before_created"] or 0),
        "last_gt_next_sameday": int(granted["last_gt_next_sameday"] or 0),
        "post_extract_co": sorted(post_cos),
        "raw_not_store": raw_not_store,
        "n_store_sched": len(store_sched),
        "n_raw_train": len(sched_train),
        "sched_rep_n": n_js,
        "sched_rep_before": n_before,
        "sched_rep_lag_p50": float(js["lag"].median()) if n_js else float("nan"),
        "prod_rep_n": int(len(jp)),
        "prod_rep_before": int((jp["lag"] < 0).sum()),
        "prod_rep_lag_p50": float(jp["lag"].median()) if len(jp) else float("nan"),
        "new_cm": int(new.sum()),
        "new_first_birth": int(first_birth.sum()),
        "panel_min_co": int(n_by_m.min()) if len(n_by_m) else 0,
        "panel_max_co": int(n_by_m.max()) if len(n_by_m) else 0,
        "no_rep_ids": sorted(sched_train - set(first_rep.company_id)),
    }


def pass11_ratio_rate(store: pd.DataFrame) -> dict:
    """f_sched_vs_obs clip + snapshot rate vs observed financing cost (train)."""
    hold = load_holdout()
    train = store.loc[~store["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    x = pd.to_numeric(train["f_sched_vs_obs"], errors="coerce")
    nn = x.dropna()
    rate = pd.to_numeric(train["f_w_rate"], errors="coerce")
    return {
        "n": int(nn.shape[0]),
        "mean": float(nn.mean()) if len(nn) else float("nan"),
        "p50": float(nn.median()) if len(nn) else float("nan"),
        "share0": float((nn == 0).mean()) if len(nn) else float("nan"),
        "share20": float((nn >= 20).mean()) if len(nn) else float("nan"),
        "p10": float(nn.quantile(0.10)) if len(nn) else float("nan"),
        "p90": float(nn.quantile(0.90)) if len(nn) else float("nan"),
        "rho_rate_fc": _spearman(rate, train["f_fc_r"]),
        "rho_rate_ds": _spearman(rate, train["f_ds_r"]),
        "rho_svo_ds": _spearman(x, train["f_ds_r"]),
        "rho_svo_fc": _spearman(x, train["f_fc_r"]),
        "rate_p50": float(rate.dropna().median()) if rate.notna().any() else float("nan"),
        "rate_min": float(rate.dropna().min()) if rate.notna().any() else float("nan"),
        "rate_max": float(rate.dropna().max()) if rate.notna().any() else float("nan"),
    }


def _fold_auroc(y: pd.Series, x: pd.Series, folds: pd.Series) -> dict:
    """Group-fold AUROC. Score is used as-is (0/1 flags). Train companies only."""
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


def pass12_groupfold(con, store: pd.DataFrame, targets: pd.DataFrame, sched_train: set[str]) -> dict:
    """Single-feature group-fold AUROC. Not a GBM."""
    hold = load_holdout()
    cos = train_companies(con)
    assert_no_holdout(cos["company_id"])
    folded = group_folds(cos, n=5, seed=FOLD_SEED)
    m = store.merge(targets, on=["company_id", "period"], how="left")
    train = m.loc[~m["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    train = train.merge(folded[["company_id", "fold"]], on="company_id", how="left")
    train["ever_sched"] = train["company_id"].isin(sched_train).astype(float)
    train["has_sched_cm"] = train["f_w_rate"].notna().astype(float)
    train["new_gt0"] = (pd.to_numeric(train["f_new_facility"], errors="coerce") > 0).astype(float)
    train["log_in3"] = np.log1p(pd.to_numeric(train["a_in3"], errors="coerce").clip(lower=0))

    rows = []
    for y in Y_COLS:
        for name, col in (
            ("ever_sched", "ever_sched"),
            ("has_sched_cm", "has_sched_cm"),
            ("new_facility_gt0", "new_gt0"),
            ("log1p_a_in3", "log_in3"),
        ):
            r = _fold_auroc(train[y], train[col], train["fold"])
            rows.append({"y": y, "feature": name, "cv": r["cv"], "sd": r["sd"], "n_folds": r["n_folds"]})
    return {"rows": rows, "n_train_co": int(cos.shape[0]), "n_folds": 5}


def pass13_flow_vs_book(con, store: pd.DataFrame) -> dict:
    """Observed f_ds_r on schedule vs repayment-without-schedule (train)."""
    hold = load_holdout()
    sched = set(
        con.execute("SELECT DISTINCT company_id FROM debt_schedule_config").df()["company_id"].astype(str)
    ) - hold
    rep = set(
        con.execute(
            """
            SELECT DISTINCT company_id FROM transactions
            WHERE category = 'debt_repayment' AND NOT coalesce(is_extreme, false)
            """
        ).df()["company_id"].astype(str)
    ) - hold
    loc = set(
        con.execute(
            """
            SELECT DISTINCT company_id FROM debt_products
            WHERE type = 'lineofcredit' AND NOT coalesce(created_after_snapshot, false)
            """
        ).df()["company_id"].astype(str)
    ) - hold
    train = store.loc[~store["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    ds = pd.to_numeric(train["f_ds_r"], errors="coerce")
    fc = pd.to_numeric(train["f_fc_r"], errors="coerce")

    def _summ(mask, name):
        return {
            "group": name,
            "n_co": int(train.loc[mask, "company_id"].nunique()),
            "n_cm": int(mask.sum()),
            "ds_p50": float(ds[mask].median()) if mask.any() else float("nan"),
            "ds_mean": float(ds[mask].mean()) if mask.any() else float("nan"),
            "ds_gt0": float((ds[mask] > 0).mean()) if mask.any() else float("nan"),
            "fc_p50": float(fc[mask].median()) if mask.any() else float("nan"),
            "fc_gt0": float((fc[mask] > 0).mean()) if mask.any() else float("nan"),
        }

    m_sched = train["company_id"].isin(sched)
    m_rep_only = train["company_id"].isin(rep - sched)
    m_neither = ~train["company_id"].isin(sched | rep)
    rows = [
        _summ(m_sched, "schedule"),
        _summ(m_rep_only, "repayment_no_schedule"),
        _summ(m_neither, "neither"),
    ]
    return {
        "rows": rows,
        "n_sched": len(sched),
        "n_rep_only": len(rep - sched),
        "n_loc": len(loc),
        "n_sched_and_loc": len(sched & loc),
        "n_rep": len(rep),
    }


def pass14_fair_y4(con, store: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """Y4 is unlabeled without debt service. Fair split = schedule vs repayment-no-schedule."""
    hold = load_holdout()
    sched = set(
        con.execute("SELECT DISTINCT company_id FROM debt_schedule_config").df()["company_id"].astype(str)
    ) - hold
    rep = set(
        con.execute(
            """
            SELECT DISTINCT company_id FROM transactions
            WHERE category = 'debt_repayment' AND NOT coalesce(is_extreme, false)
            """
        ).df()["company_id"].astype(str)
    ) - hold
    m = store.merge(targets, on=["company_id", "period"], how="left")
    train = m.loc[~m["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    train["g"] = np.where(
        train["company_id"].isin(sched),
        "schedule",
        np.where(train["company_id"].isin(rep), "rep_no_sched", "no_rep"),
    )
    rows = []
    for y in Y_COLS:
        for g, part in train.groupby("g"):
            s = pd.to_numeric(part[y], errors="coerce")
            rows.append(
                {
                    "y": y,
                    "group": g,
                    "n_labeled": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "rate": float(s.mean()) if s.notna().any() else float("nan"),
                    "n_co": int(part["company_id"].nunique()),
                }
            )
    odd = con.execute(
        """
        SELECT d.type,
               COUNT(*) AS n,
               SUM(CASE WHEN s.granted_balance = 0 THEN 1 ELSE 0 END) AS n_zero_granted,
               SUM(CASE WHEN s.outstanding_gt_granted THEN 1 ELSE 0 END) AS n_ogtg,
               SUM(CASE WHEN s.granted_balance > 1e8 THEN 1 ELSE 0 END) AS n_huge
        FROM debt_schedule_config s
        JOIN debt_products d ON s.product_id = d.product_id
        GROUP BY 1
        ORDER BY n DESC
        """
    ).df()
    return {
        "rows": rows,
        "odd": odd,
        "y4_sched": next(
            r["rate"] for r in rows if r["y"] == "y4_ds_r_double" and r["group"] == "schedule"
        ),
        "y4_rep": next(
            r["rate"] for r in rows if r["y"] == "y4_ds_r_double" and r["group"] == "rep_no_sched"
        ),
        "y4_norep_labeled": next(
            r["n_labeled"] for r in rows if r["y"] == "y4_ds_r_double" and r["group"] == "no_rep"
        ),
    }


def pass15_birth_rate(con, store: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """First connected facility vs add-on; variable vs fixed snapshot rate."""
    hold = load_holdout()
    m = store.merge(targets, on=["company_id", "period"], how="left")
    train = m.loc[~m["company_id"].isin(hold)].copy()
    assert_no_holdout(train["company_id"])
    train = train.sort_values(["company_id", "period"])
    prev = train.groupby("company_id")["f_n_facilities"].shift(1)
    new = pd.to_numeric(train["f_new_facility"], errors="coerce") > 0
    train["kind"] = np.where(~new, "none", np.where(prev.fillna(0).eq(0), "first_birth", "add_on"))
    rows = []
    for y in Y_COLS:
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
    it = con.execute("SELECT company_id, interest_type FROM debt_schedule_config").df()
    it["company_id"] = it["company_id"].astype(str)
    maj = it.groupby("company_id")["interest_type"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]
    )
    w = train.loc[train["f_w_rate"].notna()].copy()
    w["itype"] = w["company_id"].map(maj)
    itype_rows = []
    for itype, part in w.groupby("itype"):
        itype_rows.append(
            {
                "interest_type": itype,
                "n_cm": int(len(part)),
                "n_co": int(part["company_id"].nunique()),
                "rate_p50": float(pd.to_numeric(part["f_w_rate"], errors="coerce").median()),
                "fc_p50": float(pd.to_numeric(part["f_fc_r"], errors="coerce").median()),
                "ds_p50": float(pd.to_numeric(part["f_ds_r"], errors="coerce").median()),
            }
        )
    return {"rows": rows, "itype": itype_rows}


def make_png(cov: dict, inv: dict) -> bool:
    if not HAS_MPL:
        return False
    by = pd.DataFrame(cov["by_month"])
    if by.empty:
        return False
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    p = pd.to_datetime(by["period"])
    ax.plot(p, 100.0 * by["f_w_rate"], marker="o", ms=4, color="#1f4e79", label="f_w_rate / next-pay / sched_vs_obs")
    ax.plot(p, 100.0 * by["f_util_snapshot"], marker="s", ms=4, color="#c45911", label="f_util_snapshot")
    ax.plot(p, 100.0 * by["fac_gt0"], marker=".", ms=3, color="#548235", ls="--", label="f_n_facilities > 0")
    created = inv["created"]
    sch = created.loc[created["src"] == "schedule"].copy()
    if len(sch):
        ax2 = ax.twinx()
        ax2.bar(
            pd.to_datetime(sch["m"]),
            sch["n"],
            width=20,
            color="#7b7b7b",
            alpha=0.35,
            label="schedule products connected",
        )
        ax2.set_ylabel("schedule products connected")
        ax2.set_ylim(0, max(20, int(sch["n"].max()) + 4))
    ax.set_ylabel("% of train companies that month")
    ax.set_title("Who has a schedule vs when (train)")
    ax.set_ylim(0, 40)
    ax.set_xlim(pd.Timestamp("2022-12-01"), pd.Timestamp("2026-10-15"))
    ax.axvline(LAST_M, color="#c45911", ls=":", lw=1, label="last panel month")
    h1, l1 = ax.get_legend_handles_labels()
    if len(sch):
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, framealpha=0.9)
    else:
        ax.legend(loc="upper left", fontsize=7)
    fig.autofmt_xdate()
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    return True


def write_md(p1, cov, inv, flow, yx, ogtg, q6, grp, nf, p10, p11, p12, p13, p14, p15, png_ok: bool) -> None:
    lines = []
    a = lines.append
    a("# Debt schedule + utilisation snapshot QA")
    a("")
    a(f"Generated `{_utc_ts()}` UTC by `python -m analysis.evaluate.debt_schedule_qa`.")
    a("Holdout 72 (seed 20260918) is **coverage only**. Rates, tertiles, AUROC,")
    a("and PARK/CLOSE are train. No parquet rewrite. No new GBM. No 0–100.")
    a("Y10 utilisation labels stay parked — this file does not revive them.")
    a("")
    a("## Headline")
    a("")
    a(cov["verdict"])
    a("")
    a(f"- Raw `debt_schedule_config`: **{p1['n_rows']} rows / {p1['n_co']} companies / {p1['n_prod']} products**.")
    a(f"  debt.py citation (~87 rows / 40 companies) is **{'still true' if p1['still_40_87'] else 'STALE'}**.")
    a(f"  Train companies with a schedule **row**: **{p1['n_train_co']}**. Holdout (count only): **{p1['n_hold_co']}**.")
    a(f"  Monthly store trains down to **{cov['n_sched_co']}** — raw-not-store `{p10['raw_not_store']}` is the")
    a("  post-extract `created_after_snapshot` loan (Family F already drops it).")
    a(f"- Train store: `{cov['n_sched_co']}` companies / `{cov['n_sched_cm']}` company-months have `f_w_rate`")
    a(f"  ({100.0 * cov['rate_cov_cm']:.1f}% of {cov['train_n_cm']:,} train CM). Last-month share of those non-nulls: "
      f"{100.0 * cov['rate_last_share']:.1f}%.")
    a(f"- `f_util_snapshot` last-month-only: **{cov['util_last_month_only']}** "
      f"(coverage {100.0 * cov['util_cov_cm']:.1f}%). `f_outstanding_gt_granted` last-month-only: "
      f"**{cov['ogtg_last_month_only']}**.")
    a(f"- Inventory (`f_n_facilities` / `f_new_facility` / `f_has_loc`) is a **panel**: "
      f"{'yes' if inv['inventory_is_panel'] else 'no'}. Facility count rises {inv['n_fac_rise']} times and drops "
      f"{inv['n_fac_drop']} times (drops should be rare — as-of `created_at`).")
    a(f"- PARK snapshot columns as GBM X and as a health Y. Keep the *flow* `f_ds_r`.")
    a(f"- Q6: `next_payment_date` / `total_periods` are **not** lead time. {q6['reason']}")
    a("")
    a("## Brief questions")
    a("")
    a("5. **Why did it change?** — only if a schedule vs no-schedule split is a real trail,")
    a("   not a 40-company bookkeeping tag. CM AUROC of ever-schedule vs accepted Ys is ~0.50.")
    a("   Y4 is a bit higher on schedule companies (they have a debt service by construction).")
    a("   That is not a why-trail. Do not treat schedule presence as Q5.")
    a("6. **How many months earlier?** — `next_payment_date` is the last-book snapshot,")
    a("   usually already past. Countdown `f_months_to_next_pay` is not visibility of a turn.")
    a("   Q6 stays closed for these columns.")
    a("")
    a("## 1. Raw `debt_schedule_config` + join to `debt_products`")
    a("")
    a(f"| item | n |")
    a(f"| --- | ---: |")
    a(f"| schedule rows | {p1['n_rows']} |")
    a(f"| distinct companies | {p1['n_co']} (train {p1['n_train_co']} / holdout {p1['n_hold_co']} `{', '.join(p1['hold_ids'])}`) |")
    a(f"| distinct product_id | {p1['n_prod']} |")
    a(f"| debt_products rows / companies / products | {p1['n_dp']} / {p1['n_dp_co']} / {p1['n_dp_prod']} |")
    a(f"| schedule rows joined on product_id | {p1['joined']} |")
    a(f"| schedule with no debt_products row | {p1['sched_no_dp']} |")
    a(f"| company_id mismatch | {p1['company_mismatch']} |")
    a(f"| schedule product `created_after_snapshot` | {p1['created_after_snapshot']} |")
    a("")
    a("Join is clean. One schedule product is post-extract and Family F already drops")
    a("`created_after_snapshot` in `_schedule_asof`.")
    a("")
    a("### Product-type mix (schedule ∩ debt_products)")
    a("")
    a("No LOC and no factoring on the schedule table. Formal amortisation is almost all loans.")
    a("")
    type_rows = p1["by_type"].to_dict("records")
    a(_md_table(type_rows, [("type", "type"), ("n_rows", "rows"), ("n_co", "companies"), ("n_prod", "products")]))
    a("")
    a("## 2. Which months have non-null schedule fields")
    a("")
    a("Train panel: "
      f"**{cov['train_n_cm']:,}** company-months / **{cov['train_n_co']}** companies "
      f"(2024-09 … 2026-08). Holdout coverage only: {cov['hold_n_cm']:,} / {cov['hold_n_co']}.")
    a("")
    a(_md_table(
        cov["rows"],
        [
            ("split", "split"),
            ("col", "column"),
            ("nn", "n non-null"),
            ("cov_cm", "cov CM"),
            ("n_co", "n companies"),
            ("cov_co", "cov companies"),
            ("last_n", "n in last month"),
            ("last_share", "share of non-nulls in last month"),
            ("last_month_only", "last-month only?"),
        ],
    ))
    a("")
    a("### Train coverage by calendar month")
    a("")
    a("| period | n companies | f_w_rate | f_months_to_next_pay | f_sched_vs_obs | f_util_snapshot | fac>0 | new>0 | ogtg non-null |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in cov["by_month"]:
        a(
            f"| {pd.Timestamp(r['period']).date()} "
            f"| {r['n_co']} "
            f"| {100.0 * r['f_w_rate']:.1f}% "
            f"| {100.0 * r['f_months_to_next_pay']:.1f}% "
            f"| {100.0 * r['f_sched_vs_obs']:.1f}% "
            f"| {100.0 * r['f_util_snapshot']:.1f}% "
            f"| {100.0 * r['fac_gt0']:.1f}% "
            f"| {100.0 * r['new_gt0']:.1f}% "
            f"| {100.0 * r['ogtg_nn']:.1f}% |"
        )
    a("")
    a("Read: rate / next-pay / sched_vs_obs grow from ~0.9% to 3.1% as `created_at`")
    a("brings products onto the book. That is a **thin panel**, not a last-month still.")
    a("`f_util_snapshot` and `f_outstanding_gt_granted` are NaN until 2026-08 — Family F")
    a("blanks snapshot amounts before `AS_OF`. NORTH_STAR's '~1.7% last-month' sentence")
    a("mixes those two facts. The train panel itself is **unbalanced** "
      f"({p10['panel_min_co']} companies in 2024-09 → {p10['panel_max_co']} by 2026-05),")
    a("so the `fac>0` share can dip when new companies enter without a connected facility.")
    a("Within company, `f_n_facilities` never falls.")
    a("")
    a("## 3. Inventory is a panel; amounts are a still")
    a("")
    a("`created_at` decides which facilities exist as-of `period_end`. Counts and type")
    a("flags therefore move through 2024–2026. Granted / outstanding / rate / next-pay")
    a("on those rows are still the 2026-09-01 extract (Family F already names them snapshot).")
    a("")
    a(_md_table(
        inv["inv_stats"],
        [
            ("col", "column"),
            ("cov_cm", "cov CM"),
            ("share_gt0", "share > 0"),
            ("n_co_gt0", "n companies > 0"),
            ("acf1", "acf1"),
            ("acf3", "acf3"),
            ("acf6", "acf6"),
        ],
    ))
    a("")
    a(f"- `f_n_facilities` month-to-month: {inv['n_fac_rise']} rises, {inv['n_fac_drop']} drops, "
      f"{inv['n_fac_flat']} flats. A real as-of inventory path.")
    a(f"- `f_w_rate` is almost a company constant: {inv['rate_nuniq_1']} / "
      f"{inv['rate_nuniq_1'] + inv['rate_nuniq_gt1']} train schedule companies have one unique rate "
      f"({inv['rate_nuniq_gt1']} change when a second product is created).")
    a(f"- `f_months_to_next_pay` has median {inv['next_nuniq_median']:.0f} distinct values per company")
    a("  — the frozen `next_payment_date` minus a moving `period_end`.")
    a(f"- Months of non-null `f_w_rate` per train schedule company: median "
      f"{inv['months_per_sched_co_p50']:.0f}, max {inv['months_per_sched_co_max']} (of 24).")
    a("")
    a("## 4. Schedule companies vs observed financing flows")
    a("")
    a(f"Train companies: **{flow['n_train']}**. Ever a schedule row: **{flow['n_sched_train']}** "
      f"({100.0 * flow['sched_share']:.1f}%). Ever a repayment / interest / fee transaction: "
      f"**{flow['n_flow']}**.")
    a(f"Schedule ∩ `debt_repayment`: {flow['sched_and_rep']}. Schedule with **no** repayment: "
      f"{flow['sched_no_rep']}. Schedule ∩ (fee ∪ interest): {flow['sched_and_fee']}.")
    a("")
    a(_md_table(
        flow["rows"],
        [
            ("set", "set"),
            ("n_co", "n train companies"),
            ("share", "share of train"),
            ("n_and_sched", "∩ schedule"),
            ("sched_share", "share that have a schedule"),
        ],
    ))
    a("")
    a("A schedule row is rare. Observed debt service is common. The health-relevant")
    a("financing trail is already in Family F as `f_ds_r` / `f_fc_r`, not in the 40-row book.")
    a("")
    a("## 5. Accepted Y base rates — schedule vs not")
    a("")
    a("Company-month rates on train. `ever_sched` = company has any `debt_schedule_config` row.")
    a("`log1p(a_in3)` is the size control (AUROC vs Y; not a model).")
    a("")
    a(_md_table(
        [r for r in yx["cm_rows"] if not str(r["group"]).startswith("auroc")],
        [
            ("y", "Y"),
            ("group", "group"),
            ("n_labeled", "n labeled CM"),
            ("n_pos", "n pos"),
            ("rate", "base rate"),
            ("n_co", "n companies"),
        ],
    ))
    a("")
    a(_md_table(
        yx["aucs"],
        [
            ("y", "Y"),
            ("auc_ever", "AUROC ever-schedule"),
            ("auc_cm", "AUROC has-schedule this month"),
            ("auc_size", "AUROC log1p(a_in3)"),
        ],
    ))
    a("")
    a("Company-ever (max of the Y on that company):")
    a("")
    a(_md_table(
        yx["ever_rows"],
        [
            ("y", "Y"),
            ("group", "group"),
            ("n_co", "n companies with a label"),
            ("n_pos", "n ever-positive"),
            ("rate", "ever rate"),
        ],
    ))
    a("")
    a("Size tertiles (train company-median `log1p(a_in3)`):")
    a("")
    a(_md_table(
        yx["tert_rows"],
        [
            ("tertile", "tertile"),
            ("y", "Y"),
            ("group", "group"),
            ("n_labeled", "n labeled CM"),
            ("n_pos", "n pos"),
            ("rate", "base rate"),
            ("n_co", "n companies"),
        ],
    ))
    a("")
    a("Read: ever-schedule is not a classifier (AUROC ≈ 0.50). The naive Y4 gap")
    a("(18.9% schedule vs 13.3% no-schedule) is **not** schedule vs everyone — Y4 is")
    a(f"unlabeled on companies with no repayment ({p14['y4_norep_labeled']} labeled no-rep CM).")
    a(f"Fair split: schedule {100.0 * p14['y4_sched']:.1f}% vs repayment-no-schedule "
      f"{100.0 * p14['y4_rep']:.1f}%. Small-n bump, still CLOSE. Y2 is *lower* on schedule.")
    a("Company-ever Y4 59% vs 36% sits on n=27 labeled schedule companies — do not")
    a("promote a split that small. Size-controlled tertiles do not flip the PARK.")
    a("")
    a("## 6. `outstanding > granted`")
    a("")
    a(f"- `debt_products`: {ogtg['dp_true']} true / {ogtg['dp_false']} false / {ogtg['dp_null']} null "
      f"({ogtg['dp_co']} companies).")
    a(f"- `debt_schedule_config`: {ogtg['sched_true']} true ({ogtg['sched_co']} companies).")
    a(f"- Train last month `{ogtg['last_period']}`: `{ogtg['n_ogtg_lm_train']}` / `{ogtg['n_lm_train']}` "
      f"companies have `f_outstanding_gt_granted` = 1 ({100.0 * ogtg['ogtg_share_lm']:.1f}%).")
    a("- Last-month labeled Y counts (horizon → almost empty): "
      + ", ".join(f"`{k}`={v}" for k, v in ogtg["last_y_nn"].items())
      + ".")
    a("  Overlap is therefore company-ever Y among the 32 last-month OGTG companies.")
    a("")
    a(_md_table(
        ogtg["y_rows"],
        [
            ("y", "Y"),
            ("group", "group"),
            ("n_co", "n companies with a label"),
            ("n_pos", "n ever-positive"),
            ("rate", "ever rate"),
        ],
    ))
    a("")
    a("No material Y4 / Y9 enrichment. Same extract-still as Y10 `y10_ogtg_last_month` (PARK).")
    a("")
    a("## 7. Q6 honesty — lead time")
    a("")
    a(f"| item | value |")
    a("| --- | --- |")
    a(f"| next_payment_date range | {q6['min_next']} → {q6['max_next']} |")
    a(f"| last_payment_date range | {q6['min_last']} → {q6['max_last']} |")
    a(f"| next before / on-or-after / after 2026-09-01 | {q6['next_before_asof']} / {q6['next_on_or_after']} / {q6['next_after']} |")
    a(f"| last_payment after extract | {q6['last_after_asof']} |")
    a(f"| last_payment > next_payment | {q6['last_after_next']} "
      f"(all {p10['last_gt_next_sameday']} are the same calendar day — timestamp, not a broken book) |")
    a(f"| total_periods min / mean / max | {q6['min_tp']} / {q6['mean_tp']:.1f} / {q6['max_tp']} |")
    a(f"| granted_balance + / 0 / − | {q6['g_pos']} / {q6['g_zero']} / {q6['g_neg']} |")
    a(f"| f_months_to_next_pay train mean / p50 | {q6['months_mean']:.2f} / {q6['months_p50']:.2f} |")
    a(f"| share of non-null months_to_next < 0 | {100.0 * q6['share_neg']:.1f}% |")
    a(f"| median within-company month step | {q6['med_step']:.2f} |")
    a(f"| usable as lead time? | **no** |")
    a("")
    a("Amortising frequency / interest type (all 87 rows):")
    a("")
    a(_md_table(q6["freq"].to_dict("records"), [("amortising_frequency", "frequency"), ("n", "n")]))
    a("")
    a(_md_table(q6["itype"].to_dict("records"), [("interest_type", "interest"), ("n", "n")]))
    a("")
    a(q6["reason"])
    a("")
    a("## 8. Do groups share a schedule?")
    a("")
    a(f"{grp['n_groups']} groups have at least one schedule company ({grp['n_groups_train']} with a train member).")
    a(f"{grp['n_solo']} groups have exactly one schedule company; {grp['n_multi']} have 2–{grp['max_in_group']}.")
    a(f"Those groups contain {grp['n_sib']} companies of which {grp['n_sib_sched']} have a schedule.")
    a("Schedule is a company-level tag, not a group book. Siblings usually do **not** share it.")
    a("")
    a(f"Group-size counts (all / train-member groups): `{grp['sizes']}` / `{grp['train_sizes']}`.")
    a("")
    a("## 9. `f_new_facility` as a Q3 turning flag (no tree)")
    a("")
    a(f"Train: {nf['n_cm_gt0']} company-months ({100.0 * nf['share_gt0']:.1f}%) across "
      f"{nf['n_co_gt0']} companies have `f_new_facility` > 0.")
    a(f"Median company acf1 of the count: {nf['acf1']:.3f}; of the >0 flag: {nf['acf1_flag']:.3f} "
      f"(acf3 {nf['acf3']:.3f}, acf6 {nf['acf6']:.3f}).")
    a(f"AUROC of `log1p(a_in3)` vs the new-facility flag: {nf['size_auc_vs_new']:.3f} — "
      "larger firms connect more products (size-tilted, not a health signal).")
    a("")
    a("Single-feature AUROC of the new-facility flag vs accepted Ys (train, not a GBM):")
    a("")
    a(_md_table(
        nf["aucs"],
        [
            ("y", "Y"),
            ("auc_new", "AUROC f_new_facility>0"),
            ("auc_size", "AUROC log1p(a_in3)"),
            ("n_labeled", "n labeled"),
        ],
    ))
    a("")
    a("Same-month coincidence (descriptive):")
    a("")
    a(_md_table(
        nf["coin"],
        [
            ("y", "Y"),
            ("n_new_and_y", "n new ∧ Y"),
            ("rate_y_given_new", "rate Y | new"),
            ("rate_y_given_none", "rate Y | no new"),
        ],
    ))
    a("")
    a("`f_new_facility` is a usable *inventory clock* (rare, low persistence, already on the")
    a("feature-report keep list as a rare-event flag). It is **not** a new Y and it does")
    a("not rescue the snapshot rate / util columns. Dictionary `created_at` is *when the")
    a(f"product was connected*: {p10['sched_rep_before']}/{p10['sched_rep_n']} train schedule")
    a("companies with a repayment paid **before** their first schedule product was connected")
    a(f"(median lag {p10['sched_rep_lag_p50']:.0f} days). Among any debt_product + repayment, "
      f"{p10['prod_rep_before']}/{p10['prod_rep_n']} also paid first. "
      f"{p10['new_first_birth']}/{p10['new_cm']} new-facility months are a first-ever connected facility.")
    a("Leave it as X inventory with a **CAUTION** on Q3 (connection ≠ origination).")
    a("Do not build a turning label from `created_at` here (Y10 already parked `y10_new_loc_after_stress`).")
    a("")
    a("## 10. Settlement, granted holes, 39 vs 38")
    a("")
    a(f"- Settlement: **{p10['hit_bank']}/87** hit `banking_products` (all {p10['bank_same_co']} same-company checking).")
    a(f"  The other {p10['miss_bank']}: {p10['miss_to_debt']} settle to another of the company's **debt** products")
    a(f"  (LOC), {p10['orphan']} orphan `settlement_product_id`s (not in bank or debt).")
    if len(p10["dest_type"]):
        a("")
        a(_md_table(p10["dest_type"].to_dict("records"), [("dest_type", "non-bank settlement dest"), ("n", "n")]))
        a("")
    a(f"- `granted_balance` vs `abs(debt_products.granted)` match within €1 on **{p10['n_abs_match']}/87** rows.")
    a(f"  Sign convention: {p10['sched_pos_dp_neg']} schedule rows have positive granted and negative product granted.")
    a(f"  **{p10['n_zero_granted']}** schedule rows have `granted_balance = 0` but a large outstanding")
    a("  (Family F then emits scheduled installment 0, so `f_sched_vs_obs` can be 0 rather than null).")
    a("  Not a debt.py bug — the schedule extract is just incomplete. Still PARK the column.")
    a(f"- Raw train schedule companies {p10['n_raw_train']} vs store {p10['n_store_sched']}: missing `{p10['raw_not_store']}`.")
    a(f"  Post-extract schedule companies: `{p10['post_extract_co']}`.")
    a(f"- Schedule with no `debt_repayment` transaction (train): `{p10['no_rep_ids']}` (n={len(p10['no_rep_ids'])}).")
    a("  All five have fees; the book is not the flow.")
    a(f"- `next_payment_date` before `created_at`: {p10['next_before_created']} rows.")
    a("")
    a("## 11. `f_sched_vs_obs` clip and snapshot rate vs observed cost")
    a("")
    a(f"Train non-null `f_sched_vs_obs`: n={p11['n']}, mean={p11['mean']:.2f}, p50={p11['p50']:.2f},")
    a(f"share exactly 0 = {100.0 * p11['share0']:.1f}%, share clipped at 20 = {100.0 * p11['share20']:.1f}%.")
    a(f"p10={p11['p10']:.2f}, p90={p11['p90']:.2f}. A third of the thin panel sits on the clip —")
    a("the snapshot installment is not a historical expected path.")
    a("")
    a(f"Snapshot `f_w_rate` on the same months: p50={p11['rate_p50']:.3f} "
      f"(min {p11['rate_min']:.3f}, max {p11['rate_max']:.3f}) — a 0–11% annual rate, not bps.")
    a(f"Spearman vs observed `f_fc_r` = {p11['rho_rate_fc']:.3f}; vs `f_ds_r` = {p11['rho_rate_ds']:.3f}.")
    a(f"`f_sched_vs_obs` vs `f_ds_r` = {p11['rho_svo_ds']:.3f}; vs `f_fc_r` = {p11['rho_svo_fc']:.3f}.")
    a("The extract rate does not track observed financing-cost pressure. Another PARK nail.")
    a("")
    a("## 12. Group-fold single-feature AUROC (protocol.py, not a GBM)")
    a("")
    a(f"5 group folds, seed {FOLD_SEED}, {p12['n_train_co']} train companies. Score used as-is.")
    a("")
    a(_md_table(
        p12["rows"],
        [
            ("y", "Y"),
            ("feature", "feature"),
            ("cv", "CV AUROC"),
            ("sd", "sd"),
            ("n_folds", "n folds"),
        ],
    ))
    a("")
    a("Ever-schedule and same-month schedule stay at chance. `f_new_facility>0` stays at chance.")
    a("`log1p(a_in3)` is the only column here that moves (and it is the size control, not a debt signal).")
    a("Confirm CLOSE on schedule-as-Y / Q5.")
    a("")
    a("## 13. Flow vs book")
    a("")
    a(f"Train companies with a LOC on `debt_products`: {p13['n_loc']}. "
      f"Schedule ∩ LOC: {p13['n_sched_and_loc']} — the amortisation table is not the credit-line book.")
    a(f"Repayment without a schedule: {p13['n_rep_only']} companies. The observed `f_ds_r` trail is there.")
    a("")
    a(_md_table(
        p13["rows"],
        [
            ("group", "group"),
            ("n_co", "n companies"),
            ("n_cm", "n CM"),
            ("ds_p50", "f_ds_r p50"),
            ("ds_mean", "f_ds_r mean"),
            ("ds_gt0", "share f_ds_r>0"),
            ("fc_p50", "f_fc_r p50"),
            ("fc_gt0", "share f_fc_r>0"),
        ],
    ))
    a("")
    a("Schedule companies look like the repayment-no-schedule majority on `f_ds_r`, just smaller n.")
    a("The book does not add a second financing trail.")
    a("")
    a("## 14. Fair Y4 split + odd schedule rows")
    a("")
    a(_md_table(
        p14["rows"],
        [
            ("y", "Y"),
            ("group", "group"),
            ("n_labeled", "n labeled CM"),
            ("n_pos", "n pos"),
            ("rate", "base rate"),
            ("n_co", "n companies"),
        ],
    ))
    a("")
    a("Type mix / holes on the 87-row book (not a model input):")
    a("")
    a(_md_table(
        p14["odd"].to_dict("records"),
        [
            ("type", "type"),
            ("n", "n"),
            ("n_zero_granted", "granted=0"),
            ("n_ogtg", "ogtg"),
            ("n_huge", "granted>1e8"),
        ],
    ))
    a("")
    a("Guarantee / renting / one 300M loan sit on this table. COMP_0415 is a 300M loan with")
    a("no `debt_repayment` transaction. Do not treat the 87 rows as a clean amortising book.")
    a("")
    a("## 15. First-birth vs add-on; variable vs fixed")
    a("")
    a("Same-month Y rates when `f_new_facility>0` is a first connected facility vs an add-on.")
    a("")
    a(_md_table(
        p15["rows"],
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
    a("Add-on months have **zero** labeled Y3 recoveries (n=112). First-birth Y9 is a bit")
    a("higher (small n). Neither is a turning label. Reinforces Q3 CAUTION: connection clock.")
    a("")
    a("Snapshot rate type on schedule company-months (train):")
    a("")
    a(_md_table(
        p15["itype"],
        [
            ("interest_type", "interest"),
            ("n_cm", "n CM"),
            ("n_co", "n companies"),
            ("rate_p50", "f_w_rate p50"),
            ("fc_p50", "f_fc_r p50"),
            ("ds_p50", "f_ds_r p50"),
        ],
    ))
    a("")
    a("Variable vs fixed does not split observed `f_fc_r`. PARK `f_w_rate` stands.")
    a("")
    a("## PARK / CLOSE")
    a("")
    a("| object | decision | why |")
    a("| --- | --- | --- |")
    a("| `f_w_rate` as GBM X | **PARK** | 1.7% CM; almost a company constant; snapshot rate |")
    a("| `f_months_to_next_pay` as GBM X | **PARK** | 1.7% CM; countdown to a frozen, usually past, date |")
    a("| `f_sched_vs_obs` as GBM X | **PARK** | 1.7% CM; numerator is snapshot granted/total_periods |")
    a("| `f_util_snapshot` as GBM X or Y | **PARK** | last-month only; Y10 already parked utilisation |")
    a("| `f_outstanding_gt_granted` as Y | **PARK** | last-month extract flag; no Y4/Y9 trail |")
    a("| schedule presence as Y / Q5 | **CLOSE** | AUROC ~0.50; n=39 train companies |")
    a("| `next_payment_date` / `total_periods` as Q6 lead | **CLOSE** | last-book snapshot, not remaining tenor |")
    a("| `f_ds_r` / `f_fc_r` flow | **KEEP** | already Family F; this is the observed financing trail |")
    a("| `f_n_facilities` / `f_has_*` / `f_new_facility` | **KEEP** (inventory, Q3 CAUTION) | real connection panel; not origination |")
    a("")
    a("A *flow* of observed repayments vs expected installment is already sketched as")
    a("`f_sched_vs_obs`. The expected side is the snapshot book, so the ratio cannot be")
    a("a historical miss/hit path. Do not invent a new Y from it.")
    a("")
    if png_ok:
        a("## Plot")
        a("")
        a("- `analysis/outputs/debt_schedule_who_when.png` — train % with a schedule field")
        a("  vs % with any facility, and when schedule products were connected.")
        a("")
    a("## Closed in this module (no leftover cut)")
    a("")
    a("- Settlement join: 77 checking + 8 LOC + 2 orphan. Not a trail.")
    a("- 4 `granted_balance = 0` rows: leave `f_sched_vs_obs` as-is; column stays PARK.")
    a("- `f_n_facilities` drops: **0**. Inventory is monotone within company.")
    a("- 39 vs 38: COMP_1027 post-extract only. Not a bug.")
    a("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_registry(p1, cov, flow, yx, ogtg, q6, nf, p10, p11, p12, p13, p14, p15) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    ts = _utc_ts()
    rows = [
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "-", "model": "debt_schedule_qa",
            "split": "train", "metric": "n_sched_rows",
            "value": _fmt(p1["n_rows"]), "coverage": "1.0000",
            "notes": f"40 companies still; train_co={p1['n_train_co']}; hold_co={p1['n_hold_co']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "-", "model": "debt_schedule_qa",
            "split": "train", "metric": "f_w_rate_cov_cm",
            "value": _fmt(cov["rate_cov_cm"]), "coverage": _fmt(cov["rate_cov_cm"]),
            "notes": f"n_cm={cov['n_sched_cm']} n_co={cov['n_sched_co']} last_share={cov['rate_last_share']:.3f} NOT last-month-only",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "-", "model": "debt_schedule_qa",
            "split": "train", "metric": "f_util_snapshot_last_month_only",
            "value": "1" if cov["util_last_month_only"] else "0",
            "coverage": _fmt(cov["util_cov_cm"]),
            "notes": "PARK snapshot X/Y; utilisation already parked in Y10",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "-", "model": "debt_schedule_qa",
            "split": "train", "metric": "sched_co_share",
            "value": _fmt(flow["sched_share"]), "coverage": _fmt(flow["sched_share"]),
            "notes": f"n={flow['n_sched_train']}/{flow['n_train']}; flow_cos={flow['n_flow']}; sched_no_rep={flow['sched_no_rep']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "y4_ds_r_double", "model": "debt_schedule_qa",
            "split": "train", "metric": "auroc_ever_sched",
            "value": _fmt(next(r["auc_ever"] for r in yx["aucs"] if r["y"] == "y4_ds_r_double")),
            "coverage": _fmt(flow["sched_share"]),
            "notes": "single-feature diagnostic; not a GBM",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "-", "model": "debt_schedule_qa",
            "split": "train", "metric": "ogtg_last_month_n",
            "value": _fmt(ogtg["n_ogtg_lm_train"]), "coverage": _fmt(ogtg["ogtg_share_lm"]),
            "notes": f"debt_products true={ogtg['dp_true']} cos={ogtg['dp_co']}; schedule true={ogtg['sched_true']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "-", "model": "debt_schedule_qa",
            "split": "train", "metric": "q6_usable_lead",
            "value": "0", "coverage": _fmt(cov["rate_cov_cm"]),
            "notes": f"next_before_asof={q6['next_before_asof']}/87; months_p50={q6['months_p50']:.2f}; CLOSE",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "-", "model": "debt_schedule_qa",
            "split": "train", "metric": "f_new_facility_share_gt0",
            "value": _fmt(nf["share_gt0"]), "coverage": "1.0000",
            "notes": f"n_cm={nf['n_cm_gt0']} n_co={nf['n_co_gt0']} acf1={nf['acf1']:.3f} KEEP inventory Q3-CAUTION connection!=origination",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "-", "model": "debt_schedule_qa",
            "split": "train", "metric": "sched_created_after_repay_share",
            "value": _fmt(p10["sched_rep_before"] / p10["sched_rep_n"] if p10["sched_rep_n"] else float("nan")),
            "coverage": _fmt(flow["sched_share"]),
            "notes": f"{p10['sched_rep_before']}/{p10['sched_rep_n']} paid before first schedule connected; raw_not_store={p10['raw_not_store']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "-", "model": "debt_schedule_qa",
            "split": "train", "metric": "f_sched_vs_obs_clip20_share",
            "value": _fmt(p11["share20"]), "coverage": _fmt(cov["rate_cov_cm"]),
            "notes": f"p50={p11['p50']:.3f} share0={p11['share0']:.3f} rho_rate_fc={p11['rho_rate_fc']:.3f} PARK",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "y4_ds_r_double", "model": "debt_schedule_qa",
            "split": "train_cv", "metric": "auroc_ever_sched_gfold",
            "value": _fmt(next(r["cv"] for r in p12["rows"] if r["y"] == "y4_ds_r_double" and r["feature"] == "ever_sched")),
            "coverage": _fmt(flow["sched_share"]),
            "notes": "protocol group-fold 5; CLOSE schedule-as-Y",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "-", "model": "debt_schedule_qa",
            "split": "train", "metric": "n_rep_no_schedule",
            "value": _fmt(p13["n_rep_only"]), "coverage": _fmt(p13["n_rep_only"] / flow["n_train"] if flow["n_train"] else float("nan")),
            "notes": f"sched_and_loc={p13['n_sched_and_loc']}/{p13['n_loc']} loc; flow exists without the book",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "y4_ds_r_double", "model": "debt_schedule_qa",
            "split": "train", "metric": "y4_rate_sched_vs_rep_only",
            "value": _fmt(p14["y4_sched"]),
            "coverage": _fmt(flow["sched_share"]),
            "notes": f"sched={p14['y4_sched']:.4f} vs rep_no_sched={p14['y4_rep']:.4f}; no_rep labeled={p14['y4_norep_labeled']}; CLOSE",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "y3_recover_cash_6m", "model": "debt_schedule_qa",
            "split": "train", "metric": "y3_rate_new_addon",
            "value": _fmt(next(r["rate"] for r in p15["rows"] if r["y"] == "y3_recover_cash_6m" and r["kind"] == "add_on")),
            "coverage": "1.0000",
            "notes": "add-on new_facility months; 0 Y3 recoveries labeled; Q3 CAUTION",
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
        print("registry: nothing new to append")
        return
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in fresh:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"appended {len(fresh)} registry rows")


def main() -> int:
    print(f"debt_schedule_qa start seed={FOLD_SEED} holdout=72")
    if not STORE.exists():
        raise FileNotFoundError(STORE)
    if not TARGETS.exists():
        raise FileNotFoundError(TARGETS)
    con = connect()
    store = load_store()
    targets = load_targets()

    print("pass 1 raw + join")
    p1 = pass1_raw(con)
    print(f"  rows={p1['n_rows']} cos={p1['n_co']} still_40_87={p1['still_40_87']}")

    print("pass 2 month coverage")
    cov = pass2_month_coverage(store)
    print(" ", cov["verdict"][:160])

    print("pass 3 inventory panel")
    inv = pass3_inventory(con, store)
    print(f"  inventory_is_panel={inv['inventory_is_panel']} fac_rise={inv['n_fac_rise']} fac_drop={inv['n_fac_drop']}")

    print("pass 4 flow overlap")
    flow = pass4_flow_overlap(con)
    print(f"  sched_train={flow['n_sched_train']}/{flow['n_train']} flow={flow['n_flow']} no_rep={flow['sched_no_rep']}")

    print("pass 5 Y base rates")
    yx = pass5_y_rates(store, targets, flow["sched_train"])
    for r in yx["aucs"]:
        print(f"  {r['y']} auc_ever={r['auc_ever']:.3f} auc_size={r['auc_size']:.3f}")

    print("pass 6 OGTG")
    ogtg = pass6_ogtg(con, store, targets)
    print(f"  last-month ogtg train={ogtg['n_ogtg_lm_train']}/{ogtg['n_lm_train']}")

    print("pass 7 Q6 lead")
    q6 = pass7_q6(con, store)
    print(f"  usable_lead={q6['usable_lead']} next_before={q6['next_before_asof']} months_p50={q6['months_p50']:.2f}")

    print("pass 8 groups")
    grp = pass8_groups(con, flow["sched_train"])
    print(f"  groups={grp['n_groups']} multi={grp['n_multi']} sib={grp['n_sib_sched']}/{grp['n_sib']}")

    print("pass 9 new_facility")
    nf = pass9_new_facility(store, targets)
    print(f"  new>0 {nf['n_cm_gt0']} cm / {nf['n_co_gt0']} co acf1={nf['acf1']:.3f}")

    print("pass 10 settlement + created_at clock")
    p10 = pass10_settlement_clock(con, store, flow["sched_train"])
    print(
        f"  settle bank={p10['hit_bank']} orphan={p10['orphan']} "
        f"raw_not_store={p10['raw_not_store']} paid_before={p10['sched_rep_before']}/{p10['sched_rep_n']}"
    )

    print("pass 11 sched_vs_obs + rate vs flow")
    p11 = pass11_ratio_rate(store)
    print(f"  clip20={p11['share20']:.3f} rho_rate_fc={p11['rho_rate_fc']:.3f}")

    print("pass 12 group-fold single-feature")
    p12 = pass12_groupfold(con, store, targets, flow["sched_train"])
    for r in p12["rows"]:
        if r["feature"] in ("ever_sched", "log1p_a_in3"):
            print(f"  {r['y']} {r['feature']} cv={r['cv']:.3f} sd={r['sd']:.3f}")

    print("pass 13 flow vs book")
    p13 = pass13_flow_vs_book(con, store)
    print(f"  rep_only={p13['n_rep_only']} sched∩loc={p13['n_sched_and_loc']}/{p13['n_loc']}")

    print("pass 14 fair Y4 + odd rows")
    p14 = pass14_fair_y4(con, store, targets)
    print(
        f"  y4 sched={p14['y4_sched']:.3f} rep_only={p14['y4_rep']:.3f} "
        f"no_rep_labeled={p14['y4_norep_labeled']}"
    )

    print("pass 15 first-birth vs add-on")
    p15 = pass15_birth_rate(con, store, targets)
    y3a = next(r for r in p15["rows"] if r["y"] == "y3_recover_cash_6m" and r["kind"] == "add_on")
    print(f"  y3 add_on rate={y3a['rate']} n_lab={y3a['n_labeled']}")

    png_ok = make_png(cov, inv)
    print(f"png={OUT_PNG if png_ok else 'skipped'}")

    write_md(p1, cov, inv, flow, yx, ogtg, q6, grp, nf, p10, p11, p12, p13, p14, p15, png_ok)
    print(f"wrote {OUT_MD}")
    append_registry(p1, cov, flow, yx, ogtg, q6, nf, p10, p11, p12, p13, p14, p15)
    con.close()
    print("debt_schedule_qa done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
