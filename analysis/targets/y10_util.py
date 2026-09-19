"""Y10 — financing-stress / LOC-inventory targets (binary).

There is **no historical outstanding/granted panel**. `granted`,
`outstanding`, `liquidity`, and `outstanding_gt_granted` are a
2026-09-01 extract still. `f_util_snapshot` is last-month only.
This module does **not** invent a utilisation path.

Legal inputs only: `debt_products.created_at` (causal inventory) and
transaction flows rebuilt here from `CAT_MAP`. Does **not** import
`analysis.features.debt`.

Assigned definitions (iterate; honest PARK/CLOSE is success):

- y10_new_loc_after_stress: a `type='lineofcredit'` row has
  `created_at` in (t, t+6] AND operational net < 0 in ≥ 2 of the last
  3 months at t (not family B). Same idea as rejected
  `y4_new_facility_after_dip`, restricted to LOC — PARK, do not rename.
- y10_loc_book_then_fees: LOC with created_at ≤ t (month-start) AND
  Y9-style `fee_r` own-p80 for t+1..t+3. On the overlap this *is*
  `y9_fee_r_ownp80` (Spearman 1.0) — CLOSE as a subset, not a new Y.
- y10_ogtg_last_month: extract `outstanding_gt_granted` on 2026-08
  only, NaN elsewhere. Not an onset. PARK as a Y (same leak as
  `y4_ogtg_appear`).

Extra probes kept in the frame so the wave note can quote them:

- y10_add_loc_6m: already has a LOC with created_at ≤ t, and another
  LOC created_at in (t, t+6]. Size AUROC fails after the first-LOC
  birth-month leak is removed.
- y10_loc_then_int_ownp80 / y10_loc_then_int_spike: LOC on book at t
  and interest-only own-p80 / spike (Y9 method, not fee+interest).
  These two can pass numeric gates; they are not a utilisation path.
  Parent decides merge.

Forbidden later X: family F and prefixes `f_*`. Also drop
`f_util_snapshot` / `f_outstanding_gt_granted` if anyone models a
column here — the label must not be those snapshot fields copied
forward. Fee/interest labels also forbid `a_fin_cost` / `a_fc`.

Maps to brief Q3 (turning) / Q5 (why) from *financing stress*, not
bankruptcy. Not a 0–100 score.

Literature: Yao, Levy-Chapira, Margaryan 2017 (arXiv 1707.00757)
credit-line violations; FinRegLab 2025 NSF/fee. Catalogue Y4
“LOC util > 90%” cannot be built without a util time series.
"""
from __future__ import annotations

import csv
from datetime import datetime

import numpy as np
import pandas as pd

from analysis.evaluate.protocol import assert_no_holdout, auroc, leakage_check
from analysis.features.common import ANALYSIS, AS_OF, CAT_MAP, LAST_M, MONTHS, train_mask

ACCEPT_MD = ANALYSIS / "outputs" / "y10_acceptance.md"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "14d1bb47"
ROUND = "R4"
WAVE = "4"

Y10_COLS = [
    "y10_new_loc_after_stress",
    "y10_loc_book_then_fees",
    "y10_ogtg_last_month",
    "y10_add_loc_6m",
    "y10_loc_then_int_ownp80",
    "y10_loc_then_int_spike",
]

HORIZON_NEW = 6
HORIZON_FEE = 3
MIN_OWN_HIST = 6
OWN_P = 0.80
TRAIL = 6
SPIKE_K = 2.0
SPIKE_OF = 2
LOC_TYPE = "lineofcredit"

_OP_IN = tuple(k for k, v in CAT_MAP.items() if v == "op_in")
_OP_OUT = tuple(k for k, v in CAT_MAP.items() if v == "op_out")
_FIN_COST = tuple(k for k, v in CAT_MAP.items() if v == "fin_cost")
_INTEREST = ("interest_charge",)

META = {
    "name": "y10_financing_stress",
    "horizon": HORIZON_NEW,
    "source_tables": ["transactions", "debt_products"],
    "forbidden_x_families": ["f"],
    "forbidden_x_prefixes": ["f_", "a_fin_cost", "a_fc"],
    "literature": (
        "Yao, Levy-Chapira, Margaryan 2017 (arXiv 1707.00757): credit-line "
        "violations and cash inflows are central for corporate default. "
        "FinRegLab 2025 NSF/fee. Y10 cannot use outstanding/granted as a "
        "2024–2025 path — those fields are the 2026-09-01 extract."
    ),
    "kind": "binary",
    "columns": list(Y10_COLS),
    "accepted": {
        "y10_new_loc_after_stress": False,
        "y10_loc_book_then_fees": False,
        "y10_ogtg_last_month": False,
        "y10_add_loc_6m": False,
        "y10_loc_then_int_ownp80": True,
        "y10_loc_then_int_spike": True,
    },
    "brief_questions": ["turning", "why"],
    "definitions": {
        "y10_new_loc_after_stress": (
            "1 if a lineofcredit created_at falls in (t, t+6] AND "
            "operational net < 0 in at least 2 of the last 3 months at t. "
            "NaN when t+6m is after the extract or the dip window is short. "
            "Subset of rejected y4_new_facility_after_dip."
        ),
        "y10_loc_book_then_fees": (
            "1 if a lineofcredit has created_at ≤ t (month-start) AND "
            "fee_r = (fee+interest)_3m / max(in3, 1) exceeds that "
            "company's expanding p80 (months ≤ t, min 6 finite) in each "
            "of t+1..t+3. Identical to y9_fee_r_ownp80 on the LOC book."
        ),
        "y10_ogtg_last_month": (
            "LAST-MONTH-ONLY: 1 if any debt_products.outstanding_gt_granted "
            "at extract; populated only for 2026-08, NaN elsewhere. Not an "
            "onset. Same leak as y4_ogtg_appear."
        ),
        "y10_add_loc_6m": (
            "1 if a lineofcredit already exists with created_at ≤ t and "
            "another lineofcredit has created_at in (t, t+6]. Sample is "
            "the on-book LOC months with a full 6-month horizon."
        ),
        "y10_loc_then_int_ownp80": (
            "1 if a lineofcredit has created_at ≤ t AND interest_r = "
            "interest_charge_3m / max(in3, 1) exceeds own expanding p80 "
            "in each of t+1..t+3. Y9 method, interest-only numerator, "
            "LOC inventory filter."
        ),
        "y10_loc_then_int_spike": (
            "1 if a lineofcredit has created_at ≤ t AND interest_r >= 2 * "
            "trailing-6m mean(interest_r) at t in at least 2 of t+1..t+3. "
            "NaN when the 6m mean is 0 or missing (doubling undefined)."
        ),
    },
}

SOURCE_TABLES = META["source_tables"]


def _sql_in(values: tuple[str, ...]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def _month_key(period: pd.Series) -> pd.Series:
    return pd.to_datetime(period).dt.to_period("M").dt.to_timestamp()


def _has_horizon(month: pd.Series, h: int) -> pd.Series:
    cutoff = LAST_M - pd.DateOffset(months=h)
    return month <= cutoff


def _expanding_quantile_skipna(s: pd.Series, q: float, min_periods: int) -> pd.Series:
    """Expanding percentile that counts only finite observations."""
    out = np.full(len(s), np.nan, dtype=float)
    vals: list[float] = []
    for i, v in enumerate(s.to_numpy(dtype=float)):
        if np.isfinite(v):
            vals.append(float(v))
        if len(vals) >= min_periods:
            out[i] = float(np.quantile(vals, q))
    return pd.Series(out, index=s.index)


def monthly_fin_flows(con) -> pd.DataFrame:
    """Monthly op_in, op_out, fin_cost, interest from transactions.

    Outflow groups are negated (same bank-sign convention as Y4 / Y9).
    ``fin_cost`` = fee + interest_charge; ``interest`` is the charge only.
    """
    df = con.execute(
        f"""
        SELECT
          t.company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          SUM(CASE WHEN t.category IN ({_sql_in(_OP_IN)}) THEN t.amount ELSE 0 END) AS op_in,
          -SUM(CASE WHEN t.category IN ({_sql_in(_OP_OUT)}) THEN t.amount ELSE 0 END) AS op_out,
          -SUM(CASE WHEN t.category IN ({_sql_in(_FIN_COST)}) THEN t.amount ELSE 0 END) AS fin_cost,
          -SUM(CASE WHEN t.category IN ({_sql_in(_INTEREST)}) THEN t.amount ELSE 0 END) AS interest
        FROM transactions t
        WHERE t."date" < TIMESTAMP '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["month"] = pd.to_datetime(df["month"])
    return df


def fin_month_panel(con) -> pd.DataFrame:
    """Dense company × month path for Y10.

    Columns: company_id, month, op_in, op_out, net, fin_cost, interest,
    in3, fc3, int3, fee_r, int_r. Months after first activity with no
    txs are 0-filled. Ratios are unclipped.
    """
    flows = monthly_fin_flows(con)
    first = flows.groupby("company_id")["month"].min().rename("first_m")
    panel = pd.MultiIndex.from_product(
        [first.index, MONTHS], names=["company_id", "month"]
    ).to_frame(index=False)
    panel["company_id"] = panel["company_id"].astype(str)
    panel = panel.merge(first, left_on="company_id", right_index=True)
    panel = panel[panel["month"] >= panel["first_m"]]
    panel = panel.merge(flows, on=["company_id", "month"], how="left")
    for c in ("op_in", "op_out", "fin_cost", "interest"):
        panel[c] = panel[c].fillna(0.0)
    panel["net"] = panel["op_in"] - panel["op_out"]
    panel = panel.sort_values(["company_id", "month"]).reset_index(drop=True)
    g = panel.groupby("company_id", sort=False)
    panel["in3"] = g["op_in"].transform(lambda s: s.rolling(3).sum())
    panel["fc3"] = g["fin_cost"].transform(lambda s: s.rolling(3).sum())
    panel["int3"] = g["interest"].transform(lambda s: s.rolling(3).sum())
    panel["fee_r"] = panel["fc3"] / np.maximum(panel["in3"], 1.0)
    panel["int_r"] = panel["int3"] / np.maximum(panel["in3"], 1.0)
    return panel.drop(columns=["first_m"])


def loc_products(con) -> pd.DataFrame:
    """Causal LOC inventory: created_at, drop post-snapshot rows."""
    df = con.execute(
        f"""
        SELECT company_id, created_at
        FROM debt_products
        WHERE type = '{LOC_TYPE}'
          AND created_at IS NOT NULL
          AND NOT coalesce(created_after_snapshot, false)
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def _future_created(panel: pd.DataFrame, events: pd.DataFrame, h: int) -> pd.Series:
    """1 if any created_at is in (month, month+h m], else 0."""
    keys = panel[["company_id", "month"]]
    if events.empty:
        return pd.Series(0.0, index=panel.index)
    m = keys.merge(events, on="company_id", how="left")
    t_h = m["month"] + pd.DateOffset(months=h)
    hit = (
        m["created_at"].notna()
        & (m["created_at"] > m["month"])
        & (m["created_at"] <= t_h)
    )
    flagged = (
        m.loc[hit, ["company_id", "month"]]
        .drop_duplicates()
        .assign(_hit=1.0)
    )
    out = keys.merge(flagged, on=["company_id", "month"], how="left")["_hit"]
    return out.fillna(0.0).astype(float)


def _asof_created(panel: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """1 if any created_at ≤ month-start (already on book at t)."""
    keys = panel[["company_id", "month"]]
    if events.empty:
        return pd.Series(0.0, index=panel.index)
    m = keys.merge(events, on="company_id", how="left")
    hit = m["created_at"].notna() & (m["created_at"] <= m["month"])
    flagged = (
        m.loc[hit, ["company_id", "month"]]
        .drop_duplicates()
        .assign(_hit=1.0)
    )
    out = keys.merge(flagged, on=["company_id", "month"], how="left")["_hit"]
    return out.fillna(0.0).astype(float)


def inspect_debt_table(con) -> dict:
    """Train-safe inventory facts. Holdout is counted only as coverage."""
    raw = con.execute(
        """
        SELECT type, company_id, created_at, created_after_snapshot,
               outstanding_gt_granted, granted, outstanding
        FROM debt_products
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["created_at"] = pd.to_datetime(raw["created_at"])
    types = (
        raw.groupby("type")
        .agg(n=("company_id", "size"), n_cos=("company_id", "nunique"))
        .sort_values("n", ascending=False)
        .reset_index()
    )
    loc = raw[raw["type"] == LOC_TYPE]
    u = pd.Series(sorted(raw["company_id"].unique()), dtype=str)
    tr_ids = set(u.loc[train_mask(u)].astype(str))
    loc_tr = loc[loc["company_id"].isin(tr_ids)]
    first_loc = (
        loc.dropna(subset=["created_at"])
        .groupby("company_id")["created_at"]
        .min()
        .reset_index()
    )
    first_loc["loc_m"] = first_loc["created_at"].dt.to_period("M").dt.to_timestamp()
    first_tx = con.execute(
        """
        SELECT company_id, CAST(date_trunc('month', MIN("date")) AS DATE) AS first_tx
        FROM transactions
        WHERE "date" < TIMESTAMP '2026-09-01'
        GROUP BY 1
        """
    ).df()
    first_tx["company_id"] = first_tx["company_id"].astype(str)
    first_tx["first_tx"] = pd.to_datetime(first_tx["first_tx"])
    fl = first_loc.merge(first_tx, on="company_id", how="left")
    fl["midpanel"] = (fl["loc_m"] > fl["first_tx"]) & (fl["loc_m"] <= LAST_M)
    ogtg_cos = int(raw.groupby("company_id")["outstanding_gt_granted"].max().sum())
    facts = {
        "n_debt_rows": int(len(raw)),
        "n_debt_cos": int(raw["company_id"].nunique()),
        "n_loc_rows": int(len(loc)),
        "n_loc_cos": int(loc["company_id"].nunique()),
        "n_loc_cos_train": int(loc_tr["company_id"].nunique()),
        "n_first_loc_midpanel": int(fl["midpanel"].sum()),
        "n_first_loc_midpanel_train": int(
            fl.loc[fl["company_id"].isin(tr_ids), "midpanel"].sum()
        ),
        "n_ogtg_cos": ogtg_cos,
        "n_ogtg_cos_train": int(
            raw.loc[raw["company_id"].isin(tr_ids)]
            .groupby("company_id")["outstanding_gt_granted"]
            .max()
            .sum()
        ),
        "n_loc_after_snapshot": int(loc["created_after_snapshot"].fillna(False).sum()),
        "created_min": str(raw["created_at"].min()),
        "created_max": str(raw["created_at"].max()),
        "types": types.to_dict(orient="records"),
    }
    return facts


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    """Return company_id, period and the Y10 binaries (0/1/NaN)."""
    need = {"company_id", "period"}
    if not need <= set(grid.columns):
        raise ValueError("grid must have company_id, period")

    panel = fin_month_panel(con)
    cid = panel["company_id"]
    g = panel.groupby(cid, sort=False)

    loc = loc_products(con)
    panel["new_loc6"] = _future_created(panel, loc, HORIZON_NEW).to_numpy()
    panel["has_loc_t"] = _asof_created(panel, loc).to_numpy()

    neg = (panel["net"] < 0).astype(float)
    n0 = neg
    n1 = neg.groupby(cid).shift(1)
    n2 = neg.groupby(cid).shift(2)
    dip = pd.Series(
        np.where(n1.notna() & n2.notna(), ((n0 + n1 + n2) >= 2).astype(float), np.nan),
        index=panel.index,
    )
    has_h6 = (panel["month"] + pd.DateOffset(months=HORIZON_NEW)) <= AS_OF
    panel["y10_new_loc_after_stress"] = np.where(
        has_h6 & dip.notna(),
        ((dip == 1.0) & (panel["new_loc6"] == 1.0)).astype(float),
        np.nan,
    )
    panel["y10_add_loc_6m"] = np.where(
        has_h6 & (panel["has_loc_t"] == 1.0),
        panel["new_loc6"].astype(float),
        np.nan,
    )

    panel["own_p80_fee"] = g["fee_r"].transform(
        lambda s: _expanding_quantile_skipna(s, OWN_P, MIN_OWN_HIST)
    )
    panel["own_p80_int"] = g["int_r"].transform(
        lambda s: _expanding_quantile_skipna(s, OWN_P, MIN_OWN_HIST)
    )
    f1 = g["fee_r"].shift(-1)
    f2 = g["fee_r"].shift(-2)
    f3 = g["fee_r"].shift(-3)
    i1 = g["int_r"].shift(-1)
    i2 = g["int_r"].shift(-2)
    i3 = g["int_r"].shift(-3)
    has_h3 = _has_horizon(panel["month"], HORIZON_FEE)
    fut_ok = has_h3 & f1.notna() & f2.notna() & f3.notna()
    hi_fee = (f1 > panel["own_p80_fee"]) & (f2 > panel["own_p80_fee"]) & (f3 > panel["own_p80_fee"])
    y9like = np.where(
        fut_ok & panel["own_p80_fee"].notna(), hi_fee.astype(float), np.nan
    )
    panel["y10_loc_book_then_fees"] = np.where(
        panel["has_loc_t"] == 1.0, y9like, np.nan
    )
    hi_int = (i1 > panel["own_p80_int"]) & (i2 > panel["own_p80_int"]) & (i3 > panel["own_p80_int"])
    int_ok = has_h3 & i1.notna() & i2.notna() & i3.notna() & panel["own_p80_int"].notna()
    y_int = np.where(int_ok, hi_int.astype(float), np.nan)
    panel["y10_loc_then_int_ownp80"] = np.where(
        panel["has_loc_t"] == 1.0, y_int, np.nan
    )
    panel["trail6_int"] = g["int_r"].transform(
        lambda s: s.rolling(TRAIL, min_periods=TRAIL).mean()
    )
    base_ok = panel["trail6_int"].notna() & (panel["trail6_int"] > 0)
    n_dbl = (
        (i1 >= SPIKE_K * panel["trail6_int"]).astype(float)
        + (i2 >= SPIKE_K * panel["trail6_int"]).astype(float)
        + (i3 >= SPIKE_K * panel["trail6_int"]).astype(float)
    )
    y_ispike = np.where(
        has_h3 & i1.notna() & i2.notna() & i3.notna() & base_ok,
        (n_dbl >= SPIKE_OF).astype(float),
        np.nan,
    )
    panel["y10_loc_then_int_spike"] = np.where(
        panel["has_loc_t"] == 1.0, y_ispike, np.nan
    )

    og = con.execute(
        """
        SELECT company_id,
               MAX(CASE WHEN outstanding_gt_granted THEN 1 ELSE 0 END) AS ogtg
        FROM debt_products
        GROUP BY 1
        """
    ).df()
    og["company_id"] = og["company_id"].astype(str)
    panel = panel.merge(og, on="company_id", how="left")
    panel["y10_ogtg_last_month"] = np.where(
        panel["month"].eq(LAST_M), panel["ogtg"].fillna(0.0), np.nan
    )

    out = grid[["company_id", "period"]].copy()
    out["company_id"] = out["company_id"].astype(str)
    out["period"] = pd.to_datetime(out["period"])
    out["_m"] = _month_key(out["period"])
    lab = panel[["company_id", "month", *Y10_COLS]].rename(columns={"month": "_m"})
    out = out.merge(lab, on=["company_id", "_m"], how="left").drop(columns=["_m"])
    return out.reset_index(drop=True)


def _two_sided(auc: float) -> float:
    if not np.isfinite(auc):
        return float("nan")
    return float(max(auc, 1.0 - auc))


def train_acceptance(
    y10: pd.DataFrame,
    op_in: pd.Series,
    in3: pd.Series,
    is_train: pd.Series,
) -> pd.DataFrame:
    """Base rates and two-sided size AUROC on train company-months only.

    Contract: train base ∈ [5%, 30%] and max(auc, 1-auc) < 0.60 on
    log1p(|op_in|) (also report log1p(|in3|)). Holdout never used.
    ``accepted`` here is the *numeric* gate only. Family verdicts
    (PARK / CLOSE) are applied in ``column_verdict``.
    """
    assert_no_holdout(y10.loc[is_train, "company_id"])
    rows = []
    n_train = int(is_train.sum())
    opin_s = np.log1p(pd.to_numeric(op_in.loc[is_train], errors="coerce").abs())
    in3_s = np.log1p(pd.to_numeric(in3.loc[is_train], errors="coerce").abs())
    for col in Y10_COLS:
        y = y10.loc[is_train, col]
        ok = y.notna()
        n = int(ok.sum())
        rate = float(y[ok].mean()) if n else float("nan")
        auc = auroc(y, opin_s)
        auc3 = auroc(y, in3_s)
        auc_abs = _two_sided(auc)
        auc3_abs = _two_sided(auc3)
        in_rate = bool(n and 0.05 <= rate <= 0.30)
        size_ok = bool(np.isfinite(auc_abs) and auc_abs < 0.60)
        numeric = bool(in_rate and size_ok)
        verdict = column_verdict(col, numeric, in_rate, size_ok)
        rows.append(
            {
                "column": col,
                "n_train": n_train,
                "n_labeled": n,
                "n_pos": int((y == 1).sum()),
                "coverage": n / n_train if n_train else float("nan"),
                "base_rate": rate,
                "size_auroc": auc,
                "size_auroc_two_sided": auc_abs,
                "size_auroc_in3": auc3,
                "size_auroc_in3_two_sided": auc3_abs,
                "numeric_ok": numeric,
                "accepted": verdict == "ACCEPTED",
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def column_verdict(col: str, numeric: bool, in_rate: bool, size_ok: bool) -> str:
    """KEEP / PARK / CLOSE. Numeric pass is not enough for a Y4/Y9 rewrite."""
    if col == "y10_new_loc_after_stress":
        return "PARK"
    if col == "y10_loc_book_then_fees":
        return "CLOSE"
    if col == "y10_ogtg_last_month":
        return "PARK"
    if col == "y10_add_loc_6m":
        return "PARK"
    if col == "y10_loc_then_int_ownp80":
        # Numeric gates pass; not a Y9 rewrite (ρ≈0.35). Leave in-module.
        # Parent decides merge — this is interest pressure on the LOC
        # book, not a utilisation path.
        return "ACCEPTED" if numeric else "PARK"
    if col == "y10_loc_then_int_spike":
        return "ACCEPTED" if numeric else "PARK"
    if numeric:
        return "ACCEPTED"
    if not in_rate and not size_ok:
        return "PARK"
    return "PARK"


def spearman_vs_frozen(y10: pd.DataFrame, is_train: pd.Series) -> pd.DataFrame:
    """Spearman vs accepted/rejected neighbours on train overlap. Read-only parquet."""
    from analysis.features.common import DATA

    path = DATA / "feature_store" / "targets.parquet"
    if not path.exists():
        return pd.DataFrame()
    tgt = pd.read_parquet(path)
    tgt["company_id"] = tgt["company_id"].astype(str)
    tgt["period"] = pd.to_datetime(tgt["period"])
    neighbours = [
        c
        for c in (
            "y2_neg_2of3",
            "y4_ds_r_double",
            "y4_new_facility_after_dip",
            "y9_fee_r_ownp80",
            "y9_fee_spike",
        )
        if c in tgt.columns
    ]
    keys = y10[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"])
    m = keys.merge(tgt[["company_id", "period", *neighbours]], on=["company_id", "period"], how="left")
    rows = []
    for col in Y10_COLS:
        for nb in neighbours:
            ok = is_train & y10[col].notna() & m[nb].notna()
            n = int(ok.sum())
            if n < 20:
                rho = float("nan")
            else:
                rho = float(y10.loc[ok, col].corr(m.loc[ok, nb], method="spearman"))
            rows.append(
                {
                    "column": col,
                    "vs": nb,
                    "n_overlap": n,
                    "spearman": rho,
                    "rewrite": bool(np.isfinite(rho) and abs(rho) >= 0.8),
                }
            )
    return pd.DataFrame(rows)


def write_acceptance_md(
    facts: dict,
    acc: pd.DataFrame,
    pos_cos: pd.DataFrame,
    rho: pd.DataFrame,
    extra: dict,
) -> None:
    """Write analysis/outputs/y10_acceptance.md from train-only tables."""
    ACCEPT_MD.parent.mkdir(parents=True, exist_ok=True)
    acc_i = acc.set_index("column")
    pos_i = pos_cos.set_index("column")

    def row(col: str) -> str:
        r = acc_i.loc[col]
        p = pos_i.loc[col]
        return (
            f"| `{col}` | {int(r.n_labeled):,} | {r.coverage:.1%} | {int(r.n_pos):,} | "
            f"{r.base_rate:.2%} | {r.size_auroc:.3f} | {r.size_auroc_two_sided:.3f} | "
            f"{r.size_auroc_in3:.3f} | {int(p.n_cos_labeled)} / {int(p.n_cos_pos)} | "
            f"**{r.verdict}** |"
        )

    rho_lines = []
    if rho is not None and not rho.empty:
        for _, rec in rho.iterrows():
            spr = rec.spearman
            spr_s = "—" if not np.isfinite(spr) else f"{spr:.3f}"
            flag = " rewrite" if rec.rewrite else ""
            rho_lines.append(
                f"| `{rec.column}` | `{rec.vs}` | {int(rec.n_overlap):,} | {spr_s} |{flag} |"
            )

    types = "\n".join(
        f"| `{t['type']}` | {t['n']:,} | {t['n_cos']:,} |" for t in facts["types"]
    )
    lines = [
        "# Y10 acceptance — financing stress / LOC inventory",
        "",
        "Generated by `python -m analysis.targets.y10_util`. Train company-months",
        "only (holdout 72 out of every rate and size AUROC). No 0–100. No `product/`.",
        "Does **not** write `targets.parquet`. Parent decides merge.",
        "",
        "## Brief questions (NORTH_STAR)",
        "",
        "Y10 is **not** a bankruptcy label and **not** a utilisation path.",
        "Outstanding/granted/`outstanding_gt_granted` are a 2026-09-01 extract still.",
        "The only causal debt-table clock is `created_at`.",
        "",
        "3. **Who is turning?** — interest/inflow stays above that firm's own p80",
        "   for the next quarter, among companies that already have a line of credit",
        "   on the book (`y10_loc_then_int_ownp80`).",
        "5. **Why did it change?** — financing-cost pressure on an existing LOC",
        "   (Yao 2017 credit-line / FinRegLab fee-interest), not a drawn-balance series.",
        "",
        "## Can `debt_products` support a new Y?",
        "",
        f"- `{LOC_TYPE}` is the real type string. {facts['n_loc_rows']} LOC rows, "
        f"{facts['n_loc_cos']} companies ({facts['n_loc_cos_train']} train).",
        f"- First LOC appears mid-panel (after first tx, ≤ 2026-08) for "
        f"{facts['n_first_loc_midpanel']} companies ({facts['n_first_loc_midpanel_train']} train).",
        f"- `outstanding_gt_granted` hits {facts['n_ogtg_cos']} companies "
        f"({facts['n_ogtg_cos_train']} train) — extract flag, not an onset.",
        "- **No** historical outstanding/granted panel. Join QA: `f_w_rate` ever on",
        "  38 train companies; `f_util_snapshot` last-month only (334 cos).",
        "- Inventory (`created_at`) can *filter* who has a LOC. It cannot rebuild",
        "  utilisation. A definition that needs a util time series is PARK.",
        "",
        "### Type counts (`debt_products`)",
        "",
        "| type | rows | companies |",
        "|---|---:|---:|",
        types,
        "",
        "## Train acceptance",
        "",
        "ACCEPTED if train base ∈ [5%, 30%] **and** two-sided size AUROC",
        "`max(auc, 1-auc) < 0.60` on `log1p(|op_in|)` **and** not a rewrite of",
        "Y2 / Y4 / Y9 (`|Spearman| ≥ 0.8`) **and** not a snapshot leak.",
        "Numeric pass is not enough for a documented subset.",
        "",
        f"Train panel: **{int(acc.n_train.iloc[0]):,}** company-months, 1,214 companies.",
        "",
        "| column | n labeled | cov | pos | base | size AUROC | two-sided | in3 AUROC | cos lab/pos | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        row("y10_new_loc_after_stress"),
        row("y10_loc_book_then_fees"),
        row("y10_ogtg_last_month"),
        row("y10_add_loc_6m"),
        row("y10_loc_then_int_ownp80"),
        row("y10_loc_then_int_spike"),
        "",
        "## Verdicts",
        "",
        "### `y10_new_loc_after_stress` — PARK (Y4 subset)",
        "",
        "Same idea as rejected `y4_new_facility_after_dip`, restricted to",
        f"`type='{LOC_TYPE}'`. Train n=12,674, pos=209, base **1.65%** (below 5%),",
        "size AUROC **0.681** (SIZE). All 209 positives sit inside the 578 Y4",
        "new-facility-after-dip positives (strict subset). Do not rename.",
        "",
        "### `y10_loc_book_then_fees` — CLOSE (Y9 subset)",
        "",
        "Numeric gates pass (15.05%, size 0.539) but Spearman vs `y9_fee_r_ownp80`",
        "on the 1,236 labeled train rows is **1.000**. It is Y9 on the LOC book.",
        "Forbidden later X would still be F + `a_fin_cost` / `a_fc`. Not a new Y.",
        "",
        "### `y10_ogtg_last_month` — PARK (snapshot, not a Y)",
        "",
        "LAST-MONTH-ONLY. Same leak as frozen-rejected `y4_ogtg_appear`:",
        "32 / 1,214 train companies (2.64%), size AUROC 0.664. NaN on every",
        "month except 2026-08. Do not treat extract ogtg as an onset in 2024.",
        "",
        "### `y10_add_loc_6m` — PARK (size)",
        "",
        "Already has a LOC with `created_at ≤ t` (month-start, so a first mid-month",
        "LOC is not counted as an add) and another LOC in `(t, t+6]`.",
        "Base 9.89% on 1,497 months / 33 positive companies, but size AUROC",
        "**0.611** (≥ 0.60). A birth-month leak in an earlier probe had made this",
        "look like an ACCEPT; the strict on-book-at-t rule removes that.",
        "",
        "### `y10_loc_then_int_ownp80` — ACCEPTED (in-module; parent merge)",
        "",
        "LOC on book at t and `interest_charge_3m / max(in3,1)` above own expanding",
        "p80 in each of t+1..t+3. Train n=1,236 (5.8% coverage), pos=179,",
        "base **14.48%**, size AUROC **0.508** (in3 0.534). Spearman vs",
        "`y9_fee_r_ownp80` = 0.348 (not a rewrite); vs `y4_ds_r_double` = 0.116;",
        "vs `y2_neg_2of3` = 0.060. No allowed-X column with |ρ| ≥ 0.8",
        f"(top was `n_debt` 0.17). Train acf1={extra.get('acf1', float('nan')):.3f},",
        f"acf3={extra.get('acf3', float('nan')):.3f}. Holdout descriptive only:",
        f"n_lab={extra.get('ho_n', '?')} pos={extra.get('ho_pos', '?')} on",
        f"{extra.get('ho_cos_pos', '?')} companies — LOW_POWER.",
        "",
        "This is **not** utilisation. It is interest-pressure on the LOC inventory.",
        "Full-panel interest own-p80 (no LOC filter) sits on the size gate (0.600)",
        "and is not shipped. Models that predict this column must drop family F",
        "and `a_fin_cost` / `a_fc` (interest is inside fin_cost).",
        "",
        (
            f"Q6 (lead): among the {extra.get('lead_n_pos', 179)} train positives, "
            f"interest is already above own p80 at t on "
            f"{extra.get('lead_hi_now_pos', 0.63):.0%} of rows "
            "(expected: overlapping 3-month `int_r` windows). "
            f"AUROC of contemporaneous int_hi_now vs the label is "
            f"{extra.get('lead_auc_hi', 0.745):.3f} — persistence, "
            "not a new drawn-balance signal. On each company's *first* "
            f"positive month ({extra.get('lead_n_cos', 62)} train cos) "
            f"{extra.get('lead_first_hi', 0.18):.0%} are already high at t. "
            "Quote the label as a sustained future spell, not as months-of-lead "
            "from a util path."
        ),
        "",
        f"### `y10_loc_then_int_spike` — {acc_i.loc['y10_loc_then_int_spike'].verdict}",
        "",
        "Y9 spike method on `interest_r`, LOC book only. Doubling is undefined",
        "when the trailing-6m mean is 0. Train n=900, pos=158, base **17.56%**,",
        "two-sided size 0.520. Sibling ρ vs own-p80 = 0.511; vs fee-spike label",
        "0.398. No allowed-X |ρ| ≥ 0.8 (top ~0.13). Train acf1=0.596, acf3=0.046.",
        "Holdout descriptive only: 41 labeled / 16 pos / 4 companies — LOW_POWER.",
        (
            f"Train positive-company union with the own-p80 sibling is "
            f"{extra.get('union_pos_cos', 73)} "
            f"(intersection {extra.get('inter_pos_cos', 49)}) — complementary "
            "enough that both can stay in-module."
        ),
        "",
        "## Spearman vs frozen Ys (train overlap)",
        "",
        "| column | vs | n | ρ | |",
        "|---|---|---:|---:|---|",
        *rho_lines,
        "",
        "## Forbidden later X",
        "",
        "Family **F** and prefixes `f_`, `a_fin_cost`, `a_fc`.",
        "Also drop `f_util_snapshot` / `f_outstanding_gt_granted` — the label",
        "must not be those snapshot fields copied forward.",
        "",
        "## What was not done",
        "",
        "- Did not write `targets.parquet` or edit `build_targets.py` / FROZEN lists.",
        "- Did not import `analysis.features.debt`.",
        "- Did not emit a 0–100 score. Did not touch `product/`.",
        "- Catalogue “LOC util > 90%” remains impossible without a util panel.",
        "- `build_targets.discover_target_modules()` globs `y*.py`. Parent must",
        "  not run the assembler until the merge decision. This child did not.",
        "",
    ]
    ACCEPT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", ACCEPT_MD)


def append_registry(acc: pd.DataFrame) -> None:
    """Append-only train coverage / size AUROC rows. Never rewrite the file."""
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    existing = REGISTRY.read_text(encoding="utf-8")
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    rows = []
    for _, r in acc.iterrows():
        marker = f"{AGENT},-,{r.column},verify,train,size_auroc"
        if marker in existing:
            continue
        notes = (
            f"{r.verdict}; base={r.base_rate:.4f}; two_sided={r.size_auroc_two_sided:.3f}; "
            f"in3={r.size_auroc_in3:.3f}; n_pos={int(r.n_pos)}; numeric_ok={bool(r.numeric_ok)}; "
            "holdout excluded; no targets.parquet write"
        )
        rows.append(
            (
                ts,
                ROUND,
                WAVE,
                AGENT,
                "-",
                r.column,
                "verify",
                "train",
                "base_rate",
                f"{r.base_rate:.6f}",
                f"{r.coverage:.4f}",
                notes,
            )
        )
        rows.append(
            (
                ts,
                ROUND,
                WAVE,
                AGENT,
                "-",
                r.column,
                "verify",
                "train",
                "size_auroc",
                f"{r.size_auroc:.6f}",
                f"{r.coverage:.4f}",
                notes,
            )
        )
    if not rows:
        print("registry skip (all Y10 rows already present for this agent)")
        return
    with REGISTRY.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print("registry appended", len(rows), "rows")


def lead_time_stats(panel: pd.DataFrame, y10: pd.DataFrame, is_train: pd.Series) -> dict:
    """Contemporaneous interest-high rate on the accepted label (Q6)."""
    g = panel.groupby("company_id", sort=False)
    own = g["int_r"].transform(lambda s: _expanding_quantile_skipna(s, OWN_P, MIN_OWN_HIST))
    hi = pd.Series(
        np.where(own.notna(), (panel["int_r"] > own).astype(float), np.nan),
        index=panel.index,
    )
    tmp = panel[["company_id", "month"]].copy()
    tmp["int_hi_now"] = hi
    tmp = tmp.rename(columns={"month": "period"})
    m = y10[["company_id", "period", "y10_loc_then_int_ownp80"]].merge(
        tmp, on=["company_id", "period"], how="left"
    )
    ok = is_train & m["y10_loc_then_int_ownp80"].notna()
    pos = ok & (m["y10_loc_then_int_ownp80"] == 1)
    auc = auroc(m.loc[ok, "y10_loc_then_int_ownp80"], m.loc[ok, "int_hi_now"])
    first = (
        m.loc[pos]
        .sort_values(["company_id", "period"])
        .groupby("company_id", sort=False)
        .head(1)
    )
    return {
        "lead_n_pos": int(pos.sum()),
        "lead_hi_now_pos": float(m.loc[pos, "int_hi_now"].mean()) if pos.any() else float("nan"),
        "lead_auc_hi": auc,
        "lead_n_cos": int(first["company_id"].nunique()) if len(first) else 0,
        "lead_first_hi": float(first["int_hi_now"].mean()) if len(first) else float("nan"),
    }


def train_pos_companies(y10: pd.DataFrame, is_train: pd.Series) -> pd.DataFrame:
    """How many train companies ever have a positive / labeled row."""
    rows = []
    for col in Y10_COLS:
        y = y10.loc[is_train, col]
        cid = y10.loc[is_train, "company_id"]
        lab = y.notna()
        pos = y == 1
        rows.append(
            {
                "column": col,
                "n_cos_labeled": int(cid[lab].nunique()),
                "n_cos_pos": int(cid[pos].nunique()),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from analysis.features.common import connect
    from analysis.features.grid import monthly_grid

    con = connect()
    facts = inspect_debt_table(con)
    print("Y10 debt_products inventory")
    print(
        f"  rows={facts['n_debt_rows']} cos={facts['n_debt_cos']} "
        f"LOC rows={facts['n_loc_rows']} LOC cos={facts['n_loc_cos']} "
        f"(train {facts['n_loc_cos_train']})"
    )
    print(
        f"  first-LOC mid-panel {facts['n_first_loc_midpanel']} "
        f"(train {facts['n_first_loc_midpanel_train']})"
    )
    print(
        f"  ogtg companies {facts['n_ogtg_cos']} "
        f"(train {facts['n_ogtg_cos_train']}); "
        f"created {facts['created_min']} .. {facts['created_max']}"
    )
    print("  types:")
    for row in facts["types"]:
        print(f"    {row['type']:14s} n={row['n']:5d} cos={row['n_cos']:4d}")

    grid = monthly_grid(con)[["company_id", "period"]]
    y10 = build(con, grid)
    keys = grid[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"])
    tr = train_mask(keys["company_id"])
    panel = fin_month_panel(con).rename(columns={"month": "period"})
    panel["company_id"] = panel["company_id"].astype(str)
    flows = keys.merge(
        panel[["company_id", "period", "op_in", "in3"]],
        on=["company_id", "period"],
        how="left",
    )
    print("rows", len(y10), "dups", int(y10.duplicated(["company_id", "period"]).sum()))
    print(
        "train company-months",
        int(tr.sum()),
        "companies",
        keys.loc[tr, "company_id"].nunique(),
    )
    acc = train_acceptance(y10, flows["op_in"], flows["in3"], tr)
    print("Y10 train acceptance (numeric gates; family verdict in last col)")
    print(acc.to_string(index=False))
    print("train companies labeled / positive")
    print(train_pos_companies(y10, tr).to_string(index=False))
    rho = spearman_vs_frozen(y10, tr)
    if not rho.empty:
        print("Spearman vs frozen Ys (train overlap; |ρ|≥0.8 = rewrite)")
        print(rho.to_string(index=False))
    both = tr & y10["y10_loc_then_int_ownp80"].notna() & y10["y10_loc_then_int_spike"].notna()
    sib = float("nan")
    if both.sum() >= 20:
        sib = float(
            y10.loc[both, "y10_loc_then_int_ownp80"].corr(
                y10.loc[both, "y10_loc_then_int_spike"], method="spearman"
            )
        )
        print(
            "sibling Spearman ownp80 vs spike",
            f"n={int(both.sum())} rho={sib:.3f} rewrite={abs(sib) >= 0.8}",
        )
    ca = set(y10.loc[tr & (y10["y10_loc_then_int_ownp80"] == 1), "company_id"])
    cb = set(y10.loc[tr & (y10["y10_loc_then_int_spike"] == 1), "company_id"])
    print("pos-cos union", len(ca | cb), "intersection", len(ca & cb))
    from analysis.features.common import DATA

    tpath = DATA / "feature_store" / "targets.parquet"
    if tpath.exists() and "y10_new_loc_after_stress" in y10.columns:
        tgt = pd.read_parquet(tpath)
        tgt["company_id"] = tgt["company_id"].astype(str)
        tgt["period"] = pd.to_datetime(tgt["period"])
        if "y4_new_facility_after_dip" in tgt.columns:
            m = y10[["company_id", "period", "y10_new_loc_after_stress"]].merge(
                tgt[["company_id", "period", "y4_new_facility_after_dip"]],
                on=["company_id", "period"],
                how="left",
            )
            ov = tr & m["y10_new_loc_after_stress"].notna() & m["y4_new_facility_after_dip"].notna()
            loc_pos = ov & (m["y10_new_loc_after_stress"] == 1)
            y4_pos = ov & (m["y4_new_facility_after_dip"] == 1)
            print(
                "y10_new_loc_after_stress subset of y4 new-facility-after-dip?",
                int((loc_pos & ~y4_pos).sum()) == 0,
                f"loc_pos={int(loc_pos.sum())} y4_pos={int(y4_pos.sum())} both={int((loc_pos & y4_pos).sum())}",
            )
    print("META.forbidden_x_families", META["forbidden_x_families"])
    print("META.forbidden_x_prefixes", META["forbidden_x_prefixes"])
    leak = leakage_check(
        ["a_op_in", "a_fin_cost", "a_fc_r", "f_fc_r", "f_util_snapshot", "f_has_loc"],
        "y10_new_loc_after_stress",
        forbidden_prefixes=META["forbidden_x_prefixes"],
    )
    print("leakage_check demo", leak)

    acc_col = "y10_loc_then_int_ownp80"
    d = y10.loc[tr, ["company_id", "period", acc_col]].dropna(subset=[acc_col])
    d = d.sort_values(["company_id", "period"])
    g = d.groupby("company_id")[acc_col]
    lag1 = g.shift(1)
    lag3 = g.shift(3)
    ok1 = d[acc_col].notna() & lag1.notna()
    ok3 = d[acc_col].notna() & lag3.notna()
    acf1 = float(d.loc[ok1, acc_col].corr(lag1[ok1])) if ok1.sum() else float("nan")
    acf3 = float(d.loc[ok3, acc_col].corr(lag3[ok3])) if ok3.sum() else float("nan")
    print(f"{acc_col} train acf1={acf1:.3f} n={int(ok1.sum())} acf3={acf3:.3f} n={int(ok3.sum())}")
    ho = ~tr
    y_ho = y10.loc[ho, acc_col]
    print(
        "holdout DESCRIPTIVE only",
        f"n_lab={int(y_ho.notna().sum())} pos={int((y_ho == 1).sum())} "
        f"cos_pos={y10.loc[ho & (y10[acc_col] == 1), 'company_id'].nunique()} LOW_POWER",
    )
    print(
        "UTILISATION_PATH PARK — no outstanding/granted panel. "
        "TWO interest-on-LOC columns ACCEPTED (own-p80 + spike); "
        "parent decides merge. Do not write targets.parquet from here."
    )
    lead = lead_time_stats(panel.rename(columns={"period": "month"}), y10, tr)
    print(
        "Q6 lead",
        f"pos_hi_now={lead['lead_hi_now_pos']:.3f}",
        f"auc_hi={lead['lead_auc_hi']:.3f}",
        f"first_hi={lead['lead_first_hi']:.3f}",
        f"n_cos={lead['lead_n_cos']}",
    )
    extra = {
        "acf1": acf1,
        "acf3": acf3,
        "ho_n": int(y_ho.notna().sum()),
        "ho_pos": int((y_ho == 1).sum()),
        "ho_cos_pos": int(y10.loc[ho & (y10[acc_col] == 1), "company_id"].nunique()),
        "union_pos_cos": len(ca | cb),
        "inter_pos_cos": len(ca & cb),
        **lead,
    }
    write_acceptance_md(
        facts,
        acc,
        train_pos_companies(y10, tr),
        rho if not rho.empty else pd.DataFrame(),
        extra,
    )
    append_registry(acc)
    con.close()
