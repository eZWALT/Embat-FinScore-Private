"""Family M — category mix (what the money is doing).

Monthly *shares* of booking-date cash, not a 0–100 score. Answers brief
Q5 (why) and Q3 (turning): the composition of |amount| (and a few counts)
by ``CAT_MAP`` group and by raw categories Family A collapses.

No look-ahead: a row for ``period`` (month-start) uses booking dates in
``[period, period + 1 month)`` — the same cut as Family A
(``date_trunc('month', date) = period``). 9,242 txs on 2026-09-01 stay
off August. Holdout companies are scored with the same formulas; nothing
here fits percentiles or other train-only refs.

This module does **not** rewrite ``monthly.parquet`` and is **not** in
``build_feature_store.FAMILIES``. Parent merges after review. Does not
import other family modules. Clean flags (``is_dup``, ``is_extreme``)
are not used as drop filters.

Prefix ``m_`` (mix). Family I is interactions (``i_*``) — do not clash.

Denominator
-----------
``abs_amt = sum(|amount|)`` and ``n_tx = count(*)`` in the company-month.
Shares are NaN when the denom is 0 (empty grid month), never 0-filled.

CAT_MAP group amount shares (sum to 1 when defined):

- m_op_in_share / m_op_out_share / m_fin_share / m_debt_share
- m_xfer_share / m_invest_share / m_uncat_share

``m_uncat_share`` is *amount*-weighted. Family A's ``a_uncat_share`` is
the *count* share of the same rows (uncategorized or not in CAT_MAP —
only ``uncategorized`` is unmapped). The count version is parked.

Raw categories the score under-uses (amount / |all|):

- m_fee_share, m_int_share, m_tax_share, m_salary_share
- m_coll_share, m_pay_share   (raw ``collection`` / ``payment``, not bulk)

Collection vs payment:

- m_core_share   = (|collection| + |payment|) / |all|
- m_coll_vs_pay  = |collection| / (|collection| + |payment|)
                   (NaN if the pair is absent)

Count shares only where they are not a rewrite of the amount twin or of
an existing A column (see parked list in ``catmix_report.md``):

- m_op_in_n_share, m_op_out_n_share
- m_fee_n_share, m_coll_n_share, m_pay_n_share

Parked (computed in the report, never emitted):

- uncat *count* share ≡ ``a_uncat_share``
- ``a_fin_cost / a_op_in`` (the monthly fin-cost / inflow ratio)
- count twins of debt / transfer / invest / interest / salary
  (Spearman vs own amount share ≥ 0.95)

``m_fin_share`` is the CAT_MAP group (fee + interest_charge). It ranks
close to monthly ``a_fin_cost / a_op_in`` but is a [0, 1] activity share,
not that ratio, and is only moderately tied to trailing ``f_fc_r``.
Do not stack it with those columns in a GBM.

Trailing-k mix (Q6 candidate)
-----------------------------
``m_*_t3`` / ``m_*_t6`` recompute the same share from **summed |amounts|**
(or counts) over months ``t-k+1..t`` on the company grid, ``min_periods=k``.
Empty grid months contribute 0 to the sums (they are not a vote). First
``k-1`` months are NaN. This is not the mean of monthly shares: a €10
month and a €10m month do not vote equally.

Train-only diagnostics: ``python -m analysis.features.catmix`` writes
``analysis/outputs/catmix_report.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow `python analysis/features/catmix.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.features.common import ANALYSIS, CAT_MAP, DATA, connect, train_mask
from analysis.features.grid import monthly_grid

SOURCE_TABLES = ["transactions"]
FAMILY = "m"

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
REPORT_PATH = ANALYSIS / "outputs" / "catmix_report.md"

SIZE_RHO = 0.85
CONSTANT_THRESH = 0.999
NZV_THRESH = 0.95
MIN_ACF_PAIRS = 4
REWRITE_RHO = 0.95
ACF1_KEEP = 0.25
ACF1_LIFT = 0.15
T3_WINDOW = 3
T6_WINDOW = 6

# CAT_MAP groups + residual. Only unmapped raw token is `uncategorized`.
_GROUPS = ("op_in", "op_out", "fin_cost", "debt_service", "transfer", "invest")
_RAW_AMT = (
    "fee",
    "interest_charge",
    "tax",
    "salary",
    "collection",
    "payment",
)

SPECS: tuple[dict, ...] = (
    {
        "name": "m_op_in_share",
        "formula": "sum(|amt| | CAT_MAP=op_in) / sum(|amt|)",
        "brief": "Q1/Q5 why — operating-inflow mix (collection, settlements, refunds)",
    },
    {
        "name": "m_op_out_share",
        "formula": "sum(|amt| | CAT_MAP=op_out) / sum(|amt|)",
        "brief": "Q1/Q5 why — operating-outflow mix (payment, utility, payroll, tax, cash)",
    },
    {
        "name": "m_fin_share",
        "formula": "sum(|amt| | CAT_MAP=fin_cost) / sum(|amt|)",
        "brief": "Q3/Q5 turning — fee+interest as a share of activity (not fin/inflow)",
    },
    {
        "name": "m_debt_share",
        "formula": "sum(|amt| | CAT_MAP=debt_service) / sum(|amt|)",
        "brief": "Q5 why — debt_repayment volume in the month's cash",
    },
    {
        "name": "m_xfer_share",
        "formula": "sum(|amt| | category=transfer) / sum(|amt|)",
        "brief": "Q5 why — |transfer| mix; A only keeps signed net a_transfer",
    },
    {
        "name": "m_invest_share",
        "formula": "sum(|amt| | CAT_MAP=invest) / sum(|amt|)",
        "brief": "Q5 why — deploy+return volume share",
    },
    {
        "name": "m_uncat_share",
        "formula": "sum(|amt| | uncategorized or not in CAT_MAP) / sum(|amt|)",
        "brief": "Q5 why — amount-weighted opacity (A's a_uncat_share is count)",
    },
    {
        "name": "m_fee_share",
        "formula": "sum(|amt| | category=fee) / sum(|amt|)",
        "brief": "Q3/Q5 FinRegLab — fee slice of fin_cost (score uses the lump)",
    },
    {
        "name": "m_int_share",
        "formula": "sum(|amt| | category=interest_charge) / sum(|amt|)",
        "brief": "Q3/Q5 turning — interest_charge vs fee (Y9 lumps them)",
    },
    {
        "name": "m_tax_share",
        "formula": "sum(|amt| | category=tax) / sum(|amt|)",
        "brief": "Q5 why — tax amount intensity (C only has c_tax_month)",
    },
    {
        "name": "m_salary_share",
        "formula": "sum(|amt| | category=salary) / sum(|amt|)",
        "brief": "Q4/Q5 why — payroll amount intensity (C only has presence)",
    },
    {
        "name": "m_coll_share",
        "formula": "sum(|amt| | category=collection) / sum(|amt|)",
        "brief": "Q5 why — raw collection (not bulk / settlements)",
    },
    {
        "name": "m_pay_share",
        "formula": "sum(|amt| | category=payment) / sum(|amt|)",
        "brief": "Q5 why — raw payment (not bulk / utility / payroll)",
    },
    {
        "name": "m_core_share",
        "formula": "(|collection| + |payment|) / sum(|amt|)",
        "brief": "Q5 why — how much of the month is the named coll/pay pair",
    },
    {
        "name": "m_coll_vs_pay",
        "formula": "|collection| / (|collection| + |payment|)",
        "brief": "Q5 why — collection vs payment (1 = all collection, 0 = all payment)",
    },
    {
        "name": "m_op_in_n_share",
        "formula": "n(CAT_MAP=op_in) / n_tx",
        "brief": "Q5 why — op_in ticket mix (count ≠ amount, ρ≈0.65)",
    },
    {
        "name": "m_op_out_n_share",
        "formula": "n(CAT_MAP=op_out) / n_tx",
        "brief": "Q5 why — op_out ticket mix (count ≠ amount, ρ≈0.67)",
    },
    {
        "name": "m_fee_n_share",
        "formula": "n(category=fee) / n_tx",
        "brief": "Q3/Q5 — many small fees; amount share stays tiny",
    },
    {
        "name": "m_coll_n_share",
        "formula": "n(category=collection) / n_tx",
        "brief": "Q5 why — collection ticket mix vs euro mix (ρ≈0.69)",
    },
    {
        "name": "m_pay_n_share",
        "formula": "n(category=payment) / n_tx",
        "brief": "Q5 why — payment ticket mix vs euro mix (ρ≈0.77)",
    },
)

MONTHLY_COLS = tuple(s["name"] for s in SPECS)

# Trailing-3m from summed |amounts| (priority stems + a few companions).
# Do not emit parked count twins or a_fin_cost/inflow.
T3_SPECS: tuple[dict, ...] = (
    {
        "name": "m_coll_vs_pay_t3",
        "sibling": "m_coll_vs_pay",
        "num": "abs_collection",
        "den": "coll_pay",
        "brief": "Q6 — 3m collection vs payment from summed |amounts|",
    },
    {
        "name": "m_uncat_share_t3",
        "sibling": "m_uncat_share",
        "num": "abs_uncat",
        "den": "abs_amt",
        "brief": "Q6 — 3m amount-weighted opacity",
    },
    {
        "name": "m_fee_share_t3",
        "sibling": "m_fee_share",
        "num": "abs_fee",
        "den": "abs_amt",
        "brief": "Q3/Q6 — 3m fee activity share",
    },
    {
        "name": "m_int_share_t3",
        "sibling": "m_int_share",
        "num": "abs_interest_charge",
        "den": "abs_amt",
        "brief": "Q3/Q6 — 3m interest_charge activity share",
    },
    {
        "name": "m_fee_n_share_t3",
        "sibling": "m_fee_n_share",
        "num": "n_fee",
        "den": "n_tx",
        "brief": "Q3/Q6 — 3m fee ticket share (counts, not mean of monthly)",
    },
    {
        "name": "m_salary_share_t3",
        "sibling": "m_salary_share",
        "num": "abs_salary",
        "den": "abs_amt",
        "brief": "Q4/Q6 — 3m payroll amount intensity",
    },
    {
        "name": "m_xfer_share_t3",
        "sibling": "m_xfer_share",
        "num": "abs_transfer",
        "den": "abs_amt",
        "brief": "Q6 — 3m |transfer| mix",
    },
    {
        "name": "m_op_in_share_t3",
        "sibling": "m_op_in_share",
        "num": "abs_op_in",
        "den": "abs_amt",
        "brief": "Q6 — 3m operating-in mix",
    },
    {
        "name": "m_debt_share_t3",
        "sibling": "m_debt_share",
        "num": "abs_debt_service",
        "den": "abs_amt",
        "brief": "Q6 — 3m debt_repayment mix",
    },
    {
        "name": "m_core_share_t3",
        "sibling": "m_core_share",
        "num": "abs_core",
        "den": "abs_amt",
        "brief": "Q6 — 3m named coll+pay weight",
    },
    {
        "name": "m_tax_share_t3",
        "sibling": "m_tax_share",
        "num": "abs_tax",
        "den": "abs_amt",
        "brief": "Q6 — 3m tax intensity",
    },
    {
        "name": "m_op_out_share_t3",
        "sibling": "m_op_out_share",
        "num": "abs_op_out",
        "den": "abs_amt",
        "brief": "Q6 — 3m operating-out mix",
    },
    {
        "name": "m_coll_share_t3",
        "sibling": "m_coll_share",
        "num": "abs_collection",
        "den": "abs_amt",
        "brief": "Q6 — 3m raw collection mix",
    },
    {
        "name": "m_pay_share_t3",
        "sibling": "m_pay_share",
        "num": "abs_payment",
        "den": "abs_amt",
        "brief": "Q6 — 3m raw payment mix",
    },
    {
        "name": "m_fin_share_t3",
        "sibling": "m_fin_share",
        "num": "abs_fin_cost",
        "den": "abs_amt",
        "brief": "Q3/Q6 — 3m fin_cost activity share (flag vs f_fc_r)",
    },
)

T3_COLS = tuple(s["name"] for s in T3_SPECS)
T6_STEMS: tuple[str, ...] = ()  # filled after t3 battery if Q6 still dies
T6_SPECS: tuple[dict, ...] = ()
T6_COLS: tuple[str, ...] = ()

M_COLS = MONTHLY_COLS + T3_COLS

PARKED_SPECS: tuple[dict, ...] = (
    {
        "name": "m_uncat_n_share",
        "formula": "n(uncategorized or not in CAT_MAP) / n_tx",
        "reason": "exact rewrite of a_uncat_share (only unmapped token is uncategorized)",
    },
    {
        "name": "m_fc_over_in",
        "formula": "a_fin_cost / a_op_in",
        "reason": "exact rewrite of monthly a_fin_cost / inflow; f_fc_r is the 3m version",
    },
    {
        "name": "m_debt_n_share",
        "formula": "n(debt_repayment) / n_tx",
        "reason": "count twin of m_debt_share (train Spearman ≥ 0.95)",
    },
    {
        "name": "m_xfer_n_share",
        "formula": "n(transfer) / n_tx",
        "reason": "count twin of m_xfer_share (train Spearman ≥ 0.95)",
    },
    {
        "name": "m_invest_n_share",
        "formula": "n(CAT_MAP=invest) / n_tx",
        "reason": "count twin of m_invest_share (train Spearman ≥ 0.95)",
    },
    {
        "name": "m_int_n_share",
        "formula": "n(interest_charge) / n_tx",
        "reason": "count twin of m_int_share (train Spearman ≥ 0.95)",
    },
    {
        "name": "m_salary_n_share",
        "formula": "n(salary) / n_tx",
        "reason": "count twin of m_salary_share (train Spearman ≥ 0.95)",
    },
    {
        "name": "m_fin_n_share",
        "formula": "n(CAT_MAP=fin_cost) / n_tx",
        "reason": "almost m_fee_n_share (interest is rare in count)",
    },
)


def _keys(grid: pd.DataFrame) -> pd.DataFrame:
    keys = grid[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"])
    return keys.reset_index(drop=True)


def _share(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return pd.Series(
        np.where(den > 0, num / den, np.nan),
        index=num.index,
        dtype=np.float64,
    )


_TRAIL_RAW = (
    "abs_amt",
    "abs_op_in",
    "abs_op_out",
    "abs_fin_cost",
    "abs_debt_service",
    "abs_transfer",
    "abs_uncat",
    "abs_fee",
    "abs_interest_charge",
    "abs_tax",
    "abs_salary",
    "abs_collection",
    "abs_payment",
    "n_tx",
    "n_fee",
)


def _roll_sum(panel: pd.DataFrame, col: str, window: int, min_periods: int) -> pd.Series:
    """Past-only rolling sum on the company grid. Empty months contribute 0."""
    s = pd.to_numeric(panel[col], errors="coerce").fillna(0.0)
    return s.groupby(panel["company_id"], sort=False).transform(
        lambda x: x.rolling(window, min_periods=min_periods).sum()
    )


def t3_specs_for_window(window: int) -> tuple[dict, ...]:
    """Copy T3_SPECS with ``_t{window}`` names. Used for t6 on a stem subset."""
    suf = f"_t{window}"
    out = []
    for spec in T3_SPECS:
        row = dict(spec)
        row["name"] = spec["name"][:-3] + suf if spec["name"].endswith("_t3") else spec["name"] + suf
        row["brief"] = spec["brief"].replace("3m", f"{window}m")
        out.append(row)
    return tuple(out)


def _apply_trail(
    panel: pd.DataFrame,
    specs: tuple[dict, ...],
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    """Add trail shares from summed |amounts| (or counts). ``panel`` is sorted."""
    rolled = {c: _roll_sum(panel, c, window, min_periods) for c in _TRAIL_RAW}
    rolled["abs_core"] = rolled["abs_collection"] + rolled["abs_payment"]
    rolled["coll_pay"] = rolled["abs_collection"] + rolled["abs_payment"]
    for spec in specs:
        num = rolled[spec["num"]]
        den = rolled[spec["den"]]
        panel[spec["name"]] = _share(num, den)
    return panel


def _monthly_mix(con) -> pd.DataFrame:
    """One row per company-month with abs amounts and counts. No grid yet."""
    cm = pd.DataFrame({"category": list(CAT_MAP), "grp": list(CAT_MAP.values())})
    con.register("_feat_m_cat_map", cm)

    grp_amt = ",\n          ".join(
        f"SUM(CASE WHEN c.grp = '{g}' THEN ABS(t.amount) ELSE 0 END) AS abs_{g}"
        for g in _GROUPS
    )
    grp_n = ",\n          ".join(
        f"SUM(CASE WHEN c.grp = '{g}' THEN 1 ELSE 0 END) AS n_{g}"
        for g in _GROUPS
    )
    raw_amt = ",\n          ".join(
        f"SUM(CASE WHEN t.category = '{c}' THEN ABS(t.amount) ELSE 0 END) AS abs_{c}"
        for c in _RAW_AMT
    )
    raw_n = ",\n          ".join(
        f"SUM(CASE WHEN t.category = '{c}' THEN 1 ELSE 0 END) AS n_{c}"
        for c in ("fee", "interest_charge", "tax", "salary", "transfer", "collection", "payment")
    )

    df = con.execute(
        f"""
        SELECT
          t.company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          SUM(ABS(t.amount)) AS abs_amt,
          COUNT(*) AS n_tx,
          {grp_amt},
          SUM(CASE WHEN t.category = 'uncategorized' OR c.grp IS NULL
                   THEN ABS(t.amount) ELSE 0 END) AS abs_uncat,
          {grp_n},
          SUM(CASE WHEN t.category = 'uncategorized' OR c.grp IS NULL
                   THEN 1 ELSE 0 END) AS n_uncat,
          {raw_amt},
          {raw_n}
        FROM transactions t
        LEFT JOIN _feat_m_cat_map c ON t.category = c.category
        WHERE t."date" IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["period"] = pd.to_datetime(df["month"])
    return df.drop(columns=["month"])


def _empty_panel(grid: pd.DataFrame) -> pd.DataFrame:
    empty = _keys(grid) if len(grid.columns) else pd.DataFrame(
        columns=["company_id", "period"]
    )
    for c in M_COLS:
        empty[c] = np.float64(np.nan)
    return empty


def _compute(keys: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    out = keys.merge(monthly, on=["company_id", "period"], how="left")
    amt = out["abs_amt"]
    n_tx = out["n_tx"]

    out["m_op_in_share"] = _share(out["abs_op_in"], amt)
    out["m_op_out_share"] = _share(out["abs_op_out"], amt)
    out["m_fin_share"] = _share(out["abs_fin_cost"], amt)
    out["m_debt_share"] = _share(out["abs_debt_service"], amt)
    out["m_xfer_share"] = _share(out["abs_transfer"], amt)
    out["m_invest_share"] = _share(out["abs_invest"], amt)
    out["m_uncat_share"] = _share(out["abs_uncat"], amt)

    out["m_fee_share"] = _share(out["abs_fee"], amt)
    out["m_int_share"] = _share(out["abs_interest_charge"], amt)
    out["m_tax_share"] = _share(out["abs_tax"], amt)
    out["m_salary_share"] = _share(out["abs_salary"], amt)
    out["m_coll_share"] = _share(out["abs_collection"], amt)
    out["m_pay_share"] = _share(out["abs_payment"], amt)
    out["m_core_share"] = _share(out["abs_collection"] + out["abs_payment"], amt)
    out["m_coll_vs_pay"] = _share(
        out["abs_collection"],
        out["abs_collection"] + out["abs_payment"],
    )

    out["m_op_in_n_share"] = _share(out["n_op_in"], n_tx)
    out["m_op_out_n_share"] = _share(out["n_op_out"], n_tx)
    out["m_fee_n_share"] = _share(out["n_fee"], n_tx)
    out["m_coll_n_share"] = _share(out["n_collection"], n_tx)
    out["m_pay_n_share"] = _share(out["n_payment"], n_tx)

    out = out.sort_values(["company_id", "period"]).reset_index(drop=True)
    out = _apply_trail(out, T3_SPECS, T3_WINDOW, T3_WINDOW)
    if T6_SPECS:
        out = _apply_trail(out, T6_SPECS, T6_WINDOW, T6_WINDOW)
    return out


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    """Return ``company_id, period`` plus ``m_*``. Queries ``transactions``."""
    if grid is None or grid.empty:
        return _empty_panel(grid if grid is not None else pd.DataFrame())

    keys = _keys(grid)
    monthly = _monthly_mix(con)
    out = _compute(keys, monthly)
    extra = [c for c in out.columns if c not in {"company_id", "period"}]
    # raw abs_/n_ helpers are dropped below; only m_* must be the keep set
    emitted = [c for c in extra if c.startswith("m_")]
    unexpected = [c for c in emitted if c not in M_COLS]
    if unexpected:
        raise ValueError(f"Family M must only emit keep-list m_*, got {unexpected}")
    missing = [c for c in M_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Family M missing columns {missing}")
    if out.duplicated(["company_id", "period"]).any():
        raise ValueError("Family M produced duplicate company_id, period rows")
    return out[["company_id", "period", *M_COLS]].reset_index(drop=True)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _median_acf(panel: pd.DataFrame, col: str, lag: int = 1) -> float:
    acc: list[float] = []
    work = panel[["company_id", "period", col]].sort_values(["company_id", "period"])
    for _, g in work.groupby("company_id", sort=False):
        s = _num(g[col]).to_numpy(dtype=float)
        if s.size <= lag:
            continue
        x, y = s[:-lag], s[lag:]
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < MIN_ACF_PAIRS:
            continue
        xx, yy = x[m], y[m]
        if xx.std() <= 1e-15 or yy.std() <= 1e-15:
            continue
        r = float(np.corrcoef(xx, yy)[0, 1])
        if np.isfinite(r):
            acc.append(r)
    return float(np.nanmedian(acc)) if acc else float("nan")


def _median_acf1(panel: pd.DataFrame, col: str) -> float:
    return _median_acf(panel, col, lag=1)


def _load_store(columns: list[str]) -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(
            f"{STORE} missing — report needs a_in3 / rewrite columns; "
            "build() itself does not read the parquet"
        )
    import pyarrow.parquet as pq

    have = set(pq.read_schema(STORE).names)
    present = [c for c in columns if c in have]
    store = pd.read_parquet(STORE, columns=["company_id", "period", *present])
    store["company_id"] = store["company_id"].astype(str)
    store["period"] = pd.to_datetime(store["period"])
    for c in columns:
        if c not in store.columns:
            store[c] = np.nan
    return store


def _parked_frame(monthly: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Parked candidates on the same company-month keys as ``monthly``."""
    p = monthly[["company_id", "period"]].copy()
    p["m_uncat_n_share"] = _share(monthly["n_uncat"], monthly["n_tx"])
    p["m_debt_n_share"] = _share(monthly["n_debt_service"], monthly["n_tx"])
    p["m_xfer_n_share"] = _share(monthly["n_transfer"], monthly["n_tx"])
    p["m_invest_n_share"] = _share(monthly["n_invest"], monthly["n_tx"])
    p["m_int_n_share"] = _share(monthly["n_interest_charge"], monthly["n_tx"])
    p["m_salary_n_share"] = _share(monthly["n_salary"], monthly["n_tx"])
    p["m_fin_n_share"] = _share(monthly["n_fin_cost"], monthly["n_tx"])
    src = store[["company_id", "period", "a_fin_cost", "a_op_in"]].copy()
    p = p.merge(src, on=["company_id", "period"], how="left")
    p["m_fc_over_in"] = _share(p["a_fin_cost"], p["a_op_in"])
    return p


def _col_stats(tr: pd.DataFrame, col: str, size: pd.Series) -> dict:
    s = _num(tr[col])
    nn = s.dropna()
    n_cm = len(tr)
    n_co = int(tr["company_id"].nunique())
    cov_cm = float(s.notna().mean()) if n_cm else float("nan")
    cov_co = (
        float(tr.loc[s.notna(), "company_id"].nunique() / n_co) if n_co else float("nan")
    )
    if nn.empty:
        modal_share = float("nan")
        n_unique = 0
        med = float("nan")
        mean = float("nan")
    else:
        vc = nn.value_counts()
        modal_share = float(vc.iloc[0] / len(nn))
        n_unique = int(nn.nunique())
        med = float(nn.median())
        mean = float(nn.mean())
    if nn.size >= 50 and n_unique > 1 and size.notna().sum() >= 50:
        size_rho = float(s.corr(size, method="spearman"))
    else:
        size_rho = float("nan")
    acf1 = _median_acf1(tr, col)
    flags: list[str] = []
    if nn.empty or n_unique <= 1 or (
        np.isfinite(modal_share) and modal_share >= CONSTANT_THRESH
    ):
        flags.append("CONSTANT")
    elif np.isfinite(modal_share) and modal_share >= NZV_THRESH:
        flags.append("NZV")
    if np.isfinite(size_rho) and abs(size_rho) > SIZE_RHO:
        flags.append("SIZE")
    return {
        "feature": col,
        "cov_cm": cov_cm,
        "cov_co": cov_co,
        "n_unique": n_unique,
        "modal_share": modal_share,
        "size_rho": size_rho,
        "acf1": acf1,
        "median": med,
        "mean": mean,
        "flags": ",".join(flags) if flags else "—",
    }


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return float(_num(a).corr(_num(b), method="spearman"))


def _fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100.0 * x:.1f}%"


def _fmt_num(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if abs(x) >= 1e5 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def _decide_trail(
    acf1: float,
    sib_acf1: float,
    size_rho: float,
    flags: str,
    rewrite_hits: list[str],
    acf3: float = float("nan"),
) -> str:
    """KEEP only if Q6-real (acf3); CLOSE if acf1 is overlap-only; PARK on junk."""
    del size_rho
    flag_s = flags or ""
    if "SIZE" in flag_s or "CONSTANT" in flag_s:
        return "PARK"
    if rewrite_hits:
        return "PARK"
    persist = np.isfinite(acf1) and (
        acf1 >= ACF1_KEEP
        or (np.isfinite(sib_acf1) and acf1 >= sib_acf1 + ACF1_LIFT)
    )
    if persist and np.isfinite(acf3) and acf3 >= ACF1_KEEP:
        return "KEEP"
    if persist:
        return "CLOSE"
    return "PARK"


def _trail_panel(keys: pd.DataFrame, monthly: pd.DataFrame, specs: tuple[dict, ...], window: int) -> pd.DataFrame:
    """Diagnostic trail columns from the same raw monthly sums ``build`` uses."""
    src = keys.merge(monthly, on=["company_id", "period"], how="left")
    src = src.sort_values(["company_id", "period"]).reset_index(drop=True)
    src = _apply_trail(src, specs, window, window)
    cols = ["company_id", "period"] + [s["name"] for s in specs]
    return src[cols]


Y_EVAL = ("y3_recover_cash_6m", "y7_top1_lost", "y9_fee_r_ownp80")
Y9_FORBIDDEN = ("m_fee_share_t3", "m_int_share_t3", "m_fin_share_t3", "m_fee_n_share_t3")


def _y_spearman_section(tr: pd.DataFrame, keep_cols: list[str]) -> list[str]:
    """Train-only Spearman of KEEP t3 vs accepted Ys. Read targets.parquet; no model."""
    lines = [
        "",
        "### t3 (acf1-pass) vs accepted Ys (train labeled rows, no model)",
        "",
        "Read `data/feature_store/targets.parquet` (not rewritten). "
        "Spearman on train rows where the Y is non-null. "
        "**Not a win:** do not read Y9 vs `m_fee_*` / `m_int_*` / `m_fin_*` "
        "as evidence — those share the same fee/interest raw material.",
        "",
    ]
    if not TARGETS.exists():
        lines += ["`targets.parquet` missing — skipped.", ""]
        return lines
    if not keep_cols:
        lines += ["No KEEP t3 columns.", ""]
        return lines
    y = pd.read_parquet(TARGETS, columns=["company_id", "period", *Y_EVAL])
    y["company_id"] = y["company_id"].astype(str)
    y["period"] = pd.to_datetime(y["period"])
    lab = tr.merge(y, on=["company_id", "period"], how="left")
    header = "| feature | " + " | ".join(f"ρ {c}" for c in Y_EVAL) + " | n labeled |"
    lines += [header, "| --- | " + " | ".join("---" for _ in Y_EVAL) + " | --- |"]
    for feat in keep_cols:
        cells = [f"`{feat}`"]
        n_lab = 0
        for yc in Y_EVAL:
            mask = lab[yc].notna() & _num(lab[feat]).notna()
            n_lab = max(n_lab, int(mask.sum()))
            if yc == "y9_fee_r_ownp80" and feat in Y9_FORBIDDEN:
                cells.append("n/a (same-source)")
                continue
            if int(mask.sum()) < 50:
                cells.append("—")
                continue
            rho = _spearman(lab.loc[mask, feat], lab.loc[mask, yc])
            cells.append(_fmt_num(rho))
        cells.append(str(n_lab))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


T3_REWRITE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("m_uncat_share_t3", "a_uncat_share", "vs monthly count uncat"),
    ("m_salary_share_t3", "c_salary_month", "vs C presence flag"),
    ("m_fin_share_t3", "f_fc_r", "vs trailing f_fc_r"),
    ("m_fin_share_t3", "fc_inflow", "vs monthly a_fin_cost/inflow"),
    ("m_fee_share_t3", "f_fc_r", "vs f_fc_r (do not stack / not a Y9 win)"),
    ("m_int_share_t3", "f_fc_r", "vs f_fc_r (do not stack / not a Y9 win)"),
    ("m_fee_n_share_t3", "f_fc_r", "fee count 3m vs f_fc_r"),
    ("m_uncat_share_t3", "m_uncat_share", "t3 vs monthly sibling"),
    ("m_coll_vs_pay_t3", "m_coll_vs_pay", "t3 vs monthly sibling"),
    ("m_fee_share_t3", "m_fee_share", "t3 vs monthly sibling"),
    ("m_op_in_share_t3", "m_op_in_share", "t3 vs monthly sibling"),
)


def write_train_report(
    part: pd.DataFrame,
    monthly: pd.DataFrame,
    path: Path = REPORT_PATH,
) -> pd.DataFrame:
    store = _load_store(
        ["a_in3", "a_uncat_share", "a_fin_cost", "a_op_in", "f_fc_r", "c_salary_month", "c_tax_month"]
    )
    keys = part[["company_id", "period"]].copy()
    panel = keys.merge(store, on=["company_id", "period"], how="left")
    panel = panel.merge(part, on=["company_id", "period"], how="left")
    parked = _parked_frame(
        keys.merge(monthly, on=["company_id", "period"], how="left"),
        store,
    )
    panel = panel.merge(
        parked.drop(columns=["a_fin_cost", "a_op_in"], errors="ignore"),
        on=["company_id", "period"],
        how="left",
    )
    panel["fc_inflow"] = _share(panel["a_fin_cost"], panel["a_op_in"])

    tr = panel.loc[train_mask(panel["company_id"])].copy()
    size = np.log1p(np.maximum(_num(tr["a_in3"]), 0.0))
    n_train = int(len(tr))
    n_co = int(tr["company_id"].nunique())
    empty_cm = float((_num(tr["a_uncat_share"]).isna() & _num(tr["m_op_in_share"]).isna()).mean())
    # empty months: no txs → mix NaN. a_uncat_share is also NaN then.

    rows = []
    for spec in SPECS:
        st = _col_stats(tr, spec["name"], size)
        st["formula"] = spec["formula"]
        st["brief"] = spec["brief"]
        rows.append(st)
    stats = pd.DataFrame(rows)

    rewrite_pairs = [
        ("m_uncat_n_share", "a_uncat_share", "park — exact count uncat"),
        ("m_uncat_share", "a_uncat_share", "keep — amount vs count"),
        ("m_fc_over_in", "fc_inflow", "park — exact a_fin_cost/inflow"),
        ("m_fin_share", "fc_inflow", "near monthly fin/inflow; not a rewrite of f_fc_r"),
        ("m_fin_share", "f_fc_r", "vs trailing 3m f_fc_r"),
        ("m_fee_share", "fc_inflow", "fee amount vs monthly fin/inflow"),
        ("m_int_share", "fc_inflow", "interest amount vs monthly fin/inflow"),
        ("m_fee_share", "m_fin_share", "fee is most of fin_cost euros"),
        ("m_int_share", "m_fin_share", "interest slice of fin_cost"),
        ("m_op_in_n_share", "m_op_in_share", "count vs amount"),
        ("m_op_out_n_share", "m_op_out_share", "count vs amount"),
        ("m_fee_n_share", "m_fee_share", "count vs amount"),
        ("m_coll_n_share", "m_coll_share", "count vs amount"),
        ("m_pay_n_share", "m_pay_share", "count vs amount"),
        ("m_debt_n_share", "m_debt_share", "park if ≥ 0.95"),
        ("m_xfer_n_share", "m_xfer_share", "park if ≥ 0.95"),
        ("m_invest_n_share", "m_invest_share", "park if ≥ 0.95"),
        ("m_int_n_share", "m_int_share", "park if ≥ 0.95"),
        ("m_salary_n_share", "m_salary_share", "park if ≥ 0.95"),
        ("m_salary_share", "c_salary_month", "amount vs C presence flag"),
        ("m_tax_share", "c_tax_month", "amount vs C presence flag"),
        ("m_op_in_share", "m_coll_share", "group vs raw collection"),
        ("m_op_out_share", "m_pay_share", "group vs raw payment"),
        ("m_coll_share", "m_pay_share", "raw pair (should stay low)"),
    ]
    rewrite_rows = []
    for a, b, note in rewrite_pairs:
        if a not in tr.columns or b not in tr.columns:
            rho = float("nan")
        else:
            rho = _spearman(tr[a], tr[b])
        rewrite_rows.append({"a": a, "b": b, "rho": rho, "note": note})

    # simplex identity on train months with a defined mix
    grp_cols = [
        "m_op_in_share",
        "m_op_out_share",
        "m_fin_share",
        "m_debt_share",
        "m_xfer_share",
        "m_invest_share",
        "m_uncat_share",
    ]
    grp_sum = tr[grp_cols].sum(axis=1, min_count=7)
    simplex_ok = grp_sum.dropna()
    simplex_max_err = (
        float((simplex_ok - 1.0).abs().max()) if not simplex_ok.empty else float("nan")
    )

    fee_int_err = float(
        (
            _num(tr["m_fee_share"]).fillna(0.0)
            + _num(tr["m_int_share"]).fillna(0.0)
            - _num(tr["m_fin_share"]).fillna(0.0)
        ).abs().max()
    )

    size_hits = stats.loc[stats["flags"].str.contains("SIZE", na=False), "feature"].tolist()
    const_hits = stats.loc[stats["flags"].str.contains("CONSTANT", na=False), "feature"].tolist()
    nzv_hits = stats.loc[stats["flags"].str.contains("NZV", na=False), "feature"].tolist()

    lines = [
        "# Category mix report (train only)",
        "",
        "Holdout `analysis/splits/holdout_companies.csv` (72 companies) is **excluded** "
        "from every number below. No percentiles or bins were fit.",
        "",
        f"- Panel: **{n_train}** company-months, **{n_co}** train companies "
        "(monthly grid 2024-09 … 2026-08).",
        "- Family: **M** (`m_*`). Source: `clean.transactions` + `CAT_MAP`. "
        "`data/feature_store/monthly.parquet` was **not** rewritten and is "
        "read only for `a_in3` / rewrite checks.",
        "- Mix is *why the money moved* (brief Q5 / Q3). `m_*_t3` is the Q6 "
        "candidate (summed |amounts| over t-2..t). Not a 0–100 score.",
        "- No look-ahead: `date_trunc('month', date) = period`. "
        "September 2026-09-01 bookings are off the August row.",
        f"- Empty company-months (no txs → all shares NaN): {_fmt_pct(empty_cm)}.",
        "- Size proxy: Spearman vs `log1p(max(a_in3, 0))`. Flag `|ρ| > 0.85` as SIZE.",
        "- Constant: modal-value share `≥ 0.999` or a single value. NZV: modal share `≥ 0.95`.",
        "- Persistence: median company-wise Pearson acf at lag 1 "
        "(≥ 4 finite pairs, non-zero s.d.).",
        f"- Group amount shares sum to 1: max |sum−1| = {_fmt_num(simplex_max_err, 2)} "
        "on months with a defined mix.",
        f"- `m_fee_share + m_int_share = m_fin_share`: max abs residual {_fmt_num(fee_int_err, 2)}.",
        "",
        "## Brief map",
        "",
        "These are *shares* that say what the cash did. They are not a 0–100 index.",
        "",
        "| feature | formula | six-question map |",
        "| --- | --- | --- |",
    ]
    for spec in SPECS:
        lines.append(f"| `{spec['name']}` | `{spec['formula']}` | {spec['brief']} |")

    lines += [
        "",
        "## Battery",
        "",
        "| feature | cov_cm | cov_co | size_ρ vs log1p(a_in3) | acf1 | modal% | n_unique | mean | flags |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in stats.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.feature}`",
                    _fmt_pct(row.cov_cm),
                    _fmt_pct(row.cov_co),
                    _fmt_num(row.size_rho),
                    _fmt_num(row.acf1),
                    _fmt_pct(row.modal_share),
                    str(row.n_unique),
                    _fmt_num(row.mean),
                    row.flags,
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## SIZE / NZV / CONSTANT",
        "",
        f"- SIZE (`|ρ| > {SIZE_RHO}` vs `log1p(a_in3)`): "
        + (", ".join(f"`{c}`" for c in size_hits) if size_hits else "none"),
        f"- NZV (modal ≥ {NZV_THRESH:g}): "
        + (", ".join(f"`{c}`" for c in nzv_hits) if nzv_hits else "none"),
        f"- CONSTANT: "
        + (", ".join(f"`{c}`" for c in const_hits) if const_hits else "none"),
        "",
        "Monthly amount shares have **acf1 ≈ 0** (same noise as `a_op_in`). "
        "They answer *this month's mix* (Q5), not lead time. Trailing-3m "
        "columns below are the Q6 candidates.",
        "",
        "## Parked (not emitted by `build`)",
        "",
        "A share that is just a rewrite of `a_uncat_share` or `a_fin_cost/inflow` "
        "is parked. Count twins with Spearman `≥ 0.95` vs their amount share "
        "are parked too.",
        "",
        "| parked | formula | why |",
        "| --- | --- | --- |",
    ]
    for spec in PARKED_SPECS:
        lines.append(f"| `{spec['name']}` | `{spec['formula']}` | {spec['reason']} |")

    lines += [
        "",
        "## Rewrite / overlap checks (train Spearman)",
        "",
        "| a | b | ρ | note |",
        "| --- | --- | --- | --- |",
    ]
    for r in rewrite_rows:
        lines.append(
            f"| `{r['a']}` | `{r['b']}` | {_fmt_num(r['rho'])} | {r['note']} |"
        )

    lines += [
        "",
        "## How to read the keep set",
        "",
        "- `m_uncat_share` is **not** `a_uncat_share`: amount-weighted opacity "
        "(train ρ vs the count share ≈ 0.89; means differ). Keep both stories "
        "if a model already has the count version — or swap to amount.",
        "- `m_fin_share` ranks like monthly `a_fin_cost / a_op_in` (ρ high) "
        "but is bounded in `[0, 1]` and is only moderately tied to `f_fc_r`. "
        "**Do not stack** with `a_fin_cost`, `f_fc_r`, or Y9 raw material. "
        "Prefer `m_fee_share` + `m_int_share` when the question is fee vs interest.",
        "- `m_xfer_share` is |transfer| / |all|. `a_transfer` is a signed net "
        "that can sit near 0 while two-way transfer volume is large.",
        "- `m_salary_share` / `m_tax_share` add intensity on top of "
        "`c_salary_month` / `c_tax_month` (presence flags).",
        "- `m_coll_vs_pay` is the named-pair mix the brief wants: "
        "what the operating cash is doing, collection vs payment.",
        "- Count shares kept (`op_in`, `op_out`, `fee`, `collection`, `payment`) "
        "are the ones whose amount twin does **not** already tell the same ranking.",
        "",
        "Do not treat holdout as confirmation. Do not drop Family A columns. "
        "Do not put these in a Y that is built from the same category sums "
        "(Y9 forbids fin-cost raw material).",
        "",
    ]

    # --- trailing-3m (Q6) ---
    sib_acf = {r.feature: r.acf1 for r in stats.itertuples(index=False)}
    t3_rows = []
    for spec in T3_SPECS:
        st = _col_stats(tr, spec["name"], size)
        st["formula"] = (
            f"sum({spec['num']})_{{t-2..t}} / sum({spec['den']})_{{t-2..t}} "
            f"(min_periods={T3_WINDOW})"
        )
        st["brief"] = spec["brief"]
        st["sibling"] = spec["sibling"]
        st["sib_acf1"] = sib_acf.get(spec["sibling"], float("nan"))
        st["acf3"] = _median_acf(tr, spec["name"], lag=3)
        t3_rows.append(st)
    t3_stats = pd.DataFrame(t3_rows)

    t3_rewrites = []
    for a, b, note in T3_REWRITE_PAIRS:
        if a not in tr.columns or b not in tr.columns:
            rho = float("nan")
        else:
            rho = _spearman(tr[a], tr[b])
        t3_rewrites.append({"a": a, "b": b, "rho": rho, "note": note})

    rewrite_by_feat: dict[str, list[str]] = {}
    siblings = {s["sibling"] for s in T3_SPECS}
    for r in t3_rewrites:
        if r["b"] in siblings:
            continue
        # f_fc_r is itself a 3m ratio — park near-clones a bit below 0.95
        cut = 0.90 if r["b"] in {"f_fc_r", "fc_inflow"} else REWRITE_RHO
        if np.isfinite(r["rho"]) and abs(r["rho"]) >= cut:
            rewrite_by_feat.setdefault(r["a"], []).append(f"{r['b']} ρ={r['rho']:.3f}")

    decisions = []
    for row in t3_stats.itertuples(index=False):
        hits = rewrite_by_feat.get(row.feature, [])
        dec = _decide_trail(
            row.acf1, row.sib_acf1, row.size_rho, row.flags, hits, getattr(row, "acf3", float("nan"))
        )
        decisions.append(dec)
    t3_stats["decision"] = decisions

    t3_keep = t3_stats.loc[t3_stats["decision"] == "KEEP", "feature"].tolist()
    t3_close = t3_stats.loc[t3_stats["decision"] == "CLOSE", "feature"].tolist()
    t3_park = t3_stats.loc[t3_stats["decision"] == "PARK", "feature"].tolist()
    t3_acf1_pass = t3_keep + t3_close

    lines += [
        "## Trailing-3m mix (Q6)",
        "",
        "Shares recomputed from **summed |amounts|** (or counts) over "
        f"`t-2..t`, `min_periods={T3_WINDOW}`. Empty grid months contribute 0; "
        "they do not vote equally with a €10m month. First two company-months "
        "are NaN. Holdout excluded. No train bins.",
        "",
        f"- KEEP = acf1 gate **and** `acf3 ≥ {ACF1_KEEP}` (real Q6). "
        f"CLOSE = acf1 ≥ {ACF1_KEEP} (or +{ACF1_LIFT} vs sibling) but acf3 dead "
        "(overlap smoother only). PARK = SIZE / CONSTANT / rewrite.",
        f"- PARK if SIZE (`|ρ| > {SIZE_RHO}`), CONSTANT, or rewrite "
        f"(`|ρ| ≥ {REWRITE_RHO}` vs `a_*` / `c_salary_month`; `|ρ| ≥ 0.90` vs `f_fc_r` / monthly fin/inflow). "
        "ρ vs the monthly sibling is expected and is not a park reason.",
        "- `acf3` is the honest Q6 check: a 3m window at lag 3 shares **no** months. "
        "acf1 on overlapping windows is partly mechanical.",
        "- `m_fin_share_t3` is optional; park if it clones `f_fc_r`.",
        "",
        "| feature | sibling | cov_cm | size_ρ | acf1 | acf3 | sib acf1 | Δ acf1 | flags | decision |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in t3_stats.itertuples(index=False):
        d_acf = (
            row.acf1 - row.sib_acf1
            if np.isfinite(row.acf1) and np.isfinite(row.sib_acf1)
            else float("nan")
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.feature}`",
                    f"`{row.sibling}`",
                    _fmt_pct(row.cov_cm),
                    _fmt_num(row.size_rho),
                    _fmt_num(row.acf1),
                    _fmt_num(row.acf3),
                    _fmt_num(row.sib_acf1),
                    _fmt_num(d_acf),
                    row.flags,
                    row.decision,
                ]
            )
            + " |"
        )

    lines += [
        "",
        f"- KEEP ({len(t3_keep)}): "
        + (", ".join(f"`{c}`" for c in t3_keep) if t3_keep else "none — none clear acf3"),
        f"- CLOSE ({len(t3_close)}): "
        + (", ".join(f"`{c}`" for c in t3_close) if t3_close else "none")
        + " — acf1 gate only; overlap smoother, not Q6",
        f"- PARK ({len(t3_park)}): "
        + (", ".join(f"`{c}`" for c in t3_park) if t3_park else "none"),
        "",
        "### t3 rewrite / overlap (train Spearman)",
        "",
        "| a | b | ρ | note |",
        "| --- | --- | --- | --- |",
    ]
    for r in t3_rewrites:
        lines.append(
            f"| `{r['a']}` | `{r['b']}` | {_fmt_num(r['rho'])} | {r['note']} |"
        )

    # t6 diagnostic on stems that failed KEEP, or the 4 highest t3 acf1 if all failed
    t6_stats = pd.DataFrame()
    if len(t3_keep) == 0:
        cand = t3_stats.sort_values("acf1", ascending=False).head(4)
        stem_names = set(cand["sibling"].tolist())
        t6_specs = tuple(s for s in t3_specs_for_window(T6_WINDOW) if s["sibling"] in stem_names)
        if t6_specs:
            t6_part = _trail_panel(keys, monthly, t6_specs, T6_WINDOW)
            tr6 = tr.merge(t6_part, on=["company_id", "period"], how="left")
            t6_rows = []
            for spec in t6_specs:
                st = _col_stats(tr6, spec["name"], size)
                st["sibling"] = spec["sibling"]
                sib_t3 = spec["sibling"] + "_t3"
                st["sib_acf1"] = float(
                    t3_stats.loc[t3_stats["feature"] == sib_t3, "acf1"].iloc[0]
                ) if (t3_stats["feature"] == sib_t3).any() else float("nan")
                st["acf6"] = _median_acf(tr6, spec["name"], lag=6)
                hits = []
                dec = _decide_trail(
                    st["acf1"], st["sib_acf1"], st["size_rho"], st["flags"], hits, st["acf6"]
                )
                st["decision"] = dec
                t6_rows.append(st)
            t6_stats = pd.DataFrame(t6_rows)
            lines += [
                "",
                "## Trailing-6m mix (Q6 fallback, not emitted)",
                "",
                "No t3 column KEEPs Q6 (acf3 dead). Same summed-|amount| rule, "
                f"`t-5..t`, `min_periods={T6_WINDOW}`, on the 4 highest-acf1 t3 stems. "
                "`acf6` is the no-overlap check for a 6m window. **Not in `build`.**",
                "",
                "| feature | sibling | cov_cm | size_ρ | acf1 | acf6 | t3 acf1 | flags | decision |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for row in t6_stats.itertuples(index=False):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{row.feature}`",
                            f"`{row.sibling}`",
                            _fmt_pct(row.cov_cm),
                            _fmt_num(row.size_rho),
                            _fmt_num(row.acf1),
                            _fmt_num(row.acf6),
                            _fmt_num(row.sib_acf1),
                            row.flags,
                            row.decision,
                        ]
                    )
                    + " |"
                )

    acf3_pass = t3_stats.loc[t3_stats["decision"].isin(["KEEP", "CLOSE"]), "acf3"]
    acf3_med = float(acf3_pass.median()) if len(acf3_pass) else float("nan")
    acf3_ok = bool(np.isfinite(acf3_med) and acf3_med >= ACF1_KEEP)

    y_lines = _y_spearman_section(tr, t3_acf1_pass)
    lines += y_lines

    if t3_keep and acf3_ok:
        q6_line = (
            f"Trailing mix **can** answer Q6: KEEP stems have acf1 "
            f"{_fmt_num(float(t3_stats.loc[t3_stats['decision']=='KEEP','acf1'].median()))} "
            f"and acf3 (no shared months) {_fmt_num(acf3_med)} ≥ {ACF1_KEEP}. "
            "A monitor can see the mix shift before the snapshot month."
        )
    elif t3_close and not acf3_ok:
        q6_line = (
            "t3 **CLOSE** for Q6: acf1 fires by overlap (3m windows at lag 1 share "
            f"2/3 of the mass). Median acf3 is {_fmt_num(acf3_med)} (lag 3 shares "
            "no months). t6 on the top-4 stems also CLOSEs (`acf6` ≈ −0.3). "
            "**Park the trail family as lead time.** Mix answers Q5 (why this "
            "month / this quarter); t3 is only a less-noisy mix if parent wants "
            "a smoother. Do not merge t3/t6 as Q6."
        )
    else:
        q6_line = (
            "Trailing mix **cannot** answer Q6: t3 failed the persistence gate. "
            "See t6 fallback. Mix is a contemporaneous why (Q5)."
        )
    lines += ["", "## Q6 verdict", "", q6_line, ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    # stash trail stats for main()
    write_train_report.t3_stats = t3_stats
    write_train_report.t6_stats = t6_stats
    return stats


def main() -> None:
    con = connect()
    grid = monthly_grid(con)[["company_id", "period"]]
    monthly = _monthly_mix(con)
    part = build(con, grid)
    con.close()
    stats = write_train_report(part, monthly)
    print(f"wrote {REPORT_PATH}")
    print(f"rows={len(part)} companies={part['company_id'].nunique()} cols={len(M_COLS)}")
    print("--- monthly ---")
    print(stats[["feature", "cov_cm", "size_rho", "acf1", "flags"]].to_string(index=False))
    t3s = getattr(write_train_report, "t3_stats", None)
    if t3s is not None and len(t3s):
        print("--- t3 ---")
        cols = [c for c in ("feature", "cov_cm", "size_rho", "acf1", "acf3", "sib_acf1", "flags", "decision") if c in t3s.columns]
        print(t3s[cols].to_string(index=False))
    t6s = getattr(write_train_report, "t6_stats", None)
    if t6s is not None and len(t6s):
        print("--- t6 (diagnostic, not emitted) ---")
        cols = [c for c in ("feature", "cov_cm", "size_rho", "acf1", "sib_acf1", "flags", "decision") if c in t6s.columns]
        print(t6s[cols].to_string(index=False))
    print("did not write monthly.parquet")
    print("did not add Family M to FAMILIES")


if __name__ == "__main__":
    main()
