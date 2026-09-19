"""Family J — invoice↔tx amount-match *rates* (not pairs).

Company-month rates. Never explode the panel into matched invoice-tx rows.
No look-ahead: for period P (month-start) use invoices whose payment_date
or issuance_date is ≤ period end, and txs with date ≤ period end.

Preferred window = **payment-month**. QA KEEP number: 35.7% of paid
invoices have a same-company same-sign tx in the payment month at
|Δ| ≤ 0.01 €, vs 0.5% random. Issuance-month is 21.2% (diagnostic).

Greedy 1-1 inside the month: within each (company, period, sign,
amount-to-cents) bucket, ``n_matched = min(n_inv, n_tx)``. An invoice
matches at most one tx; a tx matches at most one invoice of that exact
cent amount. This is stricter than the QA set-overlap (any-tx-exists
counts every invoice at that amount).

``j_pay_match`` is NaN when the company-month has no paid book invoices
(do **not** fill 0 — 0 would mean “all unmatched”). The 470 never-ERP
companies stay NaN on the rate and 0 on ``j_has_book``.

This module does **not** rewrite ``monthly.parquet`` and is **not** in
``build_feature_store.FAMILIES``. Parent merges after review. Does not
import other family modules. Does not emit a row-level FK, a
COMP_*↔COUNTERPARTY_* map, or ``j_interco_*``.

Train-only diagnostics: ``python -m analysis.features.match`` writes
``analysis/outputs/match_report.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.features.common import (
    AS_OF,
    ROOT,
    connect,
    train_mask,
)
from analysis.features.grid import monthly_grid, weekly_grid

FAMILY = "j"
SOURCE_TABLES = ["transactions", "invoices"]
TOL_CENTS = 0.01  # |Δ| from QA; do not retune on holdout

STORE = ROOT / "data/feature_store/monthly.parquet"
TARGETS = ROOT / "data/feature_store/targets.parquet"
REPORT_PATH = ROOT / "analysis/outputs/match_report.md"
REGISTRY = ROOT / "analysis/experiments/registry.csv"

SIZE_RHO = 0.85
REWRITE_RHO = 0.95
NZV_THRESH = 0.95
CONSTANT_THRESH = 0.99
MIN_ACF_PAIRS = 6
ACF1_KEEP = 0.25
Y3_X_RHO = 0.08
PAIR_DROP_RHO = 0.90
AGENT = "d56ee5fe"

BOOK = """
    document_type = 'invoice'
    AND (status IS NULL OR status <> 'cancel')
    AND amount <> 0
    AND issuance_date IS NOT NULL
"""

# Emitted keep-set after pass 2:
# - drop t3 (acf3 ≈ 0; acf1 was overlap)
# - drop unmatched (ρ = −1)
# - drop counts (Y3 |ρ| dies after size / e_ar_issued control)
# - keep iss (ρ vs pay = 0.68 < 0.9)
J_COLS: tuple[str, ...] = (
    "j_pay_match",
    "j_iss_match",
    "j_has_book",
)

REWRITE_COLS = (
    "e_ar_issued",
    "e_delay_coll",
    "e_pending_amt_share",
    "e_ar_open",
    "e_dso_proxy",
)

Y_COLS = (
    "y3_recover_cash_6m",
    "y7_top1_lost",
    "y8_inv_worse_6",
    "y8_cash_worse_6",
)


def _infer_freq(periods: pd.Series) -> str:
    u = pd.to_datetime(pd.unique(periods.dropna()))
    if len(u) <= 1:
        p = pd.Timestamp(u[0]) if len(u) else pd.NaT
        if pd.isna(p):
            return "M"
        return "M" if int(p.day) == 1 else "W"
    med = pd.Series(sorted(u)).diff().median()
    return "W" if med <= pd.Timedelta(days=10) else "M"


def _empty(grid: pd.DataFrame) -> pd.DataFrame:
    keys = (
        grid[["company_id", "period"]].copy()
        if grid is not None and not grid.empty
        else pd.DataFrame(columns=["company_id", "period"])
    )
    out = keys.copy()
    out["company_id"] = out["company_id"].astype(str) if len(out) else out.get(
        "company_id", pd.Series(dtype=str)
    )
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"])
    for c in J_COLS:
        out[c] = np.nan
    return out.reset_index(drop=True)


def _trunc(freq: str) -> str:
    return "month" if freq == "M" else "week"


def _inv_buckets(con, date_col: str, trunc: str, paid_only: bool) -> pd.DataFrame:
    """Invoice counts by (company, period, sign, cents)."""
    extra = ""
    if paid_only:
        extra = """
          AND payment_date IS NOT NULL
          AND COALESCE(payment_date_invalid, FALSE) = FALSE
        """
    sql = f"""
        SELECT
            company_id,
            date_trunc('{trunc}', {date_col})::DATE AS period,
            CASE WHEN amount > 0 THEN 1 ELSE -1 END AS sgn,
            ROUND(ABS(amount), 2) AS amt_c,
            COUNT(*)::BIGINT AS n_inv
        FROM invoices
        WHERE {BOOK}
          {extra}
          AND {date_col} IS NOT NULL
          AND {date_col} < DATE '{AS_OF.date()}'
        GROUP BY 1, 2, 3, 4
    """
    return con.execute(sql).df()


def _tx_buckets(con, trunc: str) -> pd.DataFrame:
    sql = f"""
        SELECT
            company_id,
            date_trunc('{trunc}', date)::DATE AS period,
            CASE WHEN amount > 0 THEN 1 ELSE -1 END AS sgn,
            ROUND(ABS(amount), 2) AS amt_c,
            COUNT(*)::BIGINT AS n_tx
        FROM transactions
        WHERE amount <> 0
          AND date IS NOT NULL
          AND date < DATE '{AS_OF.date()}'
        GROUP BY 1, 2, 3, 4
    """
    return con.execute(sql).df()


def _first_iss(con) -> pd.DataFrame:
    sql = f"""
        SELECT
            company_id,
            MIN(issuance_date)::DATE AS first_iss
        FROM invoices
        WHERE {BOOK}
        GROUP BY 1
    """
    return con.execute(sql).df()


def _match_month(
    inv: pd.DataFrame, tx: pd.DataFrame
) -> pd.DataFrame:
    """Greedy 1-1 + set-overlap + wrong-sign at company-period."""
    if inv.empty:
        return pd.DataFrame(
            columns=[
                "company_id",
                "period",
                "n_paid",
                "n_matched",
                "n_set_overlap",
                "n_wrong",
            ]
        )
    inv = inv.copy()
    inv["company_id"] = inv["company_id"].astype(str)
    inv["period"] = pd.to_datetime(inv["period"])
    tx = tx.copy()
    tx["company_id"] = tx["company_id"].astype(str)
    tx["period"] = pd.to_datetime(tx["period"])

    keys = ["company_id", "period", "sgn", "amt_c"]
    m = inv.merge(tx, on=keys, how="left")
    n_tx = m["n_tx"].fillna(0)
    m["n_1to1"] = np.minimum(m["n_inv"].to_numpy(), n_tx.to_numpy())
    m["n_set"] = np.where(n_tx.to_numpy() > 0, m["n_inv"].to_numpy(), 0)

    wrong_tx = tx.rename(columns={"n_tx": "n_tx_w", "sgn": "sgn_tx"})
    w = inv.merge(
        wrong_tx,
        left_on=["company_id", "period", "amt_c"],
        right_on=["company_id", "period", "amt_c"],
        how="left",
    )
    w = w[w["sgn_tx"].notna() & (w["sgn"] != w["sgn_tx"])]
    if w.empty:
        w_agg = inv.assign(n_wrong_b=0).groupby(
            ["company_id", "period"], as_index=False
        )["n_wrong_b"].sum()
        w_agg = w_agg.rename(columns={"n_wrong_b": "n_wrong"})
        w_agg["n_wrong"] = 0
    else:
        w["n_wrong_b"] = np.minimum(
            w["n_inv"].to_numpy(), w["n_tx_w"].fillna(0).to_numpy()
        )
        w_agg = w.groupby(["company_id", "period"], as_index=False)["n_wrong_b"].sum()
        w_agg = w_agg.rename(columns={"n_wrong_b": "n_wrong"})

    agg = m.groupby(["company_id", "period"], as_index=False).agg(
        n_paid=("n_inv", "sum"),
        n_matched=("n_1to1", "sum"),
        n_set_overlap=("n_set", "sum"),
    )
    agg = agg.merge(w_agg, on=["company_id", "period"], how="left")
    agg["n_wrong"] = agg["n_wrong"].fillna(0)
    return agg


def _rate(num: pd.Series, den: pd.Series) -> pd.Series:
    den_n = pd.to_numeric(den, errors="coerce")
    num_n = pd.to_numeric(num, errors="coerce")
    den_v = den_n.to_numpy(dtype=float)
    num_v = num_n.to_numpy(dtype=float)
    out = np.full(den_v.shape, np.nan, dtype=float)
    ok = np.isfinite(den_v) & (den_v > 0)
    out[ok] = num_v[ok] / den_v[ok]
    return pd.Series(out, index=den.index)


def _trailing_sum_rate(
    keys: pd.DataFrame, matched: pd.Series, paid: pd.Series, window: int
) -> pd.Series:
    """sum(matched) / sum(paid) over the last ``window`` grid months.

    Empty months contribute 0 (not a vote). ``min_periods=window``.
    This is not the mean of monthly rates.
    """
    work = keys[["company_id", "period"]].copy()
    work["_m"] = pd.to_numeric(matched, errors="coerce").fillna(0.0)
    work["_p"] = pd.to_numeric(paid, errors="coerce").fillna(0.0)
    work = work.sort_values(["company_id", "period"])
    g = work.groupby("company_id", sort=False)
    sm = g["_m"].transform(lambda s: s.rolling(window, min_periods=window).sum())
    sp = g["_p"].transform(lambda s: s.rolling(window, min_periods=window).sum())
    return _rate(sm, sp)


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    """Return ``company_id, period`` plus ``j_*``. Queries invoices + txs."""
    if grid is None or grid.empty:
        return _empty(grid if grid is not None else pd.DataFrame())

    keys = grid[["company_id", "period"]].copy()
    keys["company_id"] = keys["company_id"].astype(str)
    keys["period"] = pd.to_datetime(keys["period"])
    freq = _infer_freq(keys["period"])
    trunc = _trunc(freq)

    tx = _tx_buckets(con, trunc)
    pay_inv = _inv_buckets(con, "payment_date", trunc, paid_only=True)
    iss_inv = _inv_buckets(con, "issuance_date", trunc, paid_only=False)
    pay = _match_month(pay_inv, tx)
    iss = _match_month(iss_inv, tx)
    first = _first_iss(con)
    first["company_id"] = first["company_id"].astype(str)
    first["first_iss"] = pd.to_datetime(first["first_iss"])

    out = keys.merge(
        pay.rename(
            columns={
                "n_paid": "n_paid",
                "n_matched": "n_matched",
                "n_set_overlap": "n_set_overlap",
                "n_wrong": "n_wrong",
            }
        ),
        on=["company_id", "period"],
        how="left",
    )
    iss_r = iss.rename(
        columns={
            "n_paid": "n_iss",
            "n_matched": "n_iss_matched",
            "n_set_overlap": "n_iss_set",
            "n_wrong": "n_iss_wrong",
        }
    )
    out = out.merge(iss_r, on=["company_id", "period"], how="left")
    out = out.merge(first, on="company_id", how="left")

    if freq == "M":
        period_end = out["period"] + pd.offsets.MonthEnd(0)
    else:
        period_end = out["period"] + pd.Timedelta(days=6)
    out["j_pay_match"] = _rate(out["n_matched"], out["n_paid"])
    out["j_iss_match"] = _rate(out["n_iss_matched"], out["n_iss"])
    out["j_has_book"] = (
        out["first_iss"].notna() & (out["first_iss"] <= period_end)
    ).astype(float)
    # t3 computed for the report only (overlap-only acf1; not emitted)
    out["_j_pay_match_t3"] = _trailing_sum_rate(
        out, out["n_matched"], out["n_paid"], window=3
    )

    # stash internals for the report (not returned)
    build._last_internal = out[
        [
            "company_id",
            "period",
            "n_paid",
            "n_matched",
            "n_set_overlap",
            "n_wrong",
            "n_iss",
            "n_iss_matched",
            "_j_pay_match_t3",
        ]
    ].copy()

    unexpected = [c for c in J_COLS if c not in out.columns]
    if unexpected:
        raise ValueError(f"Family J missing columns {unexpected}")
    extra = [c for c in out.columns if c.startswith("j_") and c not in J_COLS]
    if extra:
        raise ValueError(f"Family J unexpected j_* {extra}")
    if out.duplicated(["company_id", "period"]).any():
        raise ValueError("Family J produced duplicate company_id, period rows")
    return out[["company_id", "period", *J_COLS]].reset_index(drop=True)


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


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x, y = _num(a), _num(b)
    m = x.notna() & y.notna()
    if int(m.sum()) < 50:
        return float("nan")
    if x[m].nunique() <= 1 or y[m].nunique() <= 1:
        return float("nan")
    return float(x[m].corr(y[m], method="spearman"))


def _rank_resid(y: pd.Series, x: pd.Series) -> pd.Series:
    a, b = _num(y), _num(x)
    m = a.notna() & b.notna()
    out = pd.Series(np.nan, index=y.index)
    if int(m.sum()) < 50:
        return out
    ra = a[m].rank().to_numpy()
    rb = b[m].rank().to_numpy()
    xb = np.column_stack([np.ones(int(m.sum())), rb])
    coef, *_ = np.linalg.lstsq(xb, ra, rcond=None)
    out.loc[m] = ra - xb @ coef
    return out


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


def _load_store(columns: list[str]) -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing — report needs a_in3 / e_*")
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


def _load_targets(columns: list[str]) -> pd.DataFrame:
    if not TARGETS.exists():
        return pd.DataFrame()
    import pyarrow.parquet as pq

    have = set(pq.read_schema(TARGETS).names)
    present = [c for c in columns if c in have]
    t = pd.read_parquet(TARGETS, columns=["company_id", "period", *present])
    t["company_id"] = t["company_id"].astype(str)
    t["period"] = pd.to_datetime(t["period"])
    return t


def _col_stats(
    tr: pd.DataFrame, col: str, size: pd.Series, ever_erp: pd.Series
) -> dict:
    s = _num(tr[col])
    nn = s.dropna()
    n_cm = len(tr)
    n_co = int(tr["company_id"].nunique())
    cov_cm = float(s.notna().mean()) if n_cm else float("nan")
    cov_co = (
        float(tr.loc[s.notna(), "company_id"].nunique() / n_co) if n_co else float("nan")
    )
    erp_mask = ever_erp.fillna(False)
    if erp_mask.any():
        cov_erp = float(s[erp_mask].notna().mean())
        n_erp_cm = int(erp_mask.sum())
    else:
        cov_erp = float("nan")
        n_erp_cm = 0
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
    acf1 = _median_acf(tr, col, lag=1)
    acf3 = _median_acf(tr, col, lag=3)
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
        "cov_erp": cov_erp,
        "n_erp_cm": n_erp_cm,
        "n_unique": n_unique,
        "modal_share": modal_share,
        "size_rho": size_rho,
        "acf1": acf1,
        "acf3": acf3,
        "median": med,
        "mean": mean,
        "flags": ",".join(flags) if flags else "—",
    }


def _decide(
    row: dict,
    rewrite_hits: list[str],
    pair_rho: float | None = None,
    is_t3: bool = False,
    y3_rho: float = float("nan"),
) -> str:
    flags = row.get("flags") or ""
    if "SIZE" in flags or "CONSTANT" in flags:
        return "PARK"
    if rewrite_hits:
        return "PARK"
    if pair_rho is not None and np.isfinite(pair_rho) and abs(pair_rho) >= PAIR_DROP_RHO:
        return "PARK"
    acf1 = row.get("acf1", float("nan"))
    acf3 = row.get("acf3", float("nan"))
    if is_t3:
        # Honesty: overlapping t3 makes acf1 look persistent. Require lag-3.
        if np.isfinite(acf3) and acf3 >= ACF1_KEEP:
            return "KEEP"
        if np.isfinite(acf1) and acf1 >= ACF1_KEEP:
            return "CLOSE"  # overlap-only persistence
        return "PARK"
    # monthly rate: KEEP as Q5 diagnostic if it has coverage + variation
    cov = row.get("cov_cm", float("nan"))
    if not np.isfinite(cov) or cov < 0.05:
        return "PARK"
    if "NZV" in flags:
        return "PARK"
    # Y3 X vs Q5: CLOSE-as-Y3-X if |ρ| < 0.08, still KEEP as diagnostic
    if np.isfinite(y3_rho) and abs(y3_rho) < Y3_X_RHO:
        return "KEEP-Q5"
    return "KEEP"


def write_train_report(part: pd.DataFrame) -> pd.DataFrame:
    tr = part.loc[train_mask(part["company_id"])].copy()
    store = _load_store(["a_in3", *REWRITE_COLS])
    tr = tr.merge(store, on=["company_id", "period"], how="left")
    targs = _load_targets(list(Y_COLS))
    if not targs.empty:
        tr = tr.merge(targs, on=["company_id", "period"], how="left")

    internal = getattr(build, "_last_internal", None)
    if internal is not None:
        inn = internal.copy()
        inn["company_id"] = inn["company_id"].astype(str)
        inn["period"] = pd.to_datetime(inn["period"])
        tr = tr.merge(inn, on=["company_id", "period"], how="left")

    tr["j_pay_unmatched"] = 1.0 - _num(tr["j_pay_match"])
    if "_j_pay_match_t3" in tr.columns:
        tr["j_pay_match_t3"] = _num(tr["_j_pay_match_t3"])
    if "n_paid" in tr.columns:
        tr["j_n_paid"] = _num(tr["n_paid"])
        tr["j_n_matched"] = _num(tr["n_matched"])
        tr["j_pay_set"] = _rate(tr["n_set_overlap"], tr["n_paid"])
        tr["j_pay_wrong"] = _rate(tr["n_wrong"], tr["n_paid"])
    size = np.log1p(np.maximum(_num(tr["a_in3"]), 0.0))
    ever_erp = (
        tr.groupby("company_id")["j_has_book"].transform("max").fillna(0) >= 1
    )

    diag_cols = list(J_COLS) + [
        c
        for c in (
            "j_pay_match_t3",
            "j_pay_unmatched",
            "j_n_paid",
            "j_n_matched",
            "j_pay_set",
            "j_pay_wrong",
        )
        if c in tr.columns
    ]
    stats_rows = [_col_stats(tr, c, size, ever_erp) for c in diag_cols]
    stats = pd.DataFrame(stats_rows)

    # rewrite ρ vs e_*
    rewrite: dict[str, list[str]] = {c: [] for c in diag_cols}
    rewrite_rows: list[str] = []
    for feat in diag_cols:
        for e in REWRITE_COLS:
            if e not in tr.columns:
                continue
            rho = _spearman(tr[feat], tr[e])
            if np.isfinite(rho) and abs(rho) >= REWRITE_RHO:
                rewrite[feat].append(f"{e}={rho:.3f}")
            rewrite_rows.append(
                f"| `{feat}` | `{e}` | {_fmt_num(rho)} |"
            )

    pay_iss = _spearman(tr["j_pay_match"], tr["j_iss_match"])
    pay_unm = _spearman(tr["j_pay_match"], tr["j_pay_unmatched"])
    pay_t3 = (
        _spearman(tr["j_pay_match"], tr["j_pay_match_t3"])
        if "j_pay_match_t3" in tr
        else float("nan")
    )
    paid_mask = _num(tr.get("n_paid", pd.Series(dtype=float))).fillna(0) > 0
    paid = tr.loc[paid_mask]
    inv_w_1to1 = float(
        _num(paid["n_matched"]).sum() / _num(paid["n_paid"]).sum()
    ) if paid_mask.any() and _num(paid["n_paid"]).sum() > 0 else float("nan")
    inv_w_set = float(
        _num(paid["n_set_overlap"]).sum() / _num(paid["n_paid"]).sum()
    ) if paid_mask.any() and _num(paid["n_paid"]).sum() > 0 else float("nan")
    size_paid = np.log1p(np.maximum(_num(paid["a_in3"]), 0.0)) if len(paid) else pd.Series(dtype=float)
    n_paid_y3_raw = float("nan")
    n_paid_y3_size = float("nan")
    n_paid_y3_ear = float("nan")
    n_match_y3_npay = float("nan")
    iss_pay_y3 = float("nan")
    if "y3_recover_cash_6m" in tr.columns and len(paid):
        y3p = paid[paid["y3_recover_cash_6m"].notna()]
        n_paid_y3_raw = _spearman(y3p["j_n_paid"], y3p["y3_recover_cash_6m"])
        r_n = _rank_resid(y3p["j_n_paid"], np.log1p(np.maximum(_num(y3p["a_in3"]), 0.0)))
        r_y = _rank_resid(y3p["y3_recover_cash_6m"], np.log1p(np.maximum(_num(y3p["a_in3"]), 0.0)))
        n_paid_y3_size = float(r_n.corr(r_y, method="spearman")) if r_n.notna().sum() >= 50 else float("nan")
        if "e_ar_issued" in y3p.columns:
            r_n2 = _rank_resid(y3p["j_n_paid"], y3p["e_ar_issued"])
            r_y2 = _rank_resid(y3p["y3_recover_cash_6m"], y3p["e_ar_issued"])
            n_paid_y3_ear = float(r_n2.corr(r_y2, method="spearman")) if r_n2.notna().sum() >= 50 else float("nan")
        r_m = _rank_resid(y3p["j_n_matched"], y3p["j_n_paid"])
        r_ym = _rank_resid(y3p["y3_recover_cash_6m"], y3p["j_n_paid"])
        n_match_y3_npay = float(r_m.corr(r_ym, method="spearman")) if r_m.notna().sum() >= 50 else float("nan")
        r_i = _rank_resid(y3p["j_iss_match"], y3p["j_pay_match"])
        iss_pay_y3 = float(r_i.corr(y3p["y3_recover_cash_6m"], method="spearman")) if r_i.notna().sum() >= 50 else float("nan")
    del size_paid

    y_rows: list[str] = []
    y_rhos: dict[str, dict[str, float]] = {}
    for feat in diag_cols:
        y_rhos[feat] = {}
        for yc in Y_COLS:
            if yc not in tr.columns:
                continue
            lab = tr[tr[yc].notna()]
            rho = _spearman(lab[feat], lab[yc]) if len(lab) else float("nan")
            y_rhos[feat][yc] = rho
            y_rows.append(
                f"| `{feat}` | `{yc}` | {len(lab):,} | {_fmt_num(rho)} |"
            )

    decisions: dict[str, str] = {}
    for _, r in stats.iterrows():
        feat = r["feature"]
        pair = None
        if feat == "j_iss_match":
            pair = pay_iss
        if feat == "j_pay_unmatched":
            pair = pay_unm
        decisions[feat] = _decide(
            r.to_dict(),
            rewrite[feat],
            pair_rho=pair,
            is_t3=feat.endswith("_t3"),
            y3_rho=y_rhos.get(feat, {}).get("y3_recover_cash_6m", float("nan")),
        )
        if feat in {"j_n_paid", "j_n_matched"}:
            # Raw Y3 |ρ| ≈ 0.15 but residual vs size / e_ar_issued ≈ 0. Prefer rates.
            decisions[feat] = "PARK"
        if feat == "j_pay_set":
            decisions[feat] = "PARK"  # near-twin of 1-1 (cm-mean 41.1% vs 39.2%)
        if feat == "j_pay_wrong":
            decisions[feat] = "PARK"  # control; mean 1.5%, stay in report
        if feat == "j_pay_match_t3":
            decisions[feat] = "CLOSE"  # overlap-only; not emitted

    # 470 NaN check
    never = ~ever_erp
    never_cos = tr.loc[never, "company_id"].nunique()
    pay_on_never = float(tr.loc[never, "j_pay_match"].notna().mean()) if never.any() else float("nan")
    has_on_never = float(tr.loc[never, "j_has_book"].fillna(0).mean()) if never.any() else float("nan")
    n_paid_pos = int((_num(tr.get("n_paid", pd.Series(dtype=float))).fillna(0) > 0).sum())
    mean_1to1 = float(_num(tr["j_pay_match"]).mean())
    mean_set = float(_num(tr["j_pay_set"]).mean()) if "j_pay_set" in tr else float("nan")
    mean_wrong = float(_num(tr["j_pay_wrong"]).mean()) if "j_pay_wrong" in tr else float("nan")

    y3_pay = y_rhos.get("j_pay_match", {}).get("y3_recover_cash_6m", float("nan"))
    y3_unm = y_rhos.get("j_pay_unmatched", {}).get("y3_recover_cash_6m", float("nan"))
    j_is_y3_x = np.isfinite(y3_pay) and abs(y3_pay) >= Y3_X_RHO

    lines = [
        "# Family J — amount-match rates",
        "",
        "Train-only. Holdout 72 out. Company-month *rates*, not invoice-tx pairs.",
        "Greedy 1-1 inside `(company, period, sign, cents)`. `|Δ| ≤ 0.01` from QA;",
        "not retuned. `build()` does not read or write the parquet.",
        "",
        "## Contract",
        "",
        "- `SOURCE_TABLES = [\"transactions\", \"invoices\"]`, `FAMILY = \"j\"`.",
        "- Period P uses payment/issuance/tx dates ≤ period end. Last raw date in",
        "  store is 2026-09-01; those rows stay off August (`date < 2026-09-01`).",
        "- `j_pay_match` = share of *paid this month* book invoices with a",
        "  same-company same-sign tx in the payment month, greedy 1-1.",
        "  **NaN if no paid invoices** (470 stay NaN, not 0).",
        "- `j_iss_match` = same on issuance-month (QA 21.2%). Diagnostic.",
        "- `j_has_book` = 1 if the company had a book invoice with",
        "  `issuance_date ≤ period end` (causal miss flag).",
        "- `j_pay_match_t3` = `sum(matched) / sum(paid)` over t-2..t,",
        "  `min_periods=3`. Not the mean of monthly rates.",
        "- Not emitted: row FK, COMP↔CP map, `j_interco_*`, D-family HHI.",
        "",
        "## 470 NaN check (hard rule)",
        "",
        f"- Never-ERP train companies (max `j_has_book` = 0): **{never_cos}**.",
        f"- `j_pay_match` non-null on those cm: **{_fmt_pct(pay_on_never)}** (must be 0).",
        f"- Mean `j_has_book` on those cm: **{_fmt_num(has_on_never)}** (must be 0).",
        f"- Train cm with ≥1 paid invoice: **{n_paid_pos:,}** / {len(tr):,}.",
        "",
        "## Coverage / size / acf",
        "",
        "| feature | cov_all_cm | cov_ever-ERP_cm | cov_co | size_ρ vs log1p(a_in3) | acf1 | acf3 | modal% | n_unique | mean | flags | decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for _, r in stats.iterrows():
        feat = r["feature"]
        lines.append(
            f"| `{feat}` | {_fmt_pct(r['cov_cm'])} | {_fmt_pct(r['cov_erp'])} "
            f"| {_fmt_pct(r['cov_co'])} | {_fmt_num(r['size_rho'])} "
            f"| {_fmt_num(r['acf1'])} | {_fmt_num(r['acf3'])} "
            f"| {_fmt_pct(r['modal_share'])} | {r['n_unique']} "
            f"| {_fmt_num(r['mean'])} | {r['flags']} | **{decisions[feat]}** |"
        )

    lines += [
        "",
        "## Pair / complement ρ",
        "",
        f"- `j_pay_match` vs `j_iss_match`: **{_fmt_num(pay_iss)}** "
        f"(drop iss if |ρ| ≥ {PAIR_DROP_RHO}).",
        f"- `j_pay_match` vs `j_pay_unmatched`: **{_fmt_num(pay_unm)}** "
        "(expect −1; emit only one).",
        f"- `j_pay_match` vs `j_pay_match_t3`: **{_fmt_num(pay_t3)}**.",
        "",
        "## 1-1 vs QA set-overlap (train cm with paid invoices)",
        "",
        f"- Cm-mean `j_pay_match` (greedy 1-1): **{_fmt_pct(mean_1to1)}**.",
        f"- Cm-mean set-overlap (QA-style any-tx-exists): **{_fmt_pct(mean_set)}**.",
        f"- Invoice-weighted 1-1 (`sum n_matched / sum n_paid`): **{_fmt_pct(inv_w_1to1)}**.",
        f"- Invoice-weighted set-overlap: **{_fmt_pct(inv_w_set)}** (QA KEEP headline 35.7%).",
        f"- Mean wrong-sign control: **{_fmt_pct(mean_wrong)}** (report only).",
        "",
        "## Rewrite ρ vs `e_*` (PARK if |ρ| ≥ 0.95)",
        "",
        "| feature | e_* | Spearman |",
        "| --- | --- | ---: |",
        *rewrite_rows,
        "",
        "## Spearman vs Y (train labeled)",
        "",
        "| feature | y | n_labeled | Spearman |",
        "| --- | --- | ---: | ---: |",
        *y_rows,
        "",
        "## Y3 X vs Q5 diagnostic",
        "",
        f"- `j_pay_match` vs `y3_recover_cash_6m`: **{_fmt_num(y3_pay)}**.",
        f"- `j_pay_unmatched` vs `y3_recover_cash_6m`: **{_fmt_num(y3_unm)}**.",
        (
            "- **J is a Y3 X** (|ρ| ≥ 0.08)."
            if j_is_y3_x
            else "- **J is not a Y3 X** (|ρ| < 0.08). **KEEP as a Q5 diagnostic / miss flag** "
            "(`j_has_book` + `j_pay_match` on ever-ERP months)."
        ),
        "",
        "## Pass-2 residuals (paid + Y3-labeled months)",
        "",
        f"- `j_n_paid` vs Y3 raw: **{_fmt_num(n_paid_y3_raw)}** (looks like a Y3 X).",
        f"- same, rank-residual after `log1p(a_in3)`: **{_fmt_num(n_paid_y3_size)}** → size, PARK.",
        f"- same, rank-residual after `e_ar_issued`: **{_fmt_num(n_paid_y3_ear)}** → PARK.",
        f"- `j_n_matched` vs Y3 | `n_paid`: **{_fmt_num(n_match_y3_npay)}** (no extra match-count signal).",
        f"- `j_iss_match` | `j_pay_match` vs Y3: **{_fmt_num(iss_pay_y3)}** (iss adds nothing for Y3).",
        "- `j_has_book` is a causal step (156 train cos flip 0→1; 470 stay 0; 588 start 1).",
        "  High acf1 (0.84) is the step, not a cycle. KEEP as a miss flag.",
        "",
        "## Decisions (this pass)",
        "",
    ]
    for feat in diag_cols:
        why = []
        if rewrite[feat]:
            why.append("rewrite " + ", ".join(rewrite[feat]))
        st = stats.loc[stats["feature"] == feat].iloc[0]
        if st["flags"] != "—":
            why.append(str(st["flags"]))
        if feat == "j_iss_match" and np.isfinite(pay_iss) and abs(pay_iss) >= PAIR_DROP_RHO:
            why.append(f"ρ vs pay={pay_iss:.3f}")
        if feat == "j_pay_unmatched" and np.isfinite(pay_unm) and abs(pay_unm) >= PAIR_DROP_RHO:
            why.append("exact complement")
        if feat.endswith("_t3"):
            why.append(f"acf1={st['acf1']:.3f} acf3={st['acf3']:.3f} (lag-3 = non-overlap)")
        extra = f" — {'; '.join(why)}" if why else ""
        lines.append(f"- `{feat}`: **{decisions[feat]}**{extra}")

    lines += [
        "",
        "## Look-ahead / construction notes",
        "",
        "- Book filter = Family E: `document_type=invoice`, `status<>cancel`,",
        "  `amount<>0`, `issuance_date` not null.",
        "- Paid = `payment_date` not null and not `payment_date_invalid`.",
        "- Monthly: `date_trunc('month', date) = period` and `date < 2026-09-01`.",
        "- Weekly: `date_trunc('week', date)` (Monday, matches `W-MON`).",
        "- Greedy 1-1 is amount-bucket `min(n_inv, n_tx)`, not a global optimizer.",
        "- `j_has_book` uses `MIN(issuance_date) ≤ period_end` only — no future ERP.",
        "- Did not merge into parquet. Did not edit FAMILIES. Did not rebuild Y8.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_train_report.stats = stats
    write_train_report.decisions = decisions
    write_train_report.y_rhos = y_rhos
    write_train_report.pay_iss = pay_iss
    write_train_report.never_cos = never_cos
    write_train_report.pay_on_never = pay_on_never
    write_train_report.j_is_y3_x = j_is_y3_x
    write_train_report.mean_1to1 = mean_1to1
    write_train_report.mean_set = mean_set
    return stats


def _smoke_lookahead(con, part: pd.DataFrame) -> None:
    """July row must ignore August payments (one train company with both)."""
    sql = f"""
        SELECT company_id
        FROM invoices
        WHERE {BOOK}
          AND payment_date IS NOT NULL
          AND COALESCE(payment_date_invalid, FALSE) = FALSE
          AND date_trunc('month', payment_date) = DATE '2026-07-01'
        INTERSECT
        SELECT company_id
        FROM invoices
        WHERE {BOOK}
          AND payment_date IS NOT NULL
          AND COALESCE(payment_date_invalid, FALSE) = FALSE
          AND date_trunc('month', payment_date) = DATE '2026-08-01'
        LIMIT 1
    """
    hit = con.execute(sql).df()
    if hit.empty:
        print("lookahead smoke: no company with Jul+Aug payments; skipped")
        return
    cid = str(hit.iloc[0]["company_id"])
    jul = con.execute(
        f"""
        SELECT COUNT(*)::BIGINT AS n
        FROM invoices
        WHERE {BOOK}
          AND company_id = ?
          AND payment_date IS NOT NULL
          AND COALESCE(payment_date_invalid, FALSE) = FALSE
          AND date_trunc('month', payment_date) = DATE '2026-07-01'
        """,
        [cid],
    ).df().iloc[0]["n"]
    internal = getattr(build, "_last_internal", None)
    if internal is None:
        return
    row = internal[
        (internal["company_id"] == cid)
        & (internal["period"] == pd.Timestamp("2026-07-01"))
    ]
    if row.empty:
        raise AssertionError(f"lookahead: missing July row for {cid}")
    got = row.iloc[0]["n_paid"]
    if pd.isna(got):
        got = 0
    if int(got) != int(jul):
        raise AssertionError(
            f"lookahead leak or miss for {cid}: July n_paid={got} vs SQL {jul}"
        )
    print(f"lookahead OK: {cid} July n_paid={int(got)} (Aug payments excluded)")


def _smoke_470(part: pd.DataFrame) -> None:
    tr = part.loc[train_mask(part["company_id"])]
    ever = tr.groupby("company_id")["j_has_book"].max()
    never_ids = ever[ever < 1].index
    rates = tr.loc[tr["company_id"].isin(never_ids), "j_pay_match"]
    if rates.notna().any():
        n = int(rates.notna().sum())
        raise AssertionError(f"470 rule broken: {n} never-ERP cm have j_pay_match")
    print(f"470 NaN OK: {len(never_ids)} never-ERP companies, all j_pay_match NaN")


def _append_registry(stats: pd.DataFrame, decisions: dict[str, str]) -> None:
    if not REGISTRY.exists():
        return
    ts = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M")
    rows = []
    for _, r in stats.iterrows():
        feat = r["feature"]
        if not str(feat).startswith("j_"):
            continue
        if feat not in J_COLS and feat not in ("j_pay_unmatched", "j_n_paid"):
            continue
        rows.append(
            ",".join(
                [
                    ts,
                    "R4",
                    "4",
                    AGENT,
                    "J",
                    "-",
                    "verify",
                    "train",
                    feat,
                    _fmt_num(r["cov_cm"], 4) if np.isfinite(r["cov_cm"]) else "",
                    _fmt_num(r["cov_cm"], 4) if np.isfinite(r["cov_cm"]) else "",
                    f"{decisions.get(feat, '')} size_ρ={_fmt_num(r['size_rho'])} acf1={_fmt_num(r['acf1'])}",
                ]
            )
        )
    if not rows:
        return
    with REGISTRY.open("a", encoding="utf-8") as f:
        if not REGISTRY.read_text(encoding="utf-8").endswith("\n"):
            f.write("\n")
        f.write("\n".join(rows) + "\n")
    print(f"registry appended {len(rows)} J rows")


def main() -> None:
    con = connect()
    grid = monthly_grid(con)[["company_id", "period"]]
    part = build(con, grid)
    _smoke_470(part)
    _smoke_lookahead(con, part)
    # Report uses build._last_internal from the monthly run — before weekly.
    stats = write_train_report(part)

    # weekly 2-company smoke (same contract, not persisted)
    sample = (
        grid.drop_duplicates("company_id")["company_id"]
        .astype(str)
        .head(2)
        .tolist()
    )
    wgrid = weekly_grid(con)
    wgrid = wgrid[wgrid["company_id"].astype(str).isin(sample)][
        ["company_id", "period"]
    ]
    if not wgrid.empty:
        wpart = build(con, wgrid)
        if wpart.duplicated(["company_id", "period"]).any():
            raise AssertionError("weekly J produced duplicate keys")
        extra = [c for c in wpart.columns if c not in {"company_id", "period", *J_COLS}]
        if extra:
            raise AssertionError(f"weekly J extra cols {extra}")
        print(
            f"weekly smoke OK: rows={len(wpart)} cos={wpart['company_id'].nunique()} "
            f"j_pay_match cov={wpart['j_pay_match'].notna().mean():.3f}"
        )

    con.close()
    decisions = getattr(write_train_report, "decisions", {})
    print(f"wrote {REPORT_PATH}")
    print(f"rows={len(part)} companies={part['company_id'].nunique()} cols={list(J_COLS)}")
    print(
        stats[
            [c for c in ("feature", "cov_cm", "cov_erp", "size_rho", "acf1", "acf3", "flags") if c in stats.columns]
        ].to_string(index=False)
    )
    print("decisions:", decisions)
    print("pay vs iss ρ:", getattr(write_train_report, "pay_iss", None))
    print("Y3 X?:", getattr(write_train_report, "j_is_y3_x", None))
    print(
        "mean 1-1 / set-overlap:",
        getattr(write_train_report, "mean_1to1", None),
        getattr(write_train_report, "mean_set", None),
    )
    print("did not write monthly.parquet")
    print("did not add Family J to FAMILIES")


if __name__ == "__main__":
    main()
