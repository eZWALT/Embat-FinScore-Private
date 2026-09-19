"""Family E — receivables / payables from `invoices`.

As-of each grid `period` (month-start or week-start) only invoices whose
issuance / due / payment dates are known by period end are used. AR is
`amount > 0`, AP is `amount < 0` (confirmed against matched bank tx).

Core book (document_type = invoice, status <> cancel, amount <> 0):

- e_ar_open / e_ap_open: unpaid |amount| stock at period end
  (issued <= end; not paid by end; `payment_date_invalid` dropped — timing unknown).
- e_ar_overdue / e_ap_overdue: overdue |amount| / open, where due_date < period end.
- e_ar_overdue_30 / e_ap_overdue_30: share of open with (period_end - due) > 30 days
  (Banque de France: only >30 days late moves PD).
- e_delay_coll / e_delay_paid: amount-weighted (paid_dt - due) days on invoices
  paid in the trailing 3 months (90 days on a weekly grid), clipped [-30, 120].
  Null on the first 6 calendar months of the panel (2024-09..2025-02): left
  truncation hides pre-sample invoices paid late, which shortens early delays.
- e_dso_proxy / e_dpo_proxy: open / this-period issued (months of billings outstanding).
- e_credit_note_ratio: |credit notes| / (|invoices| + |credit notes|) issued this
  period. Type `credit_note` if present; else ERP stand-ins `note` and `refund`
  (concepts are "Abono" / "Factura correctiva"). Null if that type is absent.
- e_pending_amt_share: reconstructed pending / |amount| on invoices issued by
  period end (0 if paid by end, else |amount|). Snapshot `pending_amount` is
  as-of extraction and would leak later collections.
- e_fx_share: this-period issued |amount| with currency <> accounting_currency.
- e_ar_issued / e_ap_issued: this-period issuance volume.

No holdout fitting (no percentiles, bins, or centroids).
"""
from __future__ import annotations

import pandas as pd

from analysis.features.common import MONTHS

SOURCE_TABLES = ["invoices"]
FAMILY = "e"

DELAY_CLIP = (-30.0, 120.0)
OVERDUE_30_DAYS = 30
# First 6 month-starts of MONTHS (2024-09 .. 2025-02); same mask as score_pipeline.
DELAY_MASK_BEFORE = MONTHS[6]

CREDIT_NOTE_CANONICAL = ("credit_note", "creditnote")
CREDIT_NOTE_FALLBACK = ("note", "refund")

FEATURE_COLS = [
    "e_ar_open",
    "e_ap_open",
    "e_ar_overdue",
    "e_ap_overdue",
    "e_ar_overdue_30",
    "e_ap_overdue_30",
    "e_delay_coll",
    "e_delay_paid",
    "e_dso_proxy",
    "e_dpo_proxy",
    "e_credit_note_ratio",
    "e_pending_amt_share",
    "e_fx_share",
    "e_ar_issued",
    "e_ap_issued",
]


def _infer_freq(periods: pd.Series) -> str:
    u = pd.to_datetime(pd.unique(periods.dropna()))
    if len(u) <= 1:
        p = pd.Timestamp(u[0]) if len(u) else pd.NaT
        if pd.isna(p):
            return "M"
        return "M" if int(p.day) == 1 else "W"
    med = pd.Series(sorted(u)).diff().median()
    return "W" if med <= pd.Timedelta(days=10) else "M"


def _period_frame(grid: pd.DataFrame) -> pd.DataFrame:
    periods = pd.to_datetime(pd.Series(grid["period"].unique())).sort_values()
    df = pd.DataFrame({"period": periods.reset_index(drop=True)})
    if _infer_freq(df["period"]) == "M":
        df["period_end"] = df["period"] + pd.offsets.MonthEnd(0)
        df["pay_start"] = df["period"] - pd.DateOffset(months=2)
    else:
        df["period_end"] = df["period"] + pd.Timedelta(days=6)
        df["pay_start"] = df["period_end"] - pd.Timedelta(days=89)
    df["delay_ok"] = df["period"] >= DELAY_MASK_BEFORE
    return df


def _sql_str_list(values: tuple[str, ...]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def _credit_note_types(con) -> tuple[str, ...]:
    found = tuple(
        con.execute("SELECT DISTINCT document_type FROM invoices WHERE document_type IS NOT NULL")
        .df()["document_type"]
        .astype(str)
        .str.strip()
        .tolist()
    )
    lower = {x.lower(): x for x in found}
    canonical = tuple(lower[c] for c in CREDIT_NOTE_CANONICAL if c in lower)
    if canonical:
        return canonical
    return tuple(lower[c] for c in CREDIT_NOTE_FALLBACK if c in lower)


def _empty(grid: pd.DataFrame) -> pd.DataFrame:
    out = grid[["company_id", "period"]].copy()
    out["period"] = pd.to_datetime(out["period"])
    for c in FEATURE_COLS:
        out[c] = pd.NA
    return out


def build(con, grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return _empty(grid)

    periods = _period_frame(grid)
    cn_types = _credit_note_types(con)
    cn_sql = _sql_str_list(cn_types) if cn_types else "NULL"

    con.register("_e_periods", periods[["period", "period_end", "pay_start", "delay_ok"]])
    try:
        issued = con.execute(
            f"""
            SELECT i.company_id,
                   p.period,
                   SUM(CASE WHEN i.document_type = 'invoice' AND i.amount > 0
                            THEN abs(i.amount) ELSE 0 END) AS e_ar_issued,
                   SUM(CASE WHEN i.document_type = 'invoice' AND i.amount < 0
                            THEN abs(i.amount) ELSE 0 END) AS e_ap_issued,
                   SUM(CASE WHEN i.document_type = 'invoice'
                             AND i.currency <> i.accounting_currency
                            THEN abs(i.amount) ELSE 0 END)
                     / NULLIF(SUM(CASE WHEN i.document_type = 'invoice'
                                       THEN abs(i.amount) END), 0) AS e_fx_share,
                   SUM(CASE WHEN i.document_type IN ({cn_sql})
                            THEN abs(i.amount) ELSE 0 END)
                     / NULLIF(SUM(CASE WHEN i.document_type = 'invoice'
                                          OR i.document_type IN ({cn_sql})
                                       THEN abs(i.amount) END), 0) AS e_credit_note_ratio
            FROM invoices i
            JOIN _e_periods p
              ON CAST(i.issuance_date AS DATE) >= CAST(p.period AS DATE)
             AND CAST(i.issuance_date AS DATE) <= CAST(p.period_end AS DATE)
            WHERE i.amount <> 0
              AND i.status <> 'cancel'
              AND i.issuance_date IS NOT NULL
            GROUP BY 1, 2
            """
        ).df()

        open_book = con.execute(
            f"""
            SELECT i.company_id,
                   p.period,
                   SUM(CASE WHEN i.amount > 0 THEN abs(i.amount) ELSE 0 END) AS e_ar_open,
                   SUM(CASE WHEN i.amount < 0 THEN abs(i.amount) ELSE 0 END) AS e_ap_open,
                   SUM(CASE WHEN i.amount > 0
                             AND i.due_date IS NOT NULL
                             AND CAST(i.due_date AS DATE) < CAST(p.period_end AS DATE)
                            THEN abs(i.amount) ELSE 0 END)
                     / NULLIF(SUM(CASE WHEN i.amount > 0 THEN abs(i.amount) END), 0) AS e_ar_overdue,
                   SUM(CASE WHEN i.amount < 0
                             AND i.due_date IS NOT NULL
                             AND CAST(i.due_date AS DATE) < CAST(p.period_end AS DATE)
                            THEN abs(i.amount) ELSE 0 END)
                     / NULLIF(SUM(CASE WHEN i.amount < 0 THEN abs(i.amount) END), 0) AS e_ap_overdue,
                   SUM(CASE WHEN i.amount > 0
                             AND i.due_date IS NOT NULL
                             AND date_diff('day', CAST(i.due_date AS DATE),
                                                CAST(p.period_end AS DATE)) > {OVERDUE_30_DAYS}
                            THEN abs(i.amount) ELSE 0 END)
                     / NULLIF(SUM(CASE WHEN i.amount > 0 THEN abs(i.amount) END), 0) AS e_ar_overdue_30,
                   SUM(CASE WHEN i.amount < 0
                             AND i.due_date IS NOT NULL
                             AND date_diff('day', CAST(i.due_date AS DATE),
                                                CAST(p.period_end AS DATE)) > {OVERDUE_30_DAYS}
                            THEN abs(i.amount) ELSE 0 END)
                     / NULLIF(SUM(CASE WHEN i.amount < 0 THEN abs(i.amount) END), 0) AS e_ap_overdue_30
            FROM invoices i
            JOIN _e_periods p
              ON CAST(i.issuance_date AS DATE) <= CAST(p.period_end AS DATE)
             AND (i.payment_date IS NULL
                  OR CAST(i.payment_date AS DATE) > CAST(p.period_end AS DATE))
             AND NOT coalesce(i.payment_date_invalid, FALSE)
            WHERE i.document_type = 'invoice'
              AND i.status <> 'cancel'
              AND i.amount <> 0
              AND i.issuance_date IS NOT NULL
            GROUP BY 1, 2
            """
        ).df()

        pending = con.execute(
            """
            SELECT i.company_id,
                   p.period,
                   SUM(CASE
                         WHEN coalesce(i.payment_date_invalid, FALSE) THEN 0
                         WHEN i.payment_date IS NOT NULL
                          AND CAST(i.payment_date AS DATE) <= CAST(p.period_end AS DATE) THEN 0
                         ELSE abs(i.amount)
                       END)
                     / NULLIF(SUM(CASE WHEN coalesce(i.payment_date_invalid, FALSE)
                                       THEN 0 ELSE abs(i.amount) END), 0) AS e_pending_amt_share
            FROM invoices i
            JOIN _e_periods p
              ON CAST(i.issuance_date AS DATE) <= CAST(p.period_end AS DATE)
            WHERE i.document_type = 'invoice'
              AND i.status <> 'cancel'
              AND i.amount <> 0
              AND i.issuance_date IS NOT NULL
            GROUP BY 1, 2
            """
        ).df()

        delay = con.execute(
            f"""
            SELECT i.company_id,
                   p.period,
                   CASE WHEN p.delay_ok THEN
                     SUM(CASE WHEN i.amount > 0
                              THEN date_diff('day', CAST(i.due_date AS DATE),
                                                  CAST(i.payment_date AS DATE)) * abs(i.amount)
                         END)
                     / NULLIF(SUM(CASE WHEN i.amount > 0 THEN abs(i.amount) END), 0)
                   END AS e_delay_coll,
                   CASE WHEN p.delay_ok THEN
                     SUM(CASE WHEN i.amount < 0
                              THEN date_diff('day', CAST(i.due_date AS DATE),
                                                  CAST(i.payment_date AS DATE)) * abs(i.amount)
                         END)
                     / NULLIF(SUM(CASE WHEN i.amount < 0 THEN abs(i.amount) END), 0)
                   END AS e_delay_paid
            FROM invoices i
            JOIN _e_periods p
              ON i.payment_date IS NOT NULL
             AND CAST(i.payment_date AS DATE) >= CAST(p.pay_start AS DATE)
             AND CAST(i.payment_date AS DATE) <= CAST(p.period_end AS DATE)
             AND CAST(i.issuance_date AS DATE) <= CAST(p.period_end AS DATE)
            WHERE i.document_type = 'invoice'
              AND i.status <> 'cancel'
              AND i.amount <> 0
              AND i.due_date IS NOT NULL
              AND i.issuance_date IS NOT NULL
              AND NOT coalesce(i.payment_date_invalid, FALSE)
            GROUP BY 1, 2, p.delay_ok
            """
        ).df()
    finally:
        con.unregister("_e_periods")

    ever = con.execute(
        """
        SELECT DISTINCT company_id
        FROM invoices
        WHERE document_type = 'invoice' AND status <> 'cancel' AND amount <> 0
        """
    ).df()
    ever_ids = set(ever["company_id"].astype(str))

    out = grid[["company_id", "period"]].copy()
    out["company_id"] = out["company_id"].astype(str)
    out["period"] = pd.to_datetime(out["period"])

    for part in (issued, open_book, pending, delay):
        if part.empty:
            continue
        part = part.copy()
        part["company_id"] = part["company_id"].astype(str)
        part["period"] = pd.to_datetime(part["period"])
        out = out.merge(part, on=["company_id", "period"], how="left")

    if "e_delay_coll" in out.columns:
        out["e_delay_coll"] = out["e_delay_coll"].clip(*DELAY_CLIP)
        out["e_delay_paid"] = out["e_delay_paid"].clip(*DELAY_CLIP)
    else:
        out["e_delay_coll"] = pd.NA
        out["e_delay_paid"] = pd.NA

    for c in FEATURE_COLS:
        if c not in out.columns:
            out[c] = pd.NA

    active = out["company_id"].isin(ever_ids)
    for c in ("e_ar_issued", "e_ap_issued", "e_ar_open", "e_ap_open"):
        out.loc[active, c] = out.loc[active, c].fillna(0.0)

    out["e_dso_proxy"] = out["e_ar_open"] / out["e_ar_issued"].where(out["e_ar_issued"] > 0)
    out["e_dpo_proxy"] = out["e_ap_open"] / out["e_ap_issued"].where(out["e_ap_issued"] > 0)

    if not cn_types:
        out["e_credit_note_ratio"] = pd.NA

    return out[["company_id", "period", *FEATURE_COLS]]
