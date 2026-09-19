"""Scorecard definition: categories, weights, items, guard. Fixed a priori; nothing here is fitted.

Score = weighted mean of category scores over the categories that exist for the company-month,
where a category score is the equal-weight mean of its available item points (0-100, 100 = healthiest)
and item points are the percentile of the trailing-window value against the train reference
(`fit.py`), or a fixed a-priori mapping for the two "fixed" items.

Rationale per item lives in `ITEMS[...].why`; the night's evidence and the traps in
`.agents/persistent-memory/2026-09-19-1130-night-variables-for-score.md` and `...-1015-y3-...md`.
"""
from __future__ import annotations

from dataclasses import dataclass

# FICO weights; new credit is shrunk because the trail barely observes it (see NEW_CREDIT_SHRINK)
CATEGORY_WEIGHTS = {
    "payment_history": 35.0,
    "amounts_owed": 30.0,
    "stability": 15.0,
    "new_credit": 10.0,
    "mix": 10.0,
}
NEW_CREDIT_SHRINK = 0.5  # 10 -> 5 nominal points; the 5 points are redistributed by renormalising the rest
EFFECTIVE_WEIGHTS = {c: w * (NEW_CREDIT_SHRINK if c == "new_credit" else 1.0) for c, w in CATEGORY_WEIGHTS.items()}

CATEGORY_LABELS = {
    "payment_history": "Historial de pagos",
    "amounts_owed": "Liquidez y deuda",
    "stability": "Estabilidad",
    "new_credit": "Nuevo crédito",
    "mix": "Combinación de clientes",
}

# a category is scored when at least this share of its items is available
MIN_ITEM_COVERAGE = 0.5

WINDOW = 3          # months of the trailing window for point-in-time features
EPISODE_WINDOW = 6  # months for cash-negative onsets and outflow volatility
PRIOR_LAG = 6       # new-credit items compare the last 3 months with the 3 months ending 6 months earlier
HHI_TAIL = 0.975    # customer concentration counts only above this (night: body of the distribution is noise)
OUT_VOL_CLIP = 3.0
RUNWAY_CLIP = (-6.0, 24.0)
LENGTH_CAP_MONTHS = 12  # trail length earns full points at 12 months

# reported trail-length / coverage thresholds for the confidence flag
CONF_HIGH_COVERAGE = 0.85
CONF_MED_COVERAGE = 0.50
CONF_HIGH_MONTHS = 12
CONF_MIN_MONTHS = 6

# Going-dark guard (a priori). A company that stops moving money must score low, never high:
# outflow-based ratios (runway, debt service / inflows) and missing invoice payments improve when activity dies.
DARK_NO_TX_DAYS = 60        # no booking of any kind in the 60 days before month-end (two silent months)
FADING_INFLOW_RATIO = 0.25  # last-3-month inflow below this share of the company's own earlier 6-month mean
CAP_DARK = 30.0
CAP_FADING = 50.0
# The caps are targets, not cliffs: while a guard is on, the ceiling starts from the company's previous score and comes down at most
# GUARD_STEP points a month toward the cap (it lifts at once when the guard ends). 10 is the size of the largest ordinary one-month
# fall of the score (guard-free months: 5th percentile -7, 1st percentile -14), so the guard never moves a score more than normal
# movement does. Switching the guard on cannot then turn a gap in the data into a 20-40 point step, and a one-month gap costs at
# most GUARD_STEP points. The going_dark alert (analysis/monitor) still fires in the first month.
GUARD_STEP = 10.0


@dataclass(frozen=True)
class Item:
    name: str
    category: str
    kind: str        # "pct" (percentile vs train reference) or "fixed"
    higher_better: bool
    family: str      # feature family it is built from: used to drop items when validating an outcome built from that family
    label: str
    why: str


ITEMS: list[Item] = [
    # ---- payment history (35): both directions of the payment flow, four equal items
    Item("delay_paid", "payment_history", "pct", False, "e", "Días de pago tras el vencimiento (proveedores)",
         "Own payment behaviour is the closest analogue of FICO payment history. Amount-weighted days between due and "
         "payment on invoices paid in the trailing 3 months (e_delay_coll/e_delay_paid construction), averaged over 3 months."),
    Item("delay_coll", "payment_history", "pct", False, "e", "Días que tardan los clientes en pagar tras el vencimiento",
         "Collections lateness drives liquidity stress. Same construction on the receivables side."),
    Item("ap_overdue30", "payment_history", "pct", False, "e", "Pagos a proveedores con más de 30 días de retraso",
         "Share of open payables more than 30 days past due (Banque de France: only > 30 days moves default risk)."),
    Item("ar_overdue30", "payment_history", "pct", False, "e", "Cobros de clientes con más de 30 días de retraso",
         "Share of open receivables more than 30 days past due."),
    # ---- amounts owed (30): liquidity and debt burden
    Item("runway", "amounts_owed", "pct", True, "b", "Meses de salidas cubiertos por la caja",
         "Period-end cash / mean monthly operating outflow (6-month, 3-month when the trail is shorter), clipped -6..24, "
         "averaged over 3 months. The 6-month denominator makes it slow to react to a one-quarter outflow collapse."),
    Item("neg_liq", "amounts_owed", "pct", False, "b", "Cierres de mes con caja negativa",
         "Share of the last 3 month-ends with cash < 0 (recomputed from cash rounded to cents; the store's flag has float noise at zero)."),
    Item("neg_episodes", "amounts_owed", "pct", False, "b", "Veces que la caja pasó a negativo",
         "Onsets of negative cash in the last 6 month-ends."),
    Item("ds_ratio", "amounts_owed", "pct", False, "f", "Pago de deuda / entradas de caja",
         "3-month debt repayment over 3-month operating inflow (f_ds_r), averaged over 3 months. Debt load, the analogue of amounts owed."),
    Item("fc_ratio", "amounts_owed", "pct", False, "f", "Comisiones e intereses bancarios / entradas de caja",
         "3-month fees + interest over 3-month operating inflow (f_fc_r), averaged over 3 months."),
    # ---- length / stability (15)
    Item("months_observed", "stability", "fixed", True, "", "Meses de historial",
         "Fixed mapping: 100 x min(months observed, 12) / 12. A thin file is less certain; the confidence flag says so too."),
    Item("active_share", "stability", "fixed", True, "c", "Meses con entrada de dinero",
         "Fixed mapping: 100 x share of the last 6 months with any incoming movement. Guard item: a company that goes dark loses points here."),
    Item("out_vol", "stability", "pct", False, "a", "Volatilidad de las salidas",
         "Std / mean of monthly operating outflows over 6 months (a_out_vol, clip 3). A company trait (night: 74% between-company variance), fine for a level score."),
    # ---- new credit (10, shrunk to 5): what the trail can observe of it
    Item("ds_increase", "new_credit", "pct", False, "f", "Pago de deuda al alza",
         "Rise of debt service / inflows vs the same window 6 months earlier (0 if it fell). Facility counts are excluded: connection dates make them rise with time."),
    Item("fc_increase", "new_credit", "pct", False, "f", "Comisiones e intereses al alza",
         "Rise of fees + interest / inflows vs the same window 6 months earlier (0 if it fell)."),
    # ---- mix (10): concentration and credit notes, both from invoices
    Item("cust_tail", "mix", "pct", False, "d", "Dependencia de un solo cliente",
         "Customer HHI above 0.975 only (night: the body of the distribution is noise, the one-buyer tail is not), 3-month mean."),
    Item("credit_note", "mix", "pct", False, "e", "Notas de crédito / facturación",
         "Share of billing reversed by credit notes, 3-month mean. Experimental: no paper measures it."),
]
ITEM_BY_NAME = {i.name: i for i in ITEMS}
PCT_ITEMS = [i.name for i in ITEMS if i.kind == "pct"]
ITEM_CATEGORY = {i.name: i.category for i in ITEMS}

# Outcomes accepted for validation and the feature families their label is built from (forbidden as X).
# y9 also forbids a_fin_cost / a_fc, which the score does not use.
OUTCOME_FORBIDDEN_FAMILIES = {
    "y2_neg_2of3": {"b"},
    "y4_ds_r_double": {"f"},
    "y5_ap_od30_ownp80": {"e"},
    "y5_ar_od30_sust": {"e"},
    "y7_top1_lost": {"d"},
    "y7_top1_lost_inflow": {"d"},
    "y9_fee_r_ownp80": {"f"},
    "y9_fee_spike": {"f"},
}

# unit of the "value" reported next to an item (what the number in the reason sentence means)
UNITS = {
    "delay_paid": "days", "delay_coll": "days", "ap_overdue30": "share", "ar_overdue30": "share",
    "runway": "months", "neg_liq": "share", "neg_episodes": "count", "ds_ratio": "share_of_inflows",
    "fc_ratio": "share_of_inflows", "months_observed": "months", "active_share": "months_of_6",
    "out_vol": "ratio", "ds_increase": "share_of_inflows", "fc_increase": "share_of_inflows",
    "cust_tail": "share", "credit_note": "share",
}
