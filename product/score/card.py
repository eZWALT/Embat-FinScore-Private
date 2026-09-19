"""v0 dummy FICO-like card.

Weights are a priori (FICO mirrors). New credit is empty in this trail
(utilisation 1.6%, no NSF token, facility counts are connection artefacts),
so its 10 points move to amounts owed. Nothing here is fit except the
train-only percentile tables in ``score.py``.

Do not put social security / payroll / movement-days on the card as
"quiet = healthier" (Y3 trap). Going dark is a cap that can only lower
the score.
"""
from __future__ import annotations

from dataclasses import dataclass

VERSION = "v0-dummy"

# FICO 35 / 30 / 15 / 10 / 10 with new credit (10) folded into amounts.
CATEGORY_WEIGHTS: dict[str, float] = {
    "payment_history": 35.0,
    "amounts_owed": 40.0,
    "length_stability": 15.0,
    "mix": 10.0,
}

DARK_DAYS_CAP = 50.0
DARK_LONG_CAP = 35.0
DARK_LONG_DAYS = 45
THIN_TRAIL_MONTHS = 6
# Averaging several 0–100 percentiles shrinks the range toward 50.
# Stretch is a priori (not fit) so the product scale can reach the brief's 80s.
SCORE_STRETCH = 2.0
CONC_OK = 0.70
CONC_TAIL = 0.975
MIN_ACTIVE_WINDOW = 3
ACTIVE_WINDOW = 6
SMOOTH_WINDOW = 3
DIP_POINTS = 5.0


@dataclass(frozen=True)
class Item:
    name: str
    category: str
    direction: int  # +1 higher better, -1 lower better
    kind: str  # percentile | threshold
    lo: float | None = None
    hi: float | None = None
    label: str = ""
    reason_en: str = ""
    reason_es: str = ""


ITEMS: tuple[Item, ...] = (
    Item(
        "e_ar_overdue_30",
        "payment_history",
        -1,
        "percentile",
        label="customer invoices >30 days late",
        reason_en="{value:.0%} of open customer invoices are more than 30 days late",
        reason_es="{value:.0%} de las facturas de clientes abiertas llevan más de 30 días vencidas",
    ),
    Item(
        "e_ap_overdue_30",
        "payment_history",
        -1,
        "percentile",
        label="supplier bills >30 days late",
        reason_en="{value:.0%} of open supplier bills are more than 30 days late",
        reason_es="{value:.0%} de las facturas de proveedores abiertas llevan más de 30 días vencidas",
    ),
    Item(
        "e_delay_coll",
        "payment_history",
        -1,
        "percentile",
        label="customers pay late (days)",
        reason_en="customers pay {value:.0f} days after the due date",
        reason_es="los clientes pagan {value:.0f} días después del vencimiento",
    ),
    Item(
        "e_delay_paid",
        "payment_history",
        -1,
        "percentile",
        label="we pay suppliers late (days)",
        reason_en="the company pays suppliers {value:.0f} days after the due date",
        reason_es="la empresa paga a proveedores {value:.0f} días después del vencimiento",
    ),
    Item(
        "b_runway",
        "amounts_owed",
        1,
        "percentile",
        label="cash runway (months)",
        reason_en="cash covers {value:.1f} months of outflows",
        reason_es="la caja cubre {value:.1f} meses de salidas",
    ),
    Item(
        "f_ds_r",
        "amounts_owed",
        -1,
        "percentile",
        label="debt service / inflows",
        reason_en="debt service is {value:.0%} of inflows",
        reason_es="el servicio de deuda es el {value:.0%} de los cobros",
    ),
    Item(
        "f_fc_r",
        "amounts_owed",
        -1,
        "percentile",
        label="fees + interest / inflows",
        reason_en="fees and interest are {value:.0%} of inflows",
        reason_es="comisiones e intereses son el {value:.0%} de los cobros",
    ),
    Item(
        "active_share_6",
        "length_stability",
        1,
        "percentile",
        label="share of last 6 months with bank movement",
        reason_en="active in {value:.0%} of the last 6 months",
        reason_es="activa en el {value:.0%} de los últimos 6 meses",
    ),
    Item(
        "d_cust_top1",
        "mix",
        -1,
        "threshold",
        lo=CONC_OK,
        hi=CONC_TAIL,
        label="largest customer share of billing",
        reason_en="the largest customer is {value:.0%} of billing",
        reason_es="el mayor cliente concentra el {value:.0%} de la facturación",
    ),
)

STORE_COLUMNS = (
    "b_runway",
    "f_ds_r",
    "f_fc_r",
    "e_ar_overdue_30",
    "e_ap_overdue_30",
    "e_delay_coll",
    "e_delay_paid",
    "d_cust_top1",
    "c_n_days_with_tx",
    "c_recency_days",
    "first_month",
)

ITEM_BY_NAME = {it.name: it for it in ITEMS}
ITEMS_BY_CATEGORY = {
    cat: tuple(it for it in ITEMS if it.category == cat) for cat in CATEGORY_WEIGHTS
}


def format_reason(item: Item, value: float, lang: str = "en") -> str:
    tmpl = item.reason_es if lang == "es" else item.reason_en
    try:
        return tmpl.format(value=value)
    except (ValueError, KeyError):
        return item.label
