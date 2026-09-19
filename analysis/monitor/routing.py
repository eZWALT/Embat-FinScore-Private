"""Who gets an alert and what to do (plan step 5 wording, owners from the plan: tesorero / CFO / Cobros).

Owner ids match the contract: "treasurer" (tesorero), "cfo", "collections" (Cobros). The owner follows the item that moved
the most (the reason with the largest points), because that is where the fix lives. Actions are concrete and short. They
never say "revenue at risk" and never quote a probability.
"""
from __future__ import annotations

OWNER_LABELS = {"treasurer": "Treasurer", "cfo": "CFO", "collections": "Collections"}

ITEM_OWNER = {
    "delay_paid": "treasurer", "ap_overdue30": "treasurer", "runway": "treasurer", "neg_liq": "treasurer", "neg_episodes": "treasurer",
    "out_vol": "treasurer", "delay_coll": "collections", "ar_overdue30": "collections", "credit_note": "collections",
    "ds_ratio": "cfo", "fc_ratio": "cfo", "ds_increase": "cfo", "fc_increase": "cfo", "months_observed": "cfo", "active_share": "treasurer",
    "cust_tail": "cfo", "guard": "treasurer",
}
ITEM_ACTION = {
    "delay_paid": "Review the payment calendar: which supplier invoices are already past due, and pay or agree new dates for the largest first.",
    "ap_overdue30": "Clear or renegotiate the payables that are more than 30 days overdue, largest first, before suppliers tighten terms.",
    "runway": "Rebuild the cash forecast for the next 3 months and decide what to move (payments, collections, credit line) to keep the runway.",
    "neg_liq": "Cash went negative: check the overdraft or credit line covers it and schedule inflows against the next big outflows.",
    "neg_episodes": "Cash keeps dipping below zero: set a minimum-balance rule and a funding source for the gaps.",
    "out_vol": "Outflows are swinging: list the irregular payments and schedule them, so the cash plan can rely on a base level.",
    "delay_coll": "Chase the receivables that are past due, oldest and largest first, and check the payment terms with the slowest customers.",
    "ar_overdue30": "Run a collections round on invoices more than 30 days overdue and escalate the largest ones.",
    "credit_note": "Review why so much billing is being reversed by credit notes (disputes, errors, returns) and fix the source.",
    "ds_ratio": "Debt repayments take more of each euro received: review the repayment schedule and whether refinancing or re-timing helps.",
    "fc_ratio": "Bank fees and interest are taking a larger share: compare conditions across banks and renegotiate the costliest.",
    "ds_increase": "Debt service is rising against inflows: check what new debt drives it and whether the cash plan covers it.",
    "fc_increase": "Fees and interest are rising against inflows: find which account or facility drives it and renegotiate.",
    "months_observed": "Short history: treat the score as provisional until more months are connected.",
    "active_share": "Money is coming in fewer months: check whether billing has stopped or whether accounts are missing from the feed.",
    "cust_tail": "One customer is most of the billing: agree terms and a concentration limit, and start diversifying the pipeline.",
    "guard": "Activity has dropped sharply: confirm the bank feeds are complete, then contact the company.",
}
CATEGORY_OWNER = {"payment_history": "treasurer", "amounts_owed": "treasurer", "stability": "cfo", "new_credit": "cfo", "mix": "collections"}

DARK_ACTION = ("Check that every bank account is still connected and reporting. If the feed is complete, call the company today: "
               "no bank movement for 60 days or more.")
IMPROVE_ACTION = ("Confirm the improvement is durable (it held 3 of the last 4 months). If so, it is a moment to review credit lines and payment "
                  "terms, or to put idle cash to work.")
GROUP_ACTION = "Look at the members that fell the most (listed in the evidence) and treat each with the action of its own alert."


def top_customer_action(customer: str, open_eur: float | None) -> str:
    base = f"Contact customer {customer}: ask whether an order is pending or the relationship changed."
    if open_eur and open_eur > 0:
        return base + f" Review the €{open_eur:,.0f} still open with them and start collection if it is overdue."
    return base + " Review any exposure and the credit terms you give them."


def route_item(item: str | None) -> tuple[str, str]:
    """(owner, action) from the reason item that moved the most."""
    if item is None:
        return "cfo", "Review the score movers with the company."
    return ITEM_OWNER.get(item, "cfo"), ITEM_ACTION.get(item, "Review the score movers with the company.")
