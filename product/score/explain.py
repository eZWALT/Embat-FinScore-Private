"""Trajectory state, month-on-month change attribution and plain-language reasons with euro amounts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import spec

# Trajectory (points per month on the 0-100 scale; the score's own within-company sd is ~7 points, so a 3-month slope of 3 = 9 points over 3 months)
SLOPE3_MATERIAL = 3.0   # OLS slope of the last 3 scores
SLOPE6_MATERIAL = 1.5   # OLS slope of the last 6 scores (>= 4 scored)
PERSIST_MONTHS = 3      # the 6-month slope must hold this many months in a row to count as a trend


def _slope(y: np.ndarray, min_obs: int) -> float:
    ok = np.isfinite(y)
    if ok.sum() < min_obs:
        return np.nan
    x = np.arange(len(y), dtype=float)[ok]
    yy = y[ok]
    xc = x - x.mean()
    return float((xc * (yy - yy.mean())).sum() / (xc ** 2).sum())


def trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Adds slope3, slope6 (points/month) and `trajectory` to a frame with company_id, period, score, guard.

    Rows must be sorted by company_id, period (items.compute_items returns them so).
    improving / deteriorating: the 6-month slope has pointed the same way for 3 months in a row and the
    3-month slope confirms it now (a sustained drift). dip: the 3-month slope is down now but the 6-month
    trend has not persisted (a sharp or short fall). A company flagged dark is deteriorating by rule.
    stable: everything else, including a step that already levelled off (the level shows in the score).
    Needs 4 scored months; before that the state is 'insufficient history'.
    """
    d = df.copy()
    by = d.groupby("company_id", sort=False)
    d["slope3"] = by["score"].transform(lambda s: s.rolling(3, min_periods=3).apply(lambda a: _slope(a, 3), raw=True))
    d["slope6"] = by["score"].transform(lambda s: s.rolling(6, min_periods=4).apply(lambda a: _slope(a, 4), raw=True))
    held = {}
    for key, cond in (("down", d["slope6"] <= -SLOPE6_MATERIAL), ("up", d["slope6"] >= SLOPE6_MATERIAL)):
        flag = cond.astype(float).where(d["slope6"].notna())
        held[key] = flag.groupby(d["company_id"], sort=False).transform(
            lambda s: s.rolling(PERSIST_MONTHS, min_periods=PERSIST_MONTHS).min()) == 1
    s3 = d["slope3"]
    state = np.select(
        [d["slope6"].isna() | s3.isna(),
         held["down"] & (s3 <= -SLOPE3_MATERIAL),
         held["up"] & (s3 >= SLOPE3_MATERIAL),
         s3 <= -SLOPE3_MATERIAL],
        ["insufficient history", "deteriorating", "improving", "dip"], default="stable")
    state = np.where(d["guard"] == "dark", "deteriorating", state)
    d["trajectory"] = np.where(d["score"].isna(), "", state)
    return d


def attribution(keys: pd.DataFrame, contrib: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """Month-on-month change per item in score points. Columns d_<item> and d_guard; they sum to the score change.

    Rows aligned with `keys` (sorted by company_id, period). A missing item contributes 0, so an item that appears or
    disappears (a category becoming available, which also re-weights the others) shows up as its own change.
    `d_guard` is the cap's effect (score - raw score). NaN when the score is missing in either month.
    """
    c = pd.concat([contrib.add_prefix("d_"), (res["score"] - res["score_raw"]).rename("d_guard")], axis=1)
    by = c.groupby(keys["company_id"], sort=False)
    delta = c - by.shift(1)
    both = res["score"].notna() & res["score"].groupby(keys["company_id"], sort=False).shift(1).notna()
    return pd.concat([keys[["company_id", "period"]], delta.where(both)], axis=1)


def eur(x: float) -> str:
    if x is None or not np.isfinite(x):
        return ""
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e6:
        return f"{sign}€{a / 1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}€{a / 1e3:.0f}k"
    return f"{sign}€{a:.0f}"


def _paren(*parts: str) -> str:
    parts = [p for p in parts if p]
    return f" ({'; '.join(parts)})" if parts else ""


def item_sentence(name: str, r) -> tuple[str, float]:
    """(sentence, euro amount) for item `name` from a row `r` of items (+ amounts). Amount is NaN when unknown.

    The euro amount covers the same window as the figure quoted next to it (5 months for the delay and debt ratios,
    which average three 3-month windows; the current month-end for stocks).
    """
    g = lambda k: float(r[k]) if k in r.index and pd.notna(r[k]) else np.nan
    v = g(name)
    if name == "delay_paid":
        e = g("ap_late5")
        return f"Suppliers were paid {v:.0f} days after the due date on average over the last 5 months{_paren(eur(e) + ' paid late' if np.isfinite(e) else '')}.", e
    if name == "delay_coll":
        e = g("ar_late5")
        return f"Customers paid {v:.0f} days after the due date on average over the last 5 months{_paren(eur(e) + ' collected late' if np.isfinite(e) else '')}.", e
    if name == "ap_overdue30":
        e = g("ap_od30_eur")
        return f"On average over the last 3 month-ends {v:.0%} of open payables was more than 30 days overdue{_paren(eur(e) + ' now')}.", e
    if name == "ar_overdue30":
        e = g("ar_od30_eur")
        return f"On average over the last 3 month-ends {v:.0%} of open receivables was more than 30 days overdue{_paren(eur(e) + ' now')}.", e
    if name == "runway":
        liq, out = g("liq"), g("out_month")
        if liq < 0:
            return f"Cash is negative ({eur(liq)}).", liq
        return f"Cash covered {v:.1f} months of outflows on average over the last 3 month-ends{_paren(eur(liq) + ' now', eur(out) + ' of outflows a month')}.", liq
    if name == "neg_liq":
        e = g("min_liq3")
        return f"Cash was negative at {round(v * 3)} of the last 3 month-ends{_paren('lowest ' + eur(e))}.", e
    if name == "neg_episodes":
        e = g("min_liq3")
        return f"Cash turned negative {v:.0f} time(s) in the last 6 months{_paren('lowest month-end in the last 3: ' + eur(e))}.", e
    if name == "ds_ratio":
        e = g("ds5")
        return f"Debt repayments took {v:.0%} of inflows over the last 5 months{_paren(eur(e) + ' repaid')}.", e
    if name == "fc_ratio":
        e = g("fc5")
        return f"Bank fees and interest took {v:.1%} of inflows over the last 5 months{_paren(eur(e) + ' paid')}.", e
    if name == "out_vol":
        e = g("out_sd6")
        return f"Monthly outflows swing by {v:.0%} of their average over 6 months{_paren('std ' + eur(e))}.", e
    if name == "months_observed":
        return f"Only {int(r['trail_months'])} months of history so far.", np.nan
    if name == "active_share":
        e = g("in3")
        return f"Money came in during {round(v / 100 * 6)} of the last 6 months{_paren(eur(e) + ' received in the last 3')}.", e
    if name == "ds_increase":
        e = v * g("in5") if np.isfinite(g("in5")) else np.nan
        return f"Debt repayments rose from {g('ds_increase_prior'):.1%} to {g('ds_ratio'):.1%} of inflows compared with 6 months earlier{_paren(eur(e) + ' more over 5 months')}.", e
    if name == "fc_increase":
        e = v * g("in5") if np.isfinite(g("in5")) else np.nan
        return f"Fees and interest rose from {g('fc_increase_prior'):.1%} to {g('fc_ratio'):.1%} of inflows compared with 6 months earlier{_paren(eur(e) + ' more over 5 months')}.", e
    if name == "cust_tail":
        e = g("top1_eur")
        return f"{g('cust_top1'):.0%} of billing comes from one customer{_paren(eur(e) + ' in 3 months')}.", e
    if name == "credit_note":
        e = g("cn3")
        return f"{v:.0%} of billing was reversed by credit notes{_paren(eur(e) + ' in 3 months')}.", e
    return name, np.nan


def _guard_sentence(r) -> tuple[str, float]:
    if r["guard"] == "dark":
        return (f"No bank movement for {int(r['recency_days'])} days: the score is capped at {spec.CAP_DARK:.0f}"
                f" (it would be {r['score_raw']:.0f} otherwise)."), np.nan
    return (f"Inflows collapsed to {r['inflow_recent'] / r['inflow_older']:.0%} of the company's own earlier level "
            f"({eur(r['inflow_recent'])} a month vs {eur(r['inflow_older'])}): the score is capped at {spec.CAP_FADING:.0f}."), r["inflow_older"] - r["inflow_recent"]


def reasons(items: pd.DataFrame, scored: dict, res: pd.DataFrame, attr: pd.DataFrame, top: int = 4) -> pd.DataFrame:
    """Top-4 reasons per row: `reasons` (why the score is not higher) and `change_reasons` (why it moved).

    Reasons are ranked by score points: points lost against a perfect item for the level, points moved since the
    previous month for the change. Each carries the euro amount behind it (blank when the trail has none).
    Returns a frame with JSON-ready lists of dicts {item, points, sentence, eur}.
    """
    share, pts, active = scored["share"], scored["pts"], scored["active"]
    names = [i.name for i in active]
    cat_of = {i.name: i.category for i in active}
    n_av = {c: pts[[n for n in names if cat_of[n] == c]].notna().sum(axis=1) for c in share.columns}
    lost = pd.DataFrame({n: (share[cat_of[n]] / n_av[cat_of[n]].replace(0, np.nan) * (100.0 - pts[n])).fillna(0.0) for n in names})
    lost_arr, names_arr = lost.to_numpy(), np.array(names)
    dcols = ["d_" + n for n in names]
    delta_arr = attr[dcols].to_numpy() if all(c in attr.columns for c in dcols) else None
    d_guard = attr["d_guard"].to_numpy()
    out_level, out_change = [], []
    guard_now = res["guard"].to_numpy()
    guard_prev = res["guard"].groupby(items["company_id"], sort=False).shift(1).to_numpy()
    for k in range(len(items)):
        row = items.iloc[k]
        r = res.iloc[k]
        if not np.isfinite(r["score"]):
            out_level.append([]); out_change.append([]); continue
        lv = []
        if r["guard"]:
            s, e = _guard_sentence({**row.to_dict(), **r.to_dict()})
            lv.append({"item": "guard", "points": float(r["score"] - r["score_raw"]), "sentence": s, "eur": None if not np.isfinite(e) else float(e)})
        for j in np.argsort(-lost_arr[k])[:top]:
            if lost_arr[k, j] <= 0.05 or len(lv) >= top:
                continue
            s, e = item_sentence(names_arr[j], row)
            lv.append({"item": names_arr[j], "points": -float(lost_arr[k, j]), "sentence": s, "eur": None if not np.isfinite(e) else float(e)})
        out_level.append(lv[:top])
        ch = []
        if guard_now[k] and guard_now[k] == guard_prev[k]:
            ch = [{"item": "guard", "points": 0.0, "sentence": f"The score is held at the activity cap ({r['score']:.0f}) while activity has not resumed.", "eur": None}]
        elif delta_arr is not None and np.isfinite(delta_arr[k]).any():
            order = np.argsort(-np.abs(np.nan_to_num(delta_arr[k])))
            for j in order[:top]:
                dv = float(np.nan_to_num(delta_arr[k, j]))
                if abs(dv) < 0.5:
                    continue
                s, e = item_sentence(names[j], row)
                ch.append({"item": names[j], "points": dv, "sentence": s, "eur": None if not np.isfinite(e) else float(e)})
            if np.isfinite(d_guard[k]) and abs(d_guard[k]) >= 0.5:
                gs, ge = _guard_sentence({**row.to_dict(), **r.to_dict()}) if r["guard"] else ("The activity cap no longer applies.", np.nan)
                ch.append({"item": "guard", "points": float(d_guard[k]), "sentence": gs, "eur": None if not np.isfinite(ge) else float(ge)})
            ch = sorted(ch, key=lambda x: -abs(x["points"]))[:top]
        out_change.append(ch)
    return pd.DataFrame({"reasons": out_level, "change_reasons": out_change}, index=items.index)
