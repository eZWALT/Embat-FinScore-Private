"""Scorecard arithmetic: item points -> category scores -> 0-100 score, guard, confidence, contributions.

No fitting here. `ref` is the percentile reference from fit.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import spec


def pct_points(values: np.ndarray, q: np.ndarray, higher_better: bool) -> np.ndarray:
    """Share of the train reference that is at least as bad as the value, x100 (100 = healthiest).

    Ties count against the reference, so a company at the best value of a zero-heavy variable
    (no overdue invoices, no debt service) gets 100, not the mid-rank of the tie block.
    """
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    n = len(q)
    if higher_better:
        share = np.searchsorted(q, v[ok], side="right") / n   # reference <= value
    else:
        share = (n - np.searchsorted(q, v[ok], side="left")) / n  # reference >= value
    out[ok] = 100.0 * share
    return out


def apply_guard(raw: np.ndarray, cap: np.ndarray, company: np.ndarray, valid: np.ndarray, step: float = spec.GUARD_STEP) -> tuple[np.ndarray, np.ndarray]:
    """Going-dark / fading ceiling, coming down at most `step` points a month toward `cap` and lifted at once when the guard ends.

    Rows must be ordered by company and month. While `cap` is finite the ceiling is max(cap, previous score - step), or `cap`
    itself when the company has no previous score. Returns (score, ceiling); ceiling is inf where no guard applies.
    """
    n = len(raw)
    score = raw.astype(float).copy()
    ceiling = np.full(n, np.inf)
    prev, prev_company = np.nan, None
    for i in range(n):
        if company[i] != prev_company:
            prev, prev_company = np.nan, company[i]
        if valid[i] and np.isfinite(cap[i]):
            ceiling[i] = cap[i] if not np.isfinite(prev) else max(cap[i], prev - step)
            score[i] = min(raw[i], ceiling[i])
        prev = score[i] if valid[i] else np.nan
    return score, ceiling


def item_points(items: pd.DataFrame, ref: dict, active: list[spec.Item]) -> pd.DataFrame:
    pts = pd.DataFrame(index=items.index)
    for it in active:
        if it.kind == "fixed":
            pts[it.name] = items[it.name].astype(float)
        else:
            pts[it.name] = pct_points(items[it.name].to_numpy(), np.asarray(ref["items"][it.name]["q"]), it.higher_better)
    return pts


def score_frame(items: pd.DataFrame, ref: dict, drop_families=frozenset(), min_months: int = spec.WINDOW,
                weights: dict | None = None, guard: bool = True) -> dict:
    """Score every row of `items` (output of items.compute_items).

    drop_families: feature families whose items are removed from the scorecard (validation against an outcome
    built from those families). Returns a dict with frames aligned to `items`: `pts`, `contrib`, `cats`,
    `share`, and `res` (score, score_raw, guard, coverage, confidence).
    """
    weights = weights or spec.EFFECTIVE_WEIGHTS
    active = [i for i in spec.ITEMS if i.family not in drop_families]
    pts = item_points(items, ref, active)
    cats, share_w, n_av = {}, {}, {}
    for cat in spec.CATEGORY_WEIGHTS:
        names = [i.name for i in active if i.category == cat]
        if not names:
            continue
        block = pts[names]
        n = block.notna().sum(axis=1)
        ok = (n / len(names)) >= spec.MIN_ITEM_COVERAGE
        cats[cat] = block.mean(axis=1).where(ok)
        share_w[cat] = ok.astype(float) * weights[cat]
        n_av[cat] = n.where(ok, 0)
    cats = pd.DataFrame(cats)
    W = pd.DataFrame(share_w)
    share = W.div(W.sum(axis=1).replace(0, np.nan), axis=0)
    raw = (share * cats).sum(axis=1, min_count=1)

    contrib = pd.DataFrame(0.0, index=items.index, columns=[i.name for i in active])
    for it in active:
        if it.category not in share:
            continue
        per_item = share[it.category] / pd.Series(n_av[it.category]).replace(0, np.nan)
        contrib[it.name] = (per_item * pts[it.name]).fillna(0.0)

    dark = items["dark_level"].to_numpy()
    cap = np.select([dark == 2, dark == 1], [spec.CAP_DARK, spec.CAP_FADING], default=np.inf) if guard else np.full(len(items), np.inf)
    scored = (items["trail_months"] >= min_months).to_numpy() & np.isfinite(raw.to_numpy())
    score, ceiling = apply_guard(raw.to_numpy(), cap, items["company_id"].to_numpy(), scored)
    score = np.where(scored, score, np.nan)

    coverage = W.sum(axis=1) / sum(weights[c] for c in W.columns)
    reasons = []
    for cat, why in (("payment_history", "sin pagos de facturas en la ventana"), ("amounts_owed", "sin saldos de caja"),
                     ("mix", "sin datos de concentración de clientes"), ("new_credit", "historial demasiado corto para comparar el pago de deuda con hace 6 meses")):
        if cat in W.columns:
            reasons.append(pd.Series(np.where(W[cat] == 0, why, ""), index=items.index))
    reasons.append(pd.Series(np.where(items["trail_months"] < spec.CONF_MIN_MONTHS, f"menos de {spec.CONF_MIN_MONTHS} meses de historial", ""), index=items.index))
    quiet = (items["active_share"] < 50).to_numpy()
    reasons.append(pd.Series(np.where(quiet, "entra dinero en menos de la mitad de los últimos 6 meses", ""), index=items.index))
    reasons.append(pd.Series(np.select([dark == 2, dark == 1], ["sin actividad reciente: puntuación limitada", "entradas de caja hundidas: puntuación limitada"], default=""), index=items.index))
    note = pd.concat(reasons, axis=1).apply(lambda r: "; ".join(x for x in r if x), axis=1) if reasons else ""
    low = (coverage < spec.CONF_MED_COVERAGE) | (items["trail_months"] < spec.CONF_MIN_MONTHS) | (dark == 2)
    high = (coverage >= spec.CONF_HIGH_COVERAGE) & (items["trail_months"] >= spec.CONF_HIGH_MONTHS) & (dark == 0) & ~quiet
    confidence = np.select([low, high], ["low", "high"], default="medium")

    res = pd.DataFrame({
        "score": score, "score_raw": np.where(scored, raw, np.nan), "guard_ceiling": np.where(scored & np.isfinite(ceiling), ceiling, np.nan), "guard": np.select([dark == 2, dark == 1], ["dark", "fading"], default=""),
        "coverage": coverage.round(3), "confidence": confidence, "confidence_note": note}, index=items.index)
    return {"pts": pts, "contrib": contrib, "cats": cats, "share": share, "res": res, "active": active}
