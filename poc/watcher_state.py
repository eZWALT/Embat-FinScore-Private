"""Novelty engine for the Sentinel watcher. Deterministic, no LLM.

A watch set (companies + groups) has a state file in poc/.state/<sha1>.json holding what was already
delivered: alert ids, last delivered score / trajectory / guard per entity, last as-of month. Novelties are
what the bundle holds up to the as-of month that this watch set has not seen yet. The bundle is static, so
the as-of month is the replay control: moving it forward simulates time passing.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from poc.bundle import Bundle

INTERVAL_SEC = int(os.environ.get("POC_WATCH_INTERVAL_SEC", "3600"))
STATE_DIR = Path(__file__).resolve().parent / ".state"
SCORE_MOVE_POINTS = 8.0
FIRST_RUN_MONTHS = 3

OWNER_LABEL = {"treasurer": "Tesorero", "cfo": "CFO", "collections": "Cobros"}


# ----------------------------------------------------------------------------- helpers


def month_add(month: str, k: int) -> str:
    y, m = int(month[:4]), int(month[5:7])
    idx = y * 12 + (m - 1) + k
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def fmt_eur(v: float | None, currency: str = "€") -> str:
    if v is None:
        return "—"
    a = abs(v)
    sign = "-" if v < 0 else ""
    if a >= 1e6:
        return f"{sign}{currency}{a / 1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}{currency}{a / 1e3:.0f}k"
    return f"{sign}{currency}{a:.0f}"


def watch_key(company_ids: list[str], group_ids: list[str]) -> str:
    ids = sorted(set(company_ids)) + sorted(set(group_ids))
    return hashlib.sha1("|".join(ids).encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------- state


def empty_state() -> dict:
    return {"delivered_alert_ids": [], "last_scores": {}, "last_as_of": None, "last_check_at": None}


def load_state(key: str) -> dict:
    p = STATE_DIR / f"{key}.json"
    if not p.exists():
        return empty_state()
    with open(p, encoding="utf-8") as f:
        return {**empty_state(), **json.load(f)}


def save_state(key: str, state: dict) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    p = STATE_DIR / f"{key}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    return p


def reset_state(key: str) -> None:
    p = STATE_DIR / f"{key}.json"
    if p.exists():
        p.unlink()


# ----------------------------------------------------------------------------- snapshot at as-of


def company_at(bundle: Bundle, company_id: str, as_of: str) -> dict | None:
    """Latest scored month <= as_of with a 3-month delta computed from the same file."""
    d = bundle.company(company_id)
    months = [m for m in d["months"] if m["month"] <= as_of]
    if not months:
        return None
    cur = months[-1]
    prev3 = next((m for m in months if m["month"] == month_add(cur["month"], -3)), None)
    return {
        "company_id": company_id,
        "group_id": d["group_id"],
        "currency": d.get("currency"),
        "month": cur["month"],
        "score": cur["score"],
        "score_pre_cap": cur["score_pre_cap"],
        "trajectory": cur["trajectory"],
        "guard": cur["guard"],
        "confidence": cur["confidence"],
        "confidence_note": cur.get("confidence_note"),
        "delta_3m": None if prev3 is None else round(cur["score"] - prev3["score"], 1),
        "top_reason": (cur.get("reasons") or [{}])[0].get("sentence") if cur.get("reasons") else None,
    }


def group_at(bundle: Bundle, group_id: str, as_of: str) -> dict | None:
    g = bundle.group(group_id)
    if not g:
        return None
    months = bundle.months
    pairs = [(m, s) for m, s in zip(months, g["mean_scores"]) if m <= as_of and s is not None]
    if not pairs:
        return None
    month, mean = pairs[-1]
    return {
        "group_id": group_id,
        "n_companies": g["n_companies"],
        "month": month,
        "mean_score": mean,
        "limits_available": g["limits_available"],
    }


def snapshot(bundle: Bundle, company_ids: list[str], group_ids: list[str], as_of: str) -> dict:
    companies = [c for c in (company_at(bundle, cid, as_of) for cid in company_ids) if c]
    groups = [g for g in (group_at(bundle, gid, as_of) for gid in group_ids) if g]
    return {"companies": companies, "groups": groups}


# ----------------------------------------------------------------------------- novelties


def _entity_alerts(bundle: Bundle, company_ids: list[str], group_ids: list[str], as_of: str) -> list[dict]:
    by_id = bundle.alerts_by_id()
    out: list[dict] = []
    for cid in company_ids:
        out += [a for a in bundle.alerts_for(cid) if a["month"] <= as_of]
    for gid in group_ids:
        g = bundle.group(gid) or {}
        out += [by_id[i] for i in g.get("alert_ids", []) if i in by_id and by_id[i]["month"] <= as_of]
    out.sort(key=lambda a: (a["month"], a["entity"]["id"]), reverse=True)
    return out


def _public(alert: dict) -> dict:
    a = dict(alert)
    a.pop("rank_score", None)
    a["owner_label"] = OWNER_LABEL.get(a.get("owner"), a.get("owner"))
    return a


def _exposure(bundle: Bundle, company_id: str, as_of: str) -> dict | None:
    quiet = [a for a in bundle.alerts_for(company_id) if a["kind"] == "top_customer_quiet" and a["month"] <= as_of]
    if not quiet:
        return None
    a = quiet[0]  # feed is newest first
    ev = a.get("evidence") or {}
    return {
        "company_id": company_id,
        "alert_month": a["month"],
        "severity": a["severity"],
        "counterparty_id": ev.get("counterparty_id"),
        "share_last_quarter": ev.get("share_last_quarter"),
        "last_quarter_amount": ev.get("last_quarter_amount"),
        "open_receivable_eur": ev.get("open_receivable_eur"),
        "months_billed_of_last_3": ev.get("months_billed_of_last_3"),
        "months_quiet": ev.get("months_quiet"),
    }


def compute_novelties(bundle: Bundle, company_ids: list[str], group_ids: list[str], as_of: str, state: dict) -> dict:
    """What this watch set has not been told yet, as of `as_of`. JSON-able."""
    snap = snapshot(bundle, company_ids, group_ids, as_of)
    delivered = set(state.get("delivered_alert_ids") or [])
    last = state.get("last_scores") or {}
    first_run = state.get("last_as_of") is None
    floor_month = month_add(as_of, -(FIRST_RUN_MONTHS - 1)) if first_run else None

    new_alerts = []
    for a in _entity_alerts(bundle, company_ids, group_ids, as_of):
        if a["alert_id"] in delivered:
            continue
        if floor_month and a["month"] < floor_month:
            continue
        new_alerts.append(_public(a))

    score_moves, trajectory_changes, guard_changes = [], [], []
    for c in snap["companies"]:
        prev = last.get(c["company_id"])
        if not prev:
            continue
        if prev.get("score") is not None and abs(c["score"] - prev["score"]) >= SCORE_MOVE_POINTS:
            score_moves.append({
                "entity_id": c["company_id"], "entity_type": "company", "from_month": prev.get("month"),
                "to_month": c["month"], "score_from": prev["score"], "score_to": c["score"],
                "change": round(c["score"] - prev["score"], 1), "guard": c["guard"],
                "top_reason": c["top_reason"],
            })
        if prev.get("trajectory") != c["trajectory"]:
            trajectory_changes.append({
                "entity_id": c["company_id"], "from": prev.get("trajectory"), "to": c["trajectory"],
                "month": c["month"], "score": c["score"],
            })
        if prev.get("guard") != c["guard"]:
            guard_changes.append({
                "entity_id": c["company_id"], "from": prev.get("guard"), "to": c["guard"], "month": c["month"],
                "score": c["score"], "score_pre_cap": c["score_pre_cap"],
            })
    for g in snap["groups"]:
        prev = last.get(g["group_id"])
        if prev and prev.get("score") is not None and abs(g["mean_score"] - prev["score"]) >= SCORE_MOVE_POINTS:
            score_moves.append({
                "entity_id": g["group_id"], "entity_type": "group", "from_month": prev.get("month"),
                "to_month": g["month"], "score_from": prev["score"], "score_to": g["mean_score"],
                "change": round(g["mean_score"] - prev["score"], 1), "n_companies": g["n_companies"],
            })

    touched = {a["entity"]["id"] for a in new_alerts}
    touched |= {x["entity_id"] for x in score_moves + trajectory_changes + guard_changes}
    watched_ids = [c["company_id"] for c in snap["companies"]] + [g["group_id"] for g in snap["groups"]]
    exposures = [e for e in (_exposure(bundle, c["company_id"], as_of) for c in snap["companies"]) if e]

    return {
        "as_of": as_of,
        "first_run": first_run,
        "last_as_of": state.get("last_as_of"),
        "watched": snap,
        "new_alerts": new_alerts,
        "score_moves": score_moves,
        "trajectory_changes": trajectory_changes,
        "guard_changes": guard_changes,
        "top_customer_exposure": exposures,
        "quiet": [i for i in watched_ids if i not in touched],
        "stats": bundle.alerts_feed.get("stats", {}),
    }


def has_news(nov: dict) -> bool:
    return any(nov[k] for k in ("new_alerts", "score_moves", "trajectory_changes", "guard_changes"))


def mark_delivered(bundle: Bundle, state: dict, nov: dict, company_ids: list[str], group_ids: list[str]) -> dict:
    """Record everything up to as_of as delivered (also alerts older than the first-run window)."""
    delivered = set(state.get("delivered_alert_ids") or [])
    delivered |= {a["alert_id"] for a in _entity_alerts(bundle, company_ids, group_ids, nov["as_of"])}
    last = dict(state.get("last_scores") or {})
    for c in nov["watched"]["companies"]:
        last[c["company_id"]] = {"score": c["score"], "trajectory": c["trajectory"], "guard": c["guard"], "month": c["month"]}
    for g in nov["watched"]["groups"]:
        last[g["group_id"]] = {"score": g["mean_score"], "month": g["month"]}
    return {
        "delivered_alert_ids": sorted(delivered),
        "last_scores": last,
        "last_as_of": nov["as_of"],
        "last_check_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
