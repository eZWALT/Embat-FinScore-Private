"""LangChain tools over the bundle and the cleaned DuckDB, plus a plot registry the UI renders.

Tools return compact JSON strings. Plots are not returned to the model; `plot_series` stores a spec in
the active PlotRegistry and returns a short acknowledgement. The UI renders `registry.plots` after the turn.
"""

from __future__ import annotations

import json
import math
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import tool

from poc import db
from poc.bundle import Bundle, load

# ----------------------------------------------------------------------------- plot registry


@dataclass
class PlotRegistry:
    plots: list[dict] = field(default_factory=list)


_registry: ContextVar[PlotRegistry | None] = ContextVar("plot_registry", default=None)


def use_registry(reg: PlotRegistry):
    return _registry.set(reg)


def current_registry() -> PlotRegistry:
    reg = _registry.get()
    if reg is None:
        reg = PlotRegistry()
        _registry.set(reg)
    return reg


_as_of: ContextVar[str | None] = ContextVar("as_of", default=None)


def set_as_of(month: str | None):
    """Clamp bundle tools to months <= month (replay). None = no clamp. Returns a token for reset."""
    return _as_of.set(month)


def reset_as_of(token) -> None:
    _as_of.reset(token)


def _cut() -> str | None:
    return _as_of.get()


def _months_upto(months: list[dict]) -> list[dict]:
    c = _cut()
    return [m for m in months if c is None or m["month"] <= c]


def _bundle() -> Bundle:
    return load()


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_default)


def _default(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return str(o)


def _round(x, n=1):
    return None if x is None else round(x, n)


# ----------------------------------------------------------------------------- bundle tools


@tool
def list_companies(group_id: str | None = None, limit: int = 30) -> str:
    """List scored companies (latest month) with score, trajectory, confidence, guard, alert count.
    Optional group_id filters to one group. Use it to resolve which companies exist."""
    b = _bundle()
    rows = b.companies
    if group_id:
        rows = [c for c in rows if c.get("group_id") == group_id]
    rows = sorted(rows, key=lambda c: c["score"])[:limit]
    out = [
        {
            "company_id": c["company_id"], "group_id": c["group_id"], "score": c["score"],
            "trajectory": c["trajectory"], "confidence": c["confidence"], "guard": c["guard"],
            "delta_3m": c["delta_3m"], "n_alerts": c["n_alerts"], "max_alert_severity": c["max_alert_severity"],
        }
        for c in rows
    ]
    return _j({"as_of": b.manifest["as_of_month"], "companies": out})


@tool
def get_company(company_id: str, month: str | None = None) -> str:
    """Score, trajectory, guard, confidence, categories, items, reasons (with EUR) and change reasons of one
    company for one month (default: latest). Also returns the 24-month score history and cluster membership."""
    b = _bundle()
    d = b.company(company_id)
    months = _months_upto(d["months"])
    if not months:
        return _j({"error": f"{company_id} has no scored month up to {_cut()}"})
    rec = next((m for m in months if m["month"] == month), None) if month else months[-1]
    if rec is None:
        return _j({"error": f"{company_id} has no scored month {month}", "scored_months": [m["month"] for m in months]})
    items = rec.get("items") or {}
    return _j(
        {
            "company_id": company_id, "group_id": d["group_id"], "country": d["country"], "currency": d["currency"],
            "erp": d["erp"], "first_month": d["first_month"], "month": rec["month"],
            "score": rec["score"], "score_pre_cap": rec["score_pre_cap"], "guard": rec["guard"],
            "trajectory": rec["trajectory"], "slope3": _round(rec["slope3"], 2), "slope6": _round(rec["slope6"], 2),
            "confidence": rec["confidence"], "confidence_note": rec["confidence_note"],
            "coverage": rec["coverage"], "trail_months": rec["trail_months"],
            "categories": rec["categories"],
            "items": {k: {kk: _round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in items.items()},
            "reasons": rec.get("reasons"), "change_reasons": rec.get("change_reasons"),
            "score_history": [{"month": m["month"], "score": m["score"], "trajectory": m["trajectory"], "guard": m["guard"]} for m in months],
            "cluster": d.get("cluster"),
            "alert_ids": [a for a in d.get("alert_ids", []) if _cut() is None or a.rsplit(":", 1)[-1] <= _cut()],
            "as_of_clamp": _cut(),
        }
    )


@tool
def explain_change(company_id: str, month: str | None = None) -> str:
    """Why the score moved from the previous month: signed per-item deltas, change_reasons, guard effect.
    Default month: latest."""
    b = _bundle()
    d = b.company(company_id)
    months = _months_upto(d["months"])
    if not months:
        return _j({"error": f"no scored month up to {_cut()}"})
    idx = len(months) - 1
    if month:
        idx = next((i for i, m in enumerate(months) if m["month"] == month), -1)
        if idx < 0:
            return _j({"error": f"no scored month {month}"})
    if idx == 0:
        return _j({"error": "first scored month, no previous month"})
    cur, prev = months[idx], months[idx - 1]
    deltas = {k: _round(v.get("delta"), 2) for k, v in (cur.get("items") or {}).items() if v.get("delta") is not None}
    return _j(
        {
            "company_id": company_id, "from": prev["month"], "to": cur["month"],
            "score_from": prev["score"], "score_to": cur["score"], "change": _round(cur["score"] - prev["score"], 1),
            "trajectory_from": prev["trajectory"], "trajectory_to": cur["trajectory"],
            "guard_from": prev["guard"], "guard_to": cur["guard"], "change_guard": cur.get("change_guard"),
            "item_deltas": dict(sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)),
            "change_reasons": cur.get("change_reasons"),
        }
    )


@tool
def get_group(group_id: str) -> str:
    """Group summary: members with their latest score/trajectory/guard, group mean and min, mean-score history,
    whether funnel limits exist (3+ scored members), and the group's alert ids."""
    b = _bundle()
    g = b.group(group_id)
    if not g:
        return _j({"error": f"{group_id} not in bundle"})
    rows = {c["company_id"]: c for c in b.companies}
    members = [
        {
            "company_id": cid, "score": rows[cid]["score"], "trajectory": rows[cid]["trajectory"],
            "confidence": rows[cid]["confidence"], "guard": rows[cid]["guard"], "delta_3m": rows[cid]["delta_3m"],
            "n_alerts": rows[cid]["n_alerts"],
        }
        for cid in g["company_ids"] if cid in rows
    ]
    members.sort(key=lambda m: m["score"])
    months = b.months
    hist = [{"month": m, "mean_score": s} for m, s in zip(months, g["mean_scores"])
            if s is not None and (_cut() is None or m <= _cut())]
    return _j(
        {
            "group_id": group_id, "n_companies": g["n_companies"], "latest_mean_score": g["latest_mean_score"],
            "latest_min_score": g["latest_min_score"], "latest_min_company_id": g["latest_min_company_id"],
            "limits_available": g["limits_available"], "members": members, "mean_score_history": hist,
            "alert_ids": g["alert_ids"],
            "control_charts": [c["comparison"] for c in (g.get("control") or [])],
        }
    )


@tool
def get_alerts(entity_id: str | None = None, kinds: list[str] | None = None, severities: list[str] | None = None,
               since_month: str | None = None, limit: int = 30) -> str:
    """Alerts from the feed, newest first. Filter by entity (company or group id), kinds
    (score_deterioration, score_improvement, category_drop, going_dark, top_customer_quiet), severities (info, watch, act),
    and since_month (YYYY-MM). Each alert has title, summary, reasons with EUR, owner, action, evidence, persistence."""
    b = _bundle()
    al = b.alerts
    if entity_id:
        al = [a for a in al if a["entity"]["id"] == entity_id]
    if kinds:
        al = [a for a in al if a["kind"] in kinds]
    if severities:
        al = [a for a in al if a["severity"] in severities]
    if since_month:
        al = [a for a in al if a["month"] >= since_month]
    if _cut():
        al = [a for a in al if a["month"] <= _cut()]
    out = []
    for a in al[:limit]:
        a = dict(a)
        a.pop("rank_score", None)  # ordering only; never surfaced
        out.append(a)
    return _j({"stats": b.alerts_feed["stats"], "n_matching": len(al), "alerts": out})


@tool
def get_control_chart(entity_id: str, comparison: str = "own_history", metric: str = "score") -> str:
    """Control chart series for a company (comparison own_history|cluster; metric score|payment_history|amounts_owed|stability)
    or a group (comparison group_own_history|group_vs_groups; metric score). Returns months, values, center, lower, upper,
    ewma, signal per month and the persistent flag, so you can judge dip vs decline and plot it."""
    b = _bundle()
    charts = None
    if entity_id.startswith("GROUP"):
        g = b.group(entity_id)
        charts = (g or {}).get("control") or []
    else:
        charts = b.company(entity_id).get("control") or []
    ch = next((c for c in charts if c["comparison"] == comparison and c["metric"] == metric), None)
    if not ch:
        return _j({"error": "no such chart (needs 7 scored months; groups need 3 members)",
                   "available": [(c["comparison"], c["metric"]) for c in charts]})
    out = {k: v for k, v in ch.items() if k != "method"} | {"method": ch["method"]["name"]}
    c = _cut()
    if c:
        keep = [i for i, m in enumerate(ch["months"]) if m <= c]
        for k in ("months", "values", "center", "lower", "upper", "ewma", "cusum_low", "cusum_high", "signal", "persistent"):
            if isinstance(out.get(k), list):
                out[k] = [out[k][i] for i in keep]
        out["as_of_clamp"] = c
    return _j(out)


@tool
def compare_with_cluster(company_id: str) -> str:
    """Peer-group comparison: the company's behaviour cluster (label, description, size, quality caveat) and its
    percentile / robust z inside that cluster for score and categories at the latest month."""
    b = _bundle()
    d = b.company(company_id)
    cl = d.get("cluster")
    if not cl:
        return _j({"error": "no cluster (trail under 6 months)"})
    meta = next((c for c in b.clusters["clusters"] if c["cluster_id"] == cl["cluster_id"]), {})
    return _j({"company_id": company_id, "cluster": meta, "quality_note": b.clusters["quality"]["note"],
               "month": cl["month"], "vs_cluster": cl["vs_cluster"]})


@tool
def get_forecast(company_id: str) -> str:
    """Score fan 1-6 months ahead (median, 50% and 80% bands) with the naive-last baseline. Method is naive_last:
    it says how far the score usually moves, not which way."""
    b = _bundle()
    f = b.company(company_id).get("forecast")
    return _j(f or {"error": "no forecast (under 4 scored months)"})


# ----------------------------------------------------------------------------- records


@tool
def query_clean_db(sql: str) -> str:
    """Run one read-only SELECT over the cleaned records (schema `clean`: companies, groups, transactions, invoices,
    balances, banking_products, debt_products, debt_schedule_config, dq_log). Always filter by company_id and use LIMIT
    (max 200 rows). Use it for record-level questions (which invoices, which counterparties, monthly flows, balances,
    debt products). Never to recompute a score."""
    try:
        df = db.query(sql)
    except db.UnsafeQuery as e:
        return _j({"error": f"rejected: {e}"})
    except Exception as e:  # duckdb errors
        return _j({"error": str(e).splitlines()[0]})
    return _j({"rows": len(df), "columns": list(df.columns), "data": json.loads(df.to_json(orient="records", date_format="iso"))})


# ----------------------------------------------------------------------------- plots


@tool
def plot_series(title: str, x: list[str], series: dict[str, list[float | None]], kind: str = "line",
                markers: list[str] | None = None, band: dict[str, list[float | None]] | None = None,
                y_label: str | None = None) -> str:
    """Ask the UI to draw a chart. x: month labels (YYYY-MM). series: name -> values aligned with x (null allowed).
    kind: line | bar. markers: months to mark (e.g. alert months). band: optional {"lower": [...], "upper": [...]}
    aligned with x (control limits or forecast band). One or two plots per answer."""
    n = len(x)
    bad = [k for k, v in series.items() if len(v) != n]
    if bad:
        return _j({"error": f"series not aligned with x: {bad}"})
    if band and any(len(v) != n for v in band.values()):
        return _j({"error": "band not aligned with x"})
    current_registry().plots.append(
        {"title": title, "x": x, "series": series, "kind": kind, "markers": markers or [], "band": band, "y_label": y_label}
    )
    return _j({"ok": True, "title": title, "points": n})


BUNDLE_TOOLS = [list_companies, get_company, explain_change, get_group, get_alerts, get_control_chart,
                compare_with_cluster, get_forecast]
RECORD_TOOLS = [query_clean_db]
PLOT_TOOLS = [plot_series]

CHAT_TOOLS = BUNDLE_TOOLS + RECORD_TOOLS + PLOT_TOOLS
SENTINEL_TOOLS = [get_company, get_group, get_alerts, get_control_chart, plot_series]
