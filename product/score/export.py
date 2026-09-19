"""Export the score as a static JSON bundle for the web app (Next.js on Vercel). Run once, ship the folder.

    PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> \
      python -m product.score.export --csv-folder <dir> --out <bundle_dir> [--detail-months 12] [--sample 12]
    python -m product.score.export --validate <bundle_dir>

Contract: product/score/DATA_CONTRACT.md, types: product/score/contract/types.ts (copied into the bundle).
Sections (manifest.sections says which exist): scores, groups, control charts, clusters, alerts with owners and actions, forecast.
Customers/suppliers as entities stay blocked (counterparty IDs do not map to company IDs). Steps 3-4 live in analysis/monitor/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import spec
from .explain import PERSIST_MONTHS, SLOPE3_MATERIAL, SLOPE6_MATERIAL
from .fit import load_reference
from .run import score_folder

SCHEMA_VERSION = "1.3.0"
SCORECARD_VERSION = "v1.1"
ENGLISH_WORDS = re.compile(r"\b(the|and|of|is|was|against|months|days|with|from|than|customer|billing|score)\b")  # none of these is a Spanish word
CONTRACT_DIR = Path(__file__).resolve().parent / "contract"
ITEM_NAMES = [i.name for i in spec.ITEMS]
DISCLAIMER = ("Documented, explainable score computed from the company's own treasury trail. It is a monitoring aid, not a "
              "bankruptcy predictor: on the accepted outcomes it does not beat a company-size baseline (product/score/validation.md).")


def _f(x, nd=2):
    """float or None (NaN/inf/None -> null), rounded."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) + 0.0 if np.isfinite(x) else None  # + 0.0 turns -0.0 into 0.0


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _write(path: Path, obj) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=_np_default).encode("utf8")
    path.write_bytes(data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def item_value(name: str, r) -> float | None:
    """The number quoted next to an item, in spec.UNITS."""
    if name == "months_observed":
        return _f(r["trail_months"], 0)
    if name == "active_share":
        return _f(r["active_share"] / 100.0 * 6.0, 0)
    if name == "cust_tail":
        return _f(r["cust_hhi"], 3)
    return _f(r[name], 4)


def _reason(x: dict, row_items, digits: int = 1) -> dict:
    item = x["item"]
    if item == "guard":
        label, unit, value = "Límite por inactividad", None, None
    else:
        label, unit, value = spec.ITEM_BY_NAME[item].label, spec.UNITS[item], item_value(item, row_items)
    return {"item": item, "label": label, "points": _f(x["points"], digits), "value": value, "unit": unit,
            "eur": None if x.get("eur") is None else _f(x["eur"], 0), "sentence": x["sentence"]}


def month_record(d, it, detail: bool) -> dict:
    """One company-month. `d`: row of the detail frame, `it`: row of the items frame."""
    contrib = {n: d[f"contrib_{n}"] for n in ITEM_NAMES if f"contrib_{n}" in d.index}
    cats = {}
    for c in spec.CATEGORY_WEIGHTS:
        v = d.get(f"cat_{c}")
        avail = v is not None and np.isfinite(v)
        cats[c] = {"score": _f(v, 1) if avail else None,
                   "contribution": _f(sum(contrib.get(n, 0.0) for n in ITEM_NAMES if spec.ITEM_CATEGORY[n] == c), 2) if avail else 0.0}
    rec = {
        "month": f"{d['period']:%Y-%m}", "score": _f(d["score"], 2), "score_pre_cap": _f(d["score_raw"], 2),
        "guard": d["guard"] or None, "guard_ceiling": _f(d.get("guard_ceiling"), 2), "guard_adjustment": _f(d["score"] - d["score_raw"], 2),
        "trajectory": d["trajectory"], "slope3": _f(d["slope3"], 2), "slope6": _f(d["slope6"], 2),
        "confidence": d["confidence"], "confidence_note": d["confidence_note"] or None, "coverage": _f(d["coverage"], 3),
        "trail_months": int(it["trail_months"]), "categories": cats,
    }
    if detail:
        items = {}
        for n in ITEM_NAMES:
            pts = d.get(f"pts_{n}")
            if pts is None or not np.isfinite(pts):
                continue
            items[n] = {"value": item_value(n, it), "points": _f(pts, 1), "contribution": _f(contrib[n], 2), "delta": _f(d.get(f"d_{n}"), 2)}
        rec["items"] = items
        rec["reasons"] = [_reason(x, it) for x in d["reasons"]]
        rec["change_reasons"] = [_reason(x, it, 2) for x in d["change_reasons"]]
        rec["change_guard"] = _f(d.get("d_guard"), 2)
    return rec


def manifest_spec() -> dict:
    return {
        "categories": {c: {"label": spec.CATEGORY_LABELS[c], "nominal_weight": w, "effective_weight": spec.EFFECTIVE_WEIGHTS[c]}
                       for c, w in spec.CATEGORY_WEIGHTS.items()},
        "items": [{"id": i.name, "category": i.category, "label": i.label, "higher_is_better": i.higher_better, "unit": spec.UNITS[i.name],
                   "kind": i.kind, "why": i.why} for i in spec.ITEMS],
        "guard": {"dark_no_booking_days": spec.DARK_NO_TX_DAYS, "cap_dark": spec.CAP_DARK,
                  "fading_inflow_ratio": spec.FADING_INFLOW_RATIO, "cap_fading": spec.CAP_FADING, "max_drop_per_month": spec.GUARD_STEP},
        "trajectory": {"states": ["improving", "stable", "dip", "deteriorating", "insufficient history"],
                       "slope3_material": SLOPE3_MATERIAL, "slope6_material": SLOPE6_MATERIAL, "persist_months": PERSIST_MONTHS},
        "confidence": {"levels": ["high", "medium", "low"], "high_min_coverage": spec.CONF_HIGH_COVERAGE, "medium_min_coverage": spec.CONF_MED_COVERAGE,
                       "high_min_months": spec.CONF_HIGH_MONTHS, "low_below_months": spec.CONF_MIN_MONTHS},
        "score_needs_months": spec.WINDOW,
    }


def run_monitor_and_forecast(work_dir: Path, detail: pd.DataFrame, items: pd.DataFrame, companies: pd.DataFrame):
    """Steps 3-4 on the full scored panel: control charts, clusters, alerts (analysis/monitor) and the score fan. Returns (monitor, forecast rows, params)."""
    import duckdb

    from analysis.monitor import engine, forecast as fc, fit as mfit

    store = pd.read_parquet(Path(work_dir) / "feature_store" / "monthly.parquet")
    store["company_id"] = store["company_id"].astype(str)
    store["period"] = pd.to_datetime(store["period"])
    con = duckdb.connect(str(Path(work_dir) / "embat.duckdb"), read_only=True)
    con.execute("SET search_path = 'clean,main'")
    try:
        params = mfit.load()
        group_of = companies.assign(company_id=companies["company_id"].astype(str)).set_index("company_id")["group_id"]
        mon = engine.run_monitor(detail, items, store, con, group_of, params)
    finally:
        con.close()
    fparams = json.loads(fc.PARAMS_PATH.read_text(encoding="utf8"))
    S = mfit.wide(detail, "score", mon.grid)
    rows = dict(zip(S.index, fc.forecast_rows(S.to_numpy(dtype=float), fparams)))
    return mon, rows, {**params, "material": engine.MATERIAL}, fparams


def _points(n: int) -> str:
    return f"{n} punto{'s' if n != 1 else ''}"


def _gap(x: float, ref: str) -> str:
    """'N punto(s) por encima/por debajo de <ref>', or 'en línea con <ref>' when the gap is under a point."""
    n = round(abs(x))
    return f"en línea con {ref}" if n == 0 else f"{_points(n)} {'por encima' if x > 0 else 'por debajo'} de {ref}"


def _driver_text(factor: str, d: dict, centre: float) -> str:
    """Frase en español para una parte de los motores del pronóstico a 3 meses (d: el dict de motores más la media propia de la empresa)."""
    if factor == "own_average":
        return "la última puntuación está " + _gap(d["own_deviation"], "la media propia de la empresa (%.0f)" % d["own_average"])
    if factor == "portfolio_level":
        return "la puntuación está " + _gap(d["level_vs_portfolio"], "la empresa típica (%.0f)" % centre)
    if factor == "recent_move":
        n = round(abs(d["last_3_month_move"]))
        return "la puntuación es estable en los últimos 3 meses" if n == 0 else "la puntuación ha variado %+d %s en los últimos 3 meses" % (round(d["last_3_month_move"]), "punto" if n == 1 else "puntos")
    return "deriva típica para una empresa con esta volatilidad"


def _forecast_note(fparams: dict) -> str:
    won = [h for h, m in fparams["method_by_horizon"].items() if m == "reversion_quantile"]
    verdict = ("El abanico de reversión superó al abanico ingenuo, con el intervalo por encima de cero, en los horizontes de " + ", ".join(won) + " meses."
               if won else "El abanico de reversión no superó al abanico ingenuo en ningún horizonte, así que se usa el último valor.")
    return ("Mejora de la pérdida pinball de este abanico frente al abanico ingenuo (último valor con cuantiles agrupados del error), medida con validación cruzada "
            "a 3 meses y en cada horizonte. " + verdict + " Es un abanico de persistencia, no una predicción de resultados.")


def forecast_json(row: dict, grid, fparams: dict) -> dict:
    origin = grid[row["origin_index"]]
    pts = [{"month": f"{origin + pd.DateOffset(months=p['h']):%Y-%m}", "median": _f(p["median"], 1), "lo50": _f(p["lo50"], 1), "hi50": _f(p["hi50"], 1),
            "lo80": _f(p["lo80"], 1), "hi80": _f(p["hi80"], 1), "method": p["method"]} for p in row["points"]]
    drv = row["drivers"]
    drivers = None
    if drv:
        ctx = drv | {"own_average": row["own_average"]}
        drivers = {"horizon_months": drv["horizon"], "expected_change": _f(drv["expected_change"], 1),
                   "parts": [{"factor": k, "points": _f(v, 1), "text": _driver_text(k, ctx, fparams["level_centre"])} for k, v in drv["parts"].items()]}
    return {"metric": "score", "method": fparams["method"], "origin_month": f"{origin:%Y-%m}", "horizon_months": fparams["horizon"], "points": pts,
            "naive_last": _f(row["naive_last"], 1), "own_average": _f(row["own_average"], 1), "drivers": drivers,
            "skill_vs_naive": _f(fparams["skill_h3"], 3), "skill_by_horizon": [_f(fparams["skill_by_horizon"][str(h)], 3) for h in range(1, fparams["horizon"] + 1)],
            "note": _forecast_note(fparams)}


def pick_sample(detail: pd.DataFrame, n: int, alert_companies: dict[str, list[str]] | None = None) -> list[str]:
    """A varied handful: high/low score, each trajectory and guard state, no invoices, short trail."""
    last = detail[detail["score"].notna()].sort_values("period").groupby("company_id").tail(1).set_index("company_id")
    picks: list[str] = []

    def take(mask, k=1):
        for cid in last[mask].sort_index().index:
            if cid not in picks and len([p for p in picks]) < n:
                picks.append(cid)
                k -= 1
                if k == 0:
                    return

    for kind in ("top_customer_quiet", "score_deterioration", "score_improvement", "going_dark", "category_drop"):
        cands = [c for c in (alert_companies or {}).get(kind, []) if c in last.index and c not in picks]
        if cands and len(picks) < n:
            picks.append(cands[0])
    hi = last[(last["confidence"] == "high")]
    take(last.index.isin(hi.sort_values("score", ascending=False).index[:3]), 2)
    take(last.index.isin(hi.sort_values("score").index[:3]), 2)
    for traj in ("deteriorating", "dip", "improving", "stable"):
        take(last["trajectory"] == traj, 1)
    take(last["guard"] == "dark")
    take(last["guard"] == "fading")
    take(last["confidence_note"].str.contains("sin pagos de facturas", na=False), 2)
    take(last["trail_months"] <= 4 if "trail_months" in last else last["confidence"] == "low")
    take(last["confidence"] == "medium", n)
    return picks[:n]


def build_bundle(csv_folder: Path, out: Path, work_dir: Path | None = None, detail_months: int = 12, sample: int | None = None,
                 verbose: bool = True) -> dict:
    out = Path(out)
    work_dir = Path(work_dir) if work_dir else out.parent / (out.name + "_work")
    detail, items, companies, info = score_folder(csv_folder, work_dir, load_reference(), verbose)
    assert (detail[["company_id", "period"]].values == items[["company_id", "period"]].values).all()
    detail = detail.reset_index(drop=True)
    items = items.reset_index(drop=True)
    detail["trail_months"] = items["trail_months"]
    detail["active_share"] = items["active_share"]

    scored = detail[detail["score"].notna()]
    months = sorted(f"{p:%Y-%m}" for p in scored["period"].unique())
    as_of = months[-1]
    by = detail.groupby("company_id", sort=False)["score"]
    detail["delta_1m"] = detail["score"] - by.shift(1)
    detail["delta_3m"] = detail["score"] - by.shift(3)
    meta = companies.set_index("company_id")
    first_month = detail.groupby("company_id")["period"].min()
    keep_detail_from = pd.Timestamp(as_of + "-01") - pd.DateOffset(months=detail_months - 1)

    mon, fc_rows, mparams, fparams = run_monitor_and_forecast(work_dir, detail, items, companies)
    feed_from = keep_detail_from
    alerts_all = mon.alerts[mon.alerts["period"] >= feed_from]
    if sample:
        by_kind = {k: list(g["company_id"]) for k, g in alerts_all[alerts_all["alert"].map(lambda a: a["entity"]["type"] == "company")].groupby("kind")}
        ids = set(pick_sample(detail, sample, by_kind))
        detail, items = detail[detail["company_id"].isin(ids)], items.loc[detail["company_id"].isin(ids)]

    if out.exists():
        shutil.rmtree(out)
    files: dict[str, dict] = {}
    index_rows, group_members = [], {}
    alert_list = [a for a in alerts_all["alert"]]
    if sample:
        keep_entities = {("company", c) for c in ids} | {("group", str(meta.loc[c, "group_id"])) for c in ids if pd.notna(meta.loc[c, "group_id"])}
        alert_list = [a for a in alert_list if (a["entity"]["type"], a["entity"]["id"]) in keep_entities]
    ids_by_entity: dict[tuple[str, str], list[str]] = {}
    for a in alert_list:
        ids_by_entity.setdefault((a["entity"]["type"], a["entity"]["id"]), []).append(a["alert_id"])
    sev_rank = {"act": 3, "watch": 2, "info": 1}
    max_sev: dict[tuple[str, str], str] = {}
    for a in alert_list:
        k = (a["entity"]["type"], a["entity"]["id"])
        if sev_rank[a["severity"]] > sev_rank.get(max_sev.get(k, ""), 0):
            max_sev[k] = a["severity"]
    comp_hashes = hashlib.sha256()
    n_scored_companies = 0
    n_rows = 0
    for cid, g in detail.groupby("company_id", sort=True):
        g = g.sort_values("period")
        g = g[g["score"].notna()]
        if g.empty:
            continue
        n_scored_companies += 1
        recs = []
        for k, (_, d) in enumerate(g.iterrows()):
            it = items.loc[d.name]
            recs.append(month_record(d, it, detail=d["period"] >= keep_detail_from))
        m = meta.loc[cid]
        gid = None if pd.isna(m["group_id"]) else str(m["group_id"])
        doc = {"schema_version": SCHEMA_VERSION, "entity_type": "company", "company_id": cid, "group_id": gid,
               "country": None if pd.isna(m["country"]) else str(m["country"]), "currency": None if pd.isna(m["currency"]) else str(m["currency"]),
               "erp": None if pd.isna(m["erp"]) else str(m["erp"]),
               "first_month": f"{first_month[cid]:%Y-%m}", "latest_month": recs[-1]["month"], "months": recs}
        if cid in mon.vs_cluster:
            doc["cluster"] = mon.vs_cluster[cid]
        if mon.charts.get(cid):
            doc["control"] = mon.charts[cid]
        if fc_rows.get(cid):
            doc["forecast"] = forecast_json(fc_rows[cid], mon.grid, fparams)
        doc["alert_ids"] = ids_by_entity.get(("company", cid), [])
        n_rows += len(recs)
        h = _write(out / "companies" / f"{cid}.json", doc)
        comp_hashes.update(f"{cid}:{h['sha256']}".encode())
        last, dlast = recs[-1], g.iloc[-1]
        series = {r["month"]: r["score"] for r in recs}
        top = last["reasons"][0]["sentence"] if last.get("reasons") else None
        index_rows.append({
            "company_id": cid, "group_id": gid, "latest_month": last["month"], "score": last["score"], "trajectory": last["trajectory"],
            "confidence": last["confidence"], "guard": last["guard"], "delta_1m": _f(dlast["delta_1m"], 1), "delta_3m": _f(dlast["delta_3m"], 1),
            "top_reason": top, "cluster_id": doc["cluster"]["cluster_id"] if "cluster" in doc else None,
            "n_alerts": len(doc["alert_ids"]), "max_alert_severity": max_sev.get(("company", cid)), "scores": [series.get(mo) for mo in months]})
        group_members.setdefault(gid, []).append((cid, last["score"], series))
    files["companies_dir"] = {"count": n_scored_companies, "sha256": comp_hashes.hexdigest()}

    files["companies.json"] = _write(out / "companies.json", {"schema_version": SCHEMA_VERSION, "as_of_month": as_of, "months": months, "companies": index_rows})

    groups = []
    for gid, mem in sorted(group_members.items(), key=lambda kv: str(kv[0])):
        if gid is None:
            continue
        means = []
        for mo in months:
            v = [s[mo] for _, _, s in mem if s.get(mo) is not None]
            means.append(_f(np.mean(v), 1) if v else None)
        latest = [(sc, cid) for cid, sc, _ in mem if sc is not None]
        groups.append({"group_id": gid, "company_ids": sorted(c for c, _, _ in mem), "n_companies": len(mem),
                       "latest_mean_score": _f(np.mean([s for s, _ in latest]), 1) if latest else None,
                       "latest_min_score": _f(min(latest)[0], 1) if latest else None,
                       "latest_min_company_id": min(latest)[1] if latest else None, "mean_scores": means,
                       "control": mon.group_charts.get(str(gid)), "alert_ids": ids_by_entity.get(("group", str(gid)), []),
                       "limits_available": str(gid) in mon.group_charts})
    files["groups.json"] = _write(out / "groups.json", {"schema_version": SCHEMA_VERSION, "as_of_month": as_of, "months": months, "groups": groups})

    clusters = {"schema_version": SCHEMA_VERSION, "clusters": mon.clusters,
                "quality": {"silhouette": mparams["clusters"]["model"]["silhouette_by_k"], "chosen_k": mparams["clusters"]["model"]["k"],
                            "note": "Estructura débil (silueta por debajo de 0,2): úsalos como grupo de referencia para comparar, no como segmentos. Ajustados con empresas de entrenamiento, "
                                    "sin la señal de tamaño; la pertenencia es un rasgo de todo el historial y nunca dispara una alerta."}}
    files["clusters.json"] = _write(out / "clusters.json", clusters)
    stats = json.loads((Path(__file__).resolve().parents[2] / "analysis" / "monitor" / "monitor_stats.json").read_text(encoding="utf8"))
    vol = stats["volume_per_company_year"]
    feed = {"schema_version": SCHEMA_VERSION, "as_of_month": as_of, "from_month": f"{feed_from:%Y-%m}",
            "stats": {"false_alarm_rate": _f(stats["headline"]["false_alarm_rate"], 3), "false_alarm_rate_at_chance": _f(stats["headline"]["false_alarm_rate_at_chance"], 3),
                      "median_lead_time_months": _f(stats["headline"]["median_lead_time_months"], 1),
                      "top_customer_precision": _f(stats["top_customer"]["onset_only"]["precision"], 3), "top_customer_base_rate": _f(stats["top_customer"]["base_rate"], 3),
                      "alerts_per_company_year": _f(vol["per_company_year_risk"] + vol["per_company_year_up"], 2),
                      "note": ("Medido en empresas de entrenamiento frente a los ocho resultados aceptados. Las alertas de caída de la puntuación van seguidas de esos resultados "
                               "aproximadamente tan a menudo como una alerta en un mes al azar (lift 0,7-1,2): indican que una empresa se ha alejado de su normalidad, con el motivo y el importe, no que vaya a fallar. "
                               "La alerta de cliente principal es la única con un lift medido (unas 2 veces). Detalles: analysis/monitor/evaluation.md.")},
            "alerts": sorted(alert_list, key=lambda a: (a["month"], sev_rank[a["severity"]], a["rank_score"] or 0.0), reverse=True)}
    files["alerts.json"] = _write(out / "alerts.json", feed)

    ref = load_reference()
    manifest = {
        "schema_version": SCHEMA_VERSION, "scorecard_version": SCORECARD_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_month": as_of, "months": months, "detail_months": detail_months,
        "detail_from_month": f"{keep_detail_from:%Y-%m}",
        "counts": {"companies": n_scored_companies, "groups": len(groups), "company_months": n_rows},
        "is_sample": bool(sample),
        "reference": {"fitted_on": "train companies only", "n_companies": ref["n_companies"], "n_company_months": ref["n_company_months"]},
        "source": {"input_files": {k: v["sha256"][:16] for k, v in info["input_files"].items()},
                   "dq_log_rules_with_rows": sum(1 for v in info["dq_log_rows_affected"].values() if v)},
        "sections": {
            "scores": {"status": "available", "files": ["companies.json", "companies/{company_id}.json"]},
            "groups": {"status": "available", "files": ["groups.json"], "note": "mean/min of member scores; charts and funnel limits only for groups of at least 3 scored companies"},
            "control_charts": {"status": "available", "plan_step": 3, "files": ["companies/{company_id}.json#control", "groups.json#control"]},
            "clusters": {"status": "available", "plan_step": 3, "files": ["clusters.json", "companies/{company_id}.json#cluster"]},
            "alerts": {"status": "available", "plan_step": 3, "files": ["alerts.json"], "note": "two-sided score alerts, category drops, going dark, top customer quiet; onsets only"},
            "forecast": {"status": "available", "plan_step": 4, "files": ["companies/{company_id}.json#forecast"], "note": f"method shipped: {fparams['method']}; fan skill over the naive fan at 3 months {fparams['skill_h3']:+.1%} (cross-validated, train companies); "
                                                                                  + ("seasonality found" if fparams["seasonality"]["supported"] else "no seasonality found (tested)")},
            "owners_actions": {"status": "available", "plan_step": 5, "note": "Alert.owner and Alert.action are filled by analysis/monitor/routing.py"},
            "counterparty_entities": {"status": "blocked", "note": "customers/suppliers only if counterparty IDs map to company IDs; they do not today"},
        },
        "monitor": {"params_fitted_on": mparams["fitted_on"], "floors": mparams["control"]["floors"], "method": mparams["control"]["method"],
                    "material_points": mparams["material"], "forecast_method": fparams["method"],
                    "forecast": {"method_by_horizon": fparams["method_by_horizon"], "skill_by_horizon": fparams["skill_by_horizon"], "features": fparams["features"],
                                 "seasonality": {k: fparams["seasonality"][k] for k in ("supported", "acf_lag12", "acf_lag12_ci", "calendar_month_corr_year1_year2", "note")},
                                 "time_split": {"split_month": fparams["time_split_check"]["split_month"],
                                                "per_horizon": {h: {k: v[k] for k in ("pinball_skill", "cover50", "cover80")} for h, v in fparams["time_split_check"]["per_horizon"].items()}},
                                 "move_persistence": fparams["move_persistence"]}},
        "language": "es", "spec": manifest_spec(), "disclaimer": DISCLAIMER, "files": files,
    }
    files["manifest.json"] = None
    _write(out / "manifest.json", {k: v for k, v in manifest.items() if k != "files"} | {"files": {k: v for k, v in files.items() if v}})
    shutil.copy(CONTRACT_DIR / "types.ts", out / "types.ts")
    if verbose:
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        print(f"alerts in feed: {len(alert_list)}")
        print(f"bundle: {n_scored_companies} companies, {len(groups)} groups, months {months[0]}..{as_of}, {size / 1e6:.1f} MB raw -> {out}")
    return manifest


def validate_bundle(path: Path) -> list[str]:
    """Structural and arithmetic checks on a bundle. Returns a list of problems (empty = ok)."""
    path = Path(path)
    bad: list[str] = []
    man = json.loads((path / "manifest.json").read_text(encoding="utf8"))
    if not man["schema_version"].startswith("1."):
        bad.append(f"unknown schema_version {man['schema_version']}")
    feed = json.loads((path / "alerts.json").read_text(encoding="utf8"))
    alert_by_id = {a["alert_id"]: a for a in feed["alerts"]}
    if len(alert_by_id) != len(feed["alerts"]):
        bad.append("alert_id is not unique")
    kinds = {"score_deterioration", "score_improvement", "category_drop", "going_dark", "top_customer_quiet"}
    for a in feed["alerts"]:
        if a["kind"] not in kinds or a["severity"] not in ("info", "watch", "act") or a["direction"] not in ("risk", "opportunity"):
            bad.append(f"{a['alert_id']}: kind/severity/direction")
        if a["owner"] not in ("treasurer", "cfo", "collections") or not a["action"]:
            bad.append(f"{a['alert_id']}: owner/action missing")
        texts = [a["title"], a["summary"], a["action"]] + [r["sentence"] for r in a["reasons"]]
        if any(re.search(r"\b(nan|inf)\b|-0 %", t) for t in texts):
            bad.append(f"{a['alert_id']}: alert text prints nan/inf/-0 %")
        if any(ENGLISH_WORDS.search(t) for t in texts + [a["persistence"]["rule"]]):
            bad.append(f"{a['alert_id']}: alert text has English words")
        if len(a["reasons"]) > 4 or a["entity"]["type"] not in ("company", "group"):
            bad.append(f"{a['alert_id']}: reasons or entity type")
        if a["kind"] == "top_customer_quiet" and ("revenue at risk" in (a["summary"] + a["action"]).lower() or a["persistence"]["months_flagged"] != 1):
            bad.append(f"{a['alert_id']}: top customer wording or onset")
        if not a["month"] >= feed["from_month"] or not a["month"] <= man["as_of_month"]:
            bad.append(f"{a['alert_id']}: month outside the feed window")
    for name, sec in man["sections"].items():
        if sec["status"] == "planned":
            bad.append(f"section {name} is still planned")
    idx = json.loads((path / "companies.json").read_text(encoding="utf8"))
    files = sorted((path / "companies").glob("*.json"))
    if len(files) != len(idx["companies"]) or len(files) != man["counts"]["companies"]:
        bad.append(f"company counts differ: files {len(files)}, index {len(idx['companies'])}, manifest {man['counts']['companies']}")
    if any(len(r["scores"]) != len(idx["months"]) for r in idx["companies"]):
        bad.append("an index row's scores are not aligned with months")
    n_rows = 0
    n_linked = 0
    for f in files:
        doc = json.loads(f.read_text(encoding="utf8"))
        for aid in doc.get("alert_ids", []):
            n_linked += 1
            if aid not in alert_by_id or alert_by_id[aid]["entity"]["id"] != doc["company_id"]:
                bad.append(f"{f.stem}: alert_id {aid} not in alerts.json for this company")
        for c in doc.get("control", []):
            ln = len(c["months"])
            if any(len(c[k]) != ln for k in ("values", "center", "lower", "upper", "signal", "persistent")):
                bad.append(f"{f.stem}: control chart {c['comparison']}/{c['metric']} arrays differ in length")
        fc = doc.get("forecast")
        user_text = [m.get("confidence_note") or "" for m in doc["months"]]
        if fc:
            user_text += [fc["note"]] + [x["text"] for x in (fc.get("drivers") or {}).get("parts", [])]
        if any(ENGLISH_WORDS.search(t) for t in user_text):
            bad.append(f"{f.stem}: forecast or confidence text has English words")
        if fc:
            for p_ in fc["points"]:
                v = [p_["lo80"], p_["lo50"], p_["median"], p_["hi50"], p_["hi80"]]
                if any(x is None for x in v) or v != sorted(v) or not 0 <= v[0] or not v[-1] <= 100:
                    bad.append(f"{f.stem}: forecast fan not ordered or outside 0-100 at {p_['month']}")
            drv = fc.get("drivers")
            if drv and abs(sum(x["points"] for x in drv["parts"]) - drv["expected_change"]) > 0.3:
                bad.append(f"{f.stem}: forecast drivers do not add up to the expected change")
        prev = None
        for r in doc["months"]:
            n_rows += 1
            if r["score"] is None or not 0 <= r["score"] <= 100:
                bad.append(f"{f.stem} {r['month']}: score {r['score']}")
            if r["trajectory"] not in man["spec"]["trajectory"]["states"]:
                bad.append(f"{f.stem} {r['month']}: trajectory {r['trajectory']}")
            if r["confidence"] not in man["spec"]["confidence"]["levels"] or r["guard"] not in (None, "dark", "fading"):
                bad.append(f"{f.stem} {r['month']}: confidence/guard {r['confidence']}/{r['guard']}")
            if set(r["categories"]) != set(man["spec"]["categories"]):
                bad.append(f"{f.stem} {r['month']}: category keys")
            gd, ce = man["spec"]["guard"], r.get("guard_ceiling")
            if (r["guard"] is None) != (ce is None):
                bad.append(f"{f.stem} {r['month']}: guard {r['guard']} but guard_ceiling {ce}")
            elif ce is not None:
                cap = gd["cap_dark"] if r["guard"] == "dark" else gd["cap_fading"]
                if r["score"] > ce + 0.01 or ce < cap - 0.01:
                    bad.append(f"{f.stem} {r['month']}: score {r['score']} / ceiling {ce} / cap {cap}")
                if prev is not None and prev["score"] is not None and ce < prev["score"] - gd["max_drop_per_month"] - 0.01 and ce > cap + 0.01:
                    bad.append(f"{f.stem} {r['month']}: guard ceiling {ce} fell more than {gd['max_drop_per_month']} from the previous score {prev['score']}")
            if "items" in r:
                known = {i["id"] for i in man["spec"]["items"]}
                used = set(r["items"]) | {x["item"] for x in r["reasons"] + r["change_reasons"]} - {"guard"}
                if not used <= known:
                    bad.append(f"{f.stem} {r['month']}: unknown items {sorted(used - known)}")
                tot = sum(v["contribution"] for v in r["items"].values()) + (r["guard_adjustment"] or 0.0)
                if abs(tot - r["score"]) > 0.15:
                    bad.append(f"{f.stem} {r['month']}: contributions {tot:.2f} != score {r['score']}")
                if prev is not None and r.get("change_reasons") is not None and prev["score"] is not None:
                    d = sum(v["delta"] or 0.0 for v in r["items"].values()) + (r.get("change_guard") or 0.0)
                    # items that disappeared carry no delta here; only check when the item set is unchanged
                    if set(r["items"]) == set(prev.get("items", {})) and abs(d - (r["score"] - prev["score"])) > 0.3:
                        bad.append(f"{f.stem} {r['month']}: attribution {d:.2f} != change {r['score'] - prev['score']:.2f}")
                if len(r["reasons"]) > 4 or len(r["change_reasons"]) > 4:
                    bad.append(f"{f.stem} {r['month']}: more than 4 reasons")
            prev = r
    n_company_alerts = sum(1 for a in feed["alerts"] if a["entity"]["type"] == "company")
    if n_linked != n_company_alerts and not man["is_sample"]:
        bad.append(f"company alert links {n_linked} != company alerts in the feed {n_company_alerts}")
    if n_rows != man["counts"]["company_months"]:
        bad.append(f"company_months {n_rows} != manifest {man['counts']['company_months']}")
    for name, meta in man["files"].items():
        if name == "manifest.json" or "sha256" not in meta or name.endswith("_dir"):
            continue
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != meta["sha256"]:
            bad.append(f"{name}: sha256 mismatch")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-folder", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--detail-months", type=int, default=12, help="months (counted back from the last) that carry items, reasons and attribution")
    ap.add_argument("--sample", type=int, default=None, help="write only N varied companies (for the repo's sample bundle)")
    ap.add_argument("--validate", type=Path, default=None)
    args = ap.parse_args()
    if args.validate:
        bad = validate_bundle(args.validate)
        print("bundle ok" if not bad else "\n".join(bad[:30]) + f"\n{len(bad)} problems")
        return 1 if bad else 0
    if not (args.csv_folder and args.out):
        ap.error("--csv-folder and --out are required")
    build_bundle(args.csv_folder, args.out, args.work_dir, args.detail_months, args.sample)
    bad = validate_bundle(args.out)
    print("bundle ok" if not bad else "\n".join(bad[:30]) + f"\n{len(bad)} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
