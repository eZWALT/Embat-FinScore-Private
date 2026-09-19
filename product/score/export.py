"""Export the score as a static JSON bundle for the web app (Next.js on Vercel). Run once, ship the folder.

    PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> \
      python -m product.score.export --csv-folder <dir> --out <bundle_dir> [--detail-months 12] [--sample 12]
    python -m product.score.export --validate <bundle_dir>

Contract: product/score/DATA_CONTRACT.md, types: product/score/contract/types.ts (copied into the bundle).
Only what exists today is written (manifest.sections says which); planned sections (control charts, clusters,
alerts, forecast, owners/actions, non-company entities) have their shapes in types.ts and are added later
without changing what is here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

SCHEMA_VERSION = "1.0.0"
SCORECARD_VERSION = "v1"
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


def _write(path: Path, obj) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf8")
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
        label, unit, value = "Activity guard", None, None
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
        "guard": d["guard"] or None, "guard_adjustment": _f(d["score"] - d["score_raw"], 2),
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
                  "fading_inflow_ratio": spec.FADING_INFLOW_RATIO, "cap_fading": spec.CAP_FADING},
        "trajectory": {"states": ["improving", "stable", "dip", "deteriorating", "insufficient history"],
                       "slope3_material": SLOPE3_MATERIAL, "slope6_material": SLOPE6_MATERIAL, "persist_months": PERSIST_MONTHS},
        "confidence": {"levels": ["high", "medium", "low"], "high_min_coverage": spec.CONF_HIGH_COVERAGE, "medium_min_coverage": spec.CONF_MED_COVERAGE,
                       "high_min_months": spec.CONF_HIGH_MONTHS, "low_below_months": spec.CONF_MIN_MONTHS},
        "score_needs_months": spec.WINDOW,
    }


def pick_sample(detail: pd.DataFrame, n: int) -> list[str]:
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

    hi = last[(last["confidence"] == "high")]
    take(last.index.isin(hi.sort_values("score", ascending=False).index[:3]), 2)
    take(last.index.isin(hi.sort_values("score").index[:3]), 2)
    for traj in ("deteriorating", "dip", "improving", "stable"):
        take(last["trajectory"] == traj, 1)
    take(last["guard"] == "dark")
    take(last["guard"] == "fading")
    take(last["confidence_note"].str.contains("no invoice", na=False), 2)
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

    if sample:
        ids = set(pick_sample(detail, sample))
        detail, items = detail[detail["company_id"].isin(ids)], items.loc[detail["company_id"].isin(ids)]

    if out.exists():
        shutil.rmtree(out)
    files: dict[str, dict] = {}
    index_rows, group_members = [], {}
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
        n_rows += len(recs)
        h = _write(out / "companies" / f"{cid}.json", doc)
        comp_hashes.update(f"{cid}:{h['sha256']}".encode())
        last, dlast = recs[-1], g.iloc[-1]
        series = {r["month"]: r["score"] for r in recs}
        top = last["reasons"][0]["sentence"] if last.get("reasons") else None
        index_rows.append({
            "company_id": cid, "group_id": gid, "latest_month": last["month"], "score": last["score"], "trajectory": last["trajectory"],
            "confidence": last["confidence"], "guard": last["guard"], "delta_1m": _f(dlast["delta_1m"], 1), "delta_3m": _f(dlast["delta_3m"], 1),
            "top_reason": top, "scores": [series.get(mo) for mo in months]})
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
                       "latest_min_company_id": min(latest)[1] if latest else None, "mean_scores": means})
    files["groups.json"] = _write(out / "groups.json", {"schema_version": SCHEMA_VERSION, "as_of_month": as_of, "months": months, "groups": groups})

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
            "groups": {"status": "available", "files": ["groups.json"], "note": "plain mean/min of member scores; funnel limits arrive with control charts"},
            "control_charts": {"status": "planned", "plan_step": 3},
            "clusters": {"status": "planned", "plan_step": 3},
            "alerts": {"status": "planned", "plan_step": 3, "note": "includes 'top customer went quiet' (decided) and two-sided score alerts"},
            "forecast": {"status": "planned", "plan_step": 4, "note": "first step to be cut if time runs out"},
            "owners_actions": {"status": "planned", "plan_step": 5},
            "counterparty_entities": {"status": "blocked", "note": "customers/suppliers only if counterparty IDs map to company IDs; they do not today"},
        },
        "spec": manifest_spec(), "disclaimer": DISCLAIMER, "files": files,
    }
    files["manifest.json"] = None
    _write(out / "manifest.json", {k: v for k, v in manifest.items() if k != "files"} | {"files": {k: v for k, v in files.items() if v}})
    shutil.copy(CONTRACT_DIR / "types.ts", out / "types.ts")
    if verbose:
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        print(f"bundle: {n_scored_companies} companies, {len(groups)} groups, months {months[0]}..{as_of}, {size / 1e6:.1f} MB raw -> {out}")
    return manifest


def validate_bundle(path: Path) -> list[str]:
    """Structural and arithmetic checks on a bundle. Returns a list of problems (empty = ok)."""
    path = Path(path)
    bad: list[str] = []
    man = json.loads((path / "manifest.json").read_text(encoding="utf8"))
    if not man["schema_version"].startswith("1."):
        bad.append(f"unknown schema_version {man['schema_version']}")
    idx = json.loads((path / "companies.json").read_text(encoding="utf8"))
    files = sorted((path / "companies").glob("*.json"))
    if len(files) != len(idx["companies"]) or len(files) != man["counts"]["companies"]:
        bad.append(f"company counts differ: files {len(files)}, index {len(idx['companies'])}, manifest {man['counts']['companies']}")
    if any(len(r["scores"]) != len(idx["months"]) for r in idx["companies"]):
        bad.append("an index row's scores are not aligned with months")
    n_rows = 0
    for f in files:
        doc = json.loads(f.read_text(encoding="utf8"))
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
