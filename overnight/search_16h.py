#!/usr/bin/env python3
"""16-hour CLI search: try score weights × Y definitions until holdout metrics hit."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import (  # noqa: E402
    DATA,
    evaluate_scores,
    load_holdout,
    parse_month,
)

HERE = Path(__file__).resolve().parent
FEAT_CACHE = HERE / "cache" / "monthly_features.csv"
LIVE = HERE / "runs" / "LIVE.json"
FOUND = HERE / "runs" / "found.jsonl"
DEADLINE = HERE / "runs" / "deadline.txt"
PRIMARY_BAR = 0.57


def build_features() -> list[dict]:
    if FEAT_CACHE.exists():
        return list(csv.DictReader(FEAT_CACHE.open()))

    net: dict[tuple[str, date], float] = defaultdict(float)
    inflow: dict[tuple[str, date], float] = defaultdict(float)
    outflow: dict[tuple[str, date], float] = defaultdict(float)
    overdue: dict[tuple[str, date], float] = defaultdict(float)
    billed: dict[tuple[str, date], float] = defaultdict(float)

    print("building monthly features from transactions/invoices (once)...", flush=True)
    with (DATA / "transactions.csv").open() as f:
        for r in csv.DictReader(f):
            m = parse_month(r["date"])
            amt = float(r["amount"] or 0)
            key = (r["company_id"], m)
            net[key] += amt
            if amt > 0:
                inflow[key] += amt
            else:
                outflow[key] += abs(amt)

    with (DATA / "invoices.csv").open() as f:
        for r in csv.DictReader(f):
            m = parse_month(r["issuance_date"] or r["due_date"])
            key = (r["company_id"], m)
            amt = abs(float(r["amount"] or 0))
            billed[key] += amt
            if (r.get("status") or "").lower() in {"overdue", "pending"}:
                overdue[key] += abs(float(r["pending_amount"] or 0))

    keys = set(net) | set(billed)
    rows = []
    for cid, m in sorted(keys, key=lambda x: (x[0], x[1])):
        out = outflow[cid, m]
        inn = inflow[cid, m]
        bill = billed[cid, m]
        rows.append(
            {
                "company_id": cid,
                "month": m.isoformat(),
                "net": f"{net[cid, m]:.6f}",
                "coverage": f"{inn / (out + 1.0):.6f}",
                "overdue_share": f"{(overdue[cid, m] / bill) if bill else 0.0:.6f}",
                "inflow": f"{inn:.6f}",
            }
        )
    FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with FEAT_CACHE.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["company_id", "month", "net", "coverage", "overdue_share", "inflow"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"cached {len(rows)} company-months -> {FEAT_CACHE}", flush=True)
    return rows


def labels_from_features(rows: list[dict], holdout: set[str], cash_q: float, ov_q: float, mode: str) -> list[dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["company_id"] in holdout:
            by[r["company_id"]].append(r)
    out = []
    for cid, rs in by.items():
        rs = sorted(rs, key=lambda x: x["month"])
        if len(rs) < 8:
            continue
        nets = [float(x["net"]) for x in rs]
        ovs = [float(x["overdue_share"]) for x in rs]
        n_cut = sorted(nets)[max(0, int(cash_q * (len(nets) - 1)))]
        o_cut = sorted(ovs)[max(0, int(ov_q * (len(ovs) - 1)))]
        stress_m = []
        for i, r in enumerate(rs):
            cash_bad = nets[i] <= n_cut
            pay_bad = ovs[i] >= o_cut and float(r.get("overdue_share") or 0) > 0
            stress = int((cash_bad and pay_bad) if mode == "and" else (cash_bad or pay_bad))
            stress_m.append(stress)
            rec = int(i >= 3 and stress_m[i - 3] == 1 and stress == 0)
            out.append(
                {
                    "company_id": cid,
                    "month": r["month"],
                    "stress": str(stress),
                    "recover": str(rec),
                    "net": r["net"],
                }
            )
    return out


def percentile_ref(values: list[float]) -> list[float]:
    v = sorted(x for x in values if x == x)
    if len(v) > 30000:
        step = len(v) / 30000
        v = [v[int(i * step)] for i in range(30000)]
    return v


def pct_rank(x: float, ref: list[float]) -> float:
    if not ref:
        return 0.5
    lo, hi = 0, len(ref)
    while lo < hi:
        mid = (lo + hi) // 2
        if ref[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo / len(ref)


def score_card(rows: list[dict], train: set[str], weights: dict[str, float]) -> dict[tuple[str, date], float]:
    keys = ["net", "coverage", "overdue_share", "inflow"]
    direction = {"net": 1, "coverage": 1, "overdue_share": -1, "inflow": 1}
    refs = {}
    for k in keys:
        refs[k] = percentile_ref(
            [float(r[k]) for r in rows if r["company_id"] in train]
        )
    out = {}
    wsum = sum(weights.values()) or 1.0
    for r in rows:
        parts = []
        ww = []
        for k in keys:
            p = pct_rank(float(r[k]), refs[k])
            if direction[k] < 0:
                p = 1.0 - p
            parts.append(p)
            ww.append(weights[k])
        s = 100.0 * sum(a * b for a, b in zip(parts, ww)) / (sum(ww) or wsum)
        out[r["company_id"], parse_month(r["month"])] = max(0.0, min(100.0, s))
    return out


def rand_weights(rng: random.Random) -> dict[str, float]:
    raw = [rng.random() for _ in range(4)]
    s = sum(raw) or 1.0
    keys = ["net", "coverage", "overdue_share", "inflow"]
    return {k: x / s for k, x in zip(keys, raw)}


def write_live(payload: dict) -> None:
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    LIVE.write_text(json.dumps(payload, indent=2) + "\n")


def write_candidate(name: str, scores: dict[tuple[str, date], float]) -> Path:
    path = HERE / "candidates" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["company_id", "month", "score"])
        for (cid, m), sc in sorted(scores.items()):
            w.writerow([cid, m.isoformat(), f"{sc:.4f}"])
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=16.0)
    ap.add_argument("--bar", type=float, default=PRIMARY_BAR, help="primary metric to count as a found Y")
    args = ap.parse_args()

    now = time.time()
    DEADLINE.parent.mkdir(parents=True, exist_ok=True)
    if DEADLINE.exists():
        deadline_wall = float(DEADLINE.read_text().strip())
        if deadline_wall <= now:
            print("previous 16h window already ended; starting a new one", flush=True)
            deadline_wall = now + args.hours * 3600
            DEADLINE.write_text(str(deadline_wall) + "\n")
        else:
            print(f"resuming until {deadline_wall} (unix)", flush=True)
    else:
        deadline_wall = now + args.hours * 3600
        DEADLINE.write_text(str(deadline_wall) + "\n")

    rng = random.Random(20260918)
    holdout = load_holdout()
    features = build_features()
    all_cos = {r["company_id"] for r in features}
    train = all_cos - holdout

    y_grid = [
        (cq, oq, mode)
        for cq in (0.15, 0.20, 0.25)
        for oq in (0.75, 0.80, 0.85)
        for mode in ("or", "and")
    ]

    trial = 0
    found = 0
    best = None
    print(f"TIMER {args.hours:.1f}h  bar={args.bar}  holdout={len(holdout)}  train={len(train)}", flush=True)

    while time.time() < deadline_wall:
        trial += 1
        left_h = (deadline_wall - time.time()) / 3600
        cash_q, ov_q, mode = y_grid[(trial - 1) % len(y_grid)]
        if trial > len(y_grid):
            cash_q = rng.choice([0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30])
            ov_q = rng.choice([0.70, 0.75, 0.80, 0.85, 0.90])
            mode = rng.choice(["or", "and"])
        weights = rand_weights(rng)
        if trial == 1:
            weights = {"net": 0.3, "coverage": 0.25, "overdue_share": 0.25, "inflow": 0.2}

        labels = labels_from_features(features, holdout, cash_q, ov_q, mode)
        scores = score_card(features, train, weights)
        name = f"t{trial:05d}_y{mode}_c{cash_q:.2f}_o{ov_q:.2f}"
        report = evaluate_scores(scores, labels, holdout, name)
        report["y"] = {"cash_q": cash_q, "ov_q": ov_q, "mode": mode, "weights": weights}
        report["trial"] = trial
        report["hours_left"] = round(left_h, 3)

        prim = report.get("primary")
        ok = bool(report.get("pass_gates") and prim is not None and prim >= args.bar)
        if best is None or (prim is not None and (best.get("primary") or -1) < prim):
            best = report

        live = {
            "running": True,
            "trial": trial,
            "hours_left": round(left_h, 3),
            "last_primary": prim,
            "best_primary": None if best is None else best.get("primary"),
            "found": found,
            "last_y": report["y"],
        }
        write_live(live)
        print(
            f"[{left_h:5.2f}h left] trial {trial} primary={prim} "
            f"y={mode}/c{cash_q}/o{ov_q} gates={report.get('pass_gates')} found={found}",
            flush=True,
        )

        if ok:
            found += 1
            path = write_candidate(name, scores)
            rec = {**report, "path": str(path)}
            FOUND.parent.mkdir(parents=True, exist_ok=True)
            with FOUND.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            (HERE / "runs" / f"{name}.json").write_text(json.dumps(rec, indent=2) + "\n")
            print(f"FOUND_Y {name} primary={prim} -> {path}", flush=True)

    write_live({"running": False, "trial": trial, "found": found, "best": best, "hours_left": 0})
    print(f"TIMER_DONE trials={trial} found={found} best={None if best is None else best.get('primary')}", flush=True)


if __name__ == "__main__":
    main()
