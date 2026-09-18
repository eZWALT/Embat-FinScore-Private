#!/usr/bin/env python3
"""Score a candidate 0-100 table on holdout proxies. See overnight/README.md."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HOLD = Path(__file__).resolve().parent / "splits" / "holdout_companies.csv"
CACHE = Path(__file__).resolve().parent / "cache" / "monthly_labels.csv"


def parse_month(s: str) -> date:
    s = s[:10]
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:19] if " " in s else s, fmt).date().replace(day=1)
        except ValueError:
            continue
    if len(s) >= 7:
        return date(int(s[:4]), int(s[5:7]), 1)
    raise ValueError(s)


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """AUROC via Mann–Whitney; None if one class is missing."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = sorted(s for s, y in zip(scores, labels) if y == 0)
    if not pos or not neg:
        return None
    better = 0.0
    nneg = len(neg)
    for p in pos:
        lo, hi = 0, nneg
        while lo < hi:
            mid = (lo + hi) // 2
            if neg[mid] < p:
                lo = mid + 1
            else:
                hi = mid
        less = lo
        lo, hi = 0, nneg
        while lo < hi:
            mid = (lo + hi) // 2
            if neg[mid] <= p:
                lo = mid + 1
            else:
                hi = mid
        better += less + 0.5 * (lo - less)
    return better / (len(pos) * nneg)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def load_holdout() -> set[str]:
    return {r["company_id"] for r in csv.DictReader(HOLD.open())}


def build_labels(holdout: set[str]) -> list[dict]:
    if CACHE.exists():
        return list(csv.DictReader(CACHE.open()))

    net: dict[tuple[str, date], float] = defaultdict(float)
    overdue: dict[tuple[str, date], float] = defaultdict(float)
    billed: dict[tuple[str, date], float] = defaultdict(float)

    with (DATA / "transactions.csv").open() as f:
        for r in csv.DictReader(f):
            if r["company_id"] not in holdout:
                continue
            m = parse_month(r["date"])
            net[r["company_id"], m] += float(r["amount"] or 0)

    inv = DATA / "invoices.csv"
    if inv.exists():
        with inv.open() as f:
            for r in csv.DictReader(f):
                if r["company_id"] not in holdout:
                    continue
                m = parse_month(r["issuance_date"] or r["due_date"])
                amt = abs(float(r["amount"] or 0))
                billed[r["company_id"], m] += amt
                if (r.get("status") or "").lower() in {"overdue", "pending"}:
                    overdue[r["company_id"], m] += abs(float(r["pending_amount"] or 0))

    by_co: dict[str, list[date]] = defaultdict(list)
    keys = set(net) | set(overdue) | set(billed)
    for cid, m in keys:
        by_co[cid].append(m)
    for cid in by_co:
        by_co[cid] = sorted(set(by_co[cid]))

    rows = []
    for cid, months in by_co.items():
        nets = [net[cid, m] for m in months]
        ovs = [
            (overdue[cid, m] / billed[cid, m]) if billed[cid, m] else 0.0 for m in months
        ]
        if len(months) < 8:
            continue
        n_cut = sorted(nets)[max(0, len(nets) // 5 - 1)]
        o_cut = sorted(ovs)[max(0, int(0.8 * (len(ovs) - 1)))]
        stress_m = []
        for i, m in enumerate(months):
            cash_bad = nets[i] <= n_cut
            pay_bad = ovs[i] >= o_cut and billed[cid, m] > 0
            stress = int(cash_bad or pay_bad)
            stress_m.append(stress)
            rec = 0
            if i >= 3 and stress_m[i - 3] == 1 and stress == 0:
                rec = 1
            rows.append(
                {
                    "company_id": cid,
                    "month": m.isoformat(),
                    "stress": str(stress),
                    "recover": str(rec),
                    "net": f"{nets[i]:.6f}",
                }
            )

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["company_id", "month", "stress", "recover", "net"])
        w.writeheader()
        w.writerows(rows)
    return rows


def load_scores(path: Path) -> dict[tuple[str, date], float]:
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            out[r["company_id"], parse_month(r["month"])] = float(r["score"])
    return out


def months_before(m: date, k: int) -> date:
    y, mo = m.year, m.month - k
    while mo <= 0:
        mo += 12
        y -= 1
    return date(y, mo, 1)


def evaluate_scores(
    scores: dict[tuple[str, date], float],
    labels: list[dict],
    holdout: set[str],
    candidate_name: str,
) -> dict:
    y_s, s_s, y_r, s_r = [], [], [], []
    leads = []
    last_scores, last_absnet = [], []
    deltas = []
    by_co: dict[str, list[tuple[date, float]]] = defaultdict(list)

    for (cid, m), sc in scores.items():
        if cid not in holdout:
            continue
        if not (0 <= sc <= 100):
            return {"error": f"score outside 0-100: {sc}", "candidate": candidate_name}
        by_co[cid].append((m, sc))

    for cid, series in by_co.items():
        series.sort()
        for i in range(1, len(series)):
            deltas.append(abs(series[i][1] - series[i - 1][1]))

    lab_i = {(r["company_id"], parse_month(r["month"])): r for r in labels}
    for r in labels:
        cid = r["company_id"]
        m = parse_month(r["month"])
        prev = months_before(m, 3)
        sc = scores.get((cid, prev))
        if sc is None:
            continue
        y_s.append(int(r["stress"]))
        s_s.append(-sc)
        prev_lab = lab_i.get((cid, prev))
        if prev_lab and prev_lab["stress"] == "1":
            y_r.append(int(r["stress"] == "0"))
            s_r.append(sc)

        if r["stress"] == "1":
            hist = [sc0 for _, sc0 in by_co.get(cid, ())]
            if hist:
                q = sorted(hist)[max(0, len(hist) // 3 - 1)]
                lead = 0
                for k in range(1, 7):
                    sk = scores.get((cid, months_before(m, k)))
                    if sk is not None and sk <= q:
                        lead = k
                    else:
                        break
                leads.append(lead)

        if m.month in {8, 9} and m.year == 2026:
            last_scores.append(sc)
            last_absnet.append(abs(float(r["net"])))

    n_hold_cm = len(labels)
    n_scored = sum(1 for r in labels if (r["company_id"], parse_month(r["month"])) in scores)
    coverage = n_scored / n_hold_cm if n_hold_cm else 0.0

    a_s = auroc(s_s, y_s)
    a_r = auroc(s_r, y_r)
    primary = None
    if a_s is not None and a_r is not None:
        primary = (a_s + a_r) / 2
    elif a_s is not None:
        primary = a_s

    size_rho = spearman(last_scores, [math.log1p(x) for x in last_absnet]) if last_scores else None
    vol = sum(deltas) / len(deltas) if deltas else None

    report = {
        "candidate": candidate_name,
        "n_holdout_companies": len(holdout),
        "auroc_stress_lead3": a_s,
        "auroc_recover_lead3": a_r,
        "primary": primary,
        "lead_months": (sum(leads) / len(leads)) if leads else None,
        "coverage": coverage,
        "vol_mae": vol,
        "size_spearman": size_rho,
        "n_stress_pairs": len(y_s),
        "n_recover_pairs": len(y_r),
        "gates": {
            "coverage_ge_0.70": coverage >= 0.70,
            "range_ok": True,
            "not_size_proxy": size_rho is None or abs(size_rho) < 0.85,
        },
    }
    report["pass_gates"] = all(report["gates"].values())
    return report


def evaluate(candidate: Path) -> dict:
    holdout = load_holdout()
    labels = build_labels(holdout)
    scores = load_scores(candidate)
    return evaluate_scores(scores, labels, holdout, candidate.name)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python overnight/compare.py overnight/candidates/<name>.csv", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    report = evaluate(path)
    out = Path(__file__).resolve().parent / "runs" / (path.stem + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
