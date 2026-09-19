#!/usr/bin/env python3
"""Flatten the score JSON bundle into CSV files for Postgres COPY."""
from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from pathlib import Path

HANDOFF = Path(os.environ.get("EMBAT_HANDOFF", "/Users/ruben/Desktop/Embat-handoff"))

csv.field_size_limit(min(2**31 - 1, 10_000_000))


def pg_array(values, cast_float=False) -> str:
    if values is None:
        return ""
    parts = []
    for v in values:
        if v is None:
            parts.append("NULL")
        elif cast_float or isinstance(v, (int, float)):
            parts.append(str(v))
        elif isinstance(v, bool):
            parts.append("t" if v else "f")
        else:
            s = str(v).replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'"{s}"')
    return "{" + ",".join(parts) + "}"


def pg_bool_array(values) -> str:
    if values is None:
        return ""
    parts = []
    for v in values:
        if v is None:
            parts.append("NULL")
        else:
            parts.append("t" if v else "f")
    return "{" + ",".join(parts) + "}"


def open_csv(path: Path, header: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w", newline="", encoding="utf-8")
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(header)
    return fh, w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(HANDOFF / "bundle"))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / ".staging" / "csv"))
    parser.add_argument("--run-id", default=str(uuid.uuid5(uuid.NAMESPACE_URL, "embat-score-bundle-1.1.0")))
    args = parser.parse_args()
    bundle = Path(args.bundle)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id

    manifest = json.loads((bundle / "manifest.json").read_text())
    index = json.loads((bundle / "companies.json").read_text())
    groups = json.loads((bundle / "groups.json").read_text())
    clusters = json.loads((bundle / "clusters.json").read_text())
    alerts = json.loads((bundle / "alerts.json").read_text())

    fh_run, w_run = open_csv(
        out / "score_runs.csv",
        [
            "run_id",
            "schema_version",
            "scorecard_version",
            "generated_at",
            "as_of_month",
            "months",
            "detail_months",
            "detail_from_month",
            "is_sample",
            "n_companies",
            "n_groups",
            "n_company_months",
            "source",
            "spec",
            "monitor",
            "sections",
            "reference",
            "files",
            "disclaimer",
        ],
    )
    w_run.writerow(
        [
            run_id,
            manifest["schema_version"],
            manifest["scorecard_version"],
            manifest["generated_at"],
            manifest["as_of_month"],
            pg_array(manifest["months"]),
            manifest["detail_months"],
            manifest.get("detail_from_month"),
            manifest["is_sample"],
            manifest["counts"]["companies"],
            manifest["counts"]["groups"],
            manifest["counts"]["company_months"],
            json.dumps(manifest["source"], separators=(",", ":")),
            json.dumps(manifest["spec"], separators=(",", ":")),
            json.dumps(manifest["monitor"], separators=(",", ":")),
            json.dumps(manifest["sections"], separators=(",", ":")),
            json.dumps(manifest.get("reference"), separators=(",", ":")),
            json.dumps(manifest.get("files"), separators=(",", ":")),
            manifest["disclaimer"],
        ]
    )
    fh_run.close()

    fh_cl, w_cl = open_csv(
        out / "clusters.csv",
        ["run_id", "cluster_id", "label", "description", "n_companies", "n_companies_train_fit"],
    )
    for c in clusters["clusters"]:
        w_cl.writerow(
            [
                run_id,
                c["cluster_id"],
                c["label"],
                c.get("description"),
                c.get("n_companies"),
                c.get("n_companies_train_fit"),
            ]
        )
    fh_cl.close()
    fh_q, w_q = open_csv(out / "cluster_quality.csv", ["run_id", "chosen_k", "silhouette", "note"])
    q = clusters.get("quality") or {}
    w_q.writerow([run_id, q.get("chosen_k"), json.dumps(q.get("silhouette")), q.get("note")])
    fh_q.close()

    fh_idx, w_idx = open_csv(
        out / "company_index.csv",
        [
            "run_id",
            "company_id",
            "group_id",
            "latest_month",
            "score",
            "trajectory",
            "confidence",
            "guard",
            "delta_1m",
            "delta_3m",
            "top_reason",
            "cluster_id",
            "n_alerts",
            "max_alert_severity",
            "sparkline",
        ],
    )
    for row in index["companies"]:
        w_idx.writerow(
            [
                run_id,
                row["company_id"],
                row.get("group_id"),
                row["latest_month"],
                row["score"],
                row["trajectory"],
                row["confidence"],
                row.get("guard"),
                row.get("delta_1m"),
                row.get("delta_3m"),
                row.get("top_reason"),
                row.get("cluster_id"),
                row.get("n_alerts"),
                row.get("max_alert_severity"),
                pg_array(row.get("scores"), cast_float=True),
            ]
        )
    fh_idx.close()

    fh_prof, w_prof = open_csv(
        out / "company_profiles.csv",
        [
            "run_id",
            "company_id",
            "group_id",
            "country",
            "currency",
            "erp",
            "first_month",
            "latest_month",
            "cluster_id",
            "alert_ids",
        ],
    )
    fh_sc, w_sc = open_csv(
        out / "company_scores.csv",
        [
            "run_id",
            "company_id",
            "month",
            "score",
            "score_pre_cap",
            "guard",
            "guard_adjustment",
            "trajectory",
            "slope3",
            "slope6",
            "confidence",
            "confidence_note",
            "coverage",
            "trail_months",
            "change_guard",
        ],
    )
    fh_cat, w_cat = open_csv(
        out / "score_categories.csv",
        ["run_id", "company_id", "month", "category_id", "score", "contribution"],
    )
    fh_it, w_it = open_csv(
        out / "score_items.csv",
        ["run_id", "company_id", "month", "item_id", "value", "points", "contribution", "delta"],
    )
    fh_rs, w_rs = open_csv(
        out / "score_reasons.csv",
        [
            "run_id",
            "company_id",
            "month",
            "kind",
            "position",
            "item",
            "label",
            "points",
            "value",
            "unit",
            "eur",
            "sentence",
        ],
    )
    fh_cc, w_cc = open_csv(
        out / "company_cluster.csv",
        ["run_id", "company_id", "cluster_id", "month"],
    )
    fh_vs, w_vs = open_csv(
        out / "cluster_vs.csv",
        ["run_id", "company_id", "metric", "percentile", "robust_z"],
    )
    fh_ch, w_ch = open_csv(
        out / "control_charts.csv",
        [
            "run_id",
            "entity_type",
            "entity_id",
            "comparison",
            "metric",
            "months",
            "values",
            "center",
            "lower",
            "upper",
            "ewma",
            "cusum_low",
            "cusum_high",
            "signal",
            "persistent",
            "method",
        ],
    )
    fh_fc, w_fc = open_csv(
        out / "forecasts.csv",
        [
            "run_id",
            "company_id",
            "metric",
            "method",
            "origin_month",
            "horizon_months",
            "naive_last",
            "skill_vs_naive",
            "note",
        ],
    )
    fh_fp, w_fp = open_csv(
        out / "forecast_points.csv",
        ["run_id", "company_id", "month", "median", "lo50", "hi50", "lo80", "hi80"],
    )

    n_months = 0
    for path in sorted((bundle / "companies").glob("COMP_*.json")):
        d = json.loads(path.read_text())
        cluster = d.get("cluster") or {}
        w_prof.writerow(
            [
                run_id,
                d["company_id"],
                d.get("group_id"),
                d.get("country"),
                d.get("currency"),
                d.get("erp"),
                d.get("first_month"),
                d.get("latest_month"),
                cluster.get("cluster_id"),
                pg_array(d.get("alert_ids") or []),
            ]
        )
        if cluster:
            w_cc.writerow([run_id, d["company_id"], cluster["cluster_id"], cluster.get("month")])
            for vs in cluster.get("vs_cluster") or []:
                w_vs.writerow(
                    [run_id, d["company_id"], vs.get("metric"), vs.get("percentile"), vs.get("robust_z")]
                )
        for chart in d.get("control") or []:
            w_ch.writerow(
                [
                    run_id,
                    "company",
                    d["company_id"],
                    chart.get("comparison"),
                    chart.get("metric"),
                    pg_array(chart.get("months")),
                    pg_array(chart.get("values"), cast_float=True),
                    pg_array(chart.get("center"), cast_float=True),
                    pg_array(chart.get("lower"), cast_float=True),
                    pg_array(chart.get("upper"), cast_float=True),
                    pg_array(chart.get("ewma"), cast_float=True),
                    pg_array(chart.get("cusum_low"), cast_float=True),
                    pg_array(chart.get("cusum_high"), cast_float=True),
                    pg_array(chart.get("signal")),
                    pg_bool_array(chart.get("persistent")),
                    json.dumps(chart.get("method"), separators=(",", ":")),
                ]
            )
        fc = d.get("forecast")
        if fc:
            w_fc.writerow(
                [
                    run_id,
                    d["company_id"],
                    fc.get("metric", "score"),
                    fc.get("method"),
                    fc.get("origin_month"),
                    fc.get("horizon_months"),
                    fc.get("naive_last"),
                    fc.get("skill_vs_naive"),
                    fc.get("note"),
                ]
            )
            for pt in fc.get("points") or []:
                w_fp.writerow(
                    [
                        run_id,
                        d["company_id"],
                        pt["month"],
                        pt["median"],
                        pt["lo50"],
                        pt["hi50"],
                        pt["lo80"],
                        pt["hi80"],
                    ]
                )
        for m in d["months"]:
            n_months += 1
            w_sc.writerow(
                [
                    run_id,
                    d["company_id"],
                    m["month"],
                    m["score"],
                    m.get("score_pre_cap"),
                    m.get("guard"),
                    m.get("guard_adjustment"),
                    m["trajectory"],
                    m.get("slope3"),
                    m.get("slope6"),
                    m["confidence"],
                    m.get("confidence_note"),
                    m.get("coverage"),
                    m.get("trail_months"),
                    m.get("change_guard"),
                ]
            )
            for cid, cat in (m.get("categories") or {}).items():
                w_cat.writerow(
                    [run_id, d["company_id"], m["month"], cid, cat.get("score"), cat.get("contribution")]
                )
            for iid, item in (m.get("items") or {}).items():
                w_it.writerow(
                    [
                        run_id,
                        d["company_id"],
                        m["month"],
                        iid,
                        item.get("value"),
                        item.get("points"),
                        item.get("contribution"),
                        item.get("delta"),
                    ]
                )
            for kind, key in (("level", "reasons"), ("change", "change_reasons")):
                for pos, r in enumerate(m.get(key) or []):
                    w_rs.writerow(
                        [
                            run_id,
                            d["company_id"],
                            m["month"],
                            kind,
                            pos,
                            r.get("item"),
                            r.get("label"),
                            r.get("points"),
                            r.get("value"),
                            r.get("unit"),
                            r.get("eur"),
                            r.get("sentence"),
                        ]
                    )

    for fh in (fh_prof, fh_sc, fh_cat, fh_it, fh_rs, fh_cc, fh_vs, fh_ch, fh_fc, fh_fp):
        fh.close()

    fh_g, w_g = open_csv(
        out / "groups_index.csv",
        [
            "run_id",
            "group_id",
            "n_companies",
            "latest_mean_score",
            "latest_min_score",
            "latest_min_company_id",
            "mean_scores",
            "limits_available",
            "alert_ids",
        ],
    )
    fh_gm, w_gm = open_csv(out / "group_members.csv", ["run_id", "group_id", "company_id"])
    for g in groups["groups"]:
        w_g.writerow(
            [
                run_id,
                g["group_id"],
                g["n_companies"],
                g.get("latest_mean_score"),
                g.get("latest_min_score"),
                g.get("latest_min_company_id"),
                pg_array(g.get("mean_scores"), cast_float=True),
                g.get("limits_available"),
                pg_array(g.get("alert_ids") or []),
            ]
        )
        for cid in g.get("company_ids") or []:
            w_gm.writerow([run_id, g["group_id"], cid])
        for chart in g.get("control") or []:
            # reopen control writer already closed — append
            pass
    fh_g.close()
    fh_gm.close()

    with (out / "control_charts.csv").open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        for g in groups["groups"]:
            for chart in g.get("control") or []:
                w.writerow(
                    [
                        run_id,
                        "group",
                        g["group_id"],
                        chart.get("comparison"),
                        chart.get("metric"),
                        pg_array(chart.get("months")),
                        pg_array(chart.get("values"), cast_float=True),
                        pg_array(chart.get("center"), cast_float=True),
                        pg_array(chart.get("lower"), cast_float=True),
                        pg_array(chart.get("upper"), cast_float=True),
                        pg_array(chart.get("ewma"), cast_float=True),
                        pg_array(chart.get("cusum_low"), cast_float=True),
                        pg_array(chart.get("cusum_high"), cast_float=True),
                        pg_array(chart.get("signal")),
                        pg_bool_array(chart.get("persistent")),
                        json.dumps(chart.get("method"), separators=(",", ":")),
                    ]
                )

    fh_al, w_al = open_csv(
        out / "alerts.csv",
        [
            "run_id",
            "alert_id",
            "entity_type",
            "entity_id",
            "entity_name",
            "month",
            "kind",
            "direction",
            "severity",
            "title",
            "summary",
            "owner",
            "action",
            "persistence_rule",
            "months_flagged",
            "rank_score",
            "evidence",
        ],
    )
    fh_ar, w_ar = open_csv(
        out / "alert_reasons.csv",
        ["run_id", "alert_id", "position", "item", "label", "points", "value", "unit", "eur", "sentence"],
    )
    for a in alerts["alerts"]:
        ent = a.get("entity") or {}
        pers = a.get("persistence") or {}
        w_al.writerow(
            [
                run_id,
                a["alert_id"],
                ent.get("type"),
                ent.get("id"),
                ent.get("name"),
                a["month"],
                a["kind"],
                a["direction"],
                a["severity"],
                a["title"],
                a["summary"],
                a["owner"],
                a["action"],
                pers.get("rule"),
                pers.get("months_flagged"),
                a.get("rank_score"),
                json.dumps(a.get("evidence"), separators=(",", ":")),
            ]
        )
        for pos, r in enumerate(a.get("reasons") or []):
            w_ar.writerow(
                [
                    run_id,
                    a["alert_id"],
                    pos,
                    r.get("item"),
                    r.get("label"),
                    r.get("points"),
                    r.get("value"),
                    r.get("unit"),
                    r.get("eur"),
                    r.get("sentence"),
                ]
            )
    fh_al.close()
    fh_ar.close()

    print(f"exported bundle csv -> {out} run_id={run_id} company_months={n_months}")


if __name__ == "__main__":
    main()
