# Health Score data contract (schema 1.1.0)

For whoever builds the web app (Next.js on Vercel). The pipeline runs **once** on a folder of CSVs and writes a **static JSON bundle**; the app reads the bundle. There is no live API and no runtime dependency on Python.

- Types: [`contract/types.ts`](contract/types.ts) (also copied into every bundle). Loader example: [`contract/loader.example.ts`](contract/loader.example.ts).
- **Sample bundle to develop against, committed: [`sample_bundle/`](sample_bundle/)** (12 varied companies picked to include every alert kind, 0.6 MB, real output on the synthetic data). Its `groups.json` lists only the sampled members of each group, so `n_companies` there is smaller than in a full bundle.
- **Inspector, to check and understand a bundle visually: [`inspector/`](inspector/README.md)** (`python product/score/inspector/serve.py --bundle <bundle_dir> --open`; re-checks the score arithmetic and the sha256 in the browser).
- **What is calculated and why, in plain language: [METHOD.md](METHOD.md).**
- Method and validation: [README.md](README.md), [validation.md](validation.md). Monitor, clusters and forecast: [`analysis/monitor/`](../../analysis/monitor/README.md). Not a predictor: show `manifest.disclaimer`.

## Flow

```text
CSV folder ──python -m product.score.export──▶ bundle/ ──copy or upload──▶ Next.js (build) ──▶ Vercel
```

```bash
PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> \
  python -m product.score.export --csv-folder <dir with the 8 CSVs> --out <bundle_dir> [--detail-months 12]
python -m product.score.export --validate <bundle_dir>      # also run automatically after export
```

About 50 s on the full data. Full bundle: 1,286 companies, 52 MB raw, about 6 MB gzipped (`--detail-months 6` is smaller). Bundles are immutable: a new run is a new bundle, and `manifest.generated_at` / `scorecard_version` say which. Do not commit the full bundle; the sample is enough for development. Ship the full one with the app build or host the folder as static files (Vercel Blob, S3) and set `BUNDLE_URL` as in the loader example. One company file is about 40 KB; the two indexes the feed and portfolio need (`companies.json`, `alerts.json`) are 0.6 and 2.7 MB raw.

**Storage:** ship the bundle plus the clean-only DuckDB (`python -m product.score.clean_db --work-dir <pipeline work dir> --out <file>`, schema `clean` only, 220 MB) and nothing else; the bundle explains scores and alerts, the DuckDB answers questions about the records behind them. See `AGENTS.md`.

The monitor's parameters (chart floors, clusters, funnel laws, ranking model) and the measured alert statistics are **fitted on train companies and committed** in `analysis/monitor/` (`monitor_params.json`, `monitor_stats.json`, `y7_rank_model.txt`, `forecast_params.json`). The export applies them, it never re-fits, so a run on new CSVs uses the same rules.

## Layout

```text
bundle/
├── manifest.json          read first: version, months, counts, section status, spec (labels/weights/units), monitor settings, disclaimer, sha256 of files
├── companies.json         list/portfolio: one row per company at its latest month + 24-month sparkline + cluster and alert summary (≈0.6 MB)
├── companies/{id}.json    one company: every scored month, control charts, cluster comparison, forecast, alert ids (≈40 KB each)
├── groups.json            250 groups: member ids, latest mean/min score, mean-score series, control charts and alert ids (groups of 3+)
├── alerts.json            the proactive feed, newest month first (≈2.7 MB; the detail window, 12 months)
├── clusters.json          behaviour clusters: label, description, size, quality
└── types.ts               the types below
```

Use `manifest.spec` for labels, weights, units and tooltips (`why`) instead of hard-coding them; it is generated from the same file that computes the score.

## What each field means

| Field | Meaning |
|---|---|
| `score` | 0–100, 100 = healthiest. Percentile-based, from trailing 3–6 month windows, as-of the month (no look-ahead). |
| `trajectory` | `improving` / `stable` / `dip` / `deteriorating` / `insufficient history`. Sustained drift (6-month trend held 3 months and confirmed by the 3-month slope) vs a short fall (`dip`). `dark` companies are `deteriorating` by rule. |
| `guard` | `dark`: no bank booking for 60 days, score capped at 30. `fading`: inflows collapsed vs own history, capped at 50. `score_pre_cap` is what it would be otherwise. **Show the cap:** a raw 90 on a silent company is the trap this exists for. |
| `confidence` + `confidence_note` | `high` / `medium` / `low` and why (no invoices, short history, dark…). Companies without invoices have no payment-history or mix category and score about 5 points higher on average from the reweighting: never rank them next to full-data companies without the badge. |
| `categories.*.score` / `.contribution` | Category 0–100 (null if not observable) and the points of the 100 it brings. New credit is thin: its effective weight is 5, not 10 (`spec`). |
| `items.*` | Per variable: `value` (unit in `spec`), `points` 0–100, `contribution`, `delta` since last month. `Σ contribution + guard_adjustment = score` (±0.15 rounding). `Σ delta + change_guard = score change` when the item set is unchanged. Only the last `detail_months` months carry items and reasons. |
| `reasons` | Up to 4, why the score is not higher, biggest points lost first, each with the € behind it (`eur`, null when the item has none) and a ready English `sentence`. `value`+`unit`+`eur`+`item` let you template other languages. |
| `change_reasons` | Up to 4 movers since last month, signed points. |
| `scores` (index) | Aligned with `months`; null where the company had no score yet (needs 3 months of trail). |

Rules: months are `YYYY-MM`; missing is `null`, never NaN or `""`; ignore unknown fields; a company file only lists scored months; money is in the company's own currency, not converted; `country` is often null.

## Monitor, clusters, forecast, alerts (plan steps 3–4, since 1.1.0)

`manifest.sections` states which sections exist; all are `available` except `counterparty_entities`, which is `blocked`.

**Control charts** (`CompanyDetail.control[]`, `groups[].control`). Charts watch *change*, not level: the series is compared with its own median over the 12 months that end 3 months earlier, scaled by a robust MAD (floor fitted on train), smoothed with an EWMA and a CUSUM. `signal` is the raw crossing per month; `persistent` is the 3-of-the-last-4-months rule; a chart starts once it has 7 scored months. Company charts: `own_history` for the score and for payment history, amounts owed and stability, and `cluster` for the score (the company's gap to the median of its behaviour cluster, a change of relative standing). Group charts (groups with 3+ scored members, `limits_available`): `group_own_history`, whose floor grows as the group shrinks, and `group_vs_groups`, the group's 3-month change against funnel limits (`center ± 3·sqrt(a + b/n)`, fitted on train groups) so small groups get wide limits. Groups of 1–2 companies: draw the mean only, no limits, no alerts.

**Clusters** (`clusters.json`, `CompanyDetail.cluster`). Behaviour clusters (volatility, months without inflows, payroll/tax/social-security presence, debt service, collection delay, concentration…), size signal regressed out, fitted on train, k=4. Structure is weak (silhouette 0.19): show them as *peer groups for a comparison*, not as segments. `vs_cluster` is the company's percentile and robust z inside its cluster's train company-months, for the latest month. Companies with fewer than 6 months of trail have no cluster. Membership is a whole-trail trait and never triggers an alert.

**Alerts** (`alerts.json`, `CompanyDetail.alert_ids`, `groups[].alert_ids`). An alert is the *onset* of a persistent and material move (the smoothed level at least 8 points from the baseline for the score, 10 for a category), so a company that stays low does not alert every month and a one-month dip is not an alert. Kinds:

| Kind | Fires when | Owner (default) | Severity |
|---|---|---|---|
| `score_deterioration` / `score_improvement` | own-history score chart, persistent and material, down / up (two-sided: improvements are `direction: "opportunity"`) | from the biggest mover: payables, cash → treasurer; receivables → collections; debt and fees → CFO | `act` ≥ 20 points from baseline, `watch` ≥ 12, else `info` |
| `category_drop` | a category chart, persistent and material, when no score alert covers it | as above | as above |
| `going_dark` | first month with no bank booking for 60 days | treasurer | always `act` |
| `top_customer_quiet` | last quarter's top customer got no invoice this month (a transparent rule, first month only) | collections | `act` = top decile of the ranking model, `info` = the customer billed in all 3 of the last 3 months, else `watch` |

Group alerts use the two score kinds with `entity.type = "group"`; their `reasons` are empty and `evidence.members_moving_most` names the members. Every alert carries `owner`, a concrete `action`, up to 4 `reasons` with the € behind them (score alerts: the items that moved most since the baseline month, signed points), `evidence`, and `persistence` (`3 of the last 4 months`). `rank_score` exists only for `top_customer_quiet` and is for ordering: it is not a probability, never show it as one.

`alerts.json` also carries `stats`, measured on train companies against the eight accepted outcomes; quote them next to the alerts (they are in `analysis/monitor/evaluation.md`): score-fall alerts are followed by an accepted outcome within 6 months about as often as an alert on a random month (`false_alarm_rate` ≈ `false_alarm_rate_at_chance`), so they mean "moved away from its own normal, here is why and the amount", not "will fail". The top-customer alert is the one with a measured lift (about 2×).

**Forecast** (`CompanyDetail.forecast`, extended in 1.2.0). A fan (median, 50% and 80% intervals) for 1–6 months from the last scored month. Since 1.2.0 it is the `reversion_quantile` fan: the score is not a trending series, so the fan is pulled toward the company's own average and the portfolio level and is skewed by level, not flat. Each point says which `method` made it (`naive_last` is the fallback for a horizon where the model did not beat it, or a company whose features are missing). `own_average` is the level it is pulled toward. `drivers` explains the 3-month median: `parts` (own average, portfolio level, recent move, baseline) add up to `expected_change` in points, each with a plain-language `text`; null when the naive fan was used. `skill_vs_naive` and `skill_by_horizon` are the cross-validated pinball-loss improvement over the naive fan (train companies): +3% at 1 month to +16% at 6, with 50% and 80% intervals calibrated out of fold; when the later months are also held out the gain is +1.5% to +12% and the 80% interval covers 74–78% (`manifest.monitor.forecast.time_split`). `manifest.monitor.forecast` carries the measured dynamics to quote next to the fan: no seasonality found, and after a material 3-month move the typical fall holds (28% recover at least half) while the average one partly recovers. Before 1.2.0 the fan was flat (naive last value, `skill_vs_naive` ≈ 0). Still a persistence fan: never say it predicts failure or an outcome. Absent for companies with under 4 scored months.

**Language:** alert text (`title`, `summary`, `action`, `persistence.rule`, `reasons[].label` and `.sentence`), the score reason sentences and the item and category labels are in **Spanish** (`manifest.language` = `"es"`; decimal comma, `64 %`, `14 k€`); ids and enums (`kind`, `severity`, `direction`, `owner`, `item`, `factor`) stay English codes. Wording the UI must keep: "el cliente principal ha dejado de facturar, revisa la exposición y los cobros" (top customer stopped billing, review exposure and collections), never "ingresos en riesgo" (revenue at risk) or a probability; the score is "explainable and monitorable", never "predicts"; TellMe mapping suggestion: `info` alerts Silent (logged), `watch`/`act` Guided (shown with reasons, the owner approves, edits or rejects).

| Section | Status |
|---|---|
| `counterparty_entities` | **Blocked**: counterparty IDs (`COUNTERPARTY_*`) do not map to company IDs, so customers and suppliers cannot be scored or watched as entities. Design for companies and groups only |

## Relation to the v0 dummy card and the Streamlit POC

`poc/` reads the v0 dummy card's parquet (`product/score/outputs/monthly_scores.parquet`), not this bundle. Rough mapping if the POC is moved onto the bundle: `score_3m` has no equivalent (the v1 score already uses 3–6 month windows, use `score`); `state` → `trajectory` (v1 adds `dip`); `confidence_band` → `confidence`; `reason_n_en` → `reasons[n].sentence` (English only, with `eur`); `cat_length_stability` → `categories.stability`; `no_invoices` → `confidence_note` starts with "no invoice payments in the window" (say if you want it as a boolean field); `going_dark` → `guard === "dark"`. Which scorecard is the product's is still to be decided.

## Versioning and checks

Additive change → `1.x.0`, breaking → `2.0.0`; the loader example refuses another major. 1.2.0 extended `forecast` (per-point `method`, `own_average`, `drivers`, `skill_by_horizon`, a better fan) and added `manifest.monitor.forecast`; `Forecast.method` gained `reversion_quantile`. 1.1.0 added `alerts.json`, `clusters.json`, `manifest.monitor`, the company/group control charts, cluster, forecast and alert ids, and made `Alert.owner` / `Alert.action` non-null; it did not change anything a 1.0.0 consumer read. `export --validate` checks file counts, score range, contribution and attribution sums, at most 4 reasons, alert ids unique and linked both ways, owner/action present, control-chart array lengths, forecast fans ordered inside 0–100, no planned sections, and the sha256 recorded in `manifest.files`.
