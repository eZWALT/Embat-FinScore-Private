# Health Score data contract (schema 1.0.0)

For whoever builds the web app (Next.js on Vercel). The score pipeline runs **once** on a folder of CSVs and writes a **static JSON bundle**; the app reads the bundle. There is no live API and no runtime dependency on Python.

- Types: [`contract/types.ts`](contract/types.ts) (also copied into every bundle). Loader example: [`contract/loader.example.ts`](contract/loader.example.ts).
- **Sample bundle to develop against, committed: [`sample_bundle/`](sample_bundle/)** (12 varied companies, 0.4 MB, real output of the current scorecard on the synthetic data).
- Method and validation: [README.md](README.md), [validation.md](validation.md). Not a predictor: show `manifest.disclaimer`.

## Flow

```text
CSV folder ──python -m product.score.export──▶ bundle/ ──copy or upload──▶ Next.js (build) ──▶ Vercel
```

```bash
PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> \
  python -m product.score.export --csv-folder <dir with the 8 CSVs> --out <bundle_dir> [--detail-months 12]
python -m product.score.export --validate <bundle_dir>      # also run automatically after export
```

About 45 s on the full data. Full bundle: 1,286 companies, 41.6 MB raw, about 4 MB gzipped (`--detail-months 6` is smaller). Bundles are immutable: a new run is a new bundle, and `manifest.generated_at` / `scorecard_version` say which. Do not commit the full bundle; the sample is enough for development. Ship the full one with the app build (copy into `data/bundle`) or, if it is too big for the deploy limits of your Vercel plan (check them), host the folder as static files (Vercel Blob, S3) and set `BUNDLE_URL` as in the loader example.

## Layout

```text
bundle/
├── manifest.json          read first: version, months, counts, section status, spec (labels/weights/units), disclaimer, sha256 of files
├── companies.json         list/portfolio: one row per company at its latest month + 24-month sparkline (≈0.5 MB)
├── companies/{id}.json    one company: every scored month (≈31 KB each)
├── groups.json            250 groups: member ids, latest mean/min score, mean-score series
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

## What the web app will grow into (planned sections)

`manifest.sections` states which sections exist. Everything below has its TypeScript shape in `types.ts` already, so screens can be designed now; the files appear when the plan step is done, without changing what exists.

| Section | Plan step | Where it lands | Product surface (Health Sentinel) |
|---|---|---|---|
| `control_charts` | 3 | `CompanyDetail.control[]`, later a group equivalent | Robust EWMA/CUSUM on within-company change; four comparisons (own history, cluster, group vs own history, group vs other groups); `persistent` = the 3-of-4 rule. Two-sided: improvements alert too |
| `clusters` | 3 | `clusters.json`, `CompanyDetail.cluster` | Behaviour clusters (not size); "vs cluster" percentile on the company page |
| `alerts` | 3, 5 | `alerts.json` (`AlertFeed`), `CompanyDetail.alert_ids` | The proactive feed. Kinds: score deterioration/improvement, category drop, going dark, **top customer quiet** (decided). `stats` carries the measured false-alarm rate and lead time to quote next to alerts |
| `owners_actions` | 5 | `Alert.owner` (treasurer / CFO / collections), `Alert.action` | Who gets the alert and the concrete recommendation |
| `forecast` | 4 | `CompanyDetail.forecast` | Fan chart with the naive last value as baseline. Expect a tie; it is the first step to be cut |
| `counterparty_entities` | 5 | `EntityType` gets `customer` / `supplier` | **Blocked**: counterparty IDs (`COUNTERPARTY_*`) do not map to company IDs, so customers and suppliers cannot be scored as entities. Design for companies and groups only for now |

Wording constraints the UI must keep: "top customer stopped billing, review exposure and collections", never "revenue at risk" or a probability; the score is "explainable and monitorable", never "predicts"; groups have a median of 2 companies, so no limits or clusters for tiny groups.

## Relation to the v0 dummy card and the Streamlit POC

`poc/` reads the v0 dummy card's parquet (`product/score/outputs/monthly_scores.parquet`), not this bundle. Rough mapping if the POC is moved onto the bundle: `score_3m` has no equivalent (the v1 score already uses 3–6 month windows, use `score`); `state` → `trajectory` (v1 adds `dip`); `confidence_band` → `confidence`; `reason_n_en` → `reasons[n].sentence` (English only, with `eur`); `cat_length_stability` → `categories.stability`; `no_invoices` → `confidence_note` starts with "no invoice payments in the window" (say if you want it as a boolean field); `going_dark` → `guard === "dark"`. Which scorecard is the product's is still to be decided.

## Versioning and checks

Additive change → `1.x.0`, breaking → `2.0.0`; the loader example refuses another major. `export --validate` checks file counts, score range, contribution and attribution sums, at most 4 reasons, and the sha256 recorded in `manifest.files`.
