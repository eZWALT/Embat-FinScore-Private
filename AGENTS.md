# AGENTS

Private working repo for **HackSpain 2026 · X Ray (Embat)**.
Public face (late sync only): `../Embat-FinScore`.

Build in this repo. Do not grow this file with live status.

FICO-like **company health score**. Four goals: signals → 0–100 index → explainability → company-facing web + LLM. Details: `.agents/persistent-memory/2026-09-18-2020-four-goals.md`.

## Plan (agreed 2026-09-19; supersedes the overnight "no 0–100 formula / no `product/`" rule)

Do the steps in order. Rationale and caveats: `.agents/persistent-memory/2026-09-19-1100-plan-fico-monitor.md` (revised). Candidate variables: `2026-09-19-1130-night-variables-for-score.md`. Top-customer alert numbers: `2026-09-19-1300-y7-alert-grade-eval.md`. Mark a step done in the journal, not here.

The night did feature and outcome discovery; from here we **select and combine**. No model training, no new feature hunting.

| # | Step | Deliverable | Rules |
|---|------|-------------|-------|
| 1 | **Cleaning pipeline** | One entry point: input folder of CSVs → `clean` DuckDB + `dq_log` → feature store (`analysis/build_db.py`, `analysis/clean_db.py`, `analysis/features/`) | Re-runnable on new CSVs. New dirt goes to `dq_log`, never silently kept. Same rules on hidden data, no re-fitting. Must handle short trails (~4% of hidden companies have 24 months). Feature store already builds locally in ~7 s (`python -m analysis.features.build_feature_store`; needs lightgbm, scikit-learn, pyarrow; set `PYTHONUTF8=1` on Windows). |
| 2 | **FICO-like score 0–100** | Scorecard in `product/score/`, monthly, as-of | Categories mirror FICO: payment history (35), amounts owed (30), length/stability (15), new credit (10, thin: shrink and redistribute, say so), mix (10). Variables from the 1130 map, level score so stable traits are allowed. Weights fixed a priori, sensitivity check only. Percentiles fit on train only. No invoices or short trail: reweight over available categories + confidence flag. Going dark scores low, never high (build this guard; the night's evidence points the other way). Do not use the night's social-security/payroll/movement-days card. Validate on the eight accepted outcomes (`y2_neg_2of3`, `y4_ds_r_double`, `y5_*`, `y7_*`, `y9_*`), not `y3`; group-fold, beat the size bar. Claim explainable and monitorable, not predictive. **Outputs:** score from trailing 3–6 month windows plus a trajectory state (improving / stable / dip / deteriorating, from 3- and 6-month slope with persistence); per-variable point contributions, month-on-month change attribution, top-4 plain-language reasons with the € amount behind each; `score_new(csv_folder)` scores unseen companies (format to confirm with organizers). |
| 3 | **Clusters + control charts** | Behaviour clusters and charts in `analysis/monitor/` | Chart within-company change (levels are traits). Cluster on behaviour, not size, fit on train. Robust EWMA/CUSUM (median/MAD), persistence rule (e.g. 3 of last 4 months). Four comparisons: company vs own history, vs cluster, group vs own history, group vs other groups. Groups need a minimum size and funnel-style limits (median group has 2 companies). **Decided: "top customer went quiet" is an alert (not a score input).** Trigger = last quarter's top customer got no invoice this month (transparent rule; 58% of flags lose the customer vs 29% base, 83% recall, but flags 40% of rows); model score (night's TURNOVER card) only ranks alerts (top 10% → 53% precision, 1.8x lift). Wording: "top customer stopped billing, review exposure and collections", never "revenue at risk" (only 17% of flags see a sustained inflow drop vs 6% base). About 1 month notice; 18% of lost customers return. Never quote "75%"; cite the numbers in the 1300 entry. Untested: severity filter (customer share of billing) to cut false alarms; same alert on suppliers. Evidence: `analysis/monitor/y7_alert_eval.py`. Report false-alarm rate on train and lead time as alarm date vs later accepted outcomes. Charts are two-sided: improving companies alert too (opportunity). Monitoring claim only: the 82→68 deterioration is not explained without cash columns. |
| 4 | **Predictions over time** | Score persistence forecast with intervals | Smoothed level vs naive last value, group-fold CV; expect a tie (night: cash best forecast by last value). Fan chart, not a point estimate. First step to cut if time runs out. |
| 5 | **Product (Health Sentinel = idea 01 in `product/web/index.html`)** | Live hosted demo in `product/web/` | Embat as buyer, TellMe skill; write the buyer rationale in one paragraph (premium module on data Embat already holds). Alert + plain-language reasons + owner (tesorero / CFO / Cobros) + action, built on steps 2–4. Watches companies, and customers/suppliers only if counterparty IDs map to `company_id` (verify). Must be a live URL, not a laptop notebook: Dockerfile is empty, pick the stack when this step starts and deploy an early version. |

Guardrails: holdout `analysis/splits/holdout_companies.csv` is never fit on. Y is never built from the X that predicts it. No look-ahead. Do not reuse the Y3 recovery outcome (`2026-09-19-1015-y3-recovery-mechanical.md`). No claims on the hidden test.

## Read first

1. This file (pointers only).
2. `.agents/persistent-memory/` — journal. Start with `2026-09-18-initial-context.md`, then newer dated files.
3. `data/data_dictionary.md` when touching data.

## Pointers

| What | Where |
|------|--------|
| Challenge brief | https://claude.ai/artifact/8N8Q7QMjprCUWxGAiJaWoP?sk=5wYke4E8ukAw6afs6TrG1g |
| Track dataset zip | https://f5xe6kyx7jpysotw.public.blob.vercel-storage.com/output_hackspain_data.zip |
| Field dictionary | `data/data_dictionary.md` |
| Dataset notes | `data/README.md` |
| 1. Signals | `analysis/` |
| 2–3. Score 0–100 + explain | `product/score/` (v0 dummy card: `PYTHONPATH=. python -m product.score`) |
| 4. Web + LLM (company user) | `product/web/` (hosted demo still empty) |
| Bundle inspector (visual check of an export bundle) | `product/score/inspector/` |
| Streamlit POC | `poc/` (Sentinel + Portfolio; binds the v0 score when the parquet exists) |
| App folder | `product/README.md` — Docker/runtime still empty |
| POC (Streamlit, iterate before `product/`) | `poc/README.md` |
| Feature store / Y / models | `analysis/` + plan `.agents/persistent-memory/2026-09-18-2350-feature-store-and-y-plan.md` |
| Night run status | `overnight/README.md` + `overnight/CONTRACT.md` |
| Public sibling | `../Embat-FinScore` (GitHub: `eZWALT/Embat-FinScore`) |

## What goes to the web app's storage

Port **only two things** to the storage of the web app (and of any agent that explains the data): the **export bundle** and the **clean-only DuckDB**. Nothing else: not the raw CSVs (647 MB), not `data/embat.duckdb` (420 MB, holds the raw `main` copy too), not the feature store, not the repo's `analysis/` outputs.

| Piece | What it is | Size (full data) | Made by | Used for |
|---|---|---|---|---|
| Export bundle | Static JSON: scores, items, reasons with EUR, alerts with owner/action, control charts, clusters, forecast | 53 MB raw, ~6 MB gzipped | `python -m product.score.export --csv-folder <dir> --out <bundle>` | The app screens, and explaining a score or an alert (contract: `product/score/DATA_CONTRACT.md`) |
| Clean-only DuckDB | Schema `clean` only: cleaned tables plus `clean.dq_log` | 220 MB, ~104 MB gzipped | `python -m product.score.clean_db --work-dir <pipeline work dir> --out <file.duckdb>` | Questions about the records behind a number (invoices, transactions, balances, debt). Open it read-only. Query `clean`, never a raw copy |

Together about 273 MB. The bundle alone cannot answer questions about individual records; the DuckDB alone would make an agent recompute scores instead of reading their explanations. Both are immutable per run: a new CSV drop is a new bundle and a new file. Neither is committed except the 12-company `product/score/sample_bundle/`. Run commands need `PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo>`.

## Memory (three teammates)

After meaningful work, **add a new file** (do not rewrite history in old ones):

```text
.agents/persistent-memory/YYYY-MM-DD-HHmm-<slug>.md
```

Each entry: author, timestamp, what changed, decisions, still-unknown.
Put facts that can rot in the journal, not here.
