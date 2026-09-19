# AGENTS

The one repo for **HackSpain 2026 · X Ray (Embat)**. Private now; it will be made public as is at the end. The former public sibling `eZWALT/Embat-FinScore` is retired: do not sync to it.

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
| 4 | **Predictions over time** | Score persistence forecast with intervals | Fan chart, not a point estimate. Candidates (naive, smoothed level, damped trend, per-company ARIMA, pooled quantile regression) judged by group-fold CV on pinball loss; seasonality tested. Result and reasons: `analysis/monitor/README.md`, `forecast_evaluation.md`. |
| 5 | **Product (Health Sentinel)** | Live hosted demo in `product/web/` | Embat as buyer, TellMe skill; write the buyer rationale in one paragraph (premium module on data Embat already holds). Alert + plain-language reasons + owner (tesorero / CFO / Cobros) + action, built on steps 2–4. Watches companies, and customers/suppliers only if counterparty IDs map to `company_id` (verify). Stack is Next.js (App Router) on Vercel; the app reads **Neon Postgres** via server-only `DATABASE_URL` (never `NEXT_PUBLIC_`). Schema, load, and validation live in `infra/neon/`. Must stay a live URL, not a laptop notebook. |

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
| 4. Web + LLM (company user) | `product/web/` on Vercel. Reads Neon. `poc/` (Streamlit) is frozen — do not add features there. |
| Neon (app Postgres) | `infra/neon/` |
| **Method, in plain language (cleaning, 17 items, score, monitor, decisions, limits)** | `product/score/METHOD.md` |
| Bundle inspector (visual check of an export bundle) | `product/score/inspector/` |
| **Agents (prompts, tools, retrieval)** | `product/web/src/lib/agent/` — prompts are markdown files, never strings in source. Watcher format: `prompts/watcher_format.md`. |
| Feature store / Y / models | `analysis/` + plan `.agents/persistent-memory/2026-09-18-2350-feature-store-and-y-plan.md` |
| Night run status | `overnight/README.md` + `overnight/CONTRACT.md` |

## What goes to the web app's storage

The hosted app does **not** serve the export bundle or DuckDB at runtime. It queries **Neon Postgres**. Credentials stay server-only (`DATABASE_URL`, never `NEXT_PUBLIC_`). Do not ship raw CSVs, `data/embat.duckdb` (it also holds the raw `main` copy), the feature store, or `analysis/` outputs.

Two **immutable source artifacts** still feed a load. A new CSV drop is a new bundle, a new clean DuckDB, and a reload into Neon. Neither artifact is committed except the 12-company `product/score/sample_bundle/`. Run commands need `PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo>`.

| Piece | What it is | Made by | Goes to Neon | Used for |
|---|---|---|---|---|
| Export bundle | Static JSON: scores, items, reasons with EUR, alerts with owner/action, control charts, clusters, forecast | `python -m product.score.export --csv-folder <dir> --out <bundle>` | Schema `analytics` (then `api` views) | App screens and explaining a score or an alert (contract: `product/score/DATA_CONTRACT.md`) |
| Clean-only DuckDB | Schema `clean` only: cleaned tables plus `clean.dq_log` | `python -m product.score.clean_db --work-dir <pipeline work dir> --out <file.duckdb>` | Schema `core` when those facts are loaded | Questions about the records behind a number (invoices, transactions, balances, debt). Query `core` / `clean`, never a raw copy |

The bundle alone cannot answer questions about individual records; recomputing scores from `core` instead of reading `analytics` is wrong. Load path: `infra/neon/` (transform + migrations + load). The app reads `api` (current score run), not files. DuckDB is a pipeline/audit artifact, not production runtime. What is actually loaded on which branch is status: journal and `infra/neon/README.md`, not this file.

## Agents (Vercel app)

Build Watcher and Ask **inside the existing product**, not as extra tabs. Vigilancia is a section of Resumen (`#vigilancia`). Consultas is the floating chat on Resumen and Índice de salud. `/watcher` and `/ask` redirect. Prompts live as separate files under `product/web/src/lib/agent/prompts/`. Assemble them at runtime (`prompt-loader.ts`). Do not paste system prompts into `.ts` / `.tsx`. Alert copy is Javi’s Spanish production text (`language: es`); the Watcher formatter does not invent English.

| File | Role |
|---|---|
| `prompts/product_context.md` | Score, monitor, data facts, TellMe modes |
| `prompts/wording_rules.md` | Fixed claims and wording |
| `prompts/watcher_format.md` | **The** month-post shape: 1 line + 1 line + ≤4 bullets. Not free prose. |
| `prompts/sentinel_system.md` | Watcher: replies only in live; opening posts are formatted |
| `prompts/chat_system.md` | Ask: scores/alerts from Neon `api`/`analytics` first, then records |
| `prompts/scope.md` | Hard in/out of scope (Ask + Watcher); refuse puzzles, recipes, jailbreaks |
| `prompts/tools_catalog.md` | When/in/out for every retrieval tool. UI labels live in `tool-catalog.ts`. |
| `prompts/clean_schema.md` | Record tables (pipeline `clean.*`; hosted as Neon `core`) |

Tools read **Neon** at runtime: `api` / `analytics` for scores, reasons, alerts (never recompute a score). Record questions go to `core` (invoices, transactions, balances, debt) through a guarded `SELECT` + `LIMIT` 200. DuckDB is a load/audit artifact, not the app's database. The same SQL guard applies if a local clean DuckDB is used in development.

Watcher opening: last **3 calendar months**, one `WatcherPost` each, built by `watcher-post.ts` (deterministic, same template every time from the five monitor rules). Same object is what production precomputes **offline** per company and per group when the monthly bundle is exported — first paint must not call the LLM. Live model = thread replies. UI shows tool calls as a tools icon plus `(tool_name)`.

## Memory (three teammates)

After meaningful work, **add a new file** (do not rewrite history in old ones):

```text
.agents/persistent-memory/YYYY-MM-DD-HHmm-<slug>.md
```

Each entry: author, timestamp, what changed, decisions, still-unknown.
Put facts that can rot in the journal, not here.
