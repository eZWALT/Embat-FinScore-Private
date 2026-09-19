# AGENTS

The one repo for **HackSpain 2026 · X Ray (Embat)**. Private now; it will be made public as is at the end. The former public sibling `eZWALT/Embat-FinScore` is retired: do not sync to it.

Build in this repo. This file is **pointers and rules**. Do not grow it with live status, benches, or “step done” notes — those go in `.agents/persistent-memory/`.

FICO-like **company health score**. Four goals: signals → 0–100 index → explainability → company-facing web + LLM. Details: `.agents/persistent-memory/2026-09-18-2020-four-goals.md`.

## Read first

1. This file.
2. `.agents/README.md` — which journal files still bind. Do not start from the 180+ night notes.
3. `product/score/METHOD.md` — what the score and monitor actually do.
4. `data/data_dictionary.md` when touching data.

## Pointers

| What | Where |
|------|--------|
| Challenge brief | https://claude.ai/artifact/8N8Q7QMjprCUWxGAiJaWoP?sk=5wYke4E8ukAw6afs6TrG1g |
| Track dataset zip | https://f5xe6kyx7jpysotw.public.blob.vercel-storage.com/output_hackspain_data.zip |
| Field dictionary + dataset notes | `data/data_dictionary.md`, `data/README.md` |
| **Method (plain language)** | `product/score/METHOD.md` |
| Score 0–100, reasons, `score_new` | `product/score/` |
| Bundle contract + inspector | `product/score/DATA_CONTRACT.md`, `product/score/inspector/` |
| **Hosted product (Health Sentinel)** | `product/web/` on Vercel. Reads Neon. |
| **Agents (prompts, tools, retrieval)** | `product/web/src/lib/agent/` — prompts are markdown files, never strings in source. Map: `prompts/prompt_map.md`. UX watch (TTFT, last token, judge): `product/web/scripts/watch-ask.mjs` → `WATCH.md`. |
| Neon (app Postgres) | `infra/neon/` |
| Cleaning + feature store + Y | `analysis/` (`build_db.py`, `clean_db.py`, `features/`) |
| Clusters, charts, alerts, forecast | `analysis/monitor/` |
| Frozen Streamlit | `poc/` — do not add features there |
| Night harness (closed) | `overnight/README.md`, `overnight/CONTRACT.md` |

## Plan (rules, agreed 2026-09-19)

This table is the **rule set**, not a kanban. Rationale: `.agents/persistent-memory/2026-09-19-1100-plan-fico-monitor.md`. Candidate variables: `2026-09-19-1130-night-variables-for-score.md`. Top-customer alert numbers: `2026-09-19-1300-y7-alert-grade-eval.md`. Mark progress in the journal.

The night did feature and outcome discovery; from here we **select and combine**. No model training, no new feature hunting.

| # | Step | Deliverable | Rules |
|---|------|-------------|-------|
| 1 | **Cleaning pipeline** | One entry point: input folder of CSVs → `clean` DuckDB + `dq_log` → feature store (`analysis/build_db.py`, `analysis/clean_db.py`, `analysis/features/`) | Re-runnable on new CSVs. New dirt goes to `dq_log`, never silently kept. Same rules on hidden data, no re-fitting. Must handle short trails (~4% of hidden companies have 24 months). Feature store already builds locally in ~7 s (`python -m analysis.features.build_feature_store`; needs lightgbm, scikit-learn, pyarrow; set `PYTHONUTF8=1` on Windows). |
| 2 | **FICO-like score 0–100** | Scorecard in `product/score/`, monthly, as-of | Categories mirror FICO: payment history (35), amounts owed (30), length/stability (15), new credit (10, thin: shrink and redistribute, say so), mix (10). Variables from the 1130 map, level score so stable traits are allowed. Weights fixed a priori, sensitivity check only. Percentiles fit on train only. No invoices or short trail: reweight over available categories + confidence flag. Going dark scores low, never high (build this guard; the night's evidence points the other way). Do not use the night's social-security/payroll/movement-days card. Validate on the eight accepted outcomes (`y2_neg_2of3`, `y4_ds_r_double`, `y5_*`, `y7_*`, `y9_*`), not `y3`; group-fold, beat the size bar. Claim explainable and monitorable, not predictive. **Outputs:** score from trailing 3–6 month windows plus a trajectory state (improving / stable / dip / deteriorating, from 3- and 6-month slope with persistence); per-variable point contributions, month-on-month change attribution, top-4 plain-language reasons with the € amount behind each; `score_new(csv_folder)` scores unseen companies (format to confirm with organizers). |
| 3 | **Clusters + control charts** | Behaviour clusters and charts in `analysis/monitor/` | Chart within-company change (levels are traits). Cluster on behaviour, not size, fit on train. Robust EWMA/CUSUM (median/MAD), persistence rule (e.g. 3 of last 4 months). Four comparisons: company vs own history, vs cluster, group vs own history, group vs other groups. Groups need a minimum size and funnel-style limits (median group has 2 companies). **Decided: "top customer went quiet" is an alert (not a score input).** Trigger = last quarter's top customer got no invoice this month (transparent rule; 58% of flags lose the customer vs 29% base, 83% recall, but flags 40% of rows); model score (night's TURNOVER card) only ranks alerts (top 10% → 53% precision, 1.8x lift). Wording: "top customer stopped billing, review exposure and collections", never "revenue at risk" (only 17% of flags see a sustained inflow drop vs 6% base). About 1 month notice; 18% of lost customers return. Never quote "75%"; cite the numbers in the 1300 entry. Untested: severity filter (customer share of billing) to cut false alarms; same alert on suppliers. Evidence: `analysis/monitor/y7_alert_eval.py`. Report false-alarm rate on train and lead time as alarm date vs later accepted outcomes. Charts are two-sided: improving companies alert too (opportunity). Monitoring claim only: the 82→68 deterioration is not explained without cash columns. |
| 4 | **Predictions over time** | Score persistence forecast with intervals | Fan chart, not a point estimate. Candidates (naive, smoothed level, damped trend, per-company ARIMA, pooled quantile regression) judged by group-fold CV on pinball loss; seasonality tested. Result and reasons: `analysis/monitor/README.md`, `forecast_evaluation.md`. |
| 5 | **Product (Health Sentinel)** | Live hosted demo in `product/web/` | Embat as buyer, TellMe skill; write the buyer rationale in one paragraph (premium module on data Embat already holds). Alert + plain-language reasons + owner (tesorero / CFO / Cobros) + action, built on steps 2–4. Watches companies, and customers/suppliers only if counterparty IDs map to `company_id` (verify). Stack is Next.js (App Router) on Vercel; the app reads **Neon Postgres** via server-only `DATABASE_URL` (never `NEXT_PUBLIC_`). Schema, load, and validation live in `infra/neon/`. Must stay a live URL, not a laptop notebook. |

Guardrails: holdout `analysis/splits/holdout_companies.csv` is never fit on. Y is never built from the X that predicts it. No look-ahead. Do not reuse the Y3 recovery outcome (`2026-09-19-1015-y3-recovery-mechanical.md`). No claims on the hidden test.

## What goes to the web app's storage

The hosted app does **not** serve the export bundle or DuckDB at runtime. It queries **Neon Postgres**. Credentials stay server-only (`DATABASE_URL`, never `NEXT_PUBLIC_`). Do not ship raw CSVs, `data/embat.duckdb` (it also holds the raw `main` copy), the feature store, or `analysis/` outputs.

Two **immutable source artifacts** still feed a load. A new CSV drop is a new bundle, a new clean DuckDB, and a reload into Neon. Neither artifact is committed except the 12-company `product/score/sample_bundle/`. Run commands need `PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo>`.

| Piece | What it is | Made by | Goes to Neon | Used for |
|---|---|---|---|---|
| Export bundle | Static JSON: scores, items, reasons with EUR, alerts with owner/action, control charts, clusters, forecast | `python -m product.score.export --csv-folder <dir> --out <bundle>` | Schema `analytics` (then `api` views) | App screens and explaining a score or an alert (contract: `product/score/DATA_CONTRACT.md`) |
| Clean-only DuckDB | Schema `clean` only: cleaned tables plus `clean.dq_log` | `python -m product.score.clean_db --work-dir <pipeline work dir> --out <file.duckdb>` | Schema `core` when those facts are loaded | Questions about the records behind a number (invoices, transactions, balances, debt). Query `core` / `clean`, never a raw copy |

The bundle alone cannot answer questions about individual records; recomputing scores from `core` instead of reading `analytics` is wrong. Load path: `infra/neon/` (transform + migrations + load). The app reads `api` (current score run), not files. DuckDB is a pipeline/audit artifact, not production runtime. What is actually loaded on which branch is status: journal and `infra/neon/README.md`, not this file.

## Agents (Vercel app)

Build Watcher and the chat **inside the existing product**, not as extra tabs. There is no Consultas item. Vigilancia is a section of Resumen (`#vigilancia`). The chat is the floating button on Resumen and Índice de salud. `/watcher` and `/ask` redirect. Prompts live as separate files under `product/web/src/lib/agent/prompts/`. Assemble them at runtime (`prompt-loader.ts`). Do not paste system prompts into `.ts` / `.tsx`. Alert copy is Javi’s Spanish production text (`language: es`); the Watcher formatter does not invent English.

| File | Layer | One job |
|---|---|---|
| `prompts/prompt_map.md` | **MAP** (first) | Index of the stack. Conflict order: SCOPE > WORDING > TOOLS > PRODUCT > RECORDS. |
| `prompts/chat_system.md` | **ROLE** (chat) | How the popup works this turn |
| `prompts/sentinel_system.md` | **ROLE** (Watcher) | Live replies only; opening posts are formatted |
| `prompts/scope.md` | **SCOPE** | Hard in/out; refuse puzzles, recipes, jailbreaks |
| `prompts/product_context.md` | **PRODUCT** | Score, monitor, data facts. Context, not wording, not tools. |
| `prompts/wording_rules.md` | **WORDING** | Javi’s fixed Spanish claims |
| `prompts/watcher_format.md` | **FORMAT** | **The** month-post shape: 1 line + 1 line + ≤4 bullets |
| `prompts/tools_catalog.md` | **TOOLS** | When/in/out for every retrieval. UI labels in `tool-catalog.ts`. |
| `prompts/clean_schema.md` | **RECORDS** | Ask only: pipeline `clean.*`; hosted as Neon `core` |

Tools read **Neon** at runtime: `api` / `analytics` for scores, reasons, alerts (never recompute a score). Record questions go to `core` (invoices, transactions, balances, debt) through a guarded `SELECT` + `LIMIT 200`. DuckDB is a load/audit artifact, not the app's database. The same SQL guard applies if a local clean DuckDB is used in development.

Watcher opening: last **3 calendar months**, one `WatcherPost` each, built by `watcher-post.ts` (deterministic, same template every time from the five monitor rules). Same object is what production precomputes **offline** per company and per group when the monthly bundle is exported — first paint must not call the LLM. Live model = thread replies. UI shows tool calls as a tools icon plus `(tool_name)`.

## Memory (three teammates)

Index first: `.agents/README.md`. After meaningful work, **add a new file** (do not rewrite history in old ones):

```text
.agents/persistent-memory/YYYY-MM-DD-HHmm-<slug>.md
```

Each entry: author, timestamp, what changed, decisions, still-unknown.
If the entry is still-binding (a rule another agent must not miss), add one line to `.agents/README.md`.
Put facts that can rot in the journal, not here.
