# 2026-09-19-1250 — product definition: three parts, two of them agentic

- **Author:** Walter (with Cursor agent)
- **When:** 2026-09-19 ~12:50 CEST, after pulling `26e6037`

## Decision (Walter)

The product is three things on one bundle:

1. **Group Health Map** — analytics: heatmaps and dashboards over groups and companies (Ruben, `product/web/`, Next.js on Vercel).
2. **Sentinel watcher** (agentic, push) — you register a set of companies; it informs you of score/health, alerts, and suspicious feature drops, above all "top customer went quiet". Consumes `alerts.json` + control charts; owner and action already routed by `analysis/monitor/routing.py`.
3. **Chat with tools** (agentic, pull) — a company or group user asks questions about its own data and gets the *why*. Tools read the export bundle (scores, reasons, alerts, charts) and the clean-only DuckDB (`clean` schema, read-only) for record-level questions. The system prompt must carry very rich context (method, spec labels, disclaimer, wording rules). Never recomputes a score.

Parts 2 and 3 are iterated in `poc/` (Streamlit) before moving to `product/`.

## What the repo already gives each part (as of `26e6037`)

- Bundle schema 1.1.0 (`product/score/DATA_CONTRACT.md`): `companies.json` (index + sparkline + cluster + alert summary), `companies/{id}.json` (months with items/reasons/attribution, control charts, cluster, forecast), `groups.json` (mean/min, series, charts for 3+ members), `alerts.json` (5 kinds, owner, action, reasons with €, `stats`), `clusters.json`, `manifest.spec` (labels/weights/units/why) and `manifest.disclaimer`.
- Inspector (`product/score/inspector/`) to audit a bundle visually; port 8765 (same as my earlier POC test port; pick another for Streamlit).
- Ruben's web reads `sample_bundle/` through `ScoreRepository`; `product/web/src/lib/` is now tracked (was swallowed by the `lib/` ignore rule).

## Evidence discipline for the two agents

- Defensible as "early": top customer quiet (onset 56% vs 29% base, ~1 month notice), going dark (rule), and level persistence of runway/score (autocorrelation 0.84; ρ 0.85 at 3 months). Not defensible: "the score predicts" (AUROC 0.48–0.55, `validation.md`), score-fall alerts (lift 0.7–1.2).
- Fixed wording: "top customer stopped billing, review exposure and collections"; never "revenue at risk", never a probability; `rank_score` is for ordering only.

## Still unknown

- Model and framework for the agents; which card is the product's (bundle exports v1 only; POC still on v0 parquet).
- Whether the watcher is scheduled (new bundle per run) or on-demand over the feed.
- Full bundle hosting for Vercel; hidden-set submission format.
