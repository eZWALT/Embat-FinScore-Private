# 2026-09-19-1100 — product verdict (ideas 01 + 02) and Streamlit POC shell

- **Author:** Walter (with Cursor agent)
- **When:** 2026-09-19 ~11:00 CEST

## Decisions

- **Analysis stops here.** Night + morning showed no hidden signal to squeeze; the day goes to selecting/combining (Javi, FICO score in `product/score/`) and to product (Walter + Ruben).
- **Product = Ruben's idea 01 + 02, merged.** 01 (Health Score Sentinel) is the engine and the alerts; 02 (Group Health Map) is the entry screen. 03 (Debt Opportunity Advisor) is not a product: at most one conditional action card when a company has an improving trajectory and a known rate.
- **Split:** Ruben owns deployment (Vercel + DB) and the production `product/`. Walter iterates the assistant and the dashboard in Streamlit in a new top-level `poc/`, then hands over. Javi owns the score.
- **Assistant behaviour is not defined.** The POC ships the interface only; no tools, prompts or model assumed.

## Facts checked today (change the plan's assumptions)

- **Groups are big.** 71 groups are singletons, but 694 of 1,286 companies (54%) sit in groups of 9+, 76% in groups of 5+. The plan's "median group has 2" understates: group charts and a portfolio view cover most companies.
- **Counterparty IDs never map to companies.** Invoice `counterparty_id`s are `COUNTERPARTY_xxxxx` (124k distinct), no `COMPANY_` values. Customers and suppliers can be shown as exposures (open €, days late, went quiet) but never scored. Trim idea 01's "watches each company, customer and supplier" accordingly.
- **Idea 03 has ~no data.** 378 companies hold a debt product, but a rate exists only in `debt_schedule_config`: 87 products across 40 companies. Utilisation is a 1.6% snapshot. No balance path. Any refinancing-savings number would be fiction.

## What was built

`poc/` — Streamlit, run `python3 -m streamlit run poc/app.py` from the repo root (the `streamlit` shim on PATH points at another interpreter on this machine).

- `Overview`: what the product is, what is real vs pending.
- `Sentinel`: chat interface with a fixed placeholder reply; session-only history.
- `Portfolio`: group → companies (latest runway, 3-month change, inflows, from `data/feature_store/monthly.parquet`) → company charts (runway, inflows/outflows). No score column yet; sorted by runway.

Verified in the browser: all three pages render, chat round-trip works, portfolio shows GROUP_0142 (22 companies).

## Still unknown

- Contract between Javi's score output and the product (one row per company-month: score, 5 subscores, trajectory state, confidence, reasons with €, alerts with owner/action). Needs writing today so product does not wait on analysis.
- Group aggregation rule for a group-level score; fallback for the 71 singleton groups.
- Assistant behaviour, tools and model.
