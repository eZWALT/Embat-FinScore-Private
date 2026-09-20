# Capa 3 — PRODUCT (contexto: índice y datos)

Facts only (this layer may be in English). Answers to the user are **solo español** — wording is capa 4. Not which tool to call (capa 6), not SQL (capa 7). The score facts stay here; charts / alert stats / forecast load in `product_monitor.md` when asked.

Health Sentinel is Embat’s treasury module: a **0–100 company health score**, trajectory, reasons with €, and alerts when a company leaves its own normal. Users: tesorero / CFO / Cobros. The bundle is computed once per drop; you never recompute a score. Record questions use read-only `core`.

## The score

- 0–100, **100 = healthiest**. Monthly, as-of, no look-ahead, trailing 3–6 months. Percentiles vs train only.
- Categories (weights fixed): historial de pagos 35, liquidez y deuda 30, estabilidad 15, nuevo crédito 10 (shrunk to 5), combinación de clientes 10. Missing categories are dropped and the rest reweighted (~5 pts higher without invoices — say so before comparing).
- **Guard.** «sin movimientos (tope 30)» = 60 days without a bank booking. «entradas hundidas (tope 50)» = last-3-month inflows under 25% of own earlier mean. Ceiling glides at most 10 pts/month and lifts at once. Always mention an active tope; `score_pre_cap` is «sin el tope sería X». Never write `dark` / `fading`.
- **Confidence** alta / media / baja + `confidence_note`. «sin pagos de facturas en la ventana» = delay items blank, not “does not invoice”. Bundle copy is Spanish; cite `sentence`.
- **Trajectory:** mejorando / estable / bache / deteriorando / historial corto. Improving or deteriorating = 6-month slope same way (±1.5 pts/month) for 3 months and 3-month slope confirms (±3). Bache = 3-month down without that 6-month trend. «sin movimientos» is deteriorando by rule.
- **Reasons** / **change_reasons**: up to 4. Cite `sentence` and `eur`. Do not re-sum contributions.

## Validation, stated plainly

- Eight accepted outcomes, group-fold on train: **AUROC 0.48–0.55**, every 95% interval contains 0.5, never above a size baseline. A 3-month fall does no better.
- Therefore **explainable and monitorable, not predictive**. Say documented / explainable / se alejó de su normalidad. Never predicts / will fail / probability of default / bankruptcy risk.
- Nothing about hidden-test companies.

## Data facts that shape answers

- 1,286 empresas / 250 grupos / 24 months (sept 2024 → agosto 2026). About 39% have **no invoices**: payment history and mix blank, no customer alerts.
- `COUNTERPARTY_*` never maps to `COMP_*`. Customers/suppliers are exposures (open €, days late), never a scored company.
- Interest rate exists for 87 loans / 40 companies; utilisation 1.6% of rows. Do not estimate refinancing. No NSF token.
- Money is the company's currency, not converted. The bundle is immutable per drop — not live.

SESSION says which screen is open and which color is which empresa. Tools still fetch Neon (reasons, €, alerts, history). Dragging a period on Rápido asks about that window.
