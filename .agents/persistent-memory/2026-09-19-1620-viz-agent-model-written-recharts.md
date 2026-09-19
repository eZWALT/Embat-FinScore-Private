# 2026-09-19-1620 — chat that draws: the agent writes Recharts (prompt, sandbox, contract)

- **Author:** Claude Code (Sonnet 5) for Javier Boix
- **Where:** `product/agent/` (not `poc/`, not `product/web/`; the team wires what it needs)

## What exists

- `prompts/viz_system.md`: the agent prompt. It writes **React + Recharts code** per chart (`render_chart({title, subtitle, note, datasets, code})`), reads data as `data.ds_N` from SQL tool results, and uses helpers from a `sentinel` module (`fmt`, `tones`, `look`, `labels` in Spanish, `scoreColor`, `divergingColor`, `ChartContainer`). It carries the comparison table (22 questions → comparison → visual), 12 recipes, a section on inventing new visuals, and the wording and evidence rules.
- `prompts/api_schema.md`: the Neon `api` views the agent queries for chart data, with a SQL cookbook (all SQL parses with pglast).
- `neon_chart_views.sql`: **proposed** migration 007 (tidy views over the current run). Not applied, only syntax-checked.
- `sandbox/`: lint, compile (sucrase), jsdom dry run, runtime helpers, fixtures from the sample bundle, tests, a preview page. `npm test` renders every jsx block of the prompt with Recharts 3.8 and checks that the lint rejects network, `window`, URLs, foreign imports, forbidden wording, `rank_score`, missing note, oversized data.
- `VIZ_SPEC.md`: integration steps into `product/web` (AI SDK 7 tools with zod, per-request dataset store, `query_api`, `render_chart`, `SandboxedChart`), the browser sandbox contract (iframe `sandbox="allow-scripts"`, opaque origin, CSP `connect-src 'none'`, postMessage), the feedback loop options for render errors, theme variables.

## Decisions

- Model-written code instead of a JSON spec, because the tool should be able to draw what nobody planned. The safety comes from the sandboxed iframe, not from the lint.
- Chart data comes from SQL (`query_api`, `query_clean_db`) that returns a dataset id; the model never retypes numbers. Explanation tools keep returning JSON for the sentences around a chart.
- The host draws title, subtitle and note (required), so wording checks apply to them too.

## Still unknown

- Migration 007 was not run on Neon. The browser frame, the client-side feedback loop and the AI SDK v7 client-tool names are not built or verified.
- Records (`core.transactions`, `core.invoices`) are not loaded, so record recipes return `records not mounted`.
- The categorical chart palette (`--chart-1..5` are greys) has to be extended for multi-series charts.
- "16 items" in `product_context.md` and `chat_system.md` copies (the scorecard has 17).
