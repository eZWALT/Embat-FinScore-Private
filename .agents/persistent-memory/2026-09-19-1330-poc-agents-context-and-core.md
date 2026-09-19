# 2026-09-19-1330 — POC agents: context, prompts, shared core; sub-agents on the three features

- **Author:** Walter (with Cursor agent; three sub-agents for the pages)
- **When:** 2026-09-19 ~13:30 CEST
- **Where:** private repo only (`Embat-FinScore-Private`); public sync at the end.

## Decisions

- **LLM provider:** Helmcode, OpenAI-compatible (`https://api.helmcode.com/v1`, `HELMCODE_API_KEY`, EU inference). Model `deepseek-v4-flash` (the DeepSeek on this key; tool calling confirmed). LangChain `ChatOpenAI` with `base_url`. Config by env only: `HELMCODE_API_KEY` or `HELMCODE_API_KEY_FILE`, `HELMCODE_BASE_URL`, `POC_LLM_MODEL`, `POC_LANG` (default `es`). Key models on the account: deepseek-v4-flash, qwen3.6, gemma4, glm5.x, gemini-3.x, claude-*, gpt-5.6-*.
- **Agents read, never compute.** Tools read the export bundle (scores, reasons with €, alerts, control charts, clusters, forecast) and the cleaned DuckDB (`clean.*`, read-only, SELECT-only guard, LIMIT 200). No score, percentile or trend is recomputed by the model. Wording rules are a file the prompt includes.
- **Two agents, one context.** `poc/agent/prompts/product_context.md` (score, guard, confidence, trajectory, validation stated plainly, monitor, alerts, stats, data facts, TellMe modes), `wording_rules.md`, `clean_schema.md`, and a role prompt each (`sentinel_system.md`, `chat_system.md`). `context.py` appends the live manifest spec (labels, weights, units, `why`), disclaimer and alert stats. System prompt ≈ 24k chars.
- **Plots** are a tool (`plot_series`) that stores a spec; the UI renders with Altair. The model never draws.
- **Watcher interval** is one env constant `POC_WATCH_INTERVAL_SEC` (3600 now, 86400 later). The bundle is static, so "novelty" = not yet delivered to this watch set, with an as-of replay control for the demo.
- **Portfolio** now reads the bundle (v1 scorecard); the v0 parquet path (`poc/data.py`, `score_3m`) is removed from the POC. The v0 card itself still exists in `product/score/score.py`; retiring it is Javi's call.
- Full bundle generated locally: `data/bundle` (1,286 companies, 250 groups, 2,365 alerts, 2024-11..2026-08, 51.6 MB; gitignored with `data/bundle_work`).

## Verified

- Tool-calling round trip through Helmcode; SQL guard blocks `drop`, `main.*`, unqualified table names; a Spanish "why + what changed + plot" question on COMP_1237 produced 5 tool calls, one plot, owners and € amounts, the fixed top-customer wording and the disclaimer.
- Portfolio renders headlessly (AppTest) and switches companies without exceptions. `st.altair_chart` takes `use_container_width`, not `width`, in 1.50.
- Prompt fix after the test: `confidence_note` "no invoice payments in the window" ≠ "no invoices".

## Sub-agents (parallel, disjoint files)

1. Watcher page: `poc/views/watcher.py` + `poc/watcher_state.py` (novelty engine, per-watch-set state in `poc/.state/`).
2. Ask page: `poc/views/ask.py` (+ `poc/ask_helpers.py`).
3. Next.js Group Health Map: route `/grupos`, additive to Ruben's app (`product/web`); heatmap groups × months, member table, company panel with reasons and alerts; must not touch `health-dashboard.tsx`, `health-sidebar.tsx`, `page.tsx`, `dashboard-service.ts`.

## Still unknown

- Whether the Next.js route conflicts with Ruben's concurrent edits (kept additive to minimise it).
- Cost/latency of DeepSeek on the 24k-char prompt at hourly cadence (fine for a demo; cache the system prompt if it matters).
- Where the full bundle and the clean DuckDB will live for the hosted agents (Vercel Blob / a small Python service).
