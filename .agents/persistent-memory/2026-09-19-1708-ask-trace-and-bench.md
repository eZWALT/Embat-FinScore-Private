# 2026-09-19-1708 — Cursor-style tool trace + Ask latency bench

- **Author:** agent (Cursor / Walter)
- **When:** 2026-09-19 17:08 CEST

## What changed

- Consultas now renders each tool in stream order (`AgentTrace`): `tools (name)`, Spanish label from `tool-catalog.ts`, status, ms, expandable input/output. SQL is shown for `query_clean_db`.
- Complete tool catalog in `prompts/tools_catalog.md` (when / in / out / do-not). Loaded with the Ask system prompt.
- Each tool execute adds `timing_ms`. Runtime logs step names and total ms on Vercel.
- Bench: `product/web/scripts/bench-ask.mjs` → `src/lib/agent/BENCHMARK.md`. Helmcode `deepseek-v4-flash`, median of 3, 2026-09-19T15:05Z.

## Numbers (this machine; Helmcode key present, no local DATABASE_URL)

| | TTFT | Completion | minutes (TTFT) |
|---|---:|---:|---:|
| Tiny system (40 chars) | 300 ms | 300 ms | 0.0050 |
| Full Ask prompt + catalog (31,047 chars) | 418 ms | 418 ms | 0.0070 |

The catalog costs ~120 ms of TTFT, not minutes. The expandable UI is render-only. A later tool loop still costs one Helmcode round per step (~0.4 s here) plus Neon SQL (not measured here).

## Decisions

- No LangChain. Same AI SDK loop.
- Neon tool-SQL bench is in the script and waits for `DATABASE_URL`.

## Still unknown

- Per-tool Neon ms on the Vercel/Neon path (this laptop had no `DATABASE_URL`).
- Cold Helmcode first call was ~1.8 s; medians after warmup are the table above.
