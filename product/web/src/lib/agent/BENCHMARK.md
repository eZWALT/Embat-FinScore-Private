# Ask agent benchmark

- When: 2026-09-19T15:22:27.060Z
- Model: deepseek-v4-flash
- Prompt stack: 36,221 chars (MAP → ROLE → SCOPE → PRODUCT → WORDING → FORMAT → TOOLS → RECORDS → SESSION)
- Tool schemas sent: 10

The Cursor-style tool trace is render-only. It does not add a model round-trip.
The clock is Helmcode (TTFT / time-to-first-tool) and Neon (tool SQL). Minutes = ms / 60000.
Full prompt is kept on purpose. Do not shrink PRODUCT or TOOLS to chase TTFT.

## Prompt pieces

| File | layer | chars |
|---|---|---:|
| `prompt_map.md` | MAP | 1,462 |
| `chat_system.md` | ROLE | 5,225 |
| `scope.md` | SCOPE | 1,643 |
| `product_context.md` | PRODUCT | 9,890 |
| `wording_rules.md` | WORDING | 3,466 |
| `watcher_format.md` | FORMAT | 4,710 |
| `tools_catalog.md` | TOOLS | 5,297 |
| `clean_schema.md` | RECORDS | 4,430 |
| `session` | SESSION | 42 |

## Time to first token (Helmcode, no tools)

| System | TTFT | Completion | TTFT (min) |
|---|---:|---:|---:|
| Tiny (40 chars), median of 3 | 284 ms | 284 ms | 0.0047 |
| Full Ask prompt (no tools), median of 3 | 353 ms | 364 ms | 0.0059 |

## Time to first tool call (Helmcode, full prompt + 10 tools)

Question: `¿Por qué este índice este mes y qué alertas hay? Usa las herramientas. No inventes números.`

| Mode | First event | First tool | Stream until pause | Tools asked | finish | First tool (min) |
|---|---:|---:|---:|---|---|---:|
| Full prompt + tools, auto, median of 3 | 446 ms | 446 ms | 447 ms | get_company, explain_change, get_alerts | tool_calls | 0.0074 |
| Full prompt + tools, required, 1 shot | 2.96 s | 2.96 s | 3.08 s | get_company, explain_change, get_alerts | tool_calls | 0.0493 |

### Rounds (auto)

| # | First tool | Tools | finish | total |
|---|---:|---|---|---:|
| 1 | 1.62 s | get_company, explain_change, get_alerts | tool_calls | 1.86 s |
| 2 | 446 ms | get_company, explain_change, get_alerts | tool_calls | 447 ms |
| 3 | 309 ms | get_company, explain_change, get_alerts | tool_calls | 316 ms |

## Neon retrieval (same queries the tools run)

| Query | p50-ish (1 shot) | min | ok | note |
|---|---:|---:|---|---|
| — | — | — | — | DATABASE_URL not set in this environment |
