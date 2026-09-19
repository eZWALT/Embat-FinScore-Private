# Ask agent benchmark

- When: 2026-09-19T15:05:47.598Z
- Model: deepseek-v4-flash
- Prompt files: 31,047 chars (complete catalog included)

The Cursor-style tool trace is render-only. It does not add a model round-trip.
The clock is Helmcode (TTFT / completion) and Neon (tool SQL). Minutes = ms / 60000.

## Prompt pieces

| File | chars |
|---|---:|
| `chat_system.md` | 4,106 |
| `product_context.md` | 9,736 |
| `wording_rules.md` | 2,698 |
| `watcher_format.md` | 4,716 |
| `tools_catalog.md` | 5,285 |
| `clean_schema.md` | 4,422 |

## Time to first token (Helmcode, no tools)

| System | TTFT | Completion | TTFT (min) |
|---|---:|---:|---:|
| Tiny (40 chars), median of 3 | 300 ms | 300 ms | 0.0050 |
| Full Ask prompt + catalog, median of 3 | 418 ms | 418 ms | 0.0070 |

## Neon retrieval (same queries the tools run)

| Query | p50-ish (1 shot) | min | ok | note |
|---|---:|---:|---|---|
| — | — | — | — | DATABASE_URL not set in this environment |
