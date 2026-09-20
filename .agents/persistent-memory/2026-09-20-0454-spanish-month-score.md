# 2026-09-20 04:54 — Spanish month and score in tool payloads

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 04:54 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- Tool payloads speak `agosto 2026` / `ago 2026` and `88,5` / `+0,6`. `parseMonth` still accepts `YYYY-MM` or those labels if the model echoes them.
- WORDING / BREVITY: cite `month` and `score` as-is; never write `2026-08` or `88.45`.
- Trace quotes under «Leer índice» append ` · 34 k€` when `eur` is present.

## Decisions

- Fair sample keep. why-score opened with «cierra agosto 2026 con un índice de 88,5». Alerts said «jul 2026» / «jun 2026», not ISO months.

| Case | main tools / q / s | tick 15 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 6.4 |
| alerts | 2 / 6 / 11.0 | 1 / 6 / 6.3 |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 1.9 |
| period-4 | 11 / 2 / 17.0 | 4 / 6 / 11.6 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 7.1 |

## Still unknown

- Alert `evidence` keys are still English.
