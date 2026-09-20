# 2026-09-20 03:31 — hide bulky tool JSON; window score history

Author: Cursor Grok (overnight loop)
Timestamp: 2026-09-20 03:31 +02
Branch: `agent/overnight-optimize` (do not merge tonight)

## What changed

- `AgentTrace` no longer dumps the raw JSON output for `get_company` / `get_alerts` / `get_group` / `explain_change` when a summary exists. Errors and `query_clean_db` still show the payload.
- `score_history` is the last 18 months (or 6 months before the focus month). Reasons still ride on the focus month, last 3, or a ≥2 pt move.
- `get_group.mean_score_history` is the last 12 months.

## Decisions

- The UI already quotes bundle `sentence` under «Leer índice». The JSON dump was noise, not grounding.
- Fair bench remains `--suite sample --ignore-latency` vs `baseline-sample.json` (hack-spain). Keep: period-4 still q 6 / −6 tools vs main; this run used 5 tools (four `get_company` + one scoped `get_alerts`).

## Still unknown

- period-4 sometimes adds `get_alerts` even when the question is only the four-company period. Next tick: strip that unless the user asked for alertas.
