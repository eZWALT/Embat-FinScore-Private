# Pregunta: one get_company covers a period

- Author: agent
- Timestamp: 2026-09-20 02:15 +02:00
- Still-binding: On `agent/overnight-optimize` (not main unless merged). `get_company.score_history` includes `reasons` / `change_reasons` for the focused month, the last 3, and months that moved ≥2 pts. Pregunta does not load FORMAT (`watcher_format.md`). After 4 steps or 8 calls, strip tools and write. Same tool catalog.

## Why

Main period questions called `explain_change` ×4–11 and sometimes `get_company` per month. History already sat on the company record; the model never saw the month reasons unless it picked that month.

## Bench (keep)

Same four sample companies on production Neon vs local sample_bundle (`--suite sample --ignore-latency` for host skew):

| Case | main tools / q | candidate tools / q |
|---|---|---|
| why-score | 1 / 6 | 1 / 6 |
| alerts | 2 / 6 | 2 / 6 |
| refuse | 0 / 5 | 0 / 5 |
| period-4 | 11 / 2 | 4 / 5 |
| why-change | 2 / 6 | 1 / 6 |

Period-4 on main was `explain_change` + `get_alerts` fan-out. Candidate: four `get_company`, Spanish, €, owners. Do not treat local COMP_1186 as vs-main.

## Still-unknown

Vercel preview of this branch with Neon; whether 1186 period-4 on a preview matches the sample-suite lift.
