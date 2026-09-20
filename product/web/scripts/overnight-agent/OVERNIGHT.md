# Pregunta overnight log

Branch `agent/overnight-optimize`. Compare every keep to `baseline.json` (main @ `hack-spain.vercel.app`).

## Tick 0 — 2026-09-20 01:55 +02

First pass (before first live candidate bench):

- One retrieval per company: `toolCallKey` ignores month; parallel same-entity calls share one execute.
- `get_company` cap 4 (period chart), still 1 per empresa.
- Catalog + ROLE: no per-month fan-out; no `explain_change` if `change_reasons` is already there; one `get_alerts` with `since_month`.
- Catalog in Spanish (same tools).
- Follow-up chips `line-clamp-3`.
- Leftover Spanish-only SCOPE / WORDING / ROLE from the working tree.

## Tick 0b — baseline (main production)

`hack-spain.vercel.app` · COMP_1186 · 2026-09-19T23:59Z

| Case | tools | q | ms | notes |
|---|---:|---:|---:|---|
| why-score | 1 | 6 | 103s | one `get_company` |
| alerts | 2 | 6 | 8s | company + group |
| refuse | 0 | 5 | 3s | Spanish refuse |
| period-4 | 4 | 5 | 113s | one company + plot; text present |
| why-change | 2 | 5 | 13s | |

Local :3010 is sample_bundle (12 companies). COMP_1186 is not there — do not treat local 1186 benches as vs-main. Preview with Neon is the fair candidate. Local is for compile + sample companies (`COMP_0030`).

## Tick 1 — 2026-09-20 02:15 +02

Keep. Fair suite = `--suite sample` (COMP_0030 / 0016 / 0011 / 0176 / 0651) on production vs local.

| Case | main tools / q / s | candidate tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 6.1 |
| alerts | 2 / 6 / 11.0 | 2 / 6 / 7.1 |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 0.9 |
| period-4 | 11 / 2 / 17.0 | 4 / 5 / 17.1 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 5.5 |

Main period-4: `explain_change` ×7 + `get_alerts` ×4. Candidate: four `get_company` (one per empresa), text grounded.

What landed:

- `score_history` carries reasons on moved months / last 3 / focus month. Cluster dropped from `get_company` (still on `compare_with_cluster`).
- After a successful `get_company` with `change_reasons`, `explain_change` and `list_companies` are stripped. Force text at 4 steps.
- Pregunta no longer loads FORMAT. ROLE / PLOTS trimmed or Spanish. Tool descriptions Spanish.
- Trace: hide `×1`; hide tiny input JSON; Spanish cap / records-not-mounted labels.
- Bench: `--suite sample`, `--ignore-latency`, month_fanout = extra months per company (not “passed month”).

Loop PID 497061, every 20 min, until 05:55–08:55 +02. Do not merge to main tonight.

## Tick 1b — 2026-09-20 02:16 +02

Keep (no quality drop on why-score q=6 / 1 tool). After four successful `get_company`, force the text step. Control charts return the last 8 months + `persistent_now`, no CUSUM arrays. Local Pregunta opens; a Next hydration overlay on `app-header` can sit on the FAB (pre-existing, not this branch).
