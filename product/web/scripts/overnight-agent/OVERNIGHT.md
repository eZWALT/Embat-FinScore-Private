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

## Tick 2 — 2026-09-20 02:31 +02

Keep vs main sample suite (`--ignore-latency`).

| Case | main q / tools | tick 2 q / tools / s |
|---|---|---|
| why-score | 6 / 1 | 6 / 1 / 5.7 |
| alerts | 6 / 2 | 6 / 2 / 9.1 |
| refuse | 5 / 0 | 5 / 0 / 1.8 |
| period-4 | 2 / 11 | **6 / 5** / 17.9 |
| why-change | 6 / 2 | 6 / 1 / 6.0 |

period-4: four `get_company` (no `month`) + one `get_alerts`. why-score text 747 chars, still grounded.

What landed:

- First step only `get_company` / `get_alerts` / `get_group` / `plot_series`.
- `clean_schema.md` only if the user asked about facturas / saldos / deuda.
- Bundle `items` when Neon extras are missing.
- Period shape: 1 line + 1 bullet per empresa.

Production `/empresa` still shows the main picker; the known waste there is still `explain_change` fan-out (this branch). Loop continues. Do not merge.

## Tick 3 — 2026-09-20 02:50 +02

Keep. period-4 now **4 tools / q 6 / 12.9 s** (tick 2 was 5 / 17.9 s; main 11 / q 2). Four `get_company`, no month, no unscoped `get_alerts`. Answer still splits fading vs cobros.

What landed:

- `get_alerts` without `entity_id` is clipped to the session + named Empresa/Grupo ids. Hard cap 8 if nothing is in scope.
- `plots_catalog` / `plot_series` only when the question mentions a chart or `Periodo seleccionado`.
- Trace: «N alertas · K entidades».

## Tick 4 — 2026-09-20 03:12 +02

Keep. period-4 **4 tools / q 6 / 9.8 s** (tick 3: 12.9 s; main 11 / q 2). why-score 1 / q 6.

What landed:

- `get_company` drops null metadata and `items` when `reasons` are present.
- Chat shows up to two bundle `sentence` quotes per company under «Leer índice» (max 4).
- Catalog: cita `sentence` tal cual.

## Tick 5 — 2026-09-20 03:31 +02

Keep vs main sample suite (`--ignore-latency`).

| Case | main tools / q / s | tick 5 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 5.7 |
| alerts | 2 / 6 / 11.0 | 2 / 6 / 8.6 |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 1.9 |
| period-4 | 11 / 2 / 17.0 | **5 / 6** / 14.5 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 7.8 |

period-4: four `get_company` (no month) + one scoped `get_alerts`. Tick 4 had 4 tools / 9.8 s; this run added alerts again. Quality still 6. Next: only call `get_alerts` when the user asked.

What landed:

- Trace hides raw JSON for score tools (keep SQL + errors).
- `score_history` last 18 months (or 6 before focus).
- Group mean history last 12 months.

Loop continues. Do not merge.

## Tick 6 — 2026-09-20 03:34 +02

Keep vs main sample suite (`--ignore-latency`).

| Case | main tools / q / s | tick 6 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 6.7 |
| alerts | 2 / 6 / 11.0 | 2 / 6 / 8.6 |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 1.8 |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / 11.8 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 6.7 |

period-4 is four `get_company` again (no alerts, no month). Text says «tope por entradas hundidas», not `fading`.

What landed:

- `get_alerts` only if the question asks for alertas / dueño / quién actúa.
- Plots catalog no longer opens on «Periodo seleccionado».
- Payloads use Spanish labels; reasons drop `item`.

Next: forbid «Llamo a…» and the method recitation on alerts.
