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

## Tick 7 — 2026-09-20 03:36 +02

Keep vs main sample suite (`--ignore-latency`).

| Case | main tools / q / s | tick 7 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 4.8 |
| alerts | 2 / 6 / 11.0 | **1 / 6** / 7.8 |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 1.6 |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / 12.7 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 7.8 |

alerts is one empty-input `get_alerts` (session company + group). period-4 has no «Llamo a…».

What landed:

- `get_alerts` cap 1; filter = session ∪ entity_id.
- No announce-the-call / method-recitation in BREVITY.
- Alert entity + evidence ids as Empresa/Grupo.

Next: stats/lift only if they ask fiabilidad.

## Tick 8 — 2026-09-20 03:39 +02

Keep vs main sample suite (`--ignore-latency`).

| Case | main tools / q / s | tick 8 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 6.5 |
| alerts | 2 / 6 / 11.0 | **1 / 6** / 7.8 typical (this run 96 s TTFT stall) |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 2.1 |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / **8.5** |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 6.8 |

alerts answer dropped the lift lecture (936 chars). Members are Empresa 0461, not `COMP_*`.

What landed:

- `stats` only if they ask fiabilidad.
- Categories as Spanish `{ name, score, pts }`.
- UI quotes alert title · dueño under «Leer alertas».
- Trace summary uses Empresa/Grupo.

## Tick 9 — 2026-09-20 03:41 +02

Keep vs main sample suite (`--ignore-latency`).

| Case | main tools / q / s | tick 9 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 7.0 |
| alerts | 2 / 6 / 11.0 | **1 / 6 / 4.8** |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 1.7 |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / 11.6 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 4.9 |

What landed: after an alerts-only question, force the text step. Production Pregunta on hack-spain is still main (this branch is not live there).

## Tick 10 — 2026-09-20 03:45 +02

Visual keep (no quality drop vs last sample suite).

- Local Pregunta: «Leer índice» + two bundle sentences, then the answer. No JSON dump.
- Chart chips / legend: Empresa 0030. SESSION series use the same label and keep `COMP_*` in parens.

## Tick 11 — 2026-09-20 03:51 +02

Keep vs main sample suite (`--ignore-latency`).

| Case | main tools / q / s | tick 11 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 6.2 |
| alerts | 2 / 6 / 11.0 | **1 / 6** / 6.7 |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 1.7 |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / 10.4 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 7.9 |

What landed: force text after one company on a single-empresa question; history is 4 months unless the question is a period (18). No `alert_ids` on `get_company`.

## Tick 12 — 2026-09-20 04:12 +02

Keep vs main sample suite (`--ignore-latency`).

| Case | main tools / q / s | tick 12 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 6.1 |
| alerts | 2 / 6 / 11.0 | **1 / 6** / 6.7 |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 1.4 |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / 11.4 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 6.5 |

What landed: `product_monitor.md` only if they ask alertas / gráfico / clúster / previsión. Tool row shows «Leer índice · Empresa 0030 · índice 88» without a chevron.

## Tick 13 — 2026-09-20 04:34 +02

Keep vs main sample suite (`--ignore-latency`). Quality held; two Helmcode TTFT stalls (~95 s) on alerts and period-4.

| Case | main q / tools | tick 13 q / tools |
|---|---|---|
| why-score | 6 / 1 | 6 / 1 (611 chars, 5.4 s) |
| alerts | 6 / 2 | 6 / 1 |
| refuse | 5 / 0 | 5 / 0 |
| period-4 | 2 / 11 | **6 / 4** |
| why-change | 6 / 2 | 6 / 1 |

What landed: ROLE is procedure only. No «tus salidas».

## Tick 14 — 2026-09-20 04:51 +02

Keep vs main sample suite (`--ignore-latency`). First bench of this tick was **not** a keep: `.map(slimReason)` passed the array index as currency and alerts confessed `toUpperCase is not a function`. Re-benched after the wrap.

| Case | main tools / q / s | tick 14 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 3.3 (705 chars, `34 k€`) |
| alerts | 2 / 6 / 11.0 | **1 / 6** / 6.1 (feed, not error) |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 0.6 |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / 10.6 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 3.2 |

What landed: `eur` is `formatMoney`; WORDING cites it as-is; `formatMoney` ignores a non-string currency.

## Tick 15 — 2026-09-20 04:54 +02

Keep vs main sample suite (`--ignore-latency`).

| Case | main tools / q / s | tick 15 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 6.4 (`agosto 2026`, `88,5`) |
| alerts | 2 / 6 / 11.0 | **1 / 6** / 6.3 (`jul 2026`) |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 1.9 |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / 11.6 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 7.1 |

What landed: `speakMonth` / `speakScore` / `speakDelta`; `parseMonth` for echoed labels; quotes append ` · 34 k€`.

## Tick 16 — 2026-09-20 04:55 +02

Keep vs main sample suite (`--ignore-latency`).

| Case | main tools / q / s | tick 16 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 5.4 (606 chars) |
| alerts | 2 / 6 / 11.0 | **1 / 6** / 8.8 (Javi action quotes) |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 2.2 |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / 12.8 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 8.3 |

What landed: Spanish evidence keys; drop `alert_id`; persistence `N de los últimos 4 meses`.

## Tick 17 — 2026-09-20 04:59 +02

Keep vs main sample suite (`--ignore-latency`). Alerts TTFT ~95 s (Helmcode).

| Case | main tools / q / s | tick 17 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 5.7 |
| alerts | 2 / 6 / 11.0 | **1 / 6** / stall (1027 chars, julio 2026) |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 1.6 (canonical recusa) |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / 12.2 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 6.0 |

What landed: no unscoped feed; drop `summary`; never ask the user for `COMP_*`. Production Rápido (no company) already asked which entity, but leaked tokens.

## Tick 18 — 2026-09-20 05:01 +02

Keep vs main sample suite (`--ignore-latency`). Visual: opening chips.

| Case | main tools / q / s | tick 18 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 5.7 |
| alerts | 2 / 6 / 11.0 | **1 / 6** / 7.9 |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 0.6 |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / 11.4 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 7.3 |

What landed: Rápido without empresa opens «¿Qué empresa miro primero?» / «¿Quién está peor este mes?».

## Tick 19 — 2026-09-20 05:02 +02

Keep vs main sample suite (`--ignore-latency`).

| Case | main tools / q / s | tick 19 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 5.8 |
| alerts | 2 / 6 / 11.0 | **1 / 6** / 8.7 |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 1.4 |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / **9.0** |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 6.7 |

What landed: PRODUCT guard/trajectory labels match `speakGuard` / `speakTrajectory`.

## Tick 20 — 2026-09-20 05:08 +02

Keep vs main sample suite (`--ignore-latency`). Local chips verified (Next overlay still sits on the FAB).

| Case | main tools / q / s | tick 20 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 5.1 |
| alerts | 2 / 6 / 11.0 | **1 / 6** / 8.5 |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 1.4 |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / 12.2 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / stall |

What landed: SESSION months and focus labels in Spanish. Rápido empty → pick-empresa chips; one company on the chart → gráfico/alertas.

## Tick 21 — 2026-09-20 05:10 +02

Keep. One company on the Rápido chart now opens «¿Por qué este índice este mes?» / «¿Qué alertas hay?».

## Tick 22 — 2026-09-20 05:07 +02

Keep vs main sample suite (`--ignore-latency`).

| Case | main tools / q / s | tick 22 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 5.2 |
| alerts | 2 / 6 / 11.0 | **1 / 6** / 7.0 |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 2.3 |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / 9.4 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 7.2 |

What landed: no extras query when `reasons` exist; `score_pre_cap` only with a tope; no raw slopes.

## Tick 23 — 2026-09-20 05:10 +02

Keep vs main sample suite (`--ignore-latency`). Same 1/1/0/4/1 tools and q 6/6/5/6/6.

What landed: omit empty `change_reasons`; MAP SESSION says agosto 2026 / Empresa 0030.

## Tick 24 — 2026-09-20 05:10 +02

Keep vs main sample suite (`--ignore-latency`). Visual: `34 k€` sits next to the reason sentence under «Leer índice».

## Tick 25 — 2026-09-20 05:13 +02

Keep. PRODUCT validation shortened; AUROC / not-predictive / no hidden-test claims stay.

## Tick 26 — 2026-09-20 05:15 +02

Keep vs main sample suite (`--ignore-latency`). Data-facts / screens cut; counterparties, no-invoice, no-refi, no-NSF stay.

## Tick 27 — 2026-09-20 05:13 +02

Keep. PRODUCT score method compressed; same weights, glide, tope, trajectory rule.

## Tick 28 — 2026-09-20 05:16 +02

Keep. TOOLS “dónde” is one line (`api`/`analytics` vs `core`).

## Tick 29 — 2026-09-20 05:18 +02

Keep. Catalog matches the unscoped-alerts guard.

## Tick 30 — 2026-09-20 05:17 +02

Keep. `list_companies` catalog in Spanish. `explain_change` parses `agosto 2026`. Alerts 851 chars / q 6.

## Tick 31 — 2026-09-20 05:19 +02

Keep. Spanish missing-month errors. WORDING: media/baja; cite owner/action.

## Tick 32 — 2026-09-20 05:24 +02

Keep vs main sample suite (`--ignore-latency`).

| Case | main tools / q / s | tick 32 tools / q / s |
|---|---|---|
| why-score | 1 / 6 / 8.7 | 1 / 6 / 5.5 |
| alerts | 2 / 6 / 11.0 | **1 / 6** / 7.6 |
| refuse | 0 / 5 / 2.4 | 0 / 5 / 1.6 |
| period-4 | 11 / 2 / 17.0 | **4 / 6** / 9.9 |
| why-change | 2 / 6 / 8.7 | 1 / 6 / 5.5 |

What landed: slimmer on-demand monitor; period tope only on companies that bring `guard`; dueño/acción only after `get_alerts` (why-score / why-change no longer invent Tesorero). First bench said «tres… tope» — 0651 is cobros; second dropped the alert-only qualifier and invented a dueño on the índice; third keep is clean.

Leftover: period lead still writes bare `0651` / `0030`. Do not merge.

## Tick 33 — 2026-09-20 05:32 +02

Keep vs main sample suite (`--ignore-latency`). New scorer (invented dueño / leaked token / bare id) recomputed on both sides: period-4 d_quality +6.

What landed: tool payloads say Empresa/Grupo; period history is the dragged months + 1 (not 18); «Empresa 0651», never «la de 65 puntos». First bench paraphrased 0651 as 65 puntos — discarded that lead.

## Tick 34 — 2026-09-20 05:35 +02

Keep vs main sample suite (`--ignore-latency`).

What landed: `speakGuard` always says tope or «sin tope»; period lead is two fading companies + 0651 mora + 0030 estable; ×4 quotes show Empresa N; production main still asks the user for `COMP_*` (this branch does not).

## Tick 35 — 2026-09-20 05:40 +02

Keep. WORDING: cite `81,1` not `81`. Local Pregunta: empty → pick-empresa chips; Empresa 0011 on the chart → why-score / alertas.

Do not merge.
