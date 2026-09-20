# `.agents` — how to read the journal

`AGENTS.md` is the pointer file. This folder is the dated journal (`persistent-memory/`). There are ~180 files. **Most are night discovery notes. Do not read them first.**

Add a new dated file after meaningful work. Do not rewrite old ones. If the new file is still-binding, add a line under **Still binding** below.

## Still binding (read in this order)

### Rules

| File | Why it still binds |
|------|-------------------|
| `persistent-memory/2026-09-18-initial-context.md` | Event, dataset, four jobs, hidden test. Working name “Aura” in that file is stale; the product is **Health Sentinel**. |
| `persistent-memory/2026-09-18-2020-four-goals.md` | Signals → 0–100 → explain → web + LLM. |
| `persistent-memory/2026-09-19-1100-plan-fico-monitor.md` | The five-step plan. `AGENTS.md` is the short table. |
| `persistent-memory/2026-09-19-1130-night-variables-for-score.md` | Candidate variables for the scorecard. No new feature hunting. |
| `persistent-memory/2026-09-19-1015-y3-recovery-mechanical.md` | **Do not reuse Y3.** |
| `persistent-memory/2026-09-19-1300-y7-alert-grade-eval.md` | Top-customer-quiet numbers. Never quote “75%”. |
| `persistent-memory/2026-09-19-embat-business-context.md` | Buyer rationale (Embat premium module). |
| `persistent-memory/2026-09-19-1410-single-repo.md` | This repo only. Public sibling retired. |

### What was built (afternoon)

| File | What |
|------|------|
| `persistent-memory/2026-09-19-1420-pipeline-entry-point.md` | Step 1: clean + feature store entry point. |
| `persistent-memory/2026-09-19-1615-fico-score-built.md` | Step 2: scorecard, not predictive. |
| `persistent-memory/2026-09-19-1830-monitor-and-forecast.md` | Steps 3–4: clusters, charts, alerts, forecast. |
| `persistent-memory/2026-09-19-1815-guard-glide.md` | The going-dark / fading guard glides (10 points a month) instead of capping at once; schema 1.3.0; effect and trade-offs. |
| `persistent-memory/2026-09-19-1700-web-data-contract.md` | Bundle → Neon `analytics` / `api`. |
| `persistent-memory/2026-09-19-1450-agents-neon-storage.md` | What the hosted app may load. |
| `persistent-memory/2026-09-19-1620-alerts-in-spanish.md` | Javi production copy (`language: es`). |
| `persistent-memory/2026-09-19-1658-integrate-spanish-watcher.md` | Vigilancia in Resumen; Watcher format locked. |
| `persistent-memory/2026-09-19-1710-ask-scope-guardrails.md` | SCOPE layer: refuse off-topic / jailbreak. |
| `persistent-memory/2026-09-19-1715-consultas-design-b.md` | Consultas: interleaved tools + shimmer text. |
| `persistent-memory/2026-09-19-1720-prompt-map-and-tool-bench.md` | Prompt layers + Helmcode time-to-first-tool. |
| `persistent-memory/2026-09-19-1750-quick-same-chat.md` | Rápido and Profundo share the Pregunta popup. |
| `persistent-memory/2026-09-19-1758-chat-screen-session.md` | Pregunta SESSION includes the open dashboard (colors, period). |
| `persistent-memory/2026-09-19-1839-month-as-of-labels.md` | Month labels: `agosto 2026` / `ago 2026`; as-of badges use `formatAsOf` (`hasta agosto 2026`). Not “today”. |
| `persistent-memory/2026-09-19-1845-ask-native-streaming.md` | Helmcode via `createOpenAI().chat()` (`/v1/chat/completions` SSE). Default `createOpenAI()(id)` is Responses (`/responses`) and does not yield visible text for deepseek-v4-flash. |
| `persistent-memory/2026-09-19-1910-ask-loaders-followups.md` | Tool ring stays until first reply token; 3-dot pulse under tools; opening chips precomputed; two agentic follow-ups after Ask. Product name in chrome is Sentinel. |
| `persistent-memory/2026-09-19-2005-ui-spanish-labels.md` | Chrome: Empresa/Grupo labels, Javi money (`14 k€` / `14 k AED`), no raw `kind`/`guard`/`item`, Spanish numbers, Δ in the Índice card. |
| `persistent-memory/2026-09-19-2025-qa-sample-bundle.md` | Local web without Neon uses `product/score/sample_bundle`. Production still requires `DATABASE_URL`. |
| `persistent-memory/2026-09-19-2055-followups-visible-push.md` | Pregunta chips: 2-col row above the composer; fallback immediately; product name is Sentinel. |
| `persistent-memory/2026-09-19-2108-ask-uses-repository.md` | Pregunta does not 503 without Neon locally; tools use the score repository. |
| `persistent-memory/2026-09-19-2112-chip-wording.md` | Pregunta chips: one idea, short tap; no yes/no; Δ in the Índice card. |
| `persistent-memory/2026-09-19-2115-spanish-only-guardrails.md` | Pregunta / Sentinel / recusa: only Spanish, even if the user writes in another language. |
| `persistent-memory/2026-09-20-0025-landing-team-screen.md` | Landing Team is `#team`: same-viewport fade to portraits + CEO/CTO/CSO, not a scroll. |
| `persistent-memory/2026-09-20-0100-thinking-brain-followups.md` | Pregunta chips after the answer + tools. Brain = DeepSeek `thinking` on. |
| `persistent-memory/2026-09-20-0112-tool-call-caps.md` | Same tool catalog. Cap 2 calls/tool/turn. Identical replay is memoized. Stop-the-stream-at-8 is superseded. |
| `persistent-memory/2026-09-20-0148-tools-then-text.md` | After 6 steps or 8 calls, strip tools and write. Do not stop the stream on call count. ×n ticks. |
| `persistent-memory/2026-09-20-0155-overnight-agent-optimize.md` | Pregunta overnight: branch `agent/overnight-optimize` only. Same tools. Bench vs `main`. 4–7 h. |
| `persistent-memory/2026-09-20-0215-get-company-period-history.md` | Overnight keep: `get_company` history carries period reasons; Pregunta skips FORMAT; force text at 4 steps. |
| `persistent-memory/2026-09-20-0231-opening-tools-records-schema.md` | Overnight: first step is score tools only; `clean_schema` only on record questions. |
| `persistent-memory/2026-09-20-0250-alerts-scoped-plots-on-demand.md` | Overnight: `get_alerts` scoped to session/named entities; plots catalog only if asked. |
| `persistent-memory/2026-09-20-0312-slim-company-reason-quotes.md` | Overnight: slimmer `get_company`; UI quotes `sentence` under the tool row. |
| `persistent-memory/2026-09-20-0331-hide-tool-json-history-window.md` | Overnight: hide bulky score-tool JSON in the trace; last-18 `score_history`; last-12 group mean. |
| `persistent-memory/2026-09-20-0334-alerts-on-demand-spanish-tokens.md` | Overnight: `get_alerts` only if asked; tool payloads in Spanish labels; no `fading`/`item` in the bubble. |
| `persistent-memory/2026-09-20-0336-one-alerts-call.md` | Overnight: one `get_alerts` (session ∪ entity); no «Llamo a…»; evidence ids as Empresa/Grupo. |
| `persistent-memory/2026-09-20-0339-alert-quotes-stats-on-demand.md` | Overnight: alert title quotes in the UI; `stats` only if they ask fiabilidad; Spanish categories. |
| `persistent-memory/2026-09-20-0341-alerts-only-force-text.md` | Overnight: after a successful alerts-only `get_alerts`, strip tools and write. |
| `persistent-memory/2026-09-20-0345-chart-session-empresa-labels.md` | Overnight: Rápido chart + SESSION series say Empresa, not `COMP_*`. |
| `persistent-memory/2026-09-20-0351-company-only-slim-history.md` | Overnight: write after one company read; 4-month history unless the question is a period. |
| `persistent-memory/2026-09-20-0412-monitor-on-demand-trace-row.md` | Overnight: monitor product layer only if asked; tool row shows the summary without a chevron. |
| `persistent-memory/2026-09-20-0434-slim-chat-role.md` | Overnight: Pregunta ROLE is procedure only; no «tus salidas». |
| `persistent-memory/2026-09-20-0451-javi-money-reasons.md` | Overnight: reason `eur` is Javi money; never `.map(slimReason)` (index ≠ currency). |
| `persistent-memory/2026-09-20-0454-spanish-month-score.md` | Overnight: tool months/scores are `agosto 2026` / `88,5`; `parseMonth` accepts both. |
| `persistent-memory/2026-09-20-0455-spanish-alert-evidence.md` | Overnight: alert evidence in Spanish; no `alert_id`; persistence is `N de 4`. |
| `persistent-memory/2026-09-20-0459-alerts-no-scope-no-summary.md` | Overnight: no unscoped `get_alerts`; drop alert `summary`; never ask for `COMP_*`. |
| `persistent-memory/2026-09-20-0501-opening-chips-no-entity.md` | Overnight: Rápido without a company does not open with «¿Qué alertas hay?». |
| `persistent-memory/2026-09-20-0502-product-guard-spanish.md` | Overnight: always-on PRODUCT speaks the same guard/trajectory labels as the tools. |
| `persistent-memory/2026-09-20-0508-session-spanish-months.md` | Overnight: SESSION as_of/periodo/focus in Spanish; local chips verified. |
| `persistent-memory/2026-09-20-0507-skip-extras-precap.md` | Overnight: skip Neon extras when reasons exist; `score_pre_cap` only if tope. |
| `persistent-memory/2026-09-20-0510-omit-empty-change.md` | Overnight: omit empty `change_reasons` so `explain_change` stays available. |
| `persistent-memory/2026-09-20-0512-reason-quote-money.md` | Overnight: tool-row quotes put Javi money in tabular-nums. |
| `persistent-memory/2026-09-20-0513-slim-product-validation.md` | Overnight: PRODUCT validation is three short bullets; claims unchanged. |
| `persistent-memory/2026-09-20-0515-slim-product-facts.md` | Overnight: PRODUCT data facts are four bullets; screens live in SESSION. |
| `persistent-memory/2026-09-20-0513-slim-product-score.md` | Overnight: PRODUCT score block is six bullets; weights/glide/tope claims stay. |
| `persistent-memory/2026-09-20-0516-slim-tools-where.md` | Overnight: TOOLS where-data is one Spanish line. |
| `persistent-memory/2026-09-20-0518-alerts-catalog-no-scope.md` | Overnight: catalog matches unscoped-alerts refuse; never ask `COMP_*`. |
| `persistent-memory/2026-09-20-0517-catalog-and-parse-month.md` | Overnight: list catalog in Spanish; `explain_change` accepts `agosto 2026`. |
| `persistent-memory/2026-09-20-0519-spanish-tool-errors.md` | Overnight: tool errors in Spanish; WORDING cites owner/action as-is. |
| `persistent-memory/2026-09-20-0524-monitor-tope-owner.md` | Overnight: slim monitor; tope per company; dueño only if `get_alerts` returned it. |
| `persistent-memory/2026-09-20-0532-empresa-labels-period-span.md` | Overnight: tools speak Empresa/Grupo; period history = span + 1; no bare `0651`. |
| `persistent-memory/2026-09-20-0535-sin-tope-period-quotes.md` | Overnight: `guard` is always tope or «sin tope»; period quotes label Empresa. |
| `persistent-memory/2026-09-20-0540-cite-precap-local-chips.md` | Overnight: cite `score_pre_cap` as `81,1`; local chips switch when a company is on the chart. |
| `persistent-memory/2026-09-20-0542-no-glide-recitation.md` | Overnight: do not recite the 10 pts/month glide; cite tope + `score_pre_cap`. |
| `persistent-memory/2026-09-20-0544-eur-is-amount.md` | Overnight: `eur` is the amount; the sign is on `points`. |
| `persistent-memory/2026-09-20-0549-omit-eur-currency.md` | Overnight: omit `currency` when EUR; `eur` already says `k€`. |
| `persistent-memory/2026-09-20-0550-followup-no-comp.md` | Overnight: follow-up chips never `COMP_*` or a bare id. |
| `persistent-memory/2026-09-20-0552-slim-get-company-desc.md` | Overnight: short `get_company` tool description; same behaviour. |
| `persistent-memory/2026-09-20-0553-slim-get-alerts-desc.md` | Overnight: short `get_alerts` tool description; unscoped refuse unchanged. |
| `persistent-memory/2026-09-20-0552-trajectory-mora-not-inflow.md` | Overnight: cite trajectory; mora ≠ falta de entradas. |
| `persistent-memory/2026-09-20-0612-slim-tools-catalog.md` | Overnight: always-on tools catalog is one short block per tool. |
| `persistent-memory/2026-09-20-0633-wording-scope-pointer.md` | Overnight: WORDING 13–16 is a SCOPE pointer; no markdown tables. |
| `persistent-memory/2026-09-20-0638-period-lead-by-guard.md` | Overnight: period lead groups by guard; one quote per empresa. |
| `persistent-memory/2026-09-20-0640-period-history-scores-only.md` | Overnight: period `score_history` is scores only; no grupo. |
| `persistent-memory/2026-09-20-0642-period-no-coda-validation.md` | Overnight: period stops after bullets; AUROC only in monitor. |
| `persistent-memory/2026-09-20-0644-opening-no-group.md` | Overnight: first step has no `get_group` unless asked. |
| `persistent-memory/2026-09-20-0646-session-no-comp-tokens.md` | Overnight: SESSION series speak Empresa; tool errors in Spanish. |
| `persistent-memory/2026-09-20-0648-period-no-confidence.md` | Overnight: period payload has no confidence; never «En 0011». |
| `persistent-memory/2026-09-20-0650-slim-alert-reasons.md` | Overnight: alert payloads keep two reasons; no volume lecture. |
| `persistent-memory/2026-09-20-0652-peers-not-company-only.md` | Overnight: pares/grupo questions are not company-only. |
| `persistent-memory/2026-09-20-0654-period-no-summary.md` | Overnight: period has no closing summary; SCOPE omits `COMP_*`. |
| `persistent-memory/2026-09-20-0656-followups-owner-if-alert.md` | Overnight: follow-up owner chips only after `get_alerts`. |
| `persistent-memory/2026-09-20-0658-alert-months-long.md` | Overnight: alert months are `agosto 2026`, not `jul 2026`. |
| `persistent-memory/2026-09-20-0700-spanish-persistence-onsets.md` | Overnight: persistence and fiabilidad copy stay Spanish. |
| `persistent-memory/2026-09-20-0702-control-chart-spanish-blurb.md` | Overnight: control-chart blurb says Persistencia, not persistent. |
| `persistent-memory/2026-09-20-0704-spanish-schema-describes.md` | Overnight: tool schema describes are Spanish. |
| `persistent-memory/2026-09-20-0708-spanish-cluster-plots.md` | Overnight: cluster payload and plots catalog are Spanish. |
| `persistent-memory/2026-09-20-0710-wording4-forecast-es.md` | Overnight: WORDING 4 is one line; forecast payload is Spanish. |
| `persistent-memory/2026-09-20-0715-refuse-no-offtopic-name.md` | Overnight: refuse does not name the off-topic task; WORDING 6 slim. |
| `persistent-memory/2026-09-20-0720-product-es-session-labels.md` | Overnight: PRODUCT always-on is Spanish; SESSION has no `COMP_*`. |
| `persistent-memory/2026-09-20-0725-monitor-es-unsigned-eur.md` | Overnight: monitor layer is Spanish; bench flags signed `eur`. |
| `persistent-memory/2026-09-20-0115-thinking-no-verbose.md` | Brain on: drop the length cap. Do not ask for a longer reply. |
| `persistent-memory/2026-09-20-0120-thinking-max-exa.md` | Exa + `max` superseded. |
| `persistent-memory/2026-09-20-0130-drop-exa.md` | No Exa. No `search_web`. No `EXA_API_KEY`. |
| `persistent-memory/2026-09-20-0135-thinking-icon-high.md` | Brain on: `reasoning_effort=high`. Pensando chip like a tool row. Do not dump CoT. |
| `persistent-memory/2026-09-20-0140-tool-count-row.md` | Same tool kind = one row, count ticks `×1` `×2` `×3`. |

Plain-language method (not a journal file): `product/score/METHOD.md`.

## Night discovery (do not re-open)

Everything dated **2026-09-19 00:00–10:00** is the feature/Y hunt (families, literature seats, leftover tables). Closed by `persistent-memory/2026-09-19-0917-night-closed.md`. Harness: `overnight/`.

Do not add features, retrain, or revive Y3 from those notes. The 1130 map and the eight accepted outcomes are the residue.

## How to add a file

```text
.agents/persistent-memory/YYYY-MM-DD-HHmm-<slug>.md
```

Author, timestamp, what changed, decisions, still-unknown. Newest file wins on facts that can rot. Rules in **Still binding** win over a later “we tried X” note unless a newer still-binding line replaces them.
