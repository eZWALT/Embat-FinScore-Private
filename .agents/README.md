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

Plain-language method (not a journal file): `product/score/METHOD.md`.

## Night discovery (do not re-open)

Everything dated **2026-09-19 00:00–10:00** is the feature/Y hunt (families, literature seats, leftover tables). Closed by `persistent-memory/2026-09-19-0917-night-closed.md`. Harness: `overnight/`.

Do not add features, retrain, or revive Y3 from those notes. The 1130 map and the eight accepted outcomes are the residue.

## How to add a file

```text
.agents/persistent-memory/YYYY-MM-DD-HHmm-<slug>.md
```

Author, timestamp, what changed, decisions, still-unknown. Newest file wins on facts that can rot. Rules in **Still binding** win over a later “we tried X” note unless a newer still-binding line replaces them.
