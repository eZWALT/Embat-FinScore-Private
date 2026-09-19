# 2026-09-19-1658 — Vigilancia in Resumen, Consultas as popup; Watcher locked to Javi’s Spanish rules

- **Author:** agent (Cursor / Walter)
- **When:** 2026-09-19 16:58 CEST

## What changed

- Pulled Javi’s `2026-09-19-1620-alerts-in-spanish.md`: titles, actions, reason sentences, `eur()` (`14 k€`), owners Tesorero / CFO / Cobros, persistence «3 de los últimos 4 meses». `kind` stays an English code.
- Watcher formatter (`watcher-post.ts`) now emits that production text in a fixed shape: línea 1 (empresa · score · trayectoria), línea 2 (`{kind} · {título} · {cifra}`), viñeta = acción de `routing.py`. No English dummy phrases.
- Prompts updated (`watcher_format.md`, `wording_rules.md`, `chat_system.md`, `sentinel_system.md`, `product_context.md`) so the model quotes Javi, never rewrites.
- Product: Vigilancia is a section of Resumen (`GET /api/watcher/feed`). Consultas is the floating panel on Resumen and Índice, wired to `POST /api/ask` with company context and tool chips. `/watcher` → `/#vigilancia`, `/ask` → `/?chat=1`. Extra Product tabs removed.

## Decisions

- Do not add Watcher/Ask as primary tabs (overbloat). Grupos stays its own map page.
- Dummy `/api/chat` (no tools, no company) is unused by the UI. The popup is the real Ask agent.
- Line 2 = fact from the rule; bullet = Javi’s action. Same five kinds only.

## Still unknown

- Neon `core` still unmounted for record SQL. `HELMCODE_API_KEY` still required for live chat.
- Spanish alert text is in the export; Neon must be reloaded from the new bundle before production shows it.
