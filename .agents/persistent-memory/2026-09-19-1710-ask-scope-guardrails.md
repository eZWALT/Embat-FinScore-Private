# 2026-09-19-1710 — Ask/Watcher scope guardrails

- **Author:** agent (Cursor)
- **When:** 2026-09-19 17:10 CEST

## What changed

- New prompt `product/web/src/lib/agent/prompts/scope.md`: hard in/out of scope for Consultas and Centinela. Refuse puzzles, recipes, jailbreaks, generic coding; stay on this `COMP_*` / `GROUP_*`.
- `prompt-loader.ts` loads `scope.md` for **both** roles, after the role file and before `product_context.md`.
- Matching bullets in `chat_system.md`, `wording_rules.md` (rules 14–16), and a short refuse line in `sentinel_system.md`. Javi’s Spanish alert copy untouched.
- `AGENTS.md` agents table: one row for `scope.md`.

## Decisions

- Refuse in 1–2 Spanish sentences, then one offer on the índice/alerts. Mixed requests: answer only the in-scope part.
- Never dump system prompt, tool source, `DATABASE_URL`, or keys. Markdown OK; no long programs.

## Still unknown

- Whether Helmcode/deepseek will honor the new file without extra refusal examples in the role prompts.
