# 2026-09-19-1500 — pivot to Vercel; freeze Streamlit; lock Watcher format

- **Author:** Walter (agent)
- **When:** 2026-09-19 ~15:00 CEST
- **Reconcile:** pulled `f75c424` (Ruben, sidebar hash focus). Discarded uncommitted `poc/` UI churn. Public sibling still retired.

## Decisions

- **Product surface is `product/web/` on Vercel.** `poc/` is frozen. Do not add Streamlit features. Keep the folder as reference until Ask/Watcher exist on Next.js, then it can go.
- **Prompts are files** under `product/web/src/lib/agent/prompts/`. Loader assembles them. No system-prompt strings in source.
- **Watcher month posts are a schema, not an LLM essay.** `watcher_format.md` + `watcher-post.ts` emit `line1` / `line2` / ≤4 bullets. This kills the Headline/Story/Act drift between generations. Opening three months are deterministic now; production will write the same JSON **offline** per company and per group on each monthly bundle (see `2026-09-19-1435-offline-month-briefs.md`). Live LLM = replies only.
- Tool calls in the UI: tools icon + `(tool_name)`.
- Agents retrieve from the bundle first; `query_clean_db` is the only path to invoices/transactions. SQL guard stays (SELECT, `clean.*`, LIMIT 200).

## Still unknown

- Helmcode key on the Vercel project (`HELMCODE_API_KEY`).
- Whether the 220 MB clean DuckDB ships with the deploy or stays a `CLEAN_DB_PATH` mount.
- Ruben adding Watcher/Ask/Grupos to the company-page sidebar vs a product-level nav.
