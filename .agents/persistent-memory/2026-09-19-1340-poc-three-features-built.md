# 2026-09-19-1340 — three features built: Watcher, Ask (Streamlit POC) and `/grupos` (Next.js)

- **Author:** Walter (Cursor agent orchestrating three sub-agents)
- **When:** 2026-09-19 ~13:40 CEST. Follows `2026-09-19-1330-poc-agents-context-and-core.md`.

## What exists now

| Part | Where | Verified |
|---|---|---|
| Watcher (push) | `poc/views/watcher.py`, `poc/watcher_state.py` | Replay on 3 companies + 1 group: first run at 2026-06 delivered a `going_dark` act alert and a top-customer exposure; advancing to 2026-07 caught a +40 score move (dark cap lifting), a trajectory change, a new deterioration alert; re-check → "Sin novedades", no LLM call. Real digest: 20–46 s, Spanish, stats quoted, one plot. |
| Ask (pull) | `poc/views/ask.py` | Live model: "qué ha cambiado" → 5 tool calls + plot; "qué clientes tienen facturas vencidas" → 5 guarded `query_clean_db` calls (€3.55M open, 1,133 invoices); context switch keeps chat, reset clears. |
| Group Health Map | `product/web/src/app/grupos/`, `components/group-health-map.tsx`, `components/group/*`, `lib/data/group-service.ts`; additive extension of `types.ts` and `local-bundle-repository.ts` (`listGroups`, `getAlerts`) | Full bundle: GROUP_0142 heatmap 23 rows × 22 months, row/cell click updates URL and company panel, group alert shown for GROUP_0165. Sample bundle: defaults to largest sampled group. `pnpm lint` + `pnpm build` pass; `/grupos` is dynamic (uses searchParams). |
| Portfolio (POC) | `poc/views/portfolio.py` | AppTest headless: renders, company switch OK. |

Run: `python3 -m streamlit run poc/app.py --server.port 8601` with `HELMCODE_API_KEY` set; full bundle at `data/bundle` (regenerate with `python -m product.score.export --csv-folder data --out data/bundle`).

## Decisions

- Watcher interval: `POC_WATCH_INTERVAL_SEC` (3600). Novelty = not yet delivered to this watch set; as-of replay for the demo. Delivery marked only after the digest (or deterministic fallback) rendered. `info` alerts collapsed (Silent), `watch`/`act` shown (Guided).
- Ask keeps the last 12 LangChain messages without splitting tool-call pairs. Tool calls shown in an expander for transparency.
- `/grupos` has **no link from Ruben's sidebar** (off-limits file); reachable by URL and with a back-link to `/`. Ruben to add the nav entry.
- Disclaimer on `/grupos` is `manifest.disclaimer` verbatim (English); the existing dashboard hard-codes a Spanish text. One of the two should give.

## Environment notes

- `pnpm` is not global here; `corepack pnpm --config.minimum-release-age=0 <cmd>` (pnpm 12 rejects two lockfile entries published yesterday). Lockfile not changed.
- The IDE browser tab was not available to sub-agents; verification used Streamlit `AppTest` and headless Chrome + CDP.
- A running `next dev` rewrites `product/web/next-env.d.ts`; discard it before rebasing.

## Still unknown

- Hosting for the agents (they need Python + the bundle + the clean DuckDB): small service or Streamlit Cloud beside the Vercel app.
- Whether the Watcher should push somewhere real (email / Slack / TellMe) or stay in-app for the demo.
- Group-level "why" for the map: `groups.json` has means and charts but no reasons; the Ask agent covers it via members.
