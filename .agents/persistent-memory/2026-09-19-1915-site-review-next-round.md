# Hosted site review — next-round list only

- Author: agent
- Timestamp: 2026-09-19 19:15 +02:00
- Not still-binding. Do not start these until Walter asks.

Walked https://hack-spain.vercel.app/ (Rápido Mejores 5 + Pregunta, Profundo COMP_0462 / COMP_0202, Índice, Vigilancia, `/grupos?group=GROUP_0194`). Production was still the pre-Sentinel deploy.

## Next round (do not start)

1. Rápido copy still says «hoy» (`Mejores 5` / `Peores 5`) while the cut-off is agosto 2026.
2. Switching Rápido → Profundo drops you on the sidebar default (`COMP_0462`), not a company you were looking at.
3. Vigilancia first-paints «Aún no hay fichas…» then the three months appear. Use a skeleton, not an empty lie.
4. Only `COMP_*` / `GROUP_*` on screen. Search says «por nombre». Demo needs a readable label even if the id stays.
5. Same reason line mixes `k€` and `mil AED` (COMP_0202). Follow Javi money format in the company currency.
6. Raw tokens leak: alert chip `guard`; kind ids in the UI.
7. Pregunta dumps markdown tables; `AgentMarkdown` does not render tables. Ban tables in ROLE or render them.
8. Locale mix: `6.9` vs `6,9`, `+1.4` vs `+1,4`.
9. `/grupos` still lectures «Índice explicable y monitorizable». Theme control is two buttons, not the header icon.
10. Two «Mostrar u ocultar el menú» buttons. «53% cobertura» next to confidence is unexplained.
11. Popup covers the right side of the chart you are asking about.
12. `/ask` chips are still English.

## Still unknown

Whether Vercel production coalesces the new `/chat/completions` SSE after deploy.
