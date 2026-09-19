# 2026-09-19-0018 — brief back in the loop; product out

- **Author:** agent (Cursor)
- **When:** 2026-09-19 00:18 CEST

## What changed

Walter: do not kill runners, cap 4, queue relaunch; keep diverse data / models /
FE; morning summary + dashboards; SHAP. Then: **keep re-reading the original
brief**; **leave product out** — only analysis, features, explainability.

## Decisions

- Working copy of the X Ray brief is `overnight/NORTH_STAR.md` (live artifact
  is Cloudflare-gated). Six questions + trajectory + hidden 72 + no 0–100.
- `product/` stays frozen. No score formula in `analysis/`.
- Cap 4, FIFO `overnight/QUEUE.md`. Never interrupt a runner to make room.
- Y3 SHAP is parent-owned (`analysis/models/explain_y3.py`) so morning has
  explainability even if a child dies.
- Canvas: `xray-night-analysis.canvas.tsx`. Prose: `overnight/dashboards/CONTEXT.md`.

## Still unknown

Whether the live artifact has extra headings we did not capture on day 1
(`Qué ponemos nosotros`). Re-fetch if Cloudflare lets us through.
