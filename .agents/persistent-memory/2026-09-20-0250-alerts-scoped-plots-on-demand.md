# Pregunta: scope get_alerts; plots catalog on demand

- Author: agent
- Timestamp: 2026-09-20 02:50 +02:00
- Still-binding: On `agent/overnight-optimize`. `get_alerts` without `entity_id` is clipped to the session company/group plus `Empresa NNNN` / `COMP_` / `GROUP_` in the question — never the whole feed. `plots_catalog.md` and `plot_series` load only if the question mentions a chart or a dragged period.

## Bench (keep)

period-4: 4 `get_company` (no month, no unscoped alerts), q 6, 12.9 s (main 11 tools / q 2). why-score still 1 / q 6.
