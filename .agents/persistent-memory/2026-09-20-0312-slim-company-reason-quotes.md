# Pregunta: slimmer get_company + reason quotes in the UI

- Author: agent
- Timestamp: 2026-09-20 03:12 +02:00
- Still-binding: On `agent/overnight-optimize`. `get_company` omits null country/erp/slopes, drops `items` when `reasons` exist, and only adds `trail_months` if under 12. The chat shows up to two `sentence` quotes per company under the tool row (max 4).

## Bench (keep)

period-4: 4 tools / q 6 / 9.8 s (main 11 / q 2 / 17 s). why-score still 1 / q 6.
