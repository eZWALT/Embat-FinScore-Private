# Pregunta: opening tools + records schema only on demand

- Author: agent
- Timestamp: 2026-09-20 02:31 +02:00
- Still-binding: On `agent/overnight-optimize`. First model step only offers `get_company`, `get_alerts`, `get_group`, `plot_series` (plus `query_clean_db` if the user asked about invoices/debt). `clean_schema.md` is not in the system prompt unless that records regex matches. Same tool catalog after step 1.

## Bench (keep vs main sample suite)

period-4: main 11 tools / q 2 → candidate 5 / q 6 (four `get_company`, no month, one `get_alerts`). why-score still 1 tool / q 6, shorter text. Bundle `items` fill in when Neon extras are missing.
