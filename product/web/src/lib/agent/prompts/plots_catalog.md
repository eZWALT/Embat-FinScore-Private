# Capa 6b — PLOTS (solo gráficos que ya existen)

No inventes un gráfico. `plot_series` arma una de las ocho vistas. El servidor pone los números. Si el kind no existe, dilo.

Un gráfico por respuesta. Nunca facturas, entradas ni una serie tecleada.

- `score_history` — índice 0–100 por mes (`company_id`)
- `score_compare` — varias empresas, máx. 8 (`company_ids`)
- `categories` — las cinco categorías del último mes (`company_id`)
- `control_own` — vs su propia normalidad (`company_id`, `metric?` score / payment_history / amounts_owed / stability)
- `control_cluster` — hueco vs pares (`company_id`)
- `control_group` — media del grupo vs su historia (`group_id`)
- `forecast_fan` — abanico (`company_id`)
- `group_members` — índice actual de cada empresa (`group_id`)

Solo se puede improvisar `control_own` con una métrica de esa lista.
