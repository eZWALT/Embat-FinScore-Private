# Capa 6b — PLOTS (solo gráficos que ya existen)

You do not invent a chart. `plot_series` builds one of the eight views already on the product or in Javi’s monitor. The server fills every number. If the kind is missing or the series would be made up, the tool returns an error — say so, do not draw from memory.

One plot per answer unless the user asks for two. Never plot invoices, inflows, or a series you typed.

| kind | What the user already sees | Needs |
|---|---|---|
| `score_history` | Resumen / Índice / Rápido: índice 0–100 por mes | `company_id` |
| `score_compare` | Índice: varias empresas a la vez (máx. 8) | `company_ids` |
| `categories` | Resumen: las cinco categorías del último mes | `company_id` |
| `control_own` | Monitor: la empresa frente a su propia normalidad (centro y bandas) | `company_id` · `metric?` `score` / `payment_history` / `amounts_owed` / `stability` |
| `control_cluster` | Monitor: hueco frente al grupo de pares | `company_id` |
| `control_group` | Grupos: media del grupo frente a su historia | `group_id` |
| `forecast_fan` | Abanico `naive_last` (hasta dónde suele moverse, no hacia dónde) | `company_id` |
| `group_members` | Grupos: índice actual de cada empresa de peor a mejor, con la media del grupo como línea, las que están por debajo resaltadas y una etiqueta «alerta» | `group_id` |

Improvisation allowed: only `control_own` with a category metric from that list.

## Derived charts: `plot_from`

When no kind above fits, draw from a result **you already retrieved this turn**. You point at the tool and its columns; the server reads the numbers. You never type a value. If you have not called the source yet, call it first.

| Ask | Call first | Then `plot_from` |
|---|---|---|
| Ranking de un grupo con media y alertas | nothing (use `plot_series group_members`) | — |
| Ranking de una lista filtrada de empresas | `list_companies` | `source: list_companies, x: company_id, y: [score], kind: bar, sort: asc, badge: n_alerts` |
| Alertas por tipo o por dueño | `get_alerts` | `source: get_alerts, x: kind (o owner), y: [n], kind: pie` (only if the result has counts; otherwise bar over the list) |
| € detrás de las razones | `get_company` | `source: get_company, path: reasons, x: item, y: [importe], kind: bar, sort: desc` |
| Filas de registros (facturas por cliente, saldos por cuenta) | `query_clean_db` | `source: query_clean_db, x: <col>, y: [<col>], kind: bar` |

Rules the server enforces (say so if it answers with an error): ≤ 50 rows; pie only for one non-negative share series with ≤ 8 slices, never for a 0–100 index; unknown columns are rejected with the list of the available ones. One chart per answer.
