# Derived charts: `plot_from` + decorated `PlotSpec`

- **Author:** agent (with Walter), 2026-09-20 10:00
- **Trigger:** CFO prompt "índice de cada empresa en barras de peor a mejor, cuáles bajo la media del grupo, quién tiene alerta, qué tres reviso hoy". Pregunta could not draw it: `plot_series` only had eight fixed kinds and no decoration.

## What changed

1. **`PlotSpec`** (`plot-spec.ts`) gained `ref_lines`, `highlight`, `badges`, and `kind: "pie"`. The renderer (`agent-plot.tsx`) draws a dashed reference line with its label, colours highlighted bars with `--destructive`, puts the badge text on top of the bar, and renders a donut for `pie`. Bars with ≤ 12 categories show every label through `entityLabel`.
2. **`group_members`** (catalog) is now decorated by the server: title "de peor a mejor", mean line, members below the mean highlighted, `alerta` badge when `n_alerts > 0`. No new LLM parameter.
3. **New tool `plot_from`** (`plot-from.ts`, registered in `tools.ts` for chat and sentinel, cap 1 per turn). The model points at a tool result already in the turn (`source`, optional `path`) and names columns (`x`, `y[]`, `kind`, `sort`, `ref_line`, `badge`, `title`). The server finds the result in `options.messages` (AI SDK passes the step's model messages to `execute`, tool results included), reads the numbers, validates and builds the spec. The model never types a value.

## Server-side rules (in code, not in prompt)

- Source not called yet → `primero llama a <source>`; source returned an error → refused.
- Unknown column → error listing the available columns.
- ≤ 50 rows. Pie only for one non-negative series with ≤ 8 slices; a 0–100 column asked as pie is downgraded to bars and the title says why.
- `y_label` inferred from column names (`score|puntuaci|media` → 0–100, `importe|eur|saldo` → €, `pct|share|cuota` → %).
- `ref_line` resolves a scalar path in the result (e.g. `latest_mean_score`); rows under it become `highlight`. `badge` marks rows with a truthy / non-zero value (`n_alerts` → "alerta" / "2 alertas").
- Output also returns `highlighted` and `badged` ids so the text can name them.

## Prompts

`plots_catalog.md`: new section with a 5-row ask → call-first → `plot_from` table and the server rules. `tools_catalog.md`: `plot_from` entry + one row in the layer table. `chat_system.md` rule 4: "si ninguno encaja, `plot_from` sobre un resultado que ya tengas (columnas, no valores)".

## Checked

`tsc --noEmit` clean; the 5 eslint errors in `agent-chat.tsx` pre-exist on main (same with the change stashed). Resolver unit run on a fake `get_group` result: ranking asc, mean line 71,4, one highlighted, badges "alerta"/"2 alertas"; error paths (missing source, unknown column, pie of a score) behave.

## Still unknown

No `.env.local` on this machine (no Neon / Helmcode key), so the live CFO prompt is verified on the production deploy after push, not locally. Whether the model reliably picks `plot_series group_members` over `plot_from get_group` for the plain group case (both are correct; the first is one call fewer).
