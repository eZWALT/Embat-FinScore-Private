# Viz agent: how model-written Recharts gets on the screen safely

For whoever builds the chat that draws in `product/web` (the Ask and Watcher-reply chat that already exists). It lives in `product/agent/` on purpose: it is not part of the `poc/` prototype and not inside the web app; copy or wire what you need.

| File | What |
|---|---|
| [`prompts/viz_system.md`](prompts/viz_system.md) | The LLM-facing prompt: how to write a chart, the comparisons to make, 12 tested recipes, rules |
| [`prompts/api_schema.md`](prompts/api_schema.md) | The scores database the agent queries for chart data (Neon `api` views) with a SQL cookbook |
| [`neon_chart_views.sql`](neon_chart_views.sql) | **Proposed** migration `007`: tidy views (control chart rows, forecast rows, items, categories, group series, cluster percentiles). Not applied; syntax-checked with pglast only |
| [`sandbox/`](sandbox/README.md) | Reference lint + compile + dry run + runtime helpers + tests. `npm install && npm test` |

## Decision

The agent **writes React + Recharts code** (one component per chart), not a JSON spec. The goal is a very powerful tool that can draw what nobody planned (bump charts, Pareto, heatmaps, small multiples, custom comparisons); a fixed spec caps that at the kinds the renderer implements. Recharts is the web app's own chart library and models write it well. The existing `plot_series` (line/bar from arrays the model retypes) stays for the Watcher sparkline and becomes redundant for chat.

| Risk of model-written code | Control |
|---|---|
| Runs in the user's browser: data exfiltration, DOM access, network | **Sandboxed iframe, opaque origin, no network (CSP `connect-src 'none'`), no storage, no access to the page**; data in by `postMessage` only. This is the real boundary |
| Code that does not compile or crashes | Server lint + compile (sucrase) before it is sent; render errors reported back to the model (see "Feedback loop"); max 3 attempts |
| Off-brand charts, broken dark mode | Colours only through `tones` / `scoreColor` / CSS variables; shared `look`; the host draws title, subtitle and note |
| Wording and evidence rules (predice, probabilities, `rank_score`, "revenue at risk") | The lint scans the chart text **and the strings in the code**; the prompt states the rules; the SQL tool must refuse `rank_score` |
| Runaway size or time | 14 KB of code, 6 datasets, 2,000 rows each |
| Wrong numbers | The model never types data: it reads `data.ds_N`, which a SQL tool filed; derived display columns are computed in code from those rows |

## What exists today (read from `origin/main` at `1258be6`)

| Piece | Where | Note |
|---|---|---|
| Web app | `product/web`: Next 16, React 19, Tailwind 4, shadcn, **Recharts 3.8**, Spanish UI (Ruben) | `ChartContainer` in `components/ui/chart.tsx`; labels in `components/group/labels.ts` |
| Agent runtime | `product/web/src/lib/agent/`: Vercel **AI SDK 7** (`ai`, `@ai-sdk/react`, `@ai-sdk/openai` → Helmcode, model `deepseek-v4-flash`), `streamText` with a tool loop (`isStepCount(8)`), zod tool schemas, `POST /api/ask` and `/api/watcher/reply` (Walter) | `runtime.ts`, `tools.ts` (list_companies, get_company, explain_change, get_group, get_alerts, get_control_chart, compare_with_cluster, get_forecast, query_clean_db, plot_series), `chat-parts.ts` (`plotsFromParts`), `prompt-loader.ts` |
| Chat UI | `components/agent-chat.tsx` (`useChat`), `agent-plot.tsx` (`AgentPlot`: one Recharts `LineChart` from a `PlotSpec`) | Charts today: a line chart of `plot_series` output |
| Prompts | `src/lib/agent/prompts/*.md`; the `chat` role = `chat_system.md` + `product_context.md` + `wording_rules.md` + `watcher_format.md` + `clean_schema.md` | `chat_system.md` step 4 tells the model to use `plot_series`: replace it when `viz_system.md` is added |
| Data | **Neon Postgres** (`infra/neon`): `analytics.*` tables, `api.*` read views over the current run, `core.*` records. The app reads it with `@neondatabase/serverless`; no bundle fallback | Analytics loaded (1,286 companies, 19,658 company-months). **Records (`core.transactions`, `core.invoices`) not loaded** (Free tier size): `query_clean_db` answers `records not mounted` |
| Palette | `--chart-1..5` are greys in `globals.css` (`d0de855` polished the health-score chart; check the current values) | Multi-series charts need real hues |

## Integration in `product/web` (in order)

1. **Data views.** Apply `neon_chart_views.sql` as `infra/neon/migrations/007_chart_views.sql` (it ends with the `GRANT`, needed because `006` granted on the tables that existed then). Verify the semantics on staging: the file was only syntax-checked.
2. **SQL tools that return datasets.** In `tools.ts` add `query_api(sql)`: `checkSql` (SELECT only), **only `api.*`**, `LIMIT` at most 2,000 (add one if missing), and **reject any query that names `rank_score` or selects `*` from `api.alerts`** (that view carries it; it must never reach a chart). Make `query_api` and `query_clean_db` file their rows in a **per-request `DatasetStore`** (a `Map` created next to `chatTools()`; a turn's tool loop runs inside one request, so it is enough) and return `{dataset_id, rows, columns, preview}` (preview at most 5 rows) instead of the whole result. Dataset ids only live for the turn.
3. **`render_chart` tool** (zod: `title`, `subtitle?`, `note`, `height?`, `datasets: string[]` (max 6), `code`). `execute` on the server: `lint` and `compile` (both in `sandbox/`, plain JS, no jsdom needed), resolve the datasets from the store, and return `{ok:true, chart:{title, subtitle, note, height, js, data}}` (like `plot_series` returns `{plot}`), or `{ok:false, stage, errors}` for the model to fix.
4. **Prompts.** Add `viz_system.md` and `api_schema.md` to `src/lib/agent/prompts/` and to `prompt-loader.ts` (chat role: `viz_system` instead of the plotting step of `chat_system.md`, and `api_schema.md` beside `clean_schema.md`). Extra tokens are fine; the prompt is about 11k.
5. **UI.** `chartsFromParts` (like `plotsFromParts`) and `<SandboxedChart>` in `agent-chat.tsx` (contract below). Keep `AgentPlot` for the Watcher spark.
6. **Theme.** The CSS variables below, including eight real series colours.

## Feedback loop when the chart fails to render

The lint and compile catch syntax and forbidden code. Runtime errors and layout problems only show when the chart runs, and that happens in the user's browser. Choose one:

- **A (simplest, do this first).** The frame reports errors to the host; the host shows "No se pudo dibujar este gráfico" and a button that sends the error text to the chat as a user message; the model then fixes it. One click, no extra infrastructure.
- **B (automatic).** Make `render_chart` a **client-side tool** in the AI SDK (a tool whose result is produced in the browser and sent back to the model): the browser mounts the frame, waits for `ready` or `error`, and returns that as the tool result, so the model fixes errors on its own inside the turn. The v7 client-tool API names differ from earlier versions (this repo's `product/web/AGENTS.md` warns about that): read the SDK docs in `node_modules` before writing it.
- **C.** Run the dry run on the server (`sandbox/dryrun.mjs`: jsdom, Recharts at a fixed width). It works, but jsdom is heavy for serverless and has no layout, so keep it for CI and local testing.

## Browser sandbox contract

`<SandboxedChart title subtitle note height js data />` renders a shadcn `Card` with the title, subtitle and note drawn by the host and an `<iframe>` for the chart:

- `sandbox="allow-scripts"` and **nothing else**: no `allow-same-origin` (opaque origin: no cookies, storage or access to the parent), no popups, forms, downloads or top navigation.
- Content from `srcdoc` (or a static route on another origin) with a meta CSP:
  `default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' <runtime-url>; style-src 'unsafe-inline'; img-src data:; font-src 'none'; connect-src 'none'; frame-src 'none'; base-uri 'none'`.
  `'unsafe-eval'` is needed to evaluate the model's compiled module; acceptable **only** because the frame has no origin, no network and no storage.
- A **pinned runtime bundle** (esbuild, static file, SRI hash): React 19.3.0, Recharts 3.8.0 (the versions in `product/web/package.json`) and the `sentinel` module. `sandbox/runtime.mjs` is the reference of that module: `fmt`, `tones`, `look`, `labels` (Spanish, same wording as `labels.ts`), `scoreColor`, `divergingColor`, `ChartContainer` (responsive; fixed width only in the dry run).
- A module loader: `require("react" | "recharts" | "sentinel")` only; anything else throws. `React` is a global for JSX.
- `postMessage`, always checked with `event.source === iframe.contentWindow`: host → frame `{type:"render", js, data, height, theme:{vars, mode}}` (and again on theme change); frame → host `{type:"ready"}`, `{type:"height", value}` (ResizeObserver), `{type:"error", message}`.
- An error boundary in the frame; a "Ver código" toggle; keep `code` and the dataset ids per chart (audit log).

### Theme variables the host defines in the frame

`--background --foreground --popover --popover-foreground --muted --muted-foreground --border` (from the app), plus `--sentinel-risk` (red), `--sentinel-opportunity` (green), `--sentinel-info` (blue), `--sentinel-neutral`, `--sentinel-accent` (amber) and `--chart-1` … `--chart-8` (eight categorical colours, light and dark). `sandbox/preview.mjs` defines a working light and dark set.

## `render_chart` (the tool)

```jsonc
{
  "title": "Score frente a su normalidad",            // required
  "subtitle": "Mediana de sus 12 meses anteriores y límites de control",
  "note": "Fuente: base de scores a 2026-08. Alerta = caída persistente y material frente a su propio historial.",   // required
  "height": 300,                                      // px, optional (120-700)
  "datasets": ["ds_1"],                               // ids returned by query_api / query_clean_db this turn, at most 6
  "code": "import { … } from \"recharts\"; … export default function Chart({ data, height }) { … }"
}
```

Errors are messages a model can act on ("import from \"lodash\" is not available; only react, recharts and sentinel", "dataset ds_1 has 2001 rows; the limit is 2000. Aggregate in the tool.", "the output contains NaN/undefined/Infinity: check the columns you read and how you handle null values").

## Tests (`product/agent/sandbox`)

`npm install && npm test` (Node 20+): compiles and renders every ```` ```jsx ```` block of the prompt with Recharts 3.8 in jsdom against fixtures built from the committed sample bundle (real output; the column names match the SQL cookbook aliases) plus synthetic record-level rows, and checks that the lint rejects network access, `window`, URLs, foreign imports, forbidden wording, `rank_score`, a missing default export, syntax errors, runtime errors, a missing `note` and oversized datasets. `node preview.mjs out.html [dark]` writes all recipes into one page for a visual check (jsdom has no layout, so tick thinning and legend placement look better in a real browser). Not tested: the SQL against Neon (only parsed), the browser frame, the client-side feedback.

Known Recharts details baked into the recipes: a fixed axis domain is extended to fit the data unless `allowDataOverflow` is set (control limits can fall outside 0-100); `name="undefined"` on bar paths is harmless; range areas take an `[low, high]` array as `dataKey`.

## Open points

- `core` records are not loaded, so record recipes (Pareto of open invoices, cash flows) return `records not mounted` until they are (`infra/neon/README.md`).
- The `chat_system.md` and `product_context.md` in `product/web/src/lib/agent/prompts` say "16 items"; the scorecard has **17** (`spec.items` in the run). Fix both copies (and `poc/` if kept).
- Companies without a group, groups of 1-2 companies and companies with fewer than 7 scored months return empty chart queries; the prompt tells the model to say so.
- The Spanish labels in `sentinel.labels` duplicate `labels.ts`; keep them in sync, or import one from the other when the runtime is built.
