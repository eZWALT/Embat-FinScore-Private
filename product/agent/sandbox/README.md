# Viz sandbox: reference dry run for model-written Recharts

What the server does with a chart the agent wrote, before anything reaches the browser (contract and architecture: [`../VIZ_SPEC.md`](../VIZ_SPEC.md), prompt: [`../prompts/viz_system.md`](../prompts/viz_system.md)).

```bash
npm install
npm test                      # every jsx block of the prompt + negative tests
node preview.mjs out.html     # all recipes on one page (add `dark` for the dark theme)
```

| File | What |
|---|---|
| `runtime.mjs` | The modules a chart may import: `react`, `recharts`, and `sentinel` (`fmt`, `tones`, `look`, `labels` in Spanish, `scoreColor`, `divergingColor`, `ChartContainer`). The browser runtime bundle must expose the same names |
| `lint.mjs` | Banned APIs and imports, wording rules (also on strings in code), `rank_score`, size and dataset limits, `note` required |
| `dryrun.mjs` | lint → compile (sucrase) → evaluate the module in a `vm` context → render in jsdom at 640 px → look for errors, `NaN`, empty output. Returns `{ok, stage, errors[], warnings[]}` |
| `fixtures.mjs` | Datasets shaped like the data tools' rows, built from `product/score/sample_bundle` plus synthetic record-level ones |
| `test.mjs` | Runs the prompt's recipes and the negative cases |
| `preview.mjs` | Static page of all recipes with the theme variables |

Not a security boundary: run the dry run in a worker or child process without network in production. Pinned to the versions in `product/web/package.json` (React 19.3.0, Recharts 3.8.0).
