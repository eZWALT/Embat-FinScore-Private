# poc/

Streamlit shell to iterate on the two agentic parts of Health Sentinel (and a portfolio view) before they move to `product/` (Vercel, Ruben). Reads the **export bundle** and the **clean DuckDB**, never the raw CSVs; never recomputes a score.

```sh
pip install -r poc/requirements.txt
PYTHONUTF8=1 PYTHONPATH=. python -m product.score.export --csv-folder data --out data/bundle   # ~50 s, full bundle (gitignored)
export HELMCODE_API_KEY=...                                    # or HELMCODE_API_KEY_FILE=~/.helmcode_key
python3 -m streamlit run poc/app.py --server.port 8601        # from the repo root; `streamlit` shim may point elsewhere
```

| Page | What |
|------|------|
| Overview | What the product is; bundle and model in use |
| Watcher | Push. Pick companies/groups; every `POC_WATCH_INTERVAL_SEC` (3600) it looks for novelties (new alerts, material score moves, trajectory/guard changes) and writes a digest with plots. Replay control moves the as-of month |
| Ask | Pull. Chat with tools over the bundle (`get_company`, `explain_change`, `get_group`, `get_alerts`, `get_control_chart`, `compare_with_cluster`, `get_forecast`) and read-only SQL over `clean.*` (`query_clean_db`), plus `plot_series` |
| Portfolio | Group → members → company: score history with alerts, reasons with €, categories, control chart |

## Layout

| File | Role |
|---|---|
| `llm.py` | Helmcode (OpenAI-compatible) via LangChain. `HELMCODE_API_KEY[_FILE]`, `HELMCODE_BASE_URL`, `POC_LLM_MODEL` (default `deepseek-v4-flash`) |
| `bundle.py` | Bundle loader. `POC_BUNDLE_DIR` → `data/bundle` → `product/score/sample_bundle` |
| `db.py` | Read-only DuckDB, `clean.*` only, SELECT only, LIMIT 200. `POC_CLEAN_DB` → `data/clean.duckdb` → `data/embat.duckdb` |
| `agent/prompts/*.md` | The context: product semantics, wording rules, clean schema, role prompts (sentinel, chat) |
| `agent/context.py` | Assembles the system prompt from the prompts + the bundle manifest (spec, disclaimer, alert stats) |
| `agent/tools.py` | LangChain tools + plot registry |
| `agent/runner.py` | Tool-calling loop; returns answer, new messages, plots, tool calls |
| `plots.py` | Altair renderer for plot specs |
| `watcher_state.py` | Novelty engine and per-watch-set state (`poc/.state/`, gitignored) |
| `views/` | One file per page |

Wording and claims are fixed in `agent/prompts/wording_rules.md`: explainable and monitorable, never predictive; "top customer stopped billing, review exposure and collections".
