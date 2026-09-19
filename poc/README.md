# poc/

Streamlit shell to iterate on the assistant and the group view before they move to `product/` (Vercel + DB, Ruben).

```sh
pip install -r poc/requirements.txt
python -m analysis.features.build_feature_store   # once; writes data/feature_store/monthly.parquet
streamlit run poc/app.py                          # from the repo root
```

| Page | What it shows now |
|------|-------------------|
| Overview | What the product is, what is real and what is pending |
| Sentinel | Chat interface only. Behaviour, tools and model are not defined |
| Portfolio | Group → companies (v0 `score_3m`, trajectory, runway) → company charts |

The 0–100 is computed by `PYTHONPATH=. python -m product.score` and joined here. Rebuild that parquet after card changes. Sentinel is still an empty chat.
