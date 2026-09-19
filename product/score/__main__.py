"""Fit the v0 dummy card on train and write monthly scores.

    PYTHONPATH=. python -m product.score
"""
from __future__ import annotations

import json

from analysis.evaluate.protocol import load_holdout
from product.score.score import (
    OUT_DIR,
    derive_columns,
    fit_ref,
    load_store,
    save_ref,
    score_panel,
    summarize,
)


def main() -> None:
    holdout = load_holdout()
    panel = derive_columns(load_store())
    ref = fit_ref(panel, holdout)
    save_ref(ref)
    scored = score_panel(panel, ref)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "monthly_scores.parquet"
    scored.to_parquet(out, index=False)
    summary = summarize(scored, holdout)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"wrote {out} rows={len(scored)} companies={scored.company_id.nunique()}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
