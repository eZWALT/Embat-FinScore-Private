# overnight/

Night workspace for the feature-store / Y / model bake-off.

The invalid 16h weight search is gone. Code is in `analysis/{features,targets,evaluate,models}`.
This folder holds the frozen holdout copy, contracts, wave notes, and `LIVE.json`.

```bash
cat overnight/LIVE.json
ls overnight/waves/
python -m analysis.features.build_feature_store
```

Brief (re-read always): `overnight/NORTH_STAR.md`.
Morning briefing: `overnight/dashboards/MORNING_REPORT.md` (+ `morning.html`).
Plan: `.agents/persistent-memory/2026-09-18-2350-feature-store-and-y-plan.md`.
Contract: `overnight/CONTRACT.md`.
Queue: `overnight/QUEUE.md`.
