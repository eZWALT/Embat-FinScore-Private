# 2026-09-19-1425 — Watcher posts last 3 months, not "nothing new"

- Author: Walter (agent)
- Timestamp: 2026-09-19 14:25

## What changed

The Watcher said "nothing new" after the first check because the novelty engine (`compute_novelties` + `mark_delivered`) treats a static bundle as a live feed: once alerts are marked delivered, every later check is empty. That is the wrong loop for a demo on one export.

`#sentinel` now opens with **three month briefs** (oldest → newest) for the watch set: snapshot + alerts dated that month. Quiet months still get a brief (scores / trajectories), never "nothing new" as a dead end. Tone: high-level for a CFO, with the score, €, owner and action. Reply in the thread is unchanged.

UI cut: Watcher keeps watch-set pickers + Clear + chat. Removed replay as-of, advance month, auto-check, check now, forget delivered, per-alert dump, reliability footnote, tool expander. Ask: Company/Group only, **2** suggested questions each.

## Decisions

- Last 3 calendar months of the bundle as-of, not "3 latest alerts" across time (a company can have one alert in 3 months; a month can have several). One post per month.
- One LLM call for all three briefs (`=== YYYY-MM ===` splitter). Deterministic fallback if the model is down or the split fails, so the channel always has content.
- Novelty helpers stay in `watcher_state.py` unused by the page. Re-enable when there is a new bundle drop, not for clicking Check on the same file.

## Still unknown

- Whether a live hourly ping is worth wiring before a second bundle exists.
- Whether 3 sequential LLM calls would write better briefs than one call (latency vs quality).
