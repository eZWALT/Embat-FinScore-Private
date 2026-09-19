# 2026-09-19-1435 — production: last 3 Sentinel messages are offline

- **Author:** Walter (agent)
- **When:** 2026-09-19 ~14:35 CEST

## Decision

In **production**, the Watcher must **not** call the LLM to write the opening channel. The last **three calendar-month briefs** (oldest → newest, ending at bundle `as_of_month`) are **computed a priori, offline**, in the same monthly job that builds the export bundle. The UI only **reads** them. Live model stays for **thread replies** only.

This applies to **every company and every group**, not to an arbitrary watch-set combination. A multi-entity channel composes from those per-entity briefs. When a new CSV drop lands, the three briefs are rewritten for that as-of; older months fall off the window.

## Why

The POC (`poc/views/watcher.py` `run_briefs`) generates the three posts on first paint of a watch set (one LLM turn, spinner, 20–46 s). That is acceptable to iterate prompts. It is not a product: first open must be instant, and 1,286 companies + 250 groups cannot wait on a live call.

The deterministic payload already exists (`watcher_state.month_brief` / `last_n_month_briefs`). Offline = run the Sentinel prompt (or the deterministic fallback) **once per entity per month** at export time, persist the prose + plot specs next to the bundle, serve them.

## Contract (formal)

| | |
|---|---|
| **Unit** | One entity: `company_id` or `group_id` |
| **Artifact** | Exactly 3 messages, one per calendar month, `as_of-2`, `as_of-1`, `as_of` |
| **When** | Batch, after the bundle for that drop is valid; same cadence as the score (monthly) |
| **Who writes** | Offline Sentinel (same wording rules, same payload). Fallback brief if the model fails, so a drop never ships an empty channel |
| **Who reads** | Watcher first paint; optional seed for Ask / company panel. No LLM on open |
| **Who is still live** | User replies in the thread |
| **Not precomputed** | Arbitrary watch-set mashups (combinatorial). Compose from the per-entity three |

Do not implement the batch writer in this POC unless asked. The spinner path stays for local iteration.

## Still unknown

- Storage: extra files in the bundle (`briefs/{company_id}.json`) vs a side table next to the clean DuckDB.
- Whether a group brief is a synthesis of members or only group-level alerts/mean (group `reasons` are empty today).
- Whether plot specs are stored or rebuilt deterministically from the bundle at read time.
