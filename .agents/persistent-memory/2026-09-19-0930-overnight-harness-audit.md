# 2026-09-19-0930 — audit of the overnight harness (`overnight/`)

- **Author:** Javier Boix (with Claude Code)
- **When:** 2026-09-19 ~09:30 CEST
- **Scope:** read `compare.py`, `search_16h.py`, `watchdog.sh`; ran their own functions on this machine (Windows, `PYTHONUTF8=1`). No code changed.

## State

- Nothing had run on this machine: `runs/`, `cache/`, `candidates/` only had `.gitkeep` (all three are gitignored, so results from whoever launched the search are not in git). Ask them for `runs/LIVE.json`, `runs/found.jsonl`, `runs/search_16h.log`.
- `compare.py` / `search_16h.py` use `open()` without `encoding=`, so on Windows they crash on `transactions.csv` (cp1252). `PYTHONUTF8=1` works around it. `watchdog.sh` needs `pgrep` and `python3`, so it is Linux/macOS/WSL only.
- I built `overnight/cache/monthly_features.csv` locally (24,405 company-months, ~20 s).

## Measurements (holdout, `or`/0.20/0.80 labels, trial-1 weights 0.30/0.25/0.25/0.20)

| Score | stress AUROC | recover AUROC | primary | lead |
|---|---|---|---|---|
| trial-1 scorecard | 0.622 | 0.580 | **0.601** | 0.58 |
| same score, months shuffled within company | 0.608 | 0.587 | **0.597** | 0.48 |
| company-mean score (constant per company, no timing) | 0.626 | 0.614 | 0.620 | **5.39** |
| iid uniform | 0.492 | 0.494 | 0.493 | 0.46 |
| only `overdue_share` | 0.643 | 0.638 | 0.641 | 2.06 |
| only `net` | 0.535 | 0.423 | 0.479 | 1.15 |

Same score under other Y cutoffs: primary 0.607 (`or`/0.25/0.75), 0.712 (`and`/0.20/0.80, n_recover=46), 0.797 (`and`/0.15/0.85, n_recover=30).

## Findings

1. **The metric measures between-company level, not trajectory.** Shuffling a company's score across its own months keeps 0.597 of 0.601. The 0.57 bar is cleared by trial 1 and by a null with no timing information, so every "found" candidate is meaningless as evidence.
2. **Y and score share a source.** `overdue_share` is in the scorecard and in the stress label (`pay_bad`); `only overdue_share` is the best single feature (0.641). This is the circularity the README says to avoid.
3. **Not causal.** `overdue_share` buckets by issuance month using the invoice status at the 2026-09-01 snapshot (192,556 `overdue`, 29,717 `pending`). The score at month t uses information from after t. It also rises in the last months (unpaid share 0.275 for Jun-2026 issues, 0.362 for Sep-2026): right-truncation, the mirror of the left-truncation artifact in the 2210 entry.
4. **`lead_months` is maximized by a constant score** (5.39 for the company-mean score) because `score <= own q33` is true for a flat series and the cap is 6. Its threshold also uses the company's full history, including the future.
5. **Recover AUROC has n = 371 pairs in the default Y, 30-46 in the `and` variants**, and swings 0.58 -> 0.94 with the Y cutoff. The search varies Y and weights and keeps the best on the same 72-company holdout: selection on the holdout, no CI or permutation null.
6. **Label rows exist only for months with activity.** Dormant months are skipped, not labeled; the clearest stress (a company going dark) is invisible. 893 of 1,286 companies have fewer than 24 active months.
7. **Smaller:** `coverage` gate is trivially ~1 (scores exist for every feature row); `size_spearman` pairs the *label* month's net with the score from m-3, not the last-month score; raw CSVs are read, so `clean` rules are not applied (24 rows with |amount| >= 1e9, `exchange_rate` != 1 on 101k transactions, amounts summed without FX); `stress`/`recover` labels are within-company quantiles, so ~20-36% of months are stress by construction.

## Decisions / implications

- Do not use `primary >= 0.57` or `found.jsonl` as evidence for any score.
- `compare.py` (candidate CSV with `company_id,month,score`, holdout-only, group-aware split) is worth keeping as the interface; the metrics and Y need replacing.

## Still unknown

- What the organizers' hidden truth is; still no outcome that is independent of the score's inputs.
- Whether the 16 h search is still running elsewhere and what it found.
