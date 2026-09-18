# overnight/

Hackathon night runs. Compare **health-score candidates** when there is **no official label**.

The organizer hidden test is **not** available here. Do not invent a Y that is just our own score or our own pillars. Rank versions with the proxies below, **only on the frozen holdout**.

## What to maximize (in this order)

| # | Metric | Meaning (brief) | Better |
|---|--------|-----------------|--------|
| 1 | `auroc_stress_lead3` | Does a **low** score 3 months earlier predict a **data-defined** stress month? | Higher (0.5 = chance) |
| 2 | `auroc_recover_lead3` | Does a **rising / higher** score 3 months earlier predict leaving stress? (both directions) | Higher |
| 3 | `lead_months` | Median months the score is already in the company's worst third before a stress month | Higher (cap 6) |
| 4 | `holdout_ok` | Metrics 1–3 computed **only** on `splits/holdout_companies.csv` | Must stay true |

**Primary ranking key:** `(auroc_stress_lead3 + auroc_recover_lead3) / 2` on holdout.  
If two candidates tie within 0.01, prefer the one with larger `lead_months`, then lower `vol_mae` (see gates).

Stress / recover labels come from **invoices + bank flows**, not from the score. See `compare.py`.

## Gates (fail = do not ship, even if AUROC is high)

| Gate | Rule |
|------|------|
| Coverage | Score present for ≥ 70% of holdout company-months that have a label |
| Range | Scores in `[0, 100]` |
| Not a size proxy | Spearman of last-month score vs log1p(\|monthly net flow\|) on holdout, **\|ρ\| < 0.85** |
| Explainable | Candidate is a named formula / scorecard, not “PC1” or an unnamed fit |
| No holdout leak | Percentiles, weights, bins fitted **without** holdout `company_id`s (and without their `group_id` siblings) |

## Do not maximize

- Official leaderboard (unseen)
- Variance explained (PCA)
- PLS / any fit to a target we invented from the same pillars as the score
- In-sample AUROC on train companies
- Javier’s group-fold Spearman of “score vs itself with another reference” — that is **stability of the method**, not validity

## Candidate file

`overnight/candidates/<name>.csv`:

```text
company_id,month,score
COMP_0058,2026-06-01,61.2
```

`month` = first day of the month. Score uses **only data up to that month**.

```bash
python overnight/compare.py overnight/candidates/<name>.csv
```

Writes `overnight/runs/<name>.json`.

## 16-hour search (CLI, not the chat agent)

The Cursor chat agent does **not** stay alive for 16 hours. `search_16h.py` does: wall-clock timer, weight × Y-cutoff combinations, same holdout metrics as `compare.py`. Hits with `primary ≥ 0.57` and passing gates are written under `candidates/` and `runs/found.jsonl`.

```bash
# already started via watchdog; safe to run again (resumes same deadline)
python3 -u overnight/search_16h.py --hours 16
# or
bash overnight/watchdog.sh
```

Status:

```bash
cat overnight/runs/LIVE.json
tail -20 overnight/runs/search_16h.log
pgrep -af search_16h
```

## Layout

```text
overnight/
├── README.md
├── compare.py
├── splits/holdout_companies.csv   # 72 companies / 15 groups, seed 20260918 — do not edit
├── candidates/                    # your score tables
├── runs/                          # metric dumps
└── cache/                         # monthly labels (local)
```

Holdout is group-aware (whole `group_id` in or out) so holdings do not leak.
