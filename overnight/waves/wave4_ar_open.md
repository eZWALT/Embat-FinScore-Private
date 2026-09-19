# Wave 4 — e_ar_open leftover after days as Y3 X

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/ar_open_qa.py`
- `analysis/outputs/ar_open_qa.md`
- `analysis/outputs/ar_open_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not touch `dso_qa.*`, `issued_qa.*`, `pending_qa.*`, `ap_open_qa.*`, `pay_match_qa.*`, `ogtg_qa.*`, `invoices.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged.

## Locked verdict

| object | decision |
| --- | --- |
| `e_ar_open` as Y3 X / 15-col card | **CLOSE unused leftover** |
| `e_ar_open` as engine X on the 44 | **DROP** |
| `y_ar_open` | **PARK** |
| TURNOVER | **CLOSE** — do not grow 0.720 |

## Locked extras

- Dark 470 stay NaN (nn=0). ERP leftover after days 0.527 dies.
- vs `e_ap_open` ρ=0.682 (peek 0.682 CONFIRM) — different object. Sibling file not overwritten.
- Leftover after issued 0.433 dies — rewrite of issued volume (ρ=0.697, not a twin).
- Leftover after DSO 0.557 lives thinly; after days+DSO 0.423 dies. DSO leftover after days OLS 0.474 CONFIRM.
- Honest leftover after days rank 0.527 dies; OLS 0.719 fake days clone (ρ resid,days −0.817). Inverse days after open 0.706 lives.
- Issued leftover after days same-n 0.608 CONFIRM. Open is weaker than the issued fake-days clone.
- Zero-stock Y3 12.6% vs open>0 5.4%. Zero-dummy leftover after days 0.479 dies. Q1 dummy leftover 0.458 dies.
- open>0 leftover rank 0.586 lives thinly but OLS fake=True and Y3 0.420 fails beat-size.
- Q6 lag1 leftover 0.529 dies. lag3 leftover 0.441 dies. Y7 leftover after issued_lag1 0.456. Do not claim TURNOVER.
- Bootstrap leftover-after-days 0.397 / 0.523 / 0.578 (70% die, n=80).
- leftover after days+fx 0.541 / days+credit-note 0.531 die. overdue_30 leftover after days 0.596 is report-only — not this seat.
- leftover after delay 0.512 / overdue 0.420 both die. Not a delay twin (ρ 0.097).
- company-median leftover 0.551 dies. Size T1 leftover 0.560 / T3 0.400. Rebuilt DSO leftover OLS 0.474 CONFIRM.
- leftover after a_n_tx 0.515 / AP issued 0.530 / overdue_30 0.416 die. last6 leftover 0.471 dies. Δopen leftover 0.523 dies.
- leftover after fx 0.593 / credit-note 0.569 without days is leftover after an unrelated control; days+fx closes it.
- first6 leftover after days+issued 0.535 dies. T1 leftover after days+issued 0.538 dies.
- median company Pearson acf1 0.770 CONFIRM. ICC 0.990 BETWEEN. ρ vs own lag1 is a BETWEEN twin.
- first6 leftover 0.564 / later 0.432. short_<12 leftover 0.560 / long 0.480.
- ρ vs own lag1 0.944 TWIN — BETWEEN snapshot, not a lead.
- open==0 days p50=11 vs open>0 18; ρ(zero,days)=-0.317. Zero stock is thinner activity, leftover dies.

Y3 leftover after days rank 0.527 (dies=True, fake=True); inverse days after ar_open 0.706. Single 0.587 vs days 0.711 vs size 0.617 vs issued 0.687 vs DSO 0.564. after issued 0.433 after DSO 0.557. Q6 lag1 leftover 0.529. vs ap_open ρ=0.682. Dark 470 NaN=True. unused leftover after days: honest rank 0.527 dies (OLS 0.719 fake=True). Issued rewrite leftover 0.433. DROP from the 44 as Y3 X. Off the 15-col card. Do not grow TURNOVER.

## What failed / next

- none

Elapsed 33s.
