# Wave 4 — f_ds_r leftover after days

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/ds_r_qa.py`
- `analysis/outputs/ds_r_qa.md`
- `analysis/outputs/ds_r_leftover.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `fc_r_qa.py` / `.md`, `n_tx_qa.*`, `dso_qa.*`, `cust_hhi_qa.*`, `debt.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged. TURNDSSWAP 0.712 not a swap.

## Locked verdict

| object | decision |
| --- | --- |
| `f_ds_r` on the 15-col card | **DROP** |
| F lag3 Q6 | **CLOSE** |
| Y4 identity | **not KEEP** |
| TURNOVER swap | **not a swap** |

Y3 leftover after days rank 0.528 (dies=True, fake=False); inverse days after ds_r 0.682. After fc 0.606. Single 0.620 vs days 0.711 vs size 0.617. ρ vs fc 0.205. Y4 ρ=-0.082 not KEEP. Q6 empty-until-6 CONFIRM=True. unused leftover after days: honest rank 0.528 dies (OLS 0.709 ρ(resid,days)=0.775 fake=False). Single 0.620 loses to days 0.711 and fails beat-size (Δ=0.003). Twin of euro ds (ρ a_debt=0.881). i_lift drop-ds_r-only still 0.7525. Parent 15-col card absorbs; do not edit the card. Same unused leftover as contemp f_fc_r on the 44.

## What failed / next



Elapsed 50s.
