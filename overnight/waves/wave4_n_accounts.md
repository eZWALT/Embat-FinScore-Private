# Wave 4 — g_n_accounts leftover after days as Y3 X

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/n_accounts_qa.py`
- `analysis/outputs/n_accounts_qa.md`
- `analysis/outputs/n_accounts_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not touch `banking_g_qa.*`, `g_has_rest_qa.*`, `ar_open_qa.*`, `ap_open_qa.*`, `util_snap_qa.*`, `products.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged.

## Locked verdict

| object | decision |
| --- | --- |
| `g_n_accounts` as Y3 X / 15-col card | **CLOSE unused leftover** |
| `g_n_accounts` as engine X on the 44 | **DROP** |
| `y_n_accounts` | **PARK** |
| `g_new` | **PARK (locked)** |
| TURNOVER | **CLOSE** — do not grow 0.720 |

## Locked extras

- Rise-only 1561/0 (peek 1,561/0 CONFIRM).
- checking hole 99.1% (peek 99.1% CONFIRM). leftover after checking 0.586.
- Dark/ERP last p50 3.000/3.000 CONFIRM. Access ≠ ERP.
- Honest leftover after days rank 0.428 (dies=True, fake=False); inverse 0.707.
- Single 0.581 vs days 0.711 vs size 0.617 vs checking 0.490 vs f_n_types 0.578.
- Q6 lag1 leftover 0.460. Ever leftover 0.419 month-after-ever 0.532.
- Twin of g_n_banks ρ 0.880. leftover after banks 0.469 dies. banks leftover after days 0.542 dies.
- leftover after days+checking 0.430. leftover after f_n_types 0.533 / facilities 0.531. f_n_types leftover 0.534 CONFIRM.
- ICC 0.983 / acf1 0.804 CONFIRM. Rise dummy leftover 0.684 is fake days. Y7 leftover 0.441. Do not claim TURNOVER.
- n>=3 leftover after days 0.617 lives but Y3 0.582 fails beat-size. Long leftover 0.562 lives, Y3 0.525 fails beat-size.
- n>=3 leftover after days+banks 0.626 lives (twin leftover of banks). Count-bin 0 leftover 0.674 is the hole dummy; 5+ leftover is not KEEP.
- n_accounts−n_banks Y3 0.544 leftover after days 0.552 dies=False fake=False. after days+banks 0.534.
- ever-rise leftover 0.438 always-flat leftover 0.428. Rise-clock companies do not unlock leftover after days.
- log1p(n_accounts) Y3 0.581 leftover after days 0.428 dies=True fake=False. Transform does not revive leftover.
- banks leftover after days+n_accounts 0.548 dies=True fake=False. after days 0.542. after n_accounts 0.579. Banks has no leftover once the count is in — cluster, not a second seat.
- Δn Y3 0.492 leftover after days 0.684 dies=True fake=True. after days+count 0.631. Month shock of a rise-only clock is not leftover after days.
- varying leftover 0.438 constant leftover 0.428. BETWEEN trait still dies after days.
- leftover after days+f_n_types 0.446 dies=True. n>=3 leftover after days+checking 0.597. after days+f_n_types+banks 0.551. n>=5 Y3 0.592 leftover 0.693 dies=False beat-size FAIL.
- n>=5 Y3 0.592 n=1,574 pos=77 vs size 0.453 vs days 0.745. leftover after days 0.693 fake=False. after days+banks 0.469. after days+size 0.637. after days+checking 0.678. beat-size FAIL — n>=5 leftover does not pass KEEP-as-X vs locked size 0.617.
- days>0 leftover 0.434 Y3 0.576 n=5,536 dies=True. ratio n/banks Y3 0.472 leftover 0.558 dies=False. leftover after days+size+checking 0.444.
- leftover after days+all g_has_* 0.432 dies=True. g_has_* already DROP; count leftover after each still dies or is days.
- checking leftover after days 0.674; after days+n_accounts 0.459 dies=True fake=False. checking=1 leftover 0.428 Y3 0.596 n=5,178 pos=364 beat-size FAIL.
- ever-max>=5 Y3 0.556 n=2,199 pos=110 vs size 0.453 vs days 0.678. leftover after days 0.606 dies=False. after days+banks 0.581. beat-size FAIL.
- ex-fold4 Y3 0.621 leftover 0.542 dies=True n=4,275. banks>=2 leftover 0.611 Y3 0.545 n=3,467. Weak fold does not hide a KEEP leftover.
- g_n_types leftover after days+n_accounts 0.537 dies=True. g_new leftover after days+n_accounts 0.623 dies=False. count leftover after days+types+banks 0.556. g_new=1 leftover — g_new=0 leftover 0.433.
- g_n_types>=2 leftover — Y3 — n=1,229. last-3 labeled leftover 0.417 Y3 0.579 n=1,762. banks>=2 leftover after days+size 0.622 Y3 0.545 beat-size FAIL.
- single-bank leftover 0.619 never-zero 0.532 ever-hole 0.459 n_1_2 0.587 chk1_n>=3 0.616.
- leftover after days+a_n_tx+size 0.435 dies=True. 2024 leftover — 2025 leftover 0.464 2026 leftover 0.574.
- never-zero Y3 0.631 n=3,085 pos=189 vs same-n size 0.681 vs days 0.733. leftover after days 0.532 dies=True. after days+size 0.431. after days+banks 0.522. beat locked-size FAIL beat same-n size FAIL — leftover still dies, not KEEP.
- small leftover 0.526 large leftover 0.429 first-labeled leftover 0.459. leftover after days+checking+size+banks 0.548 dies=True.
- low-days leftover 0.446 high-days 0.486 card=1 — ge3∩never0 0.624. leftover after days+facilities+banks 0.551. 1:1 accounts=banks Y3 0.540 leftover 0.566 dies=False.
- ge3∩never0 Y3 0.563 n=1,633 pos=77 vs size 0.577 vs days 0.677. leftover after days 0.624 after days+banks 0.642 after days+size 0.642. beat-size FAIL. >=6 labeled leftover 0.425 Y3 0.573 n=4,869.
- 2026 leftover 0.574 after banks 0.575. saving=1 leftover —. rise∩n>=3 leftover 0.622 after banks 0.610. leftover after days+checking+a_n_tx 0.435.
- invest=1 leftover — sb∩n>=2 0.632 types=1 0.425. leftover after days+types+checking 0.437.
- banks>=3 leftover 0.664 low-tx 0.549 high-tx 0.555. leftover after days+banks+types+checking 0.558 dies=False.
- banks>=3 Y3 0.555 n=2,113 pos=78 vs same-n size 0.569 vs days 0.716. leftover after days 0.664 after days+size 0.658 after days+banks 0.539. beat locked-size FAIL beat same-n FAIL — slice leftover is not KEEP-as-X.
- Q1 leftover 0.530 Q2-4 leftover 0.412. leftover after days+facilities+size 0.447 dies=True.
- H1 leftover 0.455 H2 leftover 0.430. leftover after days+tx+checking+size 0.444 dies=True.
- singleton leftover — multi-group 0.425 first-2024 0.424 first-2025 0.531.
- Bootstrap leftover-after-days rank p05=0.399 p50=0.433 p95=0.545 share<0.55=96.2% n=80.

unused leftover after days: honest rank 0.428 dies (OLS 0.562 fake=False). Rise-only 1561/0. DROP from the 44 as Y3 X. Off the 15-col card. Do not grow TURNOVER.

## What failed / next

- none

Elapsed 59s.
