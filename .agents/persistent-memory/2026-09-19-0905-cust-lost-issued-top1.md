# 2026-09-19-0905 — absorb cust_lost + issued_top1

- **Author:** parent
- **When:** 2026-09-19 09:05 CEST

## What changed

- [d_cust_lost leftover QA](553c6ea4-a745-42a3-962a-2299b600d939):
  CLOSE unused leftover. Leftover after days **0.522** dies. Twin of
  `d_n_cust` ρ **0.861**. PARK `y_cust_lost`. Dark 470 NaN. Not Y7.
- [issued-to-top-1 leftover](689100e7-9d3a-41a4-b281-a0e6a8a87d65):
  KEEP as a Y7 leftover after issued_lag1 **0.789**. Thinning-to-zero
  (Y7 57.7% vs 8.1% if still billed). CLOSE as a TURNOVER add-on —
  issued_lag1 **0.626** stays the card lead. Q6 lag1 leftover 0.749
  is a footnote, not a replace.

Night quotes unchanged: Y3 0.762/0.752, days 0.711, TURNOVER 0.720/0.712.

## Decisions

- Leftover seats all idle. Deadline 08:00 has passed. Do not refill
  leftover. Do not resume [cust_lost owner](553c6ea4-a745-42a3-962a-2299b600d939).
- Same invoice literature owner continues until ~10:00 on Wave D
  `y5_net_tc`. Cash-flow extra still on the NSF hole.

## Still unknown

Whether net TC × activity leftover explains the Y5 65% neither cell.
Whether the dictionary has any NSF-like token.
