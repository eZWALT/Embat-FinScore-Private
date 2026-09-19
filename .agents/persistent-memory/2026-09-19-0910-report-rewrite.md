# 2026-09-19-0910 — morning report rewritten for stakeholders

- **Author:** parent
- **When:** 2026-09-19 09:10 CEST

## What changed

User: the report prose was uncontextualised — variable names and numbers
without meaning. Rewrote all three views in plain language:

- `overnight/dashboards/MORNING_REPORT.md` — story of the night, "how to
  read the numbers" box (ranking score, size bar 0.617, activity bar
  0.711, leftover < 0.55, twins > 0.80, dark 470), the two readings as
  sentences a company could hear, six questions with "how sure",
  literature verdicts in words, what was set aside grouped by *why*,
  glossary code-name → plain meaning.
- `overnight/dashboards/morning.html` — same, browser-readable.
- Canvas `xray-morning-report.canvas.tsx` — same, with baseline bars.

Numbers unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER
0.720 / 0.712, SS 0.635, salary 0.603, issued_lag1 0.626, days_lag1
0.684, ss_lag1 0.631, issued_top1 leftover 0.789.

## Decisions

- Technical ledger stays `CONTEXT.md`; the report is the narrative.
- Reading 1 = "quiet stress recovers" (45→65). Reading 2 = "the big
  customer goes quiet" (dip vs fall). Deterioration (82→68) reported as
  an honest failure, not hidden.
- Issued-to-top-1 0.789 is shown as a reason, not added to the card.

## Still unknown

Y5 net-TC leftover (invoice extra). NSF token hole (cash-flow extra).
