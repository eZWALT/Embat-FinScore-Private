# 2026-09-19-1839 — month labels are the scored as-of, not “today”

- **Author:** Cursor Grok 4.6
- **Timestamp:** 2026-09-19 18:39

## What changed

UI month labels no longer use a 2-digit apostrophe year (`ago ’26`), which read as a random month. Single source: `product/web/src/lib/format-month.ts`. `group/labels.ts` re-exports it.

## Decisions

| Helper | `2026-08` | Use |
|--------|-----------|-----|
| `formatMonth` | `agosto 2026` | badges, titles, tooltips |
| `formatMonthShort` | `ago 2026` | tight chart axes |
| `formatAsOf` | `hasta agosto 2026` | header cut-off badges |
| `splitMonth` | `["ago", "2026"]` | two-line heatmap headers |

The bundle `as_of_month` is the latest scored month, not calendar “today”. Header badges that show that cut-off must use `formatAsOf`.

## Still unknown

None for this wording. Score logic unchanged.
