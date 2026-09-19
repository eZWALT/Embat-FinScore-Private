# Watcher post format (fixed, not negotiable)

Every Sentinel **month post** is this shape and nothing else. The renderer draws it. The model must not invent Headline / Story / Act / Info, paragraphs, or plot captions. If you write a month post (offline batch only), emit the JSON object, not prose.

```
LINE 1   {entity} · {score}  {state}  {delta?}
LINE 2   {one money or attention fact}
• {severity}  {Owner} · {entity} — {action with €}
```

## Field rules

**Line 1** (≤ 80 characters)

| Watch | Pattern | Example |
|---|---|---|
| One company | `{COMP} · {score}  {trajectory}  ({signed pts vs prior month})` | `COMP_0208 · 56  deteriorating  (−12)` |
| One group | `{GROUP} · {mean}  {held\|down\|up}  ({signed vs prior})` | `GROUP_0003 · 66  held  (+0.6)` |
| Several | `{n} entities · mean {m}  {held\|down\|up}` | `2 groups · mean 63  down  (−5)` |

- Score/mean: integer, no decimals.
- State for a company is the trajectory token as-is: `improving` / `stable` / `dip` / `deteriorating`.
- State for a group: `down` if mean fell ≥ 2 pts, `up` if rose ≥ 2, else `held`.
- If a guard is on, **prefix** line 1 with `dark ·` or `fading ·` (the cap is the story, not the raw score).
- If confidence is `low` or the company has no invoices, **prefix** line 2 (not line 1) with `no invoices ·` or `low confidence ·`.

**Line 2** (≤ 120 characters, one fact)

Pick **one**, in this order:

1. Active guard: `capped {score} (uncapped {pre}) · no bank booking 60d` / fading equivalent.
2. An `act` or `watch` alert that month: title + the first reason €.
3. Top customer quiet: exactly `top customer stopped billing, review exposure and collections · {open €}`.
4. Else the member (or company) that moved most: first `change_reasons` or `reasons` label + €.
5. Else `no material move`.

Never a second sentence. Never "the set holds" essays.

**Bullets** (0–4, never more)

- `act` and `watch` alerts that month, then `opportunity`, then at most **two** `follow` bullets for the member that moved if **no** alert fired.
- Each bullet: `{Owner} · {COMP or GROUP} — {concrete action with €}`.
- Owners: Treasurer / CFO / Collections only.
- `info` alerts: do not list. The post may carry `n_info` for a muted count; the UI shows it as a caption, not a bullet.
- If there are no bullets: that is valid. Do not write "Act / watch: none".

## JSON (what code stores)

```json
{
  "month": "2026-08",
  "line1": "GROUP_0003 · 66  held  (+0.6)",
  "line2": "COMP_0208 deteriorating · customers 22d late on €539k",
  "bullets": [
    { "severity": "follow", "owner": "Collections", "entity": "COMP_0208", "text": "€539k collected late, €98k AR >30d" },
    { "severity": "follow", "owner": "CFO", "entity": "COMP_0072", "text": "now deteriorating · runway 1.9 mo" }
  ],
  "n_info": 0
}
```

## Forbidden

- Labels: Overview, Headline, Story, Act, Watch, Info, Plot, Reliability.
- Paragraphs, JSON dumps, alert ids, `rank_score`, funnel-limit essays.
- "Nothing new", "no novelties", "nothing to approve".
- Predicting default or "revenue at risk".

## Production

Opening three posts (`as_of-2`, `as_of-1`, `as_of`) are **built offline** by the deterministic formatter (or an offline LLM that emits only this JSON). The UI reads them. Live model = thread replies only, and those replies stay short: answer + € + owner, not a restyled month post.
