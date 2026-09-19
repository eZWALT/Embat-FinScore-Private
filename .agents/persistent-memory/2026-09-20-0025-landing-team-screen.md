# Landing Team is a full screen

- Author: agent
- Timestamp: 2026-09-20 00:25 +02:00
- Still-binding: Team on `/` is `#team`, a same-viewport crossfade, not a scroll.

## What changed

Clicking Team sets `#team` and fades the hero out. The new screen is only the three portraits plus CEO / CTO / CSO. Logo (`#hero`) returns. CSS `:target` plus `html[data-landing]` so the switch is not React-state-only. Production had a dead `#team` hash and no target.

## Still unknown

Whether they want official LinkedIn headshots later.
