# Overnight: Pregunta agent optimize (branch)

- Author: agent
- Timestamp: 2026-09-20 01:55 +02:00
- Still-binding: Overnight work lives on `agent/overnight-optimize`, not `main`. Keep the same tools and product claims. Improve latency, tool waste, grounding, Spanish, format. Compare every keep against a `main` baseline. Do not invent tools. Do not merge to `main` tonight unless Walter asks. Leftover Spanish-only prompt edits on the working tree were the starting point for the branch, not an extra main commit.

## What this night is

Watch the hosted Pregunta popup and tighten the agent: prompts, tool caps (including same-step parallel), answer length, hallucination, formatting. Function stays: score / five alerts / records / plots / refuse off-topic. No Exa. Brain stays `high`. After a tool burst, still write (0148). Same-kind rows still tick `×n` (0140).

## Baseline (main @ 1a22194)

Known waste on a period drag: `get_company` ×4 + `explain_change` ×1 + `get_alerts` ×4 + `explain_change` ×1, then empty text (fixed on main). Cap 2/tool is bypassed by four parallel calls in one step. `get_company` already returns `reasons`, `change_reasons`, `score_history`, `alert_ids` — per-month fan-out is redundant.

## Rules for the loop

- Record a `main` bench first. A change stays only if it is not worse on judge/grounding and not slower without a quality gain.
- Window: at least 4 h, at most 7 h from 01:55 +02.
- Journal each keep. Facts that rot go here, not `AGENTS.md`.

## Still-unknown

Whether the Vercel preview of the branch is up; whether `~/.helmcode_key` is present for local Helmcode benches.
