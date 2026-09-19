# Pregunta chips pinned + Sentinel chrome + push

- Author: agent
- Timestamp: 2026-09-19 20:55 +02:00
- Still-binding for Pregunta follow-up chips.

## What changed

Pregunta chips are a Perplexity-style 2-column row pinned above the composer (opening pair and post-reply pair). Fallback chips show as soon as an assistant message exists; Helmcode `/api/ask/followups` swaps them in. The sheet `Card` keeps `overflow-hidden` but the chat column is `flex-1 min-h-0` so the chip row cannot clip.

Product chrome and prompts say **Sentinel**. No `Centinela` left under `product/`.

## Decisions

Chips live in the UI, not inside the assistant markdown. Do not wait for the stream to finish before showing the fallback pair.

## Still unknown

Helmcode follow-up wording quality on live production after this push.
