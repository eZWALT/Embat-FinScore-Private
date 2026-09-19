# Ask: tool spinner stays, pulse until first token, two follow-ups

- Author: agent
- Timestamp: 2026-09-19 19:10 +02:00

## What changed

Pregunta chrome now says **Sentinel** (not Centinela). After tools finish and before the first reply token, the last tool row keeps the ring spinner and a 3-dot pulse stays under the tools block. Both go away when text starts.

Opening chips are precomputed from the open screen (`openingSuggestions`). Popup open still hits `POST /api/ask/warmup` (1 Helmcode token) so the first tap is not a cold start.

After each Ask answer, two follow-up chips are generated agentically (`POST /api/ask/followups`, `followups.md`, no Neon, no tools). The request starts once the reply has ~80 characters so latency is hidden behind the main stream. They cannot be precomputed.

## Decisions

Follow-ups are a separate stream, not a second step of the main agent (that would delay the answer). Javi Spanish alert copy is unchanged.

## Still unknown

Whether production Vercel still coalesces the UI-message SSE. Follow-up quality on a partial answer (~80 chars) vs waiting for `onFinish`.
