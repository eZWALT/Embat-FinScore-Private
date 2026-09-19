# Ask: native Helmcode SSE, not a blob

- Author: agent
- Timestamp: 2026-09-19 18:45 +02:00

## What changed

Pregunta was calling `@ai-sdk/openai` v4 as `createOpenAI()(modelId)`, which is the **Responses** API (`POST /v1/responses`). Helmcode’s documented surface is **chat completions** (`POST /v1/chat/completions`, SSE). `createOpenAI` has no `stream` flag; `streamText` is what sends `stream: true`.

`product/web/src/lib/agent/llm.ts` now uses `helmcode.chat(modelId)`. Ask / Watcher / `/api/chat` return `result.toUIMessageStreamResponse()` (exists in `ai@7.0.107`, deprecated alias of `createUIMessageStreamResponse` + `toUIMessageStream`). `smoothStream({ delayInMs: 16, chunking: "word" })` paces **already-streamed** tokens. `POST /api/ask/warmup` is a 1-token `streamText` (Helmcode only, no `DATABASE_URL`). Popup open and `/ask` mount fire it. `agent-chat.tsx` untouched. Ask prompt stack unchanged.

## Decisions

Use `.chat()`, not the default Responses model. Do not add LangChain.

## Still unknown

Whether Vercel’s edge/proxy still coalesces SSE in production. Local Helmcode `/chat/completions` does stream.
