# 2026-09-19-1715 — Consultas Design B (timeline + markdown)

- **Author:** agent (Cursor)
- **When:** 2026-09-19 17:15 CEST

## What changed

Design B for Health Sentinel Consultas (`product/web`). No new npm deps, no LangChain, no new tabs, not committed.

- `agent-chat.tsx` walks `message.parts` in stream order. Consecutive tools are a tight numbered stack (1 / 2 / 3); a text part resets the run. Tools are never regrouped to the top.
- Last in-progress assistant text (status `streaming` / `submitted`) gets a moving white sheen (`agent-md-streaming` in `globals.css`). Finished text is normal foreground.
- New `agent-markdown.tsx`: safe subset (paragraphs, **bold**, *italic*, `code`, fences, breaks, `-` lists). No raw HTML, no images. Links are https-only (`rel=noopener`) or plain text. Code wraps; does not expand the chat card. User messages stay plain text.
- `agent-trace.tsx` optional `index` for the run number. Expandable input/output/SQL unchanged.

`corepack pnpm --config.minimum-release-age=0 exec tsc --noEmit` in `product/web` passes.

## Decisions

- Local renderer instead of `react-markdown`.
- Visual grouping of *consecutive* tools only; order of parts is preserved.

## Still unknown

- Live stream look (shimmer / long fences) not checked in the browser; no local Next server in this session.
