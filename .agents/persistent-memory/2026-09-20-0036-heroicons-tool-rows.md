# Chat tool-row icons use Heroicons

- Author: Cursor agent
- Timestamp: 2026-09-20 00:36 (Europe/Madrid)

## What changed

Ask / Watcher tool rows (`AgentTrace`, `ToolCallChip`) no longer share one Lucide wrench. Each retrieval has a 24px outline Heroicon in `product/web/src/lib/agent/tool-catalog.ts` (`TOOL_ICONS` / `toolIcon`). Package: `@heroicons/react` in `product/web`. Lucide stays for the rest of the app chrome.

## Decisions

- Outline `/24/outline` so stroke weight matches the existing 12px (`size-3`) rows.
- Unknown tool names fall back to `WrenchScrewdriverIcon`.

## Still unknown

- Live chat turn not exercised in this change; confirm one tool call in the hosted popup.
