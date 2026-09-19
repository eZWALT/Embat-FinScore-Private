# 2026-09-19 12:01 — elección persistente de tema

- **Author:** Codex
- **When:** 2026-09-19 12:01 CEST

## What changed

- Added an explicit `Claro` / `Oscuro` selector to the Health Sentinel header.
- The chosen theme is stored in `localStorage`; before the user chooses, the operating-system preference is used.
- Both themes use the same shadcn semantic tokens. Chart contrast was adjusted independently for light and dark surfaces.

## Decisions

- Kept the selector visible on every viewport; labels collapse to icons on small screens.
- Used a small first-party theme initializer and `useSyncExternalStore` instead of `next-themes`. This avoids React 19 script and hydration warnings while keeping the initial paint and control state aligned.

## Verification

- `pnpm lint` and `pnpm build` pass.
- Dark and light modes render correctly in the browser.
- Theme selection survives a reload and the browser console is clean.

## Still unknown

- No additional theme variants are planned; the product currently offers only the requested white and black choices.
