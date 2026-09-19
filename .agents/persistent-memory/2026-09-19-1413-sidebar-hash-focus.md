# Sidebar hash focus

- **Author:** Codex
- **Timestamp:** 2026-09-19 14:13 CEST

## What changed

- Replaced the permanently active first sidebar item with state derived from the URL hash.
- Navigation clicks now update the active item immediately.
- Hash changes from browser back/forward keep the sidebar selection synchronized.
- Added `aria-current="location"` to the selected navigation link.

## Decisions

- The URL hash is the source of truth for section navigation. A missing or unknown hash falls back to `#resumen`.
- Scroll position alone does not change the selection because Evolución and Categorías share the same vertical row.

## Verification

- `pnpm lint` and `pnpm build` pass.
- Browser verified direct load at `#senales`, click to `#categorias`, and back navigation to `#senales`.

## Still unknown

- None for this fix.
