# Sentinel eye radar field

- Author: Cursor Grok
- Timestamp: 2026-09-19 17:10 CEST

## What changed

The Sauron eye on `/` keeps the static masked bracket texture. A second layer of **180** live cells (cap 200) runs the Select-2026-style polar sweep in greyscale, via rAF mutating span text/color/opacity — not React state, not one node per FIELD glyph.

Paused with `IntersectionObserver` and frozen when `prefers-reduced-motion: reduce`. Pupil `]` stays; its CSS pulse is disabled under reduced motion.

## Decisions

- Bracket set `{ [ ( <` / `} ] ) >` (Geist Mono), not `‹›`.
- Radar lives inside the existing SVG luminance mask.

## Still unknown

Whether 180 cells is dense enough on a 40% column at mobile widths.
