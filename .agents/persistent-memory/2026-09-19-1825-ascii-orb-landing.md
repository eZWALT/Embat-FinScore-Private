# Sentinel ASCII orb landing

- Author: Cursor Grok
- Timestamp: 2026-09-19 18:25 CEST

## What changed

Second landing at `/orb`: same Sentinel copy and nav as `/`, right column is a static 3D ASCII sphere (no motion, no SVG mask).

The glyph is a Phong-shaded disk: key + fill + specular glint, Bayer dither, greyscale `oklch` per character. Depth also comes from scaling glyphs inside a fixed `1ch` grid (bulge larger, rim smaller) so letter-spacing cannot warp the circle. A flat ASCII ellipse sits under the sphere as a contact shadow.

## Decisions

- Keep the Sauron landing at `/`; this is a parallel page.
- White page, greyscale only, Geist Mono.
- No rAF. Volume is lighting, color, glyph weight, and per-cell scale.

## Still unknown

Whether this replaces `/` after the demo, or stays an experiment.
