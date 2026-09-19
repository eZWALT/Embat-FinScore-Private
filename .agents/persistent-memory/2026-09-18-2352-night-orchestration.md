# 2026-09-18-2352 — night orchestration start

- **Author:** agent (Cursor)
- **When:** 2026-09-18 23:52 CEST
- **Until:** 2026-09-19 07:30 CEST

## What changed

- Erased the invalid 16h search tree. Rebuilt `overnight/` as night status + contract.
- Skeleton + grid + assembler under `analysis/`. Holdout kept.
- Plan: 8-agent waves, parent verifies, 20-minute loop until 07:30.

## Decisions

- Code in `analysis/`; `overnight/` is orchestration only.
- Wave 1 owns A,B,C,D,E,F + Y1/Y2 + Y5/protocol. G/H and models wait for wave 2+.
- Parent never lets children edit the same file.

## Still unknown

- DuckDB rebuild time; weekly grid cost; which Ys pass acceptance.
