# 2026-09-18-2015 — empty runtime stubs + initial sync

- **Author:** agent
- **When:** 2026-09-18

## Decisions

- `product/Dockerfile`, `.dockerignore`, and `embat_finscore` sources are **empty**. Do not assume Python image, uv-in-Docker, user, `CMD`, or a CLI.
- Initial tree is synced to the **public** sibling (`eZWALT/Embat-FinScore`) and both remotes are pushed.
- **Next edits only on this private repo.** Reconcile/push public later, at the end.

## Still unknown

- Runtime, score, buyer, demo. Do not fill those in until chosen.
