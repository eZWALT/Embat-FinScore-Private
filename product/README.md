# product/

Application folder. Nothing here is implemented.

| File | State |
|------|--------|
| `Dockerfile` | Empty. No image, base, or `CMD` |
| `.dockerignore` | Empty |
| `pyproject.toml` / `uv.lock` | Package name only. No runtime deps |
| `src/embat_finscore/` | Empty package |

```bash
uv sync
```

Do not `docker build` until there is a real Dockerfile.
