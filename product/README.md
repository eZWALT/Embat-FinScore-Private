# product/

Goals 2–4. A first dummy 0–100 card lives in [`score/`](score/).
Web and Docker are still empty.

| Path | Goal |
|------|------|
| [`score/`](score/) | v0 dummy FICO-like score (train-only percentiles, a-priori weights) |
| [`web/`](web/) | 4. Webpage + LLM. User = the company that gives the data |
| `Dockerfile` | Empty. No base image or `CMD` |
| `src/embat_finscore/` | Empty package name only |

```bash
PYTHONPATH=. python -m product.score
```

Do not `docker build`. Do not assume a web or LLM stack.
