# product/

Goals 2–4. `score/` is implemented (score, trajectory, explanations; see its README). `web/` is still a static mockup: no image, runtime or hosted page.

| Path | Goal |
|------|------|
| [`score/`](score/) | 2. Health index 0–100 (hard feature engineering + rationale). 3. Explainability |
| [`web/`](web/) | 4. Webpage + LLM. User = the company that gives the data |
| `Dockerfile` | Empty. No base image or `CMD` |
| `src/embat_finscore/` | Empty package name only |

Do not `docker build`. Do not assume a web or LLM stack.
