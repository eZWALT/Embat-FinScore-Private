# Bundle inspector

A local page to check that an export bundle is sound and to understand every number in it, from the bundle alone (no raw data, no Python beyond the tiny server). Developer tool, not the product.

```bash
python product/score/inspector/serve.py --bundle <bundle_dir> --open      # http://127.0.0.1:8765
```

Tabs: **Overview** (counts, distributions, alerts by kind and month, the shipped statistics, clusters, and a *Run full audit* button that reads every company file in the browser and re-checks the arithmetic and sha256), **Companies** (score over time with alerts marked; click a month to see categories, every item with value/points/contribution, the live check that contributions plus guard equal the score and that item deltas equal the change, reasons, cluster percentile, control charts with persistence, forecast fan, alerts), **Alerts** (filterable feed with the evidence behind each), **Groups** (mean, own-history chart, funnel vs other groups, members, alerts), **Method** (how each number is made, generated from the manifest).

Deep links: `/?company=COMP_0541&month=2026-08`, `/?group=GROUP_0037`, `/?tab=alerts`, `/?audit=1`.
