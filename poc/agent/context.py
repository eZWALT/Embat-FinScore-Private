"""System prompt assembly: role prompt + product context + wording rules + live facts from the bundle manifest."""

from __future__ import annotations

import json
import os
from pathlib import Path

from poc.bundle import Bundle

PROMPTS = Path(__file__).parent / "prompts"


def _read(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def language() -> str:
    return os.environ.get("POC_LANG", "es")


def spec_summary(manifest: dict) -> str:
    spec = manifest["spec"]
    lines = ["## Score spec (from the bundle manifest)", "", "Categories (label, nominal → effective weight):"]
    for cid, c in spec["categories"].items():
        lines.append(f"- `{cid}`: {c['label']}, {c['nominal_weight']:.0f} → {c['effective_weight']:.0f}")
    lines += ["", "Items (id · category · label · unit · direction · why):"]
    for it in spec["items"]:
        d = "higher is better" if it["higher_is_better"] else "lower is better"
        lines.append(f"- `{it['id']}` · {it['category']} · {it['label']} · {it['unit']} · {d}. {it['why']}")
    g = spec["guard"]
    lines += [
        "",
        f"Guard: dark = no booking for {g['dark_no_booking_days']} days → cap {g['cap_dark']:.0f}; "
        f"fading = inflows under {g['fading_inflow_ratio']:.0%} of own earlier mean → cap {g['cap_fading']:.0f}.",
        f"Trajectory thresholds: 3-month slope ±{spec['trajectory']['slope3_material']} pts/month, "
        f"6-month slope ±{spec['trajectory']['slope6_material']} pts/month, persistence {spec['trajectory']['persist_months']} months.",
        f"Score needs {spec['score_needs_months']} months of trail.",
    ]
    m = manifest.get("monitor", {})
    if m:
        lines += [
            "",
            f"Monitor: {m['method']['name']} {json.dumps(m['method']['params'])}; "
            f"material points {json.dumps(m['material_points'])}; forecast method {m['forecast_method']}.",
        ]
    return "\n".join(lines)


def bundle_facts(bundle: Bundle) -> str:
    m = bundle.manifest
    stats = bundle.alerts_feed.get("stats", {})
    return "\n".join(
        [
            "## This bundle",
            "",
            f"- schema {m['schema_version']}, scorecard {m['scorecard_version']}, generated {m['generated_at']}",
            f"- as-of month {m['as_of_month']}; scored months {m['months'][0]} → {m['months'][-1]}; "
            f"items and reasons exist for the last {m['detail_months']} months (from {m['detail_from_month']})",
            f"- {m['counts']['companies']} companies, {m['counts']['groups']} groups, {m['counts']['company_months']} company-months"
            + (" (SAMPLE bundle: 12 companies; groups list only sampled members)" if m.get("is_sample") else ""),
            f"- disclaimer to show with any score: \"{m['disclaimer']}\"",
            f"- alert feed stats: {json.dumps(stats)}",
        ]
    )


def system_prompt(role: str, bundle: Bundle, extra: str | None = None) -> str:
    """role: 'sentinel' or 'chat'."""
    role_file = {"sentinel": "sentinel_system.md", "chat": "chat_system.md"}[role]
    parts = [
        _read(role_file),
        _read("product_context.md"),
        _read("wording_rules.md"),
        spec_summary(bundle.manifest),
        bundle_facts(bundle),
    ]
    if role == "chat":
        parts.append(_read("clean_schema.md"))
    parts.append(f"Default language for digests and when the user's language is unclear: `{language()}`.")
    if extra:
        parts.append(extra)
    return "\n\n---\n\n".join(parts)
