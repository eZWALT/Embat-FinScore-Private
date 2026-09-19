# 2026-09-19-1620 — alert text (and the score reasons they embed) in Spanish; NaN in alert text fixed

- **Author:** Claude Code (Sonnet 5) for Javier Boix
- **Trigger:** the web app went Spanish; alerts still arrived in English (the app shows `alert.summary`, `alert.action` and `reason.sentence` straight from the data).

## What changed (source, not the UI)

- `analysis/monitor/engine.py`, `routing.py`: titles, summaries, actions (tú form), persistence rule ("3 de los últimos 4 meses"), owner labels, the top-customer sentence and label.
- `product/score/explain.py`: item and guard sentences, `eur()` (`14 k€`, `1,2 M€`, `164 €`), helpers `num`/`pct` (decimal comma, `64 %`), singular/plural (`1 vez`, `1 mes`).
- `product/score/spec.py`: item and category labels, using the same category terms the web already maps ("Historial de pagos", "Liquidez y deuda", "Estabilidad", "Nuevo crédito", "Combinación de clientes"). `why` texts (spec documentation) stay English.
- `product/score/export.py`: "Límite por inactividad" label, alert-stats note, `manifest.language = "es"`, validator check that alert text never prints `nan`/`inf`/`-0 %`.
- Ids and enums (`kind`, `severity`, `direction`, `owner`, `item`) stay English codes. Schema stays 1.2.0 (not deployed yet); `language` is additive.

## Scope decision

- The alert reasons are the score's reason sentences, so those are Spanish too (score explanations in the company files as well). Translating only inside alerts would have needed a second copy of every sentence.
- Second pass (same day): forecast `drivers[].text` and `forecast.note`, score `confidence_note` (8 phrases in `scorecard.py`; the sample picker now matches "sin pagos de facturas"), cluster labels and descriptions (phrase table in `behaviour.py`; the stored labels in `monitor_params.json` were translated in place, **no re-fit**), the clusters quality note, and the control-chart method label ("3 de los últimos 4").
- Still English on purpose: `manifest.sections[].note`, `manifest.monitor.forecast.seasonality.note`, `spec.items[].why`, disclaimers (`DISCLAIMER` in `export.py`), and the agent prompts (poc and `product/web/src/lib/agent/prompts`, which still quote the English wording rule for the top-customer alert).
- Validator: rejects `nan`/`inf`/`-0 %` and English stopwords in alert text, confidence notes and forecast text. Tested against a deliberately broken copy. (My first version of that check contained a backspace character instead of `\b`, so it silently never matched; fixed and proven. In this environment the shell tool collapses `\\` to `\`, so build backslashes with `chr(92)` or write scripts with the Write tool.)

## Bug found on the way (predates this change)

- 40 of 2,365 alerts (1.7%) in the English bundle said things like "nan% of billing comes from one customer (€0 in 3 months)": an item with no figure that month could be picked as the biggest mover. Movers and change reasons now skip items without a value (`explain.has_value`). 18 owners and 40 reason lists changed because of it; alert ids, kinds, severities, months, evidence and counts (2,365) are identical.
- A company with net-negative recent inflows printed "-0 %" in the fading-guard sentence; clamped at 0.

## Checked

- Full export validates; a scan of every title, summary, action, rule, reason label and sentence found no English left and no `nan`/`inf`/`-0 %`.
- Sample bundle regenerated; Desktop bundle and zip replaced (the earlier English bundle is superseded).

## Still unknown

- Whether Ruben's Neon load and Walter's `get_forecast`/alert tools need any change: text columns only, so no schema change is expected.
- Tone (tú vs usted) was my choice, matching the web copy; easy to switch in `routing.py`.
