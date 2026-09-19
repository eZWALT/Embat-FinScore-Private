// Datasets shaped like what the data tools return, built from the committed sample bundle (real output) plus a few synthetic record-level ones.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../score/sample_bundle");
const read = (rel) => JSON.parse(readFileSync(path.join(root, rel), "utf8"));
const manifest = read("manifest.json");
const index = read("companies.json");
const alertsFeed = read("alerts.json");

function company(id) { return read(`companies/${id}.json`); }
const withControl = index.companies.map((c) => c.company_id).find((id) => (company(id).control || []).length >= 4 && company(id).forecast);
const doc = company(withControl);
const label = Object.fromEntries(manifest.spec.items.map((i) => [i.id, i.label]));

// get_control_chart(...) rows
const chart = doc.control.find((c) => c.comparison === "own_history" && c.metric === "score");
const control = chart.months.map((m, i) => ({ month: m, value: chart.values[i], center: chart.center[i], lower: chart.lower[i], upper: chart.upper[i], ewma: chart.ewma[i], signal: chart.signal[i], persistent: chart.persistent[i] }));

// explain_change rows: the sampled company-month with the biggest item movement, and its two totals
let best = null;
for (const c of index.companies) {
  const d = company(c.company_id);
  d.months.forEach((r, i) => {
    if (!r.items || i === 0) return;
    const size = Object.values(r.items).reduce((s, v) => s + Math.abs(v.delta ?? 0), 0);
    if (!best || size > best.size) best = { size, cur: r, prev: d.months[i - 1] };
  });
}
const { cur, prev } = best;
const deltas = Object.entries(cur.items).filter(([, v]) => v.delta !== null && Math.abs(v.delta) > 0.05).map(([k, v]) => ({ item: k, label: label[k], value: Math.round(v.delta * 100) / 100 }));
if (cur.change_guard) deltas.push({ item: "guard", label: "Activity guard", value: cur.change_guard });
const totals = [{ label: prev.month, value: prev.score }, { label: cur.month, value: cur.score }];

// get_score_matrix rows: company x month
const months = index.months;
const matrix = index.companies.flatMap((c) => months.map((m, i) => ({ month: m, entity_id: c.company_id, value: c.scores[i], group_id: c.group_id })));
const target = index.companies[0].company_id;

// portfolio_query rows
const portfolio = index.companies.map((c) => ({ company_id: c.company_id, score: c.score, delta_3m: c.delta_3m ?? 0, n_alerts: c.n_alerts, trajectory: c.trajectory, cluster_id: c.cluster_id }));

// compare_with_cluster rows
const percentiles = (doc.cluster?.vs_cluster || []).filter((v) => v.percentile !== null).map((v) => ({ metric: v.metric, percentile: v.percentile }));

// get_forecast rows
const hist = doc.months.slice(-12).map((r) => ({ month: r.month, score: r.score }));
const forecast = [...hist.map((h, i) => ({ month: h.month, score: h.score, ...(i === hist.length - 1 ? { median: h.score, lo50: h.score, hi50: h.score, lo80: h.score, hi80: h.score } : {}) })),
  ...doc.forecast.points.map((p) => ({ month: p.month, median: p.median, lo50: p.lo50, hi50: p.hi50, lo80: p.lo80, hi80: p.hi80 }))].map((r) => ({ naive_last: doc.forecast.naive_last, ...r }));

// get_alerts rows
const alerts = alertsFeed.alerts.map((a) => ({ month: a.month, kind: a.kind, severity: a.severity, owner: a.owner, entity_id: a.entity.id }));
const alertsByMonth = Object.values(alerts.reduce((acc, a) => { const k = `${a.month}|${a.kind}`; acc[k] = acc[k] || { month: a.month, kind: a.kind, n: 0 }; acc[k].n++; return acc; }, {}));

// query_clean_db style rows (synthetic, record level)
const customers = [["COUNTERPARTY_63726", 13065], ["COUNTERPARTY_29052", 9450], ["COUNTERPARTY_49362", 17424], ["COUNTERPARTY_21700", 555], ["COUNTERPARTY_49838", 46126], ["COUNTERPARTY_05161", 308550], ["COUNTERPARTY_49590", 112694]]
  .map(([customer, open_eur]) => ({ customer, open_eur }));
const flows = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"].map((m, i) => ({ month: m, inflow: 120000 + i * 4000 - (i === 4 ? 30000 : 0), outflow: 110000 + i * 2500 + (i === 5 ? 20000 : 0) }));

export const fixtures = { control, deltas, totals, matrix, portfolio, percentiles, forecast, alerts, alertsByMonth, customers, flows };
export const meta = { company: withControl, target };
