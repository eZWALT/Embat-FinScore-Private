// What a model-written chart may import. In the browser the same names are provided inside the sandboxed iframe
// (product/agent/VIZ_SPEC.md); this file is the reference used by the server-side dry run and the tests.
import React from "react";
import * as Recharts from "recharts";

const monthFmt = new Intl.DateTimeFormat("es-ES", { month: "short", year: "2-digit", timeZone: "UTC" });
const compactEur = (currency) => new Intl.NumberFormat("es-ES", { style: "currency", currency: currency || "EUR", notation: "compact", maximumFractionDigits: 1 });
const isNum = (v) => typeof v === "number" && Number.isFinite(v);

export const fmt = {
  /** "2026-08" -> "ago ’26" */
  month: (m) => { const [y, mo] = String(m).split("-").map(Number); return monthFmt.format(new Date(Date.UTC(y, mo - 1, 1))).replace(" ", " ’"); },
  eur: (v, currency) => (isNum(v) ? compactEur(currency).format(v) : "–"),
  pts: (v, signed = false) => { if (!isNum(v)) return "–"; const t = Math.abs(v).toFixed(1).replace(".", ","); return signed ? (v > 0 ? `+${t}` : v < 0 ? `−${t}` : t) : t; },
  pct: (v, digits = 0) => (isNum(v) ? `${(v * 100).toFixed(digits).replace(".", ",")} %` : "–"),
  days: (v) => (isNum(v) ? `${Math.round(v)} d` : "–"),
  num: (v, digits = 0) => (isNum(v) ? v.toLocaleString("es-ES", { maximumFractionDigits: digits }) : "–"),
};

/** Semantic colours: CSS variables the host defines in light and dark. Never write hex or oklch in a chart. */
export const tones = {
  risk: "var(--sentinel-risk)",
  opportunity: "var(--sentinel-opportunity)",
  info: "var(--sentinel-info)",
  neutral: "var(--sentinel-neutral)",
  accent: "var(--sentinel-accent)",
  muted: "var(--muted-foreground)",
  grid: "var(--border)",
  text: "var(--foreground)",
  /** categorical palette for series, up to 8 */
  series: ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)", "var(--chart-6)", "var(--chart-7)", "var(--chart-8)"],
};

/** 0 = red, 50 = amber, 100 = green (same scale as the web app's heatmap). */
export function scoreColor(score) {
  if (!isNum(score)) return undefined;
  const s = Math.max(0, Math.min(100, score));
  const hue = s <= 50 ? 25 + (s / 50) * 60 : 85 + ((s - 50) / 50) * 60;
  return `oklch(0.74 0.15 ${hue.toFixed(0)})`;
}

/** Diverging scale for changes and z-scores: negative red, positive green, centred on 0 (range = the value that reaches full colour). */
export function divergingColor(value, range = 10) {
  if (!isNum(value)) return undefined;
  const t = Math.max(-1, Math.min(1, value / range));
  return t < 0 ? `oklch(${0.9 + t * 0.16} ${Math.abs(t) * 0.15} 25)` : `oklch(${0.9 - t * 0.16} ${t * 0.15} 145)`;
}

/** Sizes the chart to its panel and adds an accessible label. Child: one Recharts chart element (LineChart, BarChart, ComposedChart…).
 *  In the browser it is responsive; the server-side dry run has no layout, so it renders at a fixed width (globalThis.__DRYRUN_WIDTH). */
export function ChartContainer({ height = 300, label, children }) {
  const fixed = globalThis.__DRYRUN_WIDTH;
  const inner = fixed
    ? React.cloneElement(React.Children.only(children), { width: fixed, height })
    : React.createElement(Recharts.ResponsiveContainer, { width: "100%", height }, children);
  return React.createElement("div", { role: "img", "aria-label": label || "chart", style: { width: "100%", height } }, inner);
}

/** Shared look: spread these on Recharts elements so every chart matches the app (`<XAxis {...look.axis} />`, `<Tooltip {...look.tooltip} />`, `<CartesianGrid {...look.grid} />`). */
export const look = {
  axis: { tick: { fontSize: 11, fill: "var(--muted-foreground)" }, tickLine: false, axisLine: false },
  grid: { stroke: "var(--border)", strokeDasharray: "3 3", vertical: false },
  tooltip: { cursor: { stroke: "var(--border)" }, contentStyle: { background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12, color: "var(--popover-foreground)" } },
  legend: { iconType: "plainline", wrapperStyle: { fontSize: 11 } },
};

/** Spanish names for the bundle's identifiers, the same wording as the web app (product/web/src/components/group/labels.ts). Use them for axis, legend and tooltip text. */
export const labels = {
  item: {
    delay_paid: "Retraso en pagos a proveedores", delay_coll: "Retraso en cobros de clientes", ap_overdue30: "Facturas de proveedores vencidas",
    ar_overdue30: "Facturas de clientes vencidas", runway: "Cobertura de caja", neg_liq: "Periodos con liquidez negativa", neg_episodes: "Episodios de caja negativa",
    ds_ratio: "Servicio de deuda", fc_ratio: "Costes financieros", months_observed: "Longitud del historial", active_share: "Recurrencia de entradas de caja",
    out_vol: "Volatilidad de salidas", ds_increase: "Aumento del servicio de deuda", fc_increase: "Aumento de costes financieros",
    cust_tail: "Concentración de clientes", credit_note: "Peso de las notas de crédito", guard: "Límite de seguridad aplicado", top_customer: "Cliente principal sin facturar",
  },
  category: { payment_history: "Historial de pagos", amounts_owed: "Liquidez y deuda", stability: "Estabilidad", new_credit: "Nuevo crédito", mix: "Mix de clientes" },
  trajectory: { improving: "Mejorando", stable: "Estable", dip: "Caída puntual", deteriorating: "Deteriorándose", "insufficient history": "Historial insuficiente" },
  confidence: { high: "Alta", medium: "Media", low: "Baja" },
  guard: { dark: "Sin movimientos", fading: "Entradas en caída" },
  severity: { act: "Actuar", watch: "Vigilar", info: "Informativa" },
  owner: { treasurer: "Tesorero", cfo: "CFO", collections: "Cobros" },
  kind: {
    score_deterioration: "Deterioro del score", score_improvement: "Mejora del score", category_drop: "Caída de una categoría",
    going_dark: "Sin movimientos bancarios", top_customer_quiet: "Cliente principal sin facturar",
  },
  metric: { score: "Score", payment_history: "Historial de pagos", amounts_owed: "Liquidez y deuda", stability: "Estabilidad", new_credit: "Nuevo crédito", mix: "Mix de clientes" },
};

export const sentinel = { fmt, tones, scoreColor, divergingColor, ChartContainer, look, labels };
export const modules = { react: React, recharts: Recharts, sentinel };
export { React, Recharts };
