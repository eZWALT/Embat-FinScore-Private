// Static checks on model-written chart code and its texts. A first line of defence and a source of clear messages for the model;
// the real security boundary is the sandboxed iframe (VIZ_SPEC.md).
export const LIMITS = { codeBytes: 14000, rowsPerDataset: 2000, datasets: 6 };

const BANNED = [
  [/\bfetch\s*\(|XMLHttpRequest|WebSocket|EventSource|sendBeacon/, "no network access"],
  [/\beval\s*\(|new\s+Function|\bFunction\s*\(|setTimeout\s*\(\s*["'`]/, "no dynamic code"],
  [/\bimport\s*\(|\brequire\s*\(/, "no dynamic import or require: use static `import … from \"react\" | \"recharts\" | \"sentinel\"`"],
  [/\b(window|document|globalThis|self|parent|top|frames)\b\s*[.\[]/, "no access to window/document/parent"],
  [/localStorage|sessionStorage|indexedDB|document\.cookie|navigator\./, "no browser storage or navigator"],
  [/postMessage|addEventListener/, "the host handles messaging and events"],
  [/dangerouslySetInnerHTML|<script|<iframe|<object|<embed|<link\b|<style\b/i, "no raw HTML, scripts, iframes or style tags"],
  [/https?:\/\/|url\s*\(/i, "no URLs or external resources"],
  [/<img\b|<image\b/i, "no images"],
];
const IMPORT_FROM = /import\s+[^;]*?from\s+["']([^"']+)["']/g;
const ALLOWED_IMPORTS = new Set(["react", "recharts", "sentinel"]);
const WORDING = [
  [/predice|predicts?\b|predictive|predicci/i, "the score explains and monitors; it never predicts"],
  [/probabilidad|probability|likelihood/i, "no probabilities"],
  [/quiebra|bankrupt|default risk|impago/i, "no bankruptcy or default language"],
  [/ingresos en riesgo|revenue at risk/i, "top customer wording: \"stopped billing, review exposure and collections\""],
  [/rank_score/, "rank_score is for ordering alerts only and is never plotted"],
];

export function lint({ code, title = "", subtitle = "", note = "", datasets = {} }) {
  const errors = [];
  const bytes = Buffer.byteLength(code);
  if (bytes > LIMITS.codeBytes) errors.push(`code is ${bytes} bytes; the limit is ${LIMITS.codeBytes}. Simplify.`);
  for (const [re, msg] of BANNED) if (re.test(code)) errors.push(msg);
  for (const m of code.matchAll(IMPORT_FROM)) if (!ALLOWED_IMPORTS.has(m[1])) errors.push(`import from "${m[1]}" is not available; only "react", "recharts" and "sentinel"`);
  if (!/export\s+default\s+/.test(code)) errors.push("export a default component: `export default function Chart({ data, height }) { … }`");
  if (!note.trim()) errors.push("`note` is required: the source (bundle or records, as-of month) and any caveat");
  if (!title.trim()) errors.push("`title` is required");
  if (Object.keys(datasets).length > LIMITS.datasets) errors.push(`at most ${LIMITS.datasets} datasets per chart`);
  for (const [name, rows] of Object.entries(datasets)) {
    if (!Array.isArray(rows)) errors.push(`dataset ${name} is not an array of rows`);
    else if (rows.length > LIMITS.rowsPerDataset) errors.push(`dataset ${name} has ${rows.length} rows; the limit is ${LIMITS.rowsPerDataset}. Aggregate in the tool.`);
  }
  const texts = [title, subtitle, note, code].join("\n");
  for (const [re, msg] of WORDING) if (re.test(texts)) errors.push(`wording: ${msg}`);
  return errors;
}
