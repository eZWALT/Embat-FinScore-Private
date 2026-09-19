// Runs every ```jsx block of prompts/viz_system.md through the dry run (lint, compile, render in jsdom) against fixtures built from the real
// sample bundle, plus negative tests for the lint. Run: `npm test` in this folder.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { dryRun } from "./dryrun.mjs";
import { fixtures, meta } from "./fixtures.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const prompt = readFileSync(path.join(here, "../prompts/viz_system.md"), "utf8").replace(/\r/g, "");
const blocks = [...prompt.matchAll(/```jsx\n([\s\S]*?)```/g)].map((m) => m[1]);
let failed = 0;
const say = (ok, msg) => { console.log(`${ok ? "ok  " : "FAIL"} ${msg}`); if (!ok) failed++; };

const rename = (rows) => rows.map((r) => (r.entity_id === meta.target ? { ...r, entity_id: "COMP_0016" } : r));

for (const [i, code] of blocks.entries()) {
  const head = code.split("\n")[0];
  const spec = /fixtures:\s*(.*)$/.exec(head)?.[1] ?? "";
  const datasets = {};
  for (const part of spec.split(",").map((s) => s.trim()).filter(Boolean)) {
    const [id, name] = part.split("=");
    if (!fixtures[name]) { say(false, `block ${i + 1}: unknown fixture ${name}`); continue; }
    datasets[id] = rename(fixtures[name]);
  }
  const title = /\/\/ title:\s*(.*)$/m.exec(code)?.[1] ?? "t";
  const r = await dryRun({ code, title, subtitle: "s", note: "Fuente: bundle a 2026-08.", datasets, height: 300 });
  const detail = r.ok ? `${r.svgs} svg, ${r.bytes} bytes${r.warnings.length ? ", warnings: " + r.warnings.join(" | ") : ""}` : `${r.stage}: ${r.errors.join(" | ")}`;
  say(r.ok && r.warnings.length === 0, `block ${i + 1} (${title}) ${detail}`);
}
say(blocks.length >= 8, `${blocks.length} jsx blocks found in the prompt`);

// negative tests: the lint and the dry run must reject these with a message
const bad = {
  "network": `export default function C(){ fetch("/x"); return <div/>; }`,
  "window": `export default function C(){ return <div>{window.location.href}</div>; }`,
  "url": `export default function C(){ return <div style={{background:"url(x.png)"}}/>; }`,
  "import lodash": `import _ from "lodash"; export default function C(){ return <div/>; }`,
  "predict wording": `export default function C(){ return <div>predice quiebra</div>; }`,
  "rank_score": `export default function C({data}){ return <div>{data.ds_1[0].rank_score}</div>; }`,
  "no default export": `export function C(){ return <div/>; }`,
  "syntax": `export default function C(){ return <div>; }`,
  "throws": `export default function C({data}){ return <div>{data.ds_9.length}</div>; }`,
};
for (const [name, code] of Object.entries(bad)) {
  const r = await dryRun({ code, title: "t", note: "n", datasets: { ds_1: [{ a: 1 }] } });
  say(!r.ok, `rejects ${name}: ${r.ok ? "ACCEPTED" : r.errors[0]}`);
}
const noNote = await dryRun({ code: `export default function C(){ return <div/>; }`, title: "t", note: "", datasets: {} });
say(!noNote.ok, `rejects a chart without note: ${noNote.errors?.[0]}`);
const big = await dryRun({ code: `export default function C(){ return <div/>; }`, title: "t", note: "n", datasets: { ds_1: Array.from({ length: 2001 }, () => ({})) } });
say(!big.ok, `rejects 2001 rows: ${big.errors?.[0]}`);

console.log(failed ? `\n${failed} FAILED` : "\nall passed");
process.exit(failed ? 1 : 0);
