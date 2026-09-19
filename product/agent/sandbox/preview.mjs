// Renders every recipe of the prompt (dry run, jsdom) into one static HTML page with the app's theme variables, to eyeball the look.
//   node preview.mjs [out.html] [dark]
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { dryRun } from "./dryrun.mjs";
import { fixtures, meta } from "./fixtures.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const out = process.argv[2] ?? path.join(here, "preview.html");
const dark = process.argv[3] === "dark";
const prompt = readFileSync(path.join(here, "../prompts/viz_system.md"), "utf8").replace(/\r/g, "");
const blocks = [...prompt.matchAll(/```jsx\n([\s\S]*?)```/g)].map((m) => m[1]);
const rename = (rows) => rows.map((r) => (r.entity_id === meta.target ? { ...r, entity_id: "COMP_0016" } : r));

const light = `--background:#fff;--foreground:#171717;--popover:#fff;--popover-foreground:#171717;--muted:#f4f4f4;--muted-foreground:#737373;--border:#e5e5e5;
--sentinel-risk:oklch(0.577 0.245 27);--sentinel-opportunity:oklch(0.62 0.15 150);--sentinel-info:oklch(0.55 0.15 255);--sentinel-neutral:oklch(0.45 0 0);--sentinel-accent:oklch(0.68 0.16 70);
--chart-1:oklch(0.55 0.15 255);--chart-2:oklch(0.62 0.15 150);--chart-3:oklch(0.68 0.16 70);--chart-4:oklch(0.577 0.2 27);--chart-5:oklch(0.55 0.18 305);--chart-6:oklch(0.6 0.12 200);--chart-7:oklch(0.5 0 0);--chart-8:oklch(0.7 0.12 340);`;
const night = `--background:#171717;--foreground:#fafafa;--popover:#262626;--popover-foreground:#fafafa;--muted:#262626;--muted-foreground:#a3a3a3;--border:#ffffff1a;
--sentinel-risk:oklch(0.704 0.191 22);--sentinel-opportunity:oklch(0.75 0.15 150);--sentinel-info:oklch(0.72 0.14 255);--sentinel-neutral:oklch(0.75 0 0);--sentinel-accent:oklch(0.8 0.15 80);
--chart-1:oklch(0.72 0.14 255);--chart-2:oklch(0.75 0.15 150);--chart-3:oklch(0.8 0.15 80);--chart-4:oklch(0.7 0.18 22);--chart-5:oklch(0.72 0.16 305);--chart-6:oklch(0.75 0.11 200);--chart-7:oklch(0.7 0 0);--chart-8:oklch(0.78 0.11 340);`;

const cards = [];
for (const code of blocks) {
  const spec = /fixtures:\s*(.*)$/m.exec(code)?.[1] ?? "";
  const title = /\/\/ title:\s*(.*)$/m.exec(code)?.[1] ?? "";
  const datasets = {};
  for (const part of spec.split(",").map((s) => s.trim()).filter(Boolean)) { const [id, n] = part.split("="); datasets[id] = rename(fixtures[n]); }
  const r = await dryRun({ code, title, note: "n", datasets, height: 300 });
  cards.push(`<section><h3>${title}</h3><div class="p">${r.ok ? r.html : "ERROR " + r.errors.join(" | ")}</div></section>`);
}
writeFileSync(out, `<!doctype html><meta charset=utf-8><style>:root{${dark ? night : light}}
body{margin:0;padding:16px;background:var(--background);color:var(--foreground);font:13px system-ui,sans-serif}
.g{display:grid;grid-template-columns:repeat(2,minmax(0,640px));gap:18px}
section{border:1px solid var(--border);border-radius:10px;padding:12px}h3{margin:0 0 6px;font-size:13px}
text{fill:var(--muted-foreground)}</style><div class="g">${cards.join("")}</div>`);
console.log(`wrote ${out}`);
process.exit(0);
