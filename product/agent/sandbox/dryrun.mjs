// Server-side dry run of a model-written chart: lint -> compile (sucrase) -> evaluate the module in a fresh VM context -> render it in jsdom
// at a fixed size -> inspect the DOM. Returns errors the model can act on.
// NOT a security boundary: the module is evaluated in a vm context, but rendering runs in this process. In production run the dry run in a
// worker or child process without privileges or network; the browser iframe sandbox is the real boundary (VIZ_SPEC.md).
import vm from "node:vm";
import { JSDOM } from "jsdom";
import { transform } from "sucrase";
import { lint } from "./lint.mjs";

let env;
async function setup() {
  if (env) return env;
  const dom = new JSDOM("<!doctype html><div id=root></div>", { pretendToBeVisual: true });
  const w = dom.window;
  Object.assign(globalThis, { window: w, document: w.document, HTMLElement: w.HTMLElement, SVGElement: w.SVGElement, Node: w.Node });
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
  globalThis.requestAnimationFrame = (f) => setTimeout(f, 0);
  globalThis.cancelAnimationFrame = clearTimeout;
  globalThis.__DRYRUN_WIDTH = 640;
  const { modules, React } = await import("./runtime.mjs");
  const { createRoot } = await import("react-dom/client");
  const { flushSync } = await import("react-dom");
  env = { w, modules, React, createRoot, flushSync };
  return env;
}

export async function dryRun({ code, title, subtitle, note, datasets = {}, height = 300 }) {
  const errors = lint({ code, title, subtitle, note, datasets });
  if (errors.length) return { ok: false, stage: "lint", errors };
  let js;
  try {
    js = transform(code, { transforms: ["jsx", "imports"], production: true }).code;
  } catch (e) {
    return { ok: false, stage: "compile", errors: [`syntax error: ${String(e.message).split("\n")[0]}`] };
  }
  const { w, modules, React, createRoot, flushSync } = await setup();
  const exports = {};
  const ctx = vm.createContext({
    exports, module: { exports }, React,
    require: (name) => { if (!(name in modules)) throw new Error(`module "${name}" is not available`); return modules[name]; },
    console: { log() {}, warn() {}, error() {} },
  });
  const problems = [];
  const originalError = console.error;
  console.error = (...a) => problems.push(String(a[0]).split("\n")[0]);
  let html = "";
  try {
    vm.runInContext(js, ctx, { timeout: 1500 });
    if (typeof exports.default !== "function") return { ok: false, stage: "run", errors: ["the default export must be a component function"] };
    const host = w.document.createElement("div");
    w.document.body.append(host);
    const root = createRoot(host);
    flushSync(() => root.render(React.createElement(exports.default, { data: datasets, height })));
    await new Promise((r) => setTimeout(r, 30));
    html = host.innerHTML;
    root.unmount();
    host.remove();
  } catch (e) {
    return { ok: false, stage: "render", errors: [`${e.name}: ${String(e.message).split("\n")[0]}`] };
  } finally {
    console.error = originalError;
  }
  const react = problems.filter((p) => /key|Invalid|Unknown|cannot|Warning|Error/i.test(p));
  if (html.length < 200) return { ok: false, stage: "render", errors: ["the component rendered nothing visible", ...react] };
  const warnings = [...react.map((p) => `React: ${p}`)];
  if (/NaN|undefined|Infinity/.test(html.replace(/aria-label="[^"]*"/g, "").replace(/ name="undefined"/g, ""))) warnings.push("the output contains NaN/undefined/Infinity: check the columns you read and how you handle null values");
  if (!/recharts-wrapper|recharts-surface|<table|<svg|grid/.test(html)) warnings.push("nothing chart-like was rendered: check the panel is not empty");
  return { ok: true, warnings, bytes: html.length, svgs: (html.match(/<svg/g) || []).length, html };
}
