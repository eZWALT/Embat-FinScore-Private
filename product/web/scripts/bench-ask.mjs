/**
 * Ask latency: prompt size + Helmcode TTFT (tiny vs full catalog).
 * Neon tool times run only when DATABASE_URL is set.
 *
 *   cd product/web && node scripts/bench-ask.mjs
 */
import { readFileSync, existsSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");

function loadDotEnv(path) {
  if (!existsSync(path)) return;
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq < 1) continue;
    const key = trimmed.slice(0, eq);
    let value = trimmed.slice(eq + 1);
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (!process.env[key]) process.env[key] = value;
  }
}

function fmt(ms) {
  if (ms >= 60_000) return `${(ms / 60_000).toFixed(2)} min`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${Math.round(ms)} ms`;
}

function minutes(ms) {
  return (ms / 60_000).toFixed(4);
}

function loadPrompts() {
  const dir = resolve(root, "src/lib/agent/prompts");
  const names = [
    "chat_system.md",
    "product_context.md",
    "wording_rules.md",
    "watcher_format.md",
    "tools_catalog.md",
    "clean_schema.md",
  ];
  const parts = [];
  const sizes = [];
  for (const name of names) {
    const text = readFileSync(resolve(dir, name), "utf8");
    sizes.push({ name, chars: text.length });
    parts.push(text);
  }
  const session = "session\ncompany_id=COMP_0085\nas_of=2026-08";
  parts.push(session);
  return { system: parts.join("\n\n---\n\n"), sizes };
}

async function helmcodeTtft(system, user) {
  const apiKey = process.env.HELMCODE_API_KEY?.trim();
  if (!apiKey) throw new Error("HELMCODE_API_KEY is not set");
  const base = (process.env.HELMCODE_BASE_URL?.trim() || "https://api.helmcode.com/v1").replace(/\/$/, "");
  const model = process.env.POC_LLM_MODEL?.trim() || "deepseek-v4-flash";
  const started = Date.now();
  const response = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      stream: true,
      temperature: 0.2,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    }),
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Helmcode ${response.status}: ${body.slice(0, 200)}`);
  }
  let ttft = null;
  let chars = 0;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;
      try {
        const json = JSON.parse(payload);
        const delta = json.choices?.[0]?.delta?.content;
        if (typeof delta === "string" && delta.length) {
          if (ttft == null) ttft = Date.now() - started;
          chars += delta.length;
        }
      } catch {
        /* keep reading */
      }
    }
  }
  return { ttft_ms: ttft, total_ms: Date.now() - started, text_chars: chars };
}

async function pingNeon() {
  const url = process.env.DATABASE_URL;
  if (!url) return null;
  const { neon } = await import("@neondatabase/serverless");
  const sql = neon(url);
  const samples = [];
  for (const [name, query] of [
    ["api.manifest", "SELECT as_of_month FROM api.manifest LIMIT 1"],
    ["get_company-shaped", "SELECT company_id, month, score FROM analytics.company_scores JOIN api.current_run r ON r.run_id = analytics.company_scores.run_id WHERE company_id = 'COMP_0085' ORDER BY month DESC LIMIT 1"],
    ["get_alerts-shaped", "SELECT count(*)::int AS n FROM analytics.alerts a JOIN api.current_run r ON r.run_id = a.run_id WHERE entity_id = 'COMP_0085'"],
    ["core.invoices", "SELECT count(*)::int AS n FROM core.invoices WHERE company_id = 'COMP_0085' LIMIT 1"],
  ]) {
    const started = performance.now();
    try {
      await sql.query(query);
      samples.push({ name, ms: performance.now() - started, ok: true });
    } catch (error) {
      samples.push({
        name,
        ms: performance.now() - started,
        ok: false,
        detail: error instanceof Error ? error.message.split("\n")[0] : String(error),
      });
    }
  }
  return samples;
}

function report(input) {
  const lines = [
    "# Ask agent benchmark",
    "",
    `- When: ${input.when}`,
    `- Model: ${input.model}`,
    `- Prompt files: ${input.promptChars.toLocaleString("en")} chars (complete catalog included)`,
    "",
    "The Cursor-style tool trace is render-only. It does not add a model round-trip.",
    "The clock is Helmcode (TTFT / completion) and Neon (tool SQL). Minutes = ms / 60000.",
    "",
    "## Prompt pieces",
    "",
    "| File | chars |",
    "|---|---:|",
  ];
  for (const row of input.sizes) {
    lines.push(`| \`${row.name}\` | ${row.chars.toLocaleString("en")} |`);
  }
  lines.push(
    "",
    "## Time to first token (Helmcode, no tools)",
    "",
    "| System | TTFT | Completion | TTFT (min) |",
    "|---|---:|---:|---:|",
  );
  for (const row of input.llm) {
    lines.push(
      `| ${row.label} | ${row.ttft_ms == null ? "—" : fmt(row.ttft_ms)} | ${fmt(row.total_ms)} | ${row.ttft_ms == null ? "—" : minutes(row.ttft_ms)} |`,
    );
  }
  lines.push("", "## Neon retrieval (same queries the tools run)", "", "| Query | p50-ish (1 shot) | min | ok | note |", "|---|---:|---:|---|---|");
  if (!input.neon) {
    lines.push("| — | — | — | — | DATABASE_URL not set in this environment |");
  } else {
    for (const row of input.neon) {
      lines.push(`| \`${row.name}\` | ${fmt(row.ms)} | ${minutes(row.ms)} | ${row.ok ? "yes" : "no"} | ${row.detail ?? ""} |`);
    }
  }
  lines.push("");
  return lines.join("\n");
}

async function main() {
  loadDotEnv(resolve(root, ".env.local"));
  loadDotEnv(resolve(root, ".env"));
  const { system, sizes } = loadPrompts();
  const model = process.env.POC_LLM_MODEL?.trim() || "deepseek-v4-flash";
  const llm = [];
  if (process.env.HELMCODE_API_KEY?.trim()) {
    const tinyRounds = [];
    const fullRounds = [];
    for (let i = 0; i < 3; i += 1) {
      tinyRounds.push(await helmcodeTtft("Eres un asistente. Responde en una frase.", "di ok"));
      fullRounds.push(await helmcodeTtft(system, "di ok"));
    }
    const mid = (rows, key) => [...rows].map((row) => row[key]).filter((n) => n != null).sort((a, b) => a - b)[1] ?? rows[0][key];
    llm.push({
      label: "Tiny (40 chars), median of 3",
      ttft_ms: mid(tinyRounds, "ttft_ms"),
      total_ms: mid(tinyRounds, "total_ms"),
    });
    llm.push({
      label: "Full Ask prompt + catalog, median of 3",
      ttft_ms: mid(fullRounds, "ttft_ms"),
      total_ms: mid(fullRounds, "total_ms"),
    });
  } else {
    throw new Error("HELMCODE_API_KEY is not set");
  }
  const neon = await pingNeon();
  const text = report({
    when: new Date().toISOString(),
    model,
    promptChars: system.length,
    sizes,
    llm,
    neon,
  });
  writeFileSync(resolve(root, "src/lib/agent/BENCHMARK.md"), text);
  process.stdout.write(text);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
