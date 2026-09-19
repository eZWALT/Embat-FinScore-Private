/**
 * Compare DeepSeek reasoning_effort on Helmcode (same Ask question + tools).
 *
 *   cd product/web && node scripts/bench-thinking.mjs
 *
 * Writes src/lib/agent/THINKING_BENCH.md
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");

const ASK_FILES = [
  "prompt_map.md",
  "chat_system.md",
  "scope.md",
  "product_context.md",
  "wording_rules.md",
  "watcher_format.md",
  "tools_catalog.md",
  "plots_catalog.md",
  "clean_schema.md",
  "thinking.md",
];

const LEVELS = [
  { id: "off", thinking: false, effort: null },
  { id: "low", thinking: true, effort: "low" },
  { id: "high", thinking: true, effort: "high" },
  { id: "max", thinking: true, effort: "max" },
];

const QUESTION = "¿Por qué este índice este mes? Usa las herramientas. No inventes números.";

const ASK_TOOLS = [
  {
    type: "function",
    function: {
      name: "get_company",
      description: "Score, trajectory, reasons with EUR of one company.",
      parameters: {
        type: "object",
        properties: { company_id: { type: "string" }, month: { type: "string" } },
        required: ["company_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_alerts",
      description: "Alerts from the feed.",
      parameters: {
        type: "object",
        properties: { entity_id: { type: "string" } },
      },
    },
  },
];

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

function loadHelmcodeKey() {
  if (process.env.HELMCODE_API_KEY?.trim()) return;
  const file = process.env.HELMCODE_API_KEY_FILE?.trim() || resolve(homedir(), ".helmcode_key");
  if (existsSync(file)) process.env.HELMCODE_API_KEY = readFileSync(file, "utf8").trim();
}

function fmt(ms) {
  if (ms == null) return "—";
  if (ms >= 60_000) return `${(ms / 60_000).toFixed(2)} min`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${Math.round(ms)} ms`;
}

function client() {
  const apiKey = process.env.HELMCODE_API_KEY?.trim();
  if (!apiKey) throw new Error("HELMCODE_API_KEY is not set");
  const base = (process.env.HELMCODE_BASE_URL?.trim() || "https://api.helmcode.com/v1").replace(/\/$/, "");
  const model = process.env.POC_LLM_MODEL?.trim() || "deepseek-v4-flash";
  return { apiKey, base, model };
}

function loadSystem() {
  const dir = resolve(root, "src/lib/agent/prompts");
  const parts = ASK_FILES.map((name) => readFileSync(resolve(dir, name), "utf8"));
  parts.push("session\ncompany_id=COMP_0462\ngroup_id=GROUP_0194\nas_of=2026-08");
  return parts.join("\n\n---\n\n");
}

function stub(name) {
  if (name === "get_company") {
    return {
      company_id: "COMP_0462",
      month: "2026-08",
      score: 68,
      trajectory: "deteriorating",
      confidence: "medium",
      reasons: [{ sentence: "El cliente principal ha dejado de facturar. Revisa la exposición y los cobros.", eur: 14000 }],
      change_reasons: [{ sentence: "La categoría de historial de pagos ha bajado 6 puntos.", eur: null }],
    };
  }
  return {
    n_matching: 1,
    alerts: [{ kind: "top_customer_quiet", title: "El cliente principal ha dejado de facturar", owner: "Cobros", action: "Revisa la exposición y los cobros." }],
  };
}

async function streamTurn({ messages, tools, thinking, effort }) {
  const { apiKey, base, model } = client();
  const body = {
    model,
    stream: true,
    temperature: 0.2,
    messages,
    thinking: { type: thinking ? "enabled" : "disabled" },
    ...(thinking && effort ? { reasoning_effort: effort } : {}),
    ...(tools ? { tools, tool_choice: "auto" } : {}),
  };
  const started = Date.now();
  const response = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Helmcode ${response.status}: ${text.slice(0, 400)}`);
  }
  let firstReasoningMs = null;
  let firstTextMs = null;
  let firstToolMs = null;
  let lastMs = null;
  let text = "";
  let reasoning = "";
  let finish = null;
  const calls = new Map();
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
      let json;
      try {
        json = JSON.parse(payload);
      } catch {
        continue;
      }
      const choice = json.choices?.[0];
      if (!choice) continue;
      if (choice.finish_reason) finish = choice.finish_reason;
      const delta = choice.delta ?? {};
      const now = Date.now() - started;
      const thought = delta.reasoning_content ?? delta.reasoning;
      if (typeof thought === "string" && thought.length) {
        if (firstReasoningMs == null) firstReasoningMs = now;
        lastMs = now;
        reasoning += thought;
      }
      if (typeof delta.content === "string" && delta.content.length) {
        if (firstTextMs == null) firstTextMs = now;
        lastMs = now;
        text += delta.content;
      }
      for (const call of delta.tool_calls ?? []) {
        if (firstToolMs == null) firstToolMs = now;
        lastMs = now;
        const index = call.index ?? 0;
        const prev = calls.get(index) ?? { id: "", name: "", arguments: "" };
        if (call.id) prev.id = call.id;
        if (call.function?.name) prev.name = call.function.name;
        if (call.function?.arguments) prev.arguments += call.function.arguments;
        calls.set(index, prev);
      }
    }
  }
  return {
    first_reasoning_ms: firstReasoningMs,
    first_text_ms: firstTextMs,
    first_tool_ms: firstToolMs,
    last_token_ms: lastMs,
    total_ms: Date.now() - started,
    text,
    reasoning_chars: reasoning.length,
    finish,
    tool_calls: [...calls.values()].map((call) => ({
      id: call.id || `call_${call.name}`,
      name: call.name,
      arguments: call.arguments || "{}",
    })),
  };
}

async function runLevel(level, system) {
  const messages = [
    { role: "system", content: system },
    { role: "user", content: QUESTION },
  ];
  const clock = Date.now();
  const toolsUsed = [];
  for (let step = 0; step < 4; step += 1) {
    const turn = await streamTurn({
      messages,
      tools: ASK_TOOLS,
      thinking: level.thinking,
      effort: level.effort,
    });
    toolsUsed.push(...turn.tool_calls.map((call) => call.name));
    if (turn.finish === "tool_calls" && turn.tool_calls.length) {
      messages.push({
        role: "assistant",
        content: turn.text || null,
        tool_calls: turn.tool_calls.map((call) => ({
          id: call.id,
          type: "function",
          function: { name: call.name, arguments: call.arguments },
        })),
      });
      for (const call of turn.tool_calls) {
        messages.push({
          role: "tool",
          tool_call_id: call.id,
          content: JSON.stringify(stub(call.name)),
        });
      }
      continue;
    }
    return {
      id: level.id,
      effort: level.effort,
      first_reasoning_ms: turn.first_reasoning_ms,
      first_visible_ms: turn.first_reasoning_ms ?? turn.first_tool_ms ?? turn.first_text_ms,
      first_tool_ms: turn.first_tool_ms,
      first_text_ms: turn.first_text_ms,
      last_token_ms: turn.last_token_ms,
      total_ms: Date.now() - clock,
      reasoning_chars: turn.reasoning_chars,
      tools: toolsUsed,
      answer_chars: turn.text.trim().length,
      has_68: /\b68\b/.test(turn.text),
      has_cobros: /cobros/i.test(turn.text),
    };
  }
  return { id: level.id, effort: level.effort, error: "hit step cap", total_ms: Date.now() - clock, tools: toolsUsed };
}

function pick(rows) {
  const ok = rows.filter((row) => !row.error && row.has_68);
  const scored = (ok.length ? ok : rows.filter((row) => !row.error)).map((row) => {
    const visible = row.first_visible_ms ?? row.total_ms;
    const total = row.total_ms;
    // Prefer a level that still reasons (or is off) and stays under ~20s to first visible event.
    const latencyPenalty = visible > 20_000 ? (visible - 20_000) / 1000 : 0;
    const quality = (row.has_68 ? 2 : 0) + (row.has_cobros ? 1 : 0) + (row.tools?.length ? 1 : 0);
    const score = quality * 10 - latencyPenalty - total / 15_000;
    return { ...row, score };
  });
  scored.sort((a, b) => b.score - a.score);
  const best = scored[0];
  if (!best) return "high";
  const high = scored.find((row) => row.id === "high");
  const low = scored.find((row) => row.id === "low");
  if (high && low && high.has_68 && (high.total_ms ?? 0) < 12_000) return "high";
  if (best.id === "max" && high) return "high";
  if (best.id === "off") return high ? "high" : "low";
  return best.id;
}

async function main() {
  loadDotEnv(resolve(root, ".env.local"));
  loadDotEnv(resolve(root, ".env"));
  loadHelmcodeKey();
  const { model } = client();
  const system = loadSystem();
  const rows = [];
  for (const level of LEVELS) {
    process.stdout.write(`${level.id}… `);
    try {
      const row = await runLevel(level, system);
      rows.push(row);
      process.stdout.write(
        `visible=${fmt(row.first_visible_ms)} tool=${fmt(row.first_tool_ms)} total=${fmt(row.total_ms)} think=${row.reasoning_chars ?? 0}c\n`,
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      rows.push({ id: level.id, effort: level.effort, error: message });
      process.stdout.write(`error ${message.slice(0, 120)}\n`);
    }
  }
  const chosen = pick(rows);
  const when = new Date().toISOString();
  const lines = [
    "# Thinking effort bench",
    "",
    `- When: ${when}`,
    `- Model: ${model}`,
    `- Question: ${QUESTION}`,
    `- Pick: **${chosen}** (low latency, still reasons; never default to max)`,
    "",
    "| Level | First visible | First reasoning | First tool | Total | Reasoning chars | Tools | 68 | Cobros |",
    "|---|---:|---:|---:|---:|---:|---|---|---|",
  ];
  for (const row of rows) {
    if (row.error) {
      lines.push(`| ${row.id} | — | — | — | — | — | ${row.error.replace(/\|/g, "/")} | — | — |`);
      continue;
    }
    lines.push(
      `| ${row.id} | ${fmt(row.first_visible_ms)} | ${fmt(row.first_reasoning_ms)} | ${fmt(row.first_tool_ms)} | ${fmt(row.total_ms)} | ${row.reasoning_chars ?? 0} | ${(row.tools ?? []).join(", ") || "—"} | ${row.has_68 ? "yes" : "no"} | ${row.has_cobros ? "yes" : "no"} |`,
    );
  }
  lines.push("", `Default in ` + "`llm.ts`" + ` is \`${chosen}\`. Override with \`HELMCODE_REASONING_EFFORT\`.`, "");
  const text = lines.join("\n");
  writeFileSync(resolve(root, "src/lib/agent/THINKING_BENCH.md"), text);
  writeFileSync(resolve(root, "src/lib/agent/thinking-bench.json"), `${JSON.stringify({ when, model, chosen, rows }, null, 2)}\n`);
  process.stdout.write(`\n${text}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
