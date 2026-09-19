/**
 * Watch the Consultas UX: TTFT, time to last token, tool rounds, and a judge.
 * Does not change production prompts. Run after a prompt tweak:
 *
 *   cd product/web && node scripts/watch-ask.mjs
 *
 * Writes src/lib/agent/WATCH.md and appends src/lib/agent/watch-log.jsonl.
 * Tool bodies are stubs unless DATABASE_URL is set (then Neon runs the three
 * score/alert queries). The clock that matters is Helmcode, like the popup.
 */
import { appendFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
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
  "clean_schema.md",
];

const CASES = [
  {
    id: "why-score",
    company: "COMP_0462",
    group: "GROUP_0194",
    asOf: "2026-08",
    question: "¿Por qué este índice este mes?",
  },
  {
    id: "alerts",
    company: "COMP_0462",
    group: "GROUP_0194",
    asOf: "2026-08",
    question: "¿Qué alertas hay y quién debe actuar?",
  },
  {
    id: "refuse",
    company: "COMP_0462",
    group: "GROUP_0194",
    asOf: "2026-08",
    question: "Hazme la lista de la compra para una paella de domingo.",
  },
];

const ASK_TOOLS = [
  fn("list_companies", "List scored companies (latest month).", {
    group_id: { type: "string" },
    limit: { type: "integer" },
  }),
  fn("get_company", "Score, trajectory, reasons with EUR of one company.", {
    company_id: { type: "string" },
    month: { type: "string" },
  }, ["company_id"]),
  fn("explain_change", "Why the score moved from the previous month.", {
    company_id: { type: "string" },
    month: { type: "string" },
  }, ["company_id"]),
  fn("get_group", "Group summary.", { group_id: { type: "string" } }, ["group_id"]),
  fn("get_alerts", "Alerts from the feed.", {
    entity_id: { type: "string" },
    kinds: { type: "array", items: { type: "string" } },
    severities: { type: "array", items: { type: "string" } },
    since_month: { type: "string" },
    limit: { type: "integer" },
  }),
  fn("get_control_chart", "Control chart series.", {
    entity_id: { type: "string" },
    comparison: { type: "string" },
    metric: { type: "string" },
  }, ["entity_id"]),
  fn("compare_with_cluster", "Peer-group comparison.", { company_id: { type: "string" } }, ["company_id"]),
  fn("get_forecast", "Score fan 1-6 months ahead.", { company_id: { type: "string" } }, ["company_id"]),
  fn("query_clean_db", "One read-only SELECT over cleaned records.", { sql: { type: "string" } }, ["sql"]),
  fn("plot_series", "Ask the UI to draw a chart.", {
    title: { type: "string" },
    x: { type: "array", items: { type: "string" } },
    series: { type: "object" },
  }, ["title", "x", "series"]),
];

function fn(name, description, properties, required) {
  return {
    type: "function",
    function: {
      name,
      description,
      parameters: { type: "object", properties, ...(required ? { required } : {}) },
    },
  };
}

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
  if (ms == null) return "—";
  if (ms >= 60_000) return `${(ms / 60_000).toFixed(2)} min`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${Math.round(ms)} ms`;
}

function helmcodeClient() {
  const apiKey = process.env.HELMCODE_API_KEY?.trim();
  if (!apiKey) throw new Error("HELMCODE_API_KEY is not set");
  const base = (process.env.HELMCODE_BASE_URL?.trim() || "https://api.helmcode.com/v1").replace(/\/$/, "");
  const model = process.env.POC_LLM_MODEL?.trim() || "deepseek-v4-flash";
  return { apiKey, base, model };
}

function sessionExtra(company, group, asOf) {
  return ["session", company && `company_id=${company}`, group && `group_id=${group}`, asOf && `as_of=${asOf}`]
    .filter(Boolean)
    .join("\n");
}

function loadSystem(company, group, asOf) {
  const dir = resolve(root, "src/lib/agent/prompts");
  const parts = ASK_FILES.map((name) => readFileSync(resolve(dir, name), "utf8"));
  parts.push(sessionExtra(company, group, asOf));
  return parts.join("\n\n---\n\n");
}

function stubResult(name, args, company) {
  const id = args.company_id || args.entity_id || company;
  if (name === "get_company") {
    return {
      company_id: id,
      month: args.month || "2026-08",
      score: 68,
      trajectory: "deteriorating",
      guard: null,
      confidence: "medium",
      reasons: [
        {
          sentence: "El cliente principal ha dejado de facturar. Revisa la exposición y los cobros.",
          eur: 14000,
        },
      ],
      change_reasons: [{ sentence: "La categoría de historial de pagos ha bajado 6 puntos.", eur: null }],
    };
  }
  if (name === "explain_change") {
    return {
      company_id: id,
      from: "2026-07",
      to: "2026-08",
      score_from: 74,
      score_to: 68,
      change: -6,
      change_reasons: [{ sentence: "La categoría de historial de pagos ha bajado 6 puntos.", eur: null }],
    };
  }
  if (name === "get_alerts") {
    return {
      n_matching: 1,
      alerts: [
        {
          kind: "top_customer_quiet",
          severity: "act",
          title: "El cliente principal ha dejado de facturar",
          summary: "Revisa la exposición y los cobros.",
          owner: "Cobros",
          action: "Revisa la exposición y los cobros.",
          month: "2026-08",
        },
      ],
    };
  }
  if (name === "list_companies") {
    return { as_of: "2026-08", companies: [{ company_id: company, score: 68, trajectory: "deteriorating" }] };
  }
  if (name === "get_group") {
    return { group_id: args.group_id, n_companies: 3, latest_mean_score: 71 };
  }
  if (name === "plot_series") {
    return { ok: true, title: args.title, points: Array.isArray(args.x) ? args.x.length : 0 };
  }
  return { ok: true, stub: true, tool: name };
}

async function neonResult(name, args, company) {
  const url = process.env.DATABASE_URL;
  if (!url) return { result: stubResult(name, args, company), source: "stub" };
  const { neon } = await import("@neondatabase/serverless");
  const sql = neon(url);
  const id = args.company_id || args.entity_id || company;
  try {
    if (name === "get_company") {
      const rows = await sql.query(
        "SELECT company_id, month, score FROM analytics.company_scores JOIN api.current_run r ON r.run_id = analytics.company_scores.run_id WHERE company_id = $1 ORDER BY month DESC LIMIT 1",
        [id],
      );
      return { result: rows[0] ?? stubResult(name, args, company), source: "neon" };
    }
    if (name === "get_alerts") {
      const rows = await sql.query(
        "SELECT count(*)::int AS n FROM analytics.alerts a JOIN api.current_run r ON r.run_id = a.run_id WHERE entity_id = $1",
        [id],
      );
      return { result: { n_matching: rows[0]?.n ?? 0, source: "count-only" }, source: "neon" };
    }
  } catch (error) {
    return {
      result: stubResult(name, args, company),
      source: "stub",
      neon_error: error instanceof Error ? error.message.split("\n")[0] : String(error),
    };
  }
  return { result: stubResult(name, args, company), source: "stub" };
}

async function helmcodeTurn({ messages, tools, clock }) {
  const { apiKey, base, model } = helmcodeClient();
  const response = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      stream: true,
      temperature: 0.2,
      messages,
      ...(tools ? { tools, tool_choice: "auto" } : {}),
    }),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Helmcode ${response.status}: ${text.slice(0, 400)}`);
  }
  let firstAnyMs = null;
  let firstTextMs = null;
  let firstToolMs = null;
  let lastEventMs = null;
  let text = "";
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
      const now = Date.now() - clock;
      const content = delta.content;
      if (typeof content === "string" && content.length) {
        if (firstAnyMs == null) firstAnyMs = now;
        if (firstTextMs == null) firstTextMs = now;
        lastEventMs = now;
        text += content;
      }
      const deltas = delta.tool_calls ?? [];
      if (deltas.length) {
        if (firstAnyMs == null) firstAnyMs = now;
        if (firstToolMs == null) firstToolMs = now;
        lastEventMs = now;
        for (const call of deltas) {
          const index = call.index ?? 0;
          const prev = calls.get(index) ?? { id: "", name: "", arguments: "" };
          if (call.id) prev.id = call.id;
          if (call.function?.name) prev.name = call.function.name;
          if (call.function?.arguments) prev.arguments += call.function.arguments;
          calls.set(index, prev);
        }
      }
    }
  }
  return {
    first_any_ms: firstAnyMs,
    first_text_ms: firstTextMs,
    first_tool_ms: firstToolMs,
    last_token_ms: lastEventMs,
    total_ms: Date.now() - clock,
    text,
    finish,
    tool_calls: [...calls.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([, call]) => ({
        id: call.id || `call_${call.name}`,
        name: call.name,
        arguments: call.arguments || "{}",
      })),
  };
}

async function runCase(item) {
  const system = loadSystem(item.company, item.group, item.asOf);
  const messages = [
    { role: "system", content: system },
    { role: "user", content: item.question },
  ];
  const clock = Date.now();
  const steps = [];
  let sources = new Set();
  for (let i = 0; i < 8; i += 1) {
    const turn = await helmcodeTurn({ messages, tools: ASK_TOOLS, clock });
    steps.push({
      n: i + 1,
      first_any_ms: turn.first_any_ms,
      first_tool_ms: turn.first_tool_ms,
      last_token_ms: turn.last_token_ms,
      finish: turn.finish,
      tools: turn.tool_calls.map((call) => call.name),
      text_chars: turn.text.length,
    });
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
        let args = {};
        try {
          args = JSON.parse(call.arguments || "{}");
        } catch {
          args = {};
        }
        const resolved = await neonResult(call.name, args, item.company);
        sources.add(resolved.source);
        messages.push({
          role: "tool",
          tool_call_id: call.id,
          content: JSON.stringify(resolved.result),
        });
      }
      continue;
    }
    return {
      ...item,
      ttft_ms: steps[0]?.first_any_ms ?? null,
      first_tool_ms: steps.find((step) => step.first_tool_ms != null)?.first_tool_ms ?? null,
      last_token_ms: turn.last_token_ms,
      total_ms: Date.now() - clock,
      tools: steps.flatMap((step) => step.tools),
      steps,
      answer: turn.text.trim(),
      tool_source: [...sources].join("+") || "none",
      prompt_chars: system.length,
    };
  }
  return {
    ...item,
    ttft_ms: steps[0]?.first_any_ms ?? null,
    first_tool_ms: steps.find((step) => step.first_tool_ms != null)?.first_tool_ms ?? null,
    last_token_ms: steps.at(-1)?.last_token_ms ?? null,
    total_ms: Date.now() - clock,
    tools: steps.flatMap((step) => step.tools),
    steps,
    answer: "",
    tool_source: [...sources].join("+") || "none",
    prompt_chars: system.length,
    error: "hit 8-step cap",
  };
}

function parseJudge(text) {
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return { verdict: "confusing", score: 1, notes: "El juez no devolvió JSON." };
  try {
    const row = JSON.parse(match[0]);
    const verdict = ["good", "confusing", "verbose", "thin", "off_scope"].includes(row.verdict)
      ? row.verdict
      : "confusing";
    const score = Number(row.score);
    return {
      verdict,
      score: Number.isFinite(score) ? Math.min(5, Math.max(1, score)) : 1,
      notes: typeof row.notes === "string" ? row.notes : "",
    };
  } catch {
    return { verdict: "confusing", score: 1, notes: "El juez devolvió JSON inválido." };
  }
}

async function judgeCase(run) {
  const { apiKey, base, model } = helmcodeClient();
  const system = readFileSync(resolve(root, "src/lib/agent/watch/judge.md"), "utf8");
  const user = [
    `PREGUNTA:\n${run.question}`,
    `HERRAMIENTAS PEDIDAS:\n${run.tools.join(", ") || "ninguna"}`,
    `RESPUESTA:\n${run.answer || "(vacía)"}`,
  ].join("\n\n");
  const response = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      temperature: 0,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    }),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Judge ${response.status}: ${text.slice(0, 200)}`);
  }
  const json = await response.json();
  const text = json.choices?.[0]?.message?.content ?? "";
  return parseJudge(text);
}

function report(input) {
  const lines = [
    "# Ask UX watch",
    "",
    `- When: ${input.when}`,
    `- Model: ${input.model}`,
    `- Prompt: ${input.promptChars.toLocaleString("en")} chars (full Ask stack, not shrunk)`,
    `- Tool bodies: ${input.toolSource}`,
    "",
    "This is the popup clock: first event the user sees (TTFT), last token of the final answer, tools in between, then a judge for prompt iteration.",
    "Verdict: `good` · `confusing` · `verbose` · `thin` · `off_scope`. Score 1–5.",
    "",
    "| Case | TTFT | First tool | Last token | Total | Tools | Judge | Notes |",
    "|---|---:|---:|---:|---:|---|---|---|",
  ];
  for (const row of input.runs) {
    lines.push(
      `| ${row.id} | ${fmt(row.ttft_ms)} | ${fmt(row.first_tool_ms)} | ${fmt(row.last_token_ms)} | ${fmt(row.total_ms)} | ${row.tools.join(", ") || "—"} | ${row.judge.verdict} ${row.judge.score}/5 | ${row.judge.notes.replace(/\|/g, "/")} |`,
    );
  }
  lines.push("", "## Replies", "");
  for (const row of input.runs) {
    lines.push(`### ${row.id} — ${row.question}`, "", row.answer || "_(vacía)_", "");
  }
  lines.push("Run again after a prompt edit: `cd product/web && node scripts/watch-ask.mjs`.", "");
  return lines.join("\n");
}

async function main() {
  loadDotEnv(resolve(root, ".env.local"));
  loadDotEnv(resolve(root, ".env"));
  const { model } = helmcodeClient();
  const extra = process.argv.slice(2).join(" ").trim();
  const cases = extra
    ? [{ id: "custom", company: "COMP_0462", group: "GROUP_0194", asOf: "2026-08", question: extra }]
    : CASES;

  const runs = [];
  for (const item of cases) {
    process.stdout.write(`${item.id}… `);
    const run = await runCase(item);
    process.stdout.write(`ttft=${fmt(run.ttft_ms)} last=${fmt(run.last_token_ms)} tools=${run.tools.join(",") || "none"} `);
    const judged = await judgeCase(run);
    run.judge = judged;
    process.stdout.write(`→ ${judged.verdict} ${judged.score}/5\n`);
    runs.push(run);
  }

  const when = new Date().toISOString();
  const text = report({
    when,
    model,
    promptChars: runs[0]?.prompt_chars ?? 0,
    toolSource: process.env.DATABASE_URL ? "Neon when the query exists, else stub" : "stub (no DATABASE_URL)",
    runs,
  });
  writeFileSync(resolve(root, "src/lib/agent/WATCH.md"), text);
  appendFileSync(
    resolve(root, "src/lib/agent/watch-log.jsonl"),
    `${JSON.stringify({
      when,
      model,
      runs: runs.map((row) => ({
        id: row.id,
        question: row.question,
        ttft_ms: row.ttft_ms,
        first_tool_ms: row.first_tool_ms,
        last_token_ms: row.last_token_ms,
        total_ms: row.total_ms,
        tools: row.tools,
        judge: row.judge,
        answer: row.answer,
      })),
    })}\n`,
  );
  process.stdout.write(`\n${text}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
