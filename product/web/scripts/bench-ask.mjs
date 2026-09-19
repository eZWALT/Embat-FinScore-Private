/**
 * Ask latency: full prompt stack + Helmcode TTFT + time-to-first-tool-call.
 * Neon SQL runs only when DATABASE_URL is set.
 *
 *   cd product/web && node scripts/bench-ask.mjs
 *
 * Loader order (must match prompt-loader.ts):
 * MAP → ROLE → SCOPE → PRODUCT → WORDING → FORMAT → TOOLS → RECORDS → SESSION
 */
import { readFileSync, existsSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
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

const SESSION = "session\ncompany_id=COMP_0085\nas_of=2026-08";

const TOOL_QUESTION =
  "¿Por qué este índice este mes y qué alertas hay? Usa las herramientas. No inventes números.";

const ASK_TOOLS = [
  fn("list_companies", "List scored companies (latest month). Optional group_id.", {
    group_id: { type: "string", description: "GROUP_xxxx" },
    limit: { type: "integer", description: "Default 30" },
  }),
  fn(
    "get_company",
    "Score, trajectory, guard, confidence, categories, items, reasons (with EUR) of one company.",
    {
      company_id: { type: "string", description: "COMP_xxxx" },
      month: { type: "string", description: "YYYY-MM, default latest" },
    },
    ["company_id"],
  ),
  fn(
    "explain_change",
    "Why the score moved from the previous month: signed per-item deltas, change_reasons, guard.",
    {
      company_id: { type: "string", description: "COMP_xxxx" },
      month: { type: "string", description: "YYYY-MM, default latest" },
    },
    ["company_id"],
  ),
  fn("get_group", "Group summary: members, mean, min, alerts.", {
    group_id: { type: "string", description: "GROUP_xxxx" },
  }, ["group_id"]),
  fn(
    "get_alerts",
    "Alerts from the feed. Filter by entity_id, kinds, severities, since_month.",
    {
      entity_id: { type: "string" },
      kinds: {
        type: "array",
        items: {
          type: "string",
          enum: [
            "score_deterioration",
            "score_improvement",
            "category_drop",
            "going_dark",
            "top_customer_quiet",
          ],
        },
      },
      severities: { type: "array", items: { type: "string", enum: ["info", "watch", "act"] } },
      since_month: { type: "string", description: "YYYY-MM" },
      limit: { type: "integer", description: "Default 30" },
    },
  ),
  fn("get_control_chart", "Control chart series for a company or group.", {
    entity_id: { type: "string", description: "COMP_xxxx or GROUP_xxxx" },
    comparison: {
      type: "string",
      enum: ["own_history", "cluster", "group_own_history", "group_vs_groups"],
    },
    metric: {
      type: "string",
      enum: ["score", "payment_history", "amounts_owed", "stability", "new_credit", "mix"],
    },
  }, ["entity_id"]),
  fn("compare_with_cluster", "Peer-group comparison for one company.", {
    company_id: { type: "string", description: "COMP_xxxx" },
  }, ["company_id"]),
  fn("get_forecast", "Score fan 1-6 months ahead (naive_last).", {
    company_id: { type: "string", description: "COMP_xxxx" },
  }, ["company_id"]),
  fn(
    "query_clean_db",
    "One read-only SELECT over cleaned records (clean.*). Always filter by company_id and LIMIT.",
    { sql: { type: "string" } },
    ["sql"],
  ),
  fn("plot_series", "Ask the UI to draw a chart. One or two plots per answer.", {
    title: { type: "string" },
    x: { type: "array", items: { type: "string" } },
    series: { type: "object", additionalProperties: { type: "array", items: { type: ["number", "null"] } } },
    kind: { type: "string", enum: ["line", "bar"] },
  }, ["title", "x", "series"]),
];

function fn(name, description, properties, required) {
  return {
    type: "function",
    function: {
      name,
      description,
      parameters: {
        type: "object",
        properties,
        ...(required ? { required } : {}),
      },
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

function minutes(ms) {
  if (ms == null) return "—";
  return (ms / 60_000).toFixed(4);
}

function mid(rows, key) {
  const values = rows.map((row) => row[key]).filter((n) => n != null).sort((a, b) => a - b);
  if (!values.length) return null;
  return values[Math.floor(values.length / 2)];
}

function loadPrompts() {
  const dir = resolve(root, "src/lib/agent/prompts");
  const parts = [];
  const sizes = [];
  for (const name of ASK_FILES) {
    const text = readFileSync(resolve(dir, name), "utf8");
    sizes.push({ name, chars: text.length });
    parts.push(text);
  }
  parts.push(SESSION);
  sizes.push({ name: "session", chars: SESSION.length });
  return { system: parts.join("\n\n---\n\n"), sizes };
}

function helmcodeClient() {
  const apiKey = process.env.HELMCODE_API_KEY?.trim();
  if (!apiKey) throw new Error("HELMCODE_API_KEY is not set");
  const base = (process.env.HELMCODE_BASE_URL?.trim() || "https://api.helmcode.com/v1").replace(/\/$/, "");
  const model = process.env.POC_LLM_MODEL?.trim() || "deepseek-v4-flash";
  return { apiKey, base, model };
}

async function helmcodeStream({ system, user, tools, toolChoice }) {
  const { apiKey, base, model } = helmcodeClient();
  const started = Date.now();
  const body = {
    model,
    stream: true,
    temperature: 0.2,
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
  };
  if (tools) {
    body.tools = tools;
    body.tool_choice = toolChoice ?? "auto";
  }
  const response = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Helmcode ${response.status}: ${text.slice(0, 400)}`);
  }
  let firstAnyMs = null;
  let firstTextMs = null;
  let firstToolMs = null;
  let textChars = 0;
  let finish = null;
  const toolsSeen = new Map();
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
      const content = delta.content;
      if (typeof content === "string" && content.length) {
        if (firstAnyMs == null) firstAnyMs = now;
        if (firstTextMs == null) firstTextMs = now;
        textChars += content.length;
      }
      const calls = delta.tool_calls ?? (delta.function_call ? [delta.function_call] : null);
      if (Array.isArray(calls) && calls.length) {
        if (firstAnyMs == null) firstAnyMs = now;
        if (firstToolMs == null) firstToolMs = now;
        for (const call of calls) {
          const name = call.function?.name ?? call.name;
          const index = call.index ?? 0;
          if (name) toolsSeen.set(index, name);
        }
      }
    }
  }
  return {
    first_any_ms: firstAnyMs,
    ttft_ms: firstTextMs,
    first_tool_ms: firstToolMs,
    total_ms: Date.now() - started,
    text_chars: textChars,
    finish,
    tools: [...toolsSeen.entries()].sort((a, b) => a[0] - b[0]).map(([, name]) => name),
  };
}

async function pingNeon() {
  const url = process.env.DATABASE_URL;
  if (!url) return null;
  const { neon } = await import("@neondatabase/serverless");
  const sql = neon(url);
  const samples = [];
  for (const [name, query] of [
    ["api.manifest", "SELECT as_of_month FROM api.manifest LIMIT 1"],
    [
      "get_company-shaped",
      "SELECT company_id, month, score FROM analytics.company_scores JOIN api.current_run r ON r.run_id = analytics.company_scores.run_id WHERE company_id = 'COMP_0085' ORDER BY month DESC LIMIT 1",
    ],
    [
      "get_alerts-shaped",
      "SELECT count(*)::int AS n FROM analytics.alerts a JOIN api.current_run r ON r.run_id = a.run_id WHERE entity_id = 'COMP_0085'",
    ],
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
    `- Prompt stack: ${input.promptChars.toLocaleString("en")} chars (MAP → ROLE → SCOPE → PRODUCT → WORDING → FORMAT → TOOLS → RECORDS → SESSION)`,
    `- Tool schemas sent: ${ASK_TOOLS.length}`,
    "",
    "The Cursor-style tool trace is render-only. It does not add a model round-trip.",
    "The clock is Helmcode (TTFT / time-to-first-tool) and Neon (tool SQL). Minutes = ms / 60000.",
    "Full prompt is kept on purpose. Do not shrink PRODUCT or TOOLS to chase TTFT.",
    "",
    "## Prompt pieces",
    "",
    "| File | layer | chars |",
    "|---|---|---:|",
  ];
  const layers = {
    "prompt_map.md": "MAP",
    "chat_system.md": "ROLE",
    "scope.md": "SCOPE",
    "product_context.md": "PRODUCT",
    "wording_rules.md": "WORDING",
    "watcher_format.md": "FORMAT",
    "tools_catalog.md": "TOOLS",
    "clean_schema.md": "RECORDS",
    session: "SESSION",
  };
  for (const row of input.sizes) {
    lines.push(`| \`${row.name}\` | ${layers[row.name] ?? "—"} | ${row.chars.toLocaleString("en")} |`);
  }
  lines.push(
    "",
    "## Time to first token (Helmcode, no tools)",
    "",
    "| System | TTFT | Completion | TTFT (min) |",
    "|---|---:|---:|---:|",
  );
  for (const row of input.llm) {
    lines.push(`| ${row.label} | ${fmt(row.ttft_ms)} | ${fmt(row.total_ms)} | ${minutes(row.ttft_ms)} |`);
  }
  lines.push(
    "",
    "## Time to first tool call (Helmcode, full prompt + 10 tools)",
    "",
    `Question: \`${TOOL_QUESTION}\``,
    "",
    "| Mode | First event | First tool | Stream until pause | Tools asked | finish | First tool (min) |",
    "|---|---:|---:|---:|---|---|---:|",
  );
  for (const row of input.tooling) {
    lines.push(
      `| ${row.label} | ${fmt(row.first_any_ms)} | ${fmt(row.first_tool_ms)} | ${fmt(row.total_ms)} | ${row.tools.length ? row.tools.join(", ") : "—"} | ${row.finish ?? "—"} | ${minutes(row.first_tool_ms)} |`,
    );
  }
  if (input.toolingRounds.length) {
    lines.push("", "### Rounds (auto)", "", "| # | First tool | Tools | finish | total |", "|---|---:|---|---|---:|");
    input.toolingRounds.forEach((row, i) => {
      lines.push(
        `| ${i + 1} | ${fmt(row.first_tool_ms)} | ${row.tools.join(", ") || "—"} | ${row.finish ?? "—"} | ${fmt(row.total_ms)} |`,
      );
    });
  }
  lines.push(
    "",
    "## Neon retrieval (same queries the tools run)",
    "",
    "| Query | p50-ish (1 shot) | min | ok | note |",
    "|---|---:|---:|---|---|",
  );
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
  const { model } = helmcodeClient();
  process.stdout.write(`prompt=${system.length} chars  model=${model}\n`);

  const tinyRounds = [];
  const fullRounds = [];
  for (let i = 0; i < 3; i += 1) {
    process.stdout.write(`ttft tiny ${i + 1}/3… `);
    tinyRounds.push(await helmcodeStream({ system: "Eres un asistente. Responde en una frase.", user: "di ok" }));
    process.stdout.write(`${fmt(tinyRounds.at(-1).ttft_ms)}\n`);
    process.stdout.write(`ttft full ${i + 1}/3… `);
    fullRounds.push(await helmcodeStream({ system, user: "di ok" }));
    process.stdout.write(`${fmt(fullRounds.at(-1).ttft_ms)}\n`);
  }

  const toolingRounds = [];
  for (let i = 0; i < 3; i += 1) {
    process.stdout.write(`tool auto ${i + 1}/3… `);
    const row = await helmcodeStream({
      system,
      user: TOOL_QUESTION,
      tools: ASK_TOOLS,
      toolChoice: "auto",
    });
    toolingRounds.push(row);
    process.stdout.write(`${fmt(row.first_tool_ms)} → ${row.tools.join(",") || "none"} (${row.finish})\n`);
  }

  process.stdout.write("tool required 1/1… ");
  const required = await helmcodeStream({
    system,
    user: TOOL_QUESTION,
    tools: ASK_TOOLS,
    toolChoice: "required",
  });
  process.stdout.write(`${fmt(required.first_tool_ms)} → ${required.tools.join(",") || "none"} (${required.finish})\n`);

  const llm = [
    { label: "Tiny (40 chars), median of 3", ttft_ms: mid(tinyRounds, "ttft_ms"), total_ms: mid(tinyRounds, "total_ms") },
    {
      label: "Full Ask prompt (no tools), median of 3",
      ttft_ms: mid(fullRounds, "ttft_ms"),
      total_ms: mid(fullRounds, "total_ms"),
    },
  ];
  const tooling = [
    {
      label: "Full prompt + tools, auto, median of 3",
      first_any_ms: mid(toolingRounds, "first_any_ms"),
      first_tool_ms: mid(toolingRounds, "first_tool_ms"),
      total_ms: mid(toolingRounds, "total_ms"),
      tools: toolingRounds.find((row) => row.tools.length)?.tools ?? [],
      finish: toolingRounds.find((row) => row.finish)?.finish ?? null,
    },
    {
      label: "Full prompt + tools, required, 1 shot",
      first_any_ms: required.first_any_ms,
      first_tool_ms: required.first_tool_ms,
      total_ms: required.total_ms,
      tools: required.tools,
      finish: required.finish,
    },
  ];

  const neon = await pingNeon();
  const text = report({
    when: new Date().toISOString(),
    model,
    promptChars: system.length,
    sizes,
    llm,
    tooling,
    toolingRounds,
    neon,
  });
  writeFileSync(resolve(root, "src/lib/agent/BENCHMARK.md"), text);
  process.stdout.write(`\n${text}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
