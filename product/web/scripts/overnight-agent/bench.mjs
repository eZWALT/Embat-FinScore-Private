/**
 * Pregunta overnight bench: hit a live /api/ask the same way the popup does.
 *
 *   node scripts/overnight-agent/bench.mjs --target https://hack-spain.vercel.app --out scripts/overnight-agent/baseline.json
 *   node scripts/overnight-agent/bench.mjs --target <preview> --out scripts/overnight-agent/candidate.json --compare scripts/overnight-agent/baseline.json
 *
 * Keep a change only if compare.keep is true (not worse on grounding / empty-text, not much slower without a quality gain).
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

const PROD_CASES = [
  {
    id: "why-score",
    companyId: "COMP_1186",
    groupId: "GROUP_0135",
    asOf: "2026-08",
    question: "¿Por qué este índice este mes?",
  },
  {
    id: "alerts",
    companyId: "COMP_1186",
    groupId: "GROUP_0135",
    asOf: "2026-08",
    question: "¿Qué alertas hay y quién debe actuar?",
  },
  {
    id: "refuse",
    companyId: "COMP_1186",
    groupId: "GROUP_0135",
    asOf: "2026-08",
    question: "Hazme la lista de la compra para una paella de domingo.",
  },
  {
    id: "period-4",
    companyId: "COMP_1186",
    question:
      "Periodo seleccionado: noviembre 2025 → diciembre 2025.\nEmpresas:\n- Empresa 1186 (Grupo 0135): 87 en noviembre 2025 → 79 en diciembre 2025 (−8,0 pts), caída\n- Empresa 0012 (Grupo 0087): 100 en noviembre 2025 → 100 en diciembre 2025 (0 pts), estable\n- Empresa 0445 (Grupo 0199): 100 en noviembre 2025 → 100 en diciembre 2025 (0 pts), estable\n- Empresa 0735 (Grupo 0200): 100 en noviembre 2025 → 100 en diciembre 2025 (0 pts), estable\n\nExplica qué pasó en ese periodo y por qué.",
  },
  {
    id: "why-change",
    companyId: "COMP_1186",
    groupId: "GROUP_0135",
    asOf: "2025-12",
    question: "¿Por qué cayó Empresa 1186 en diciembre?",
  },
];

/** Same companies exist in sample_bundle and (usually) Neon. Use for local vs production quality. */
const SAMPLE_CASES = [
  {
    id: "why-score",
    companyId: "COMP_0030",
    groupId: "GROUP_0126",
    asOf: "2026-08",
    question: "¿Por qué este índice este mes?",
  },
  {
    id: "alerts",
    companyId: "COMP_0016",
    groupId: "GROUP_0070",
    asOf: "2026-08",
    question: "¿Qué alertas hay y quién debe actuar?",
  },
  {
    id: "refuse",
    companyId: "COMP_0030",
    groupId: "GROUP_0126",
    asOf: "2026-08",
    question: "Hazme la lista de la compra para una paella de domingo.",
  },
  {
    id: "period-4",
    companyId: "COMP_0030",
    question:
      "Periodo seleccionado: junio 2026 → agosto 2026.\nEmpresas:\n- Empresa 0011 (Grupo 0234): 82 en junio 2026 → 73 en agosto 2026 (−9 pts), Caída puntual\n- Empresa 0176 (Grupo 0225): 79 en junio 2026 → 59 en agosto 2026 (−20 pts), Caída puntual\n- Empresa 0651 (Grupo 0081): 42 en junio 2026 → 33 en agosto 2026 (−9 pts), Deteriorándose\n- Empresa 0030 (Grupo 0126): 87 en junio 2026 → 88 en agosto 2026 (+1 pt), Estable\n\nExplica qué pasó en ese periodo y por qué.",
  },
  {
    id: "why-change",
    companyId: "COMP_0011",
    groupId: "GROUP_0234",
    asOf: "2026-08",
    question: "¿Por qué cayó Empresa 0011 este mes?",
  },
];

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  if (i < 0) return fallback;
  return process.argv[i + 1] ?? fallback;
}

function hasFlag(name) {
  return process.argv.includes(name);
}

function entityOf(name, input) {
  if (!input || typeof input !== "object") return "";
  return String(input.company_id || input.entity_id || input.group_id || "");
}

function inventedOwner(text, tools, id) {
  if (id === "alerts" || id === "refuse") return false;
  if ((tools ?? []).some((tool) => tool.name === "get_alerts")) return false;
  return /\bTesorero\b|\bCFO\b|dueño:\s*\**Cobros/i.test(text ?? "");
}

function leakedTokens(text) {
  return /\b(fading|dark|COMP_\d{4}|GROUP_\d{4}|out_vol|fc_ratio|score_pre_cap)\b/i.test(text ?? "");
}

function describesOfftopic(text, id) {
  if (id !== "refuse") return false;
  return /lista(?:s)? de la compra|recetas?|paella|puzzle|algoritmo|malware|roleplay|invertir una lista/i.test(text ?? "");
}

function signedEur(text) {
  const src = text ?? "";
  for (const match of src.matchAll(/[-−]\s*\d[\d.,]*\s*k?\s*€/g)) {
    const start = Math.max(0, match.index ?? 0);
    const ctx = src.slice(Math.max(0, start - 48), start);
    if (/caja|saldo|disponible|posici[oó]n|negativ/i.test(ctx)) continue;
    return true;
  }
  return false;
}

function methodCoda(text) {
  return /explicable y monitorable|se ha alejado de su normalidad|baja como máximo 10 puntos/i.test(text ?? "");
}

function roundedScore(text) {
  return /limitada a 73\b|índice (?:de |en )?\*?\*?73\*?\*?\b|queda (?:en|a) 73\b/i.test(text ?? "");
}

function bareCompanyNumber(text) {
  return /(?<!Empresa )(?<!Grupo )\b0\d{3}\b/.test(text ?? "") || /la de \d+ puntos/i.test(text ?? "");
}

function scoreCase(run, id) {
  const tools = run.tools ?? [];
  const text = (run.text ?? "").trim();
  const keys = tools.map((t) => `${t.name}:${entityOf(t.name, t.input) || "_"}`);
  const dup = keys.length - new Set(keys).size;
  const monthsByCompany = new Map();
  for (const tool of tools) {
    if (tool.name !== "get_company") continue;
    const company = entityOf(tool.name, tool.input) || "_";
    const months = monthsByCompany.get(company) ?? new Set();
    months.add(tool.input && tool.input.month ? String(tool.input.month) : "latest");
    monthsByCompany.set(company, months);
  }
  const monthFanout = [...monthsByCompany.values()].reduce((sum, months) => sum + Math.max(0, months.size - 1), 0);
  const hasText = text.length >= 40;
  const emptyAfterTools = tools.length > 0 && !hasText;
  const englishHeavy = /\b(the score|because|company|here is|I will)\b/i.test(text);
  const offersNext = /¿quieres que/i.test(text);
  const refused = /fuera de Health Sentinel/i.test(text);
  return {
    tool_count: tools.length,
    unique_tools: new Set(tools.map((t) => t.name)).size,
    entity_dups: dup,
    month_fanout: monthFanout,
    has_text: hasText,
    empty_after_tools: emptyAfterTools,
    text_len: text.length,
    english_heavy: englishHeavy,
    offers_next: offersNext,
    refused,
    invented_owner: inventedOwner(text, tools, id),
    leaked_token: leakedTokens(text),
    bare_id: bareCompanyNumber(text),
    describes_offtopic: describesOfftopic(text, id),
    signed_eur: signedEur(text),
    method_coda: methodCoda(text),
    rounded_score: roundedScore(text),
    ttft_ms: run.ttft_ms,
    total_ms: run.total_ms,
  };
}

function caseQuality(metrics, id, text = "") {
  let q = 0;
  if (id === "refuse") {
    if (metrics.refused) q += 3;
    if (metrics.tool_count === 0) q += 2;
    if (metrics.english_heavy) q -= 2;
    if (metrics.describes_offtopic) q -= 2;
    return q;
  }
  if (metrics.has_text) q += 3;
  if (metrics.empty_after_tools) q -= 5;
  if (metrics.english_heavy) q -= 2;
  if (metrics.offers_next) q -= 1;
  if (metrics.invented_owner) q -= 2;
  if (metrics.leaked_token) q -= 2;
  if (metrics.bare_id) q -= 1;
  if (metrics.signed_eur) q -= 1;
  if (metrics.method_coda) q -= 1;
  if (metrics.rounded_score) q -= 1;
  if (id === "period-4") {
    if (/tres de las cuatro.{0,48}(caen por|empresas caen por).{0,40}hundid/i.test(text)) q -= 2;
    if (/tres empresas que caen.{0,100}hundid/i.test(text)) q -= 2;
    if (/\blas que caen\b.{0,80}hundid/i.test(text)) q -= 1;
  }
  if (metrics.entity_dups === 0) q += 1;
  if (metrics.month_fanout === 0) q += 1;
  if (metrics.tool_count > 0 && metrics.tool_count <= 6) q += 1;
  if (metrics.tool_count > 8) q -= 2;
  return q;
}

function qualityOf(run, id) {
  const metrics = scoreCase({ ...run, tools: run.tools ?? [], text: run.text ?? "", ttft_ms: run.metrics?.ttft_ms, total_ms: run.metrics?.total_ms }, id);
  return caseQuality({ ...run.metrics, ...metrics }, id, run.text ?? "");
}

async function consumeAsk(base, body, timeoutMs = 110_000) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  const t0 = Date.now();
  let res;
  try {
    res = await fetch(`${base.replace(/\/$/, "")}/api/ask`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: ac.signal,
    });
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HTTP ${res.status} ${err.slice(0, 240)}`);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const tools = [];
  let text = "";
  let ttft = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;
      let ev;
      try {
        ev = JSON.parse(payload);
      } catch {
        continue;
      }
      const type = ev.type;
      if (type === "text-delta" || type === "text") {
        const piece = ev.delta ?? ev.text ?? "";
        if (piece) {
          if (ttft == null) ttft = Date.now() - t0;
          text += piece;
        }
      }
      if (type === "tool-input-start") {
        tools.push({ name: ev.toolName, id: ev.toolCallId, input: undefined });
      }
      if (type === "tool-input-available") {
        const row = tools.find((t) => t.id === ev.toolCallId);
        if (row) {
          row.input = ev.input;
          if (!row.name) row.name = ev.toolName;
        } else {
          tools.push({ name: ev.toolName, id: ev.toolCallId, input: ev.input });
        }
      }
    }
  }
  return { ttft_ms: ttft, total_ms: Date.now() - t0, text, tools };
}

function compare(baseline, candidate) {
  const rows = [];
  let keep = true;
  let reasons = [];
  for (const id of new Set([...Object.keys(baseline.cases || {}), ...Object.keys(candidate.cases || {})])) {
    const a = baseline.cases[id];
    const b = candidate.cases[id];
    if (!a || !b) continue;
    const aq = qualityOf(a, id);
    const bq = qualityOf(b, id);
    const dq = bq - aq;
    const dt = (b.metrics.total_ms ?? 0) - (a.metrics.total_ms ?? 0);
    const emptyWorse = b.metrics.empty_after_tools && !a.metrics.empty_after_tools;
    const groundWorse = b.metrics.english_heavy && !a.metrics.english_heavy;
    const ownerWorse = b.metrics.invented_owner && !inventedOwner(a.text, a.tools, id);
    const leakWorse = leakedTokens(b.text) && !leakedTokens(a.text);
    const offtopicWorse = describesOfftopic(b.text, id) && !describesOfftopic(a.text, id);
    if (emptyWorse || groundWorse || ownerWorse || leakWorse || offtopicWorse || dq <= -2) {
      keep = false;
      reasons.push(`${id}: worse quality/grounding`);
    }
    if (dt > 8000 && dq <= 0 && !hasFlag("--ignore-latency")) {
      keep = false;
      reasons.push(`${id}: +${Math.round(dt / 1000)}s without quality gain`);
    }
    rows.push({
      id,
      d_quality: dq,
      d_tools: (b.metrics.tool_count ?? 0) - (a.metrics.tool_count ?? 0),
      d_ms: dt,
      empty_after_tools: b.metrics.empty_after_tools,
    });
  }
  if (!reasons.length && keep) reasons.push("not worse; keep");
  return { keep, reasons, rows };
}

async function main() {
  const target = arg("--target", "https://hack-spain.vercel.app");
  const outRaw = arg("--out", "");
  const out = outRaw ? resolve(process.cwd(), outRaw) : resolve(here, "last.json");
  const compareRaw = arg("--compare", "");
  const comparePath = compareRaw ? resolve(process.cwd(), compareRaw) : "";
  const only = arg("--only", "");
  const suite = arg("--suite", "prod");
  const pool = suite === "sample" ? SAMPLE_CASES : PROD_CASES;
  const cases = only ? pool.filter((c) => c.id === only) : pool;
  const pack = {
    at: new Date().toISOString(),
    target,
    cases: {},
  };
  for (const c of cases) {
    process.stderr.write(`→ ${c.id} @ ${target}\n`);
    try {
      const run = await consumeAsk(target, {
        messages: [{ role: "user", content: c.question }],
        companyId: c.companyId,
        groupId: c.groupId,
        asOf: c.asOf,
        thinking: false,
      });
      const metrics = scoreCase(run, c.id);
      pack.cases[c.id] = {
        metrics,
        quality: caseQuality(metrics, c.id, run.text),
        tools: run.tools.map((t) => ({ name: t.name, input: t.input })),
        text: run.text.slice(0, 2000),
      };
      process.stderr.write(
        `  tools=${metrics.tool_count} dups=${metrics.entity_dups} text=${metrics.text_len} q=${pack.cases[c.id].quality} ${metrics.total_ms}ms\n`,
      );
    } catch (error) {
      pack.cases[c.id] = { error: String(error), metrics: { empty_after_tools: true }, quality: -9 };
      process.stderr.write(`  FAIL ${error}\n`);
    }
  }
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify(pack, null, 2));
  if (comparePath) {
    const baseline = JSON.parse(readFileSync(comparePath, "utf8"));
    const verdict = compare(baseline, pack);
    pack.compare = verdict;
    writeFileSync(out, JSON.stringify(pack, null, 2));
    console.log(JSON.stringify(verdict, null, 2));
    if (hasFlag("--fail-if-worse") && !verdict.keep) process.exit(2);
  } else {
    console.log(JSON.stringify({ at: pack.at, target, ids: Object.keys(pack.cases) }, null, 2));
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
