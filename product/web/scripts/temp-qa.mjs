/**
 * Temperature sweep for Pregunta wording (no tools). Same facts, 3 draws per T.
 *
 *   cd product/web && node scripts/temp-qa.mjs
 *
 * Needs HELMCODE_API_KEY. Writes src/lib/agent/temp-qa.json.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { homedir } from "node:os";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");

const TEMPS = [0, 0.1, 0.2, 0.3];
const REPEATS = 3;

const SYSTEM = `Eres Sentinel. Español. 3 viñetas. Hallazgo + € + dueño/acción. Sin descargo del método. Sin «¿quieres que…?». No inventes cifras.`;

const USER = `Hechos fijos (no añadas otros):
COMP_0202, agosto 2026, índice 30 (tope dark; sin tope sería 76).
Sin movimientos bancarios 62 días. Dueño: Tesorero. Acción: comprueba las conexiones; si están completas, llama hoy.
Cobertura de caja 0,6 meses (765 k€ caja; 742 k€ salidas/mes).
Escribe la respuesta ahora.`;

function helmcode() {
  const apiKey = process.env.HELMCODE_API_KEY?.trim();
  if (!apiKey) throw new Error("HELMCODE_API_KEY is not set");
  const base = (process.env.HELMCODE_BASE_URL?.trim() || "https://api.helmcode.com/v1").replace(/\/$/, "");
  const model = process.env.POC_LLM_MODEL?.trim() || "deepseek-v4-flash";
  return { apiKey, base, model };
}

function tokens(text) {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .split(/[^a-z0-9€%]+/u)
    .filter(Boolean);
}

function jaccard(a, b) {
  const left = new Set(tokens(a));
  const right = new Set(tokens(b));
  let inter = 0;
  for (const t of left) if (right.has(t)) inter += 1;
  const union = left.size + right.size - inter;
  return union ? inter / union : 1;
}

function pairwiseMean(values, score) {
  if (values.length < 2) return 1;
  let sum = 0;
  let n = 0;
  for (let i = 0; i < values.length; i += 1) {
    for (let j = i + 1; j < values.length; j += 1) {
      sum += score(values[i], values[j]);
      n += 1;
    }
  }
  return n ? sum / n : 1;
}

async function complete(temperature) {
  const { apiKey, base, model } = helmcode();
  const response = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      stream: false,
      temperature,
      max_tokens: 220,
      messages: [
        { role: "system", content: SYSTEM },
        { role: "user", content: USER },
      ],
    }),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Helmcode ${response.status}: ${text.slice(0, 400)}`);
  }
  const json = await response.json();
  return String(json.choices?.[0]?.message?.content ?? "").trim();
}

function hasRequiredFacts(text) {
  const compact = text.replace(/\s+/g, " ");
  return {
    score30: /\b30\b/.test(compact),
    preCap76: /\b76\b/.test(compact),
    days62: /\b62\b/.test(compact),
    treasurer: /tesorero/i.test(compact),
    cash765: /765/.test(compact),
    noClosingOffer: !/quieres que/i.test(compact),
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

function loadKeyFile() {
  if (process.env.HELMCODE_API_KEY) return;
  const file = process.env.HELMCODE_API_KEY_FILE?.trim() || resolve(homedir(), ".helmcode_key");
  if (!existsSync(file)) return;
  process.env.HELMCODE_API_KEY = readFileSync(file, "utf8").trim();
}

async function main() {
  loadDotEnv(resolve(root, ".env.local"));
  loadDotEnv(resolve(root, ".env"));
  loadDotEnv(resolve(root, "../..", ".env"));
  loadKeyFile();
  const byTemp = [];
  for (const temperature of TEMPS) {
    const texts = [];
    for (let i = 0; i < REPEATS; i += 1) {
      const text = await complete(temperature);
      texts.push(text);
      console.log(`\nT=${temperature} #${i + 1}\n${text}\n`);
    }
    const facts = texts.map(hasRequiredFacts);
    const factHits = Object.fromEntries(
      Object.keys(facts[0]).map((key) => [key, facts.filter((row) => row[key]).length / facts.length]),
    );
    byTemp.push({
      temperature,
      meanChars: Math.round(texts.reduce((sum, text) => sum + text.length, 0) / texts.length),
      charSpread: Math.max(...texts.map((text) => text.length)) - Math.min(...texts.map((text) => text.length)),
      jaccard: Number(pairwiseMean(texts, jaccard).toFixed(3)),
      exactPairShare: Number(pairwiseMean(texts, (a, b) => (a === b ? 1 : 0)).toFixed(3)),
      factHits,
      samples: texts,
    });
  }

  const out = {
    at: new Date().toISOString(),
    model: helmcode().model,
    temps: TEMPS,
    repeats: REPEATS,
    byTemp,
  };
  const dest = resolve(root, "scripts/temp-qa.last.json");
  writeFileSync(dest, `${JSON.stringify(out, null, 2)}\n`);
  console.log(`wrote ${dest}`);
  for (const row of byTemp) {
    console.log(
      `T=${row.temperature} jaccard=${row.jaccard} exact=${row.exactPairShare} chars=${row.meanChars}±${row.charSpread} facts=${JSON.stringify(row.factHits)}`,
    );
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
