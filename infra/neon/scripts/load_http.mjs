#!/usr/bin/env node
/**
 * Apply SQL migrations and COPY CSVs through the Neon HTTP driver.
 * Use this when TCP 5432 is blocked (common on some networks). psql \\copy remains
 * the preferred path when a direct connection works.
 *
 *   DATABASE_URL=... node infra/neon/scripts/load_http.mjs [--analytics-only]
 */
import { createRequire } from "node:module";
import { readFileSync, createReadStream } from "node:fs";
import { createInterface } from "node:readline";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "../../..");
const { neon } = createRequire(import.meta.url)(
  path.join(repoRoot, "product/web/node_modules/@neondatabase/serverless"),
);

const root = path.resolve(scriptDir, "..");
const csvDir = process.env.CSV_DIR || path.join(root, ".staging/csv");
const mode = process.argv[2] || "--analytics-only";

function readUrl() {
  if (process.env.DATABASE_URL) return process.env.DATABASE_URL;
  const env = readFileSync(path.join(root, ".env"), "utf8");
  const m = env.match(/DATABASE_URL="(.*)"/);
  if (!m) throw new Error("DATABASE_URL missing");
  return m[1];
}

const sql = neon(readUrl());

function splitSql(text) {
  const stmts = [];
  let buf = "";
  let i = 0;
  let dollar = null;
  while (i < text.length) {
    const tag = text.slice(i).match(/^\$[a-zA-Z0-9_]*\$/);
    if (tag) {
      if (dollar === null) dollar = tag[0];
      else if (dollar === tag[0]) dollar = null;
      buf += tag[0];
      i += tag[0].length;
      continue;
    }
    const c = text[i];
    if (!dollar && c === "-" && text[i + 1] === "-") {
      while (i < text.length && text[i] !== "\n") i++;
      continue;
    }
    if (!dollar && c === ";") {
      const s = buf.trim();
      if (s) stmts.push(s);
      buf = "";
      i++;
      continue;
    }
    buf += c;
    i++;
  }
  const tail = buf.trim();
  if (tail) stmts.push(tail);
  return stmts;
}

async function runFile(rel) {
  const text = readFileSync(path.join(root, "migrations", rel), "utf8");
  console.log("sql", rel);
  for (const stmt of splitSql(text)) {
    await sql.query(stmt, []);
  }
}

function parseCsvLine(line, state) {
  const { fields, cur, inQ } = state;
  let buf = cur;
  let q = inQ;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (q) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          buf += '"';
          i++;
        } else q = false;
      } else buf += c;
    } else if (c === '"') q = true;
    else if (c === ",") {
      fields.push(buf);
      buf = "";
    } else buf += c;
  }
  if (q) {
    buf += "\n";
    return { fields, cur: buf, inQ: true, done: false };
  }
  fields.push(buf);
  return { fields, cur: "", inQ: false, done: true };
}

async function* csvRows(file) {
  const rl = createInterface({ input: createReadStream(file, { encoding: "utf8" }) });
  let header = null;
  let state = { fields: [], cur: "", inQ: false };
  for await (const line of rl) {
    state = parseCsvLine(line, state);
    if (!state.done) continue;
    const values = state.fields;
    state = { fields: [], cur: "", inQ: false };
    if (!header) {
      header = values;
      continue;
    }
    const row = {};
    header.forEach((h, i) => {
      row[h] = values[i] === "" || values[i] === undefined ? null : values[i];
    });
    yield row;
  }
}

function emptyArrays(n) {
  return Array.from({ length: n }, () => []);
}

async function copyTable(table, file, columns, casts) {
  const full = path.join(csvDir, file);
  console.log("load", table);
  let batch = emptyArrays(columns.length);
  let n = 0;
  const flush = async () => {
    if (!batch[0].length) return;
    const selects = columns.map((_, i) => `u.c${i}::${casts[i]}`).join(", ");
    const unnests = columns.map((_, i) => `$${i + 1}::text[]`).join(", ");
    const aliases = columns.map((_, i) => `c${i}`).join(", ");
    await sql.query(
      `INSERT INTO ${table} (${columns.join(", ")}) SELECT ${selects} FROM unnest(${unnests}) AS u(${aliases})`,
      batch,
    );
    n += batch[0].length;
    batch = emptyArrays(columns.length);
  };
  for await (const row of csvRows(full)) {
    columns.forEach((col, i) => batch[i].push(row[col]));
    if (batch[0].length >= 250) await flush();
  }
  await flush();
  console.log(" ", n, "rows");
  return n;
}

const T = {
  uuid: "uuid",
  text: "text",
  int: "integer",
  bool: "boolean",
  f8: "double precision",
  ts: "timestamptz",
  date: "date",
  jsonb: "jsonb",
  texta: "text[]",
  f8a: "double precision[]",
  boola: "boolean[]",
};

async function loadAnalytics() {
  await sql.query("TRUNCATE analytics.score_runs RESTART IDENTITY CASCADE");
  await sql.query(
    "TRUNCATE analytics.stg_score_categories, analytics.stg_score_items, analytics.stg_score_reasons, analytics.stg_forecast_points",
  );
  await copyTable("analytics.score_runs", "score_runs.csv", [
    "run_id","schema_version","scorecard_version","generated_at","as_of_month","months","detail_months","detail_from_month","is_sample","n_companies","n_groups","n_company_months","source","spec","monitor","sections","reference","files","disclaimer",
  ], [T.uuid,T.text,T.text,T.ts,T.text,T.texta,T.int,T.text,T.bool,T.int,T.int,T.int,T.jsonb,T.jsonb,T.jsonb,T.jsonb,T.jsonb,T.jsonb,T.text]);
  await copyTable("analytics.clusters", "clusters.csv", ["run_id","cluster_id","label","description","n_companies","n_companies_train_fit"], [T.uuid,T.text,T.text,T.text,T.int,T.int]);
  await copyTable("analytics.cluster_quality", "cluster_quality.csv", ["run_id","chosen_k","silhouette","note"], [T.uuid,T.int,T.jsonb,T.text]);
  await copyTable("analytics.company_profiles", "company_profiles.csv", ["run_id","company_id","group_id","country","currency","erp","first_month","latest_month","cluster_id","alert_ids"], [T.uuid,T.text,T.text,T.text,T.text,T.text,T.text,T.text,T.text,T.texta]);
  await copyTable("analytics.company_index", "company_index.csv", ["run_id","company_id","group_id","latest_month","score","trajectory","confidence","guard","delta_1m","delta_3m","top_reason","cluster_id","n_alerts","max_alert_severity","sparkline"], [T.uuid,T.text,T.text,T.text,T.f8,T.text,T.text,T.text,T.f8,T.f8,T.text,T.text,T.int,T.text,T.f8a]);
  await copyTable("analytics.company_scores", "company_scores.csv", ["run_id","company_id","month","score","score_pre_cap","guard","guard_adjustment","trajectory","slope3","slope6","confidence","confidence_note","coverage","trail_months","change_guard"], [T.uuid,T.text,T.text,T.f8,T.f8,T.text,T.f8,T.text,T.f8,T.f8,T.text,T.text,T.f8,T.int,T.f8]);
  await copyTable("analytics.stg_score_categories", "score_categories.csv", ["run_id","company_id","month","category_id","score","contribution"], [T.uuid,T.text,T.text,T.text,T.f8,T.f8]);
  await copyTable("analytics.stg_score_items", "score_items.csv", ["run_id","company_id","month","item_id","value","points","contribution","delta"], [T.uuid,T.text,T.text,T.text,T.f8,T.f8,T.f8,T.f8]);
  await copyTable("analytics.stg_score_reasons", "score_reasons.csv", ["run_id","company_id","month","kind","position","item","label","points","value","unit","eur","sentence"], [T.uuid,T.text,T.text,T.text,T.int,T.text,T.text,T.f8,T.f8,T.text,T.f8,T.text]);
  await copyTable("analytics.company_cluster", "company_cluster.csv", ["run_id","company_id","cluster_id","month"], [T.uuid,T.text,T.text,T.text]);
  await copyTable("analytics.cluster_vs", "cluster_vs.csv", ["run_id","company_id","metric","percentile","robust_z"], [T.uuid,T.text,T.text,T.f8,T.f8]);
  await copyTable("analytics.control_charts", "control_charts.csv", ["run_id","entity_type","entity_id","comparison","metric","months","values","center","lower","upper","ewma","cusum_low","cusum_high","signal","persistent","method"], [T.uuid,T.text,T.text,T.text,T.text,T.texta,T.f8a,T.f8a,T.f8a,T.f8a,T.f8a,T.f8a,T.f8a,T.texta,T.boola,T.jsonb]);
  await copyTable("analytics.forecasts", "forecasts.csv", ["run_id","company_id","metric","method","origin_month","horizon_months","naive_last","skill_vs_naive","note"], [T.uuid,T.text,T.text,T.text,T.text,T.int,T.f8,T.f8,T.text]);
  await copyTable("analytics.stg_forecast_points", "forecast_points.csv", ["run_id","company_id","month","median","lo50","hi50","lo80","hi80"], [T.uuid,T.text,T.text,T.f8,T.f8,T.f8,T.f8,T.f8]);
  await copyTable("analytics.alerts", "alerts.csv", ["run_id","alert_id","entity_type","entity_id","entity_name","month","kind","direction","severity","title","summary","owner","action","persistence_rule","months_flagged","rank_score","evidence"], [T.uuid,T.text,T.text,T.text,T.text,T.text,T.text,T.text,T.text,T.text,T.text,T.text,T.text,T.text,T.int,T.f8,T.jsonb]);
  await copyTable("analytics.alert_reasons", "alert_reasons.csv", ["run_id","alert_id","position","item","label","points","value","unit","eur","sentence"], [T.uuid,T.text,T.int,T.text,T.text,T.f8,T.f8,T.text,T.f8,T.text]);
  await copyTable("analytics.groups_index", "groups_index.csv", ["run_id","group_id","n_companies","latest_mean_score","latest_min_score","latest_min_company_id","mean_scores","limits_available","alert_ids"], [T.uuid,T.text,T.int,T.f8,T.f8,T.text,T.f8a,T.bool,T.texta]);
  await copyTable("analytics.group_members", "group_members.csv", ["run_id","group_id","company_id"], [T.uuid,T.text,T.text]);
  await runFile("003b_map_score_children.sql");
}

async function main() {
  await runFile("001_schemas.sql");
  await runFile("002_core.sql");
  await runFile("003_analytics.sql");
  if (mode !== "--core-only") await loadAnalytics();
  await runFile("004_indexes_fks.sql");
  await runFile("005_api_views.sql");
  await runFile("006_security.sql");
  await sql.query("ANALYZE");
  const counts = await sql`
    SELECT n_companies, n_groups, n_company_months FROM api.current_run
  `;
  console.log("current_run", counts[0]);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
