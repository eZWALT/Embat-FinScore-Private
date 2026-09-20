const ALLOWED_TABLES = new Set([
  "companies",
  "groups",
  "transactions",
  "invoices",
  "balances",
  "banking_products",
  "debt_products",
  "debt_schedule_config",
  "dq_log",
]);

/** Neon `core` names. Same facts as `clean.*`; a few tables were renamed on load. */
const CORE_TABLES = new Set([
  ...ALLOWED_TABLES,
  "products",
  "debt_schedule",
  "counterparties",
]);

const TABLE_ALIASES: Record<string, string> = {
  banking_products: "products",
  debt_products: "products",
  debt_schedule_config: "debt_schedule",
};

const MAX_ROWS = 200;
const FORBIDDEN =
  /\b(insert|update|delete|drop|alter|create|attach|copy|export|import|pragma|install|load|call|set|reset|truncate|vacuum|checkpoint)\b/i;
const TABLE_REF = /\b(?:from|join)\s+([a-zA-Z_][\w.]*)/gi;

export class UnsafeQuery extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnsafeQuery";
  }
}

export function checkSql(sql: string): string {
  let s = sql.trim().replace(/;+$/, "");
  if (s.includes(";")) throw new UnsafeQuery("una sola sentencia");
  if (!/^\s*(select|with)\b/i.test(s)) throw new UnsafeQuery("solo SELECT");
  if (FORBIDDEN.test(s)) throw new UnsafeQuery("solo lectura: la sentencia tiene una palabra prohibida");

  for (const match of s.matchAll(TABLE_REF)) {
    const ref = match[1];
    const low = ref.toLowerCase();
    if (low.startsWith("clean.") || low.startsWith("core.")) {
      const table = low.split(".", 2)[1];
      const allowed = low.startsWith("core.") ? CORE_TABLES : ALLOWED_TABLES;
      if (!allowed.has(table) && !CORE_TABLES.has(TABLE_ALIASES[table] ?? "")) {
        throw new UnsafeQuery(`tabla desconocida ${ref}`);
      }
    } else if (ALLOWED_TABLES.has(low) || CORE_TABLES.has(low)) {
      throw new UnsafeQuery(`usa el esquema clean: clean.${ref}`);
    } else if (low.includes(".")) {
      throw new UnsafeQuery(`solo tablas clean.* o core.*, no ${ref}`);
    }
  }

  if (!/\blimit\s+\d+/i.test(s)) s = `${s}\nLIMIT ${MAX_ROWS}`;
  return s;
}

/** Rewrite a guarded `clean.*` statement for Neon `core` (same facts, some table names differ). */
export function toCoreSql(sql: string): string {
  return sql
    .replace(/\bclean\.banking_products\b/gi, "core.products")
    .replace(/\bclean\.debt_products\b/gi, "core.products")
    .replace(/\bclean\.debt_schedule_config\b/gi, "core.debt_schedule")
    .replace(/\bclean\./gi, "core.");
}

export { MAX_ROWS, ALLOWED_TABLES, CORE_TABLES };
