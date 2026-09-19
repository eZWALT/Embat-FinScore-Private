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
  if (s.includes(";")) throw new UnsafeQuery("one statement only");
  if (!/^\s*(select|with)\b/i.test(s)) throw new UnsafeQuery("SELECT only");
  if (FORBIDDEN.test(s)) throw new UnsafeQuery("read-only: statement contains a forbidden keyword");

  for (const match of s.matchAll(TABLE_REF)) {
    const ref = match[1];
    const low = ref.toLowerCase();
    if (low.startsWith("clean.")) {
      const table = low.split(".", 2)[1];
      if (!ALLOWED_TABLES.has(table)) throw new UnsafeQuery(`unknown table ${ref}`);
    } else if (ALLOWED_TABLES.has(low)) {
      throw new UnsafeQuery(`use the clean schema: clean.${ref}`);
    } else if (low.includes(".")) {
      throw new UnsafeQuery(`only clean.* tables may be queried, not ${ref}`);
    }
  }

  if (!/\blimit\s+\d+/i.test(s)) s = `${s}\nLIMIT ${MAX_ROWS}`;
  return s;
}

export { MAX_ROWS, ALLOWED_TABLES };
