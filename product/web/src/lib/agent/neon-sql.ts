import { neon } from "@neondatabase/serverless";

type Sql = ReturnType<typeof neon>;

export function connectNeon(): Sql {
  const url = process.env.DATABASE_URL;
  if (!url) {
    throw new Error("DATABASE_URL is not set");
  }
  return neon(url);
}

export async function neonQuery<T extends Record<string, unknown>>(
  text: string,
  params: unknown[] = [],
): Promise<T[]> {
  const sql = connectNeon();
  return (await sql.query(text, params)) as T[];
}

export async function coreIsMounted(): Promise<boolean> {
  try {
    const rows = await neonQuery<{ ok: boolean }>(
      "SELECT EXISTS (SELECT 1 FROM core.companies LIMIT 1) AS ok",
    );
    return Boolean(rows[0]?.ok);
  } catch {
    return false;
  }
}
