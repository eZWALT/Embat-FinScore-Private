/** Server-only Exa search. Used by Pregunta thinking mode, never the browser. */

const EXA_SEARCH_URL = "https://api.exa.ai/search";
const TIMEOUT_MS = 15_000;

export function exaApiKey(): string | undefined {
  const key = process.env.EXA_API_KEY?.trim();
  return key || undefined;
}

export type ExaHit = {
  title: string;
  url: string;
  publishedDate: string | null;
  highlights: string[];
};

export async function searchExa(query: string): Promise<{ query: string; results: ExaHit[] } | { error: string }> {
  const key = exaApiKey();
  if (!key) return { error: "EXA_API_KEY is not set" };

  const q = query.trim().slice(0, 500);
  if (!q) return { error: "empty query" };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetch(EXA_SEARCH_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": key,
        Authorization: `Bearer ${key}`,
      },
      body: JSON.stringify({
        query: q,
        type: "auto",
        numResults: 5,
        contents: { highlights: { maxCharacters: 400 } },
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      return { error: `exa ${response.status}${detail ? `: ${detail.slice(0, 160)}` : ""}` };
    }
    const data = (await response.json()) as {
      results?: Array<{
        title?: string;
        url?: string;
        publishedDate?: string | null;
        highlights?: string[];
      }>;
    };
    const results = (data.results ?? [])
      .filter((row): row is typeof row & { url: string } => Boolean(row.url))
      .map((row) => ({
        title: row.title?.trim() || row.url,
        url: row.url,
        publishedDate: row.publishedDate ?? null,
        highlights: (row.highlights ?? []).map((line) => line.trim()).filter(Boolean).slice(0, 3),
      }))
      .slice(0, 5);
    if (!results.length) return { error: "no results" };
    return { query: q, results };
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") return { error: "exa timeout" };
    return { error: error instanceof Error ? error.message : String(error) };
  } finally {
    clearTimeout(timer);
  }
}
