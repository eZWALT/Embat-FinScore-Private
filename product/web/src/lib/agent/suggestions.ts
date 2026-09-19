import type { DashboardView } from "./view-context";

/**
 * First Pregunta chips. Precomputed from the open screen — no LLM wait.
 * Helmcode warmup covers first-token latency when one of these is tapped.
 */
export function openingSuggestions(view?: DashboardView): string[] {
  if (view?.periodFrom && view?.periodTo) {
    return [
      "¿Qué cambió en el periodo que acabo de marcar?",
      "¿Quién tiene que actuar sobre ese cambio?",
    ];
  }
  if (view?.screen === "resumen") {
    return [
      "¿Por qué el índice es el que es este mes?",
      "¿Hay alguna alerta que revisar?",
    ];
  }
  if (view?.screen === "indice") {
    return [
      "¿Qué empresa de las que ves necesita revisión?",
      "¿Qué cambió y quién tiene que actuar?",
    ];
  }
  if (view?.screen === "rapido_chart") {
    return ["¿Qué muestra este gráfico?", "¿Hay alguna alerta que revisar?"];
  }
  return ["¿Qué muestra este gráfico?", "¿Qué cambió y quién tiene que actuar?"];
}

export function cleanFollowupLine(line: string): string | null {
  const text = line
    .replace(/^\s*(?:[-*•]|\d+[.):])\s*/, "")
    .replace(/^["«“]|["»”]$/g, "")
    .replace(/\s+/g, " ")
    .trim();
  if (text.length < 8) return null;
  const clipped = text.length > 80 ? `${text.slice(0, 77).trimEnd()}…` : text;
  return /[?¿]$/.test(clipped) ? clipped : `${clipped}?`;
}

export function extractFollowups(text: string): string[] {
  const chips: string[] = [];
  const chunks = text.split(/\r?\n|(?<=[?¿])\s+(?=¿)/);
  for (const chunk of chunks) {
    const chip = cleanFollowupLine(chunk);
    if (chip && !chips.includes(chip)) chips.push(chip);
    if (chips.length === 2) break;
  }
  return chips;
}

export function fallbackFollowups(): string[] {
  return ["¿Por qué ha cambiado y quién actúa?", "¿Hay alguna alerta que revisar?"];
}
