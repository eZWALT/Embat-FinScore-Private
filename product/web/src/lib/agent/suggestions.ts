import type { DashboardView } from "./view-context";

/**
 * First Pregunta chips. Precomputed from the open screen — no LLM wait.
 * Helmcode warmup covers first-token latency when one of these is tapped.
 */
export function openingSuggestions(view?: DashboardView, companyId?: string): string[] {
  const hasEntity = Boolean(
    companyId || view?.focusCompanyId || view?.focusGroupId || (view?.series?.length ?? 0) > 0,
  );
  if (view?.periodFrom && view?.periodTo) {
    return ["¿Qué cambió en este periodo?", hasEntity ? "¿Quién tiene que actuar?" : "¿Qué empresa miro primero?"];
  }
  if (!hasEntity) {
    return ["¿Qué empresa miro primero?", "¿Quién está peor este mes?"];
  }
  if (view?.screen === "resumen") {
    return ["¿Por qué este índice este mes?", "¿Qué alertas hay?"];
  }
  if (view?.screen === "indice") {
    return ["¿Qué empresa hay que revisar?", "¿Quién tiene que actuar?"];
  }
  if (view?.screen === "rapido_chart") {
    if ((view.series?.length ?? 0) === 1) {
      return ["¿Por qué este índice este mes?", "¿Qué alertas hay?"];
    }
    return ["¿Qué explica este gráfico?", "¿Qué alertas hay?"];
  }
  return ["¿Por qué este índice este mes?", "¿Qué alertas hay?"];
}

export function cleanFollowupLine(line: string): string | null {
  const text = line
    .replace(/^\s*(?:[-*•]|\d+[.):])\s*/, "")
    .replace(/^["«“]|["»”]$/g, "")
    .replace(/\s+/g, " ")
    .trim();
  if (text.length < 8) return null;
  const clipped = text.length > 56 ? `${text.slice(0, 53).trimEnd()}…` : text;
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

export function isOwnerFollowup(text: string): boolean {
  return /qui[eé]n (debe|tiene que) actuar|qui[eé]n act[uú]a|qu[eé] debe hacer el|dueño/i.test(text);
}

export function fallbackFollowups(hadAlerts = false): string[] {
  return hadAlerts
    ? ["¿Por qué ha cambiado?", "¿Quién tiene que actuar?"]
    : ["¿Por qué ha cambiado?", "¿Qué hay detrás de esos euros?"];
}

/** Dueño/acción chips only after `get_alerts` already ran. */
export function sanitizeFollowups(chips: string[], hadAlerts: boolean): string[] {
  const kept = chips.filter((chip) => hadAlerts || !isOwnerFollowup(chip));
  if (kept.length >= 2) return kept.slice(0, 2);
  const filler = fallbackFollowups(hadAlerts).filter((chip) => !kept.includes(chip));
  return [...kept, ...filler].slice(0, 2);
}
