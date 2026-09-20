/** Drop the 10 pts/month glide lecture from a bundle reason sentence. */
export function clipGlideLecture(text: string): string {
  const clipped = text
    .replace(/\s*:?\s*la puntuación baja como máximo 10 puntos al mes hacia \d+[^.]*\.?/gi, "")
    .replace(/\s{2,}/g, " ")
    .replace(/\s*:\s*$/g, "")
    .trim();
  if (!clipped) return text;
  return /[.!?…]$/.test(clipped) ? clipped : `${clipped}.`;
}
