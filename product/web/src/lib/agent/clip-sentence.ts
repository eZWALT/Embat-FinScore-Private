/** Drop the 10 pts/month glide lecture and say «sin el tope», never límite/salvaguarda. */
export function clipGlideLecture(text: string): string {
  const clipped = text
    .replace(/\s*:?\s*la puntuación baja como máximo 10 puntos al mes hacia \d+[^.]*\.?/gi, "")
    .replace(/sin el l[ií]mite ser[ií]a/gi, "sin el tope sería")
    .replace(/sin la salvaguarda ser[ií]a/gi, "sin el tope sería")
    .replace(/\s{2,}/g, " ")
    .replace(/\s*:\s*$/g, "")
    .trim();
  if (!clipped) return text;
  return /[.!?…]$/.test(clipped) ? clipped : `${clipped}.`;
}
