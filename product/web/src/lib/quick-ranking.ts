/** A month counts half as much as the one this many months after it. */
export const HALF_LIFE_MONTHS = 3;

/** Companies with fewer scored months are left out of the rankings: their mean says too little. */
export const MIN_SCORED_MONTHS = 6;

function monthIndex(month: string) {
  const [year, monthNumber] = month.split("-").map(Number);
  return year * 12 + (monthNumber - 1);
}

/**
 * Mean of the monthly scores where recent months weigh more (exponential decay by age).
 * `asOfMonth` is age 0. Returns null when the history is too short to rank.
 */
export function recencyWeightedMean(
  history: { month: string; score: number }[],
  asOfMonth: string,
): number | null {
  const asOf = monthIndex(asOfMonth);
  let weighted = 0;
  let total = 0;
  let scored = 0;
  for (const point of history) {
    if (!Number.isFinite(point.score)) continue;
    const age = Math.max(0, asOf - monthIndex(point.month));
    const weight = 0.5 ** (age / HALF_LIFE_MONTHS);
    weighted += weight * point.score;
    total += weight;
    scored += 1;
  }
  return scored >= MIN_SCORED_MONTHS && total > 0 ? weighted / total : null;
}
