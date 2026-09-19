import {
  ADJUSTMENTS,
  POSTURES,
  PRODUCTS,
  type Posture,
  type PostureId,
  type ProductRule,
  type Range,
} from "@/config/offers";
import { CATEGORY_LABELS } from "@/lib/data/plain-language";
import type { CategoryId, DashboardCompany } from "@/lib/data/types";

export interface Suggestion {
  product: ProductRule;
  /** The conditions that held, in words, e.g. "Liquidez y deuda 82 (mínimo 65)". Shown as a tooltip. */
  because: string[];
}

export interface OfferGuidance {
  posture: Posture;
  suggestions: Suggestion[];
}

type Guidable = Pick<DashboardCompany, "score" | "delta3m" | "trajectory" | "confidence" | "guard" | "categories">;

const ORDER: PostureId[] = POSTURES.map((posture) => posture.id);

function postureById(id: PostureId): Posture {
  return POSTURES.find((posture) => posture.id === id) ?? POSTURES[POSTURES.length - 1];
}

/** The posture is never better than `limit`. */
function cappedAt(current: PostureId, limit: PostureId): PostureId {
  return ORDER.indexOf(current) < ORDER.indexOf(limit) ? limit : current;
}

function inRange(value: number | null, range: Range | undefined): boolean {
  if (!range) return true;
  if (value === null) return false;
  return (range.min === undefined || value >= range.min) && (range.max === undefined || value <= range.max);
}

function describe(label: string, value: number, range: Range): string {
  const bound =
    range.min !== undefined && range.max !== undefined
      ? `entre ${range.min} y ${range.max}`
      : range.min !== undefined
        ? `mínimo ${range.min}`
        : `máximo ${range.max}`;
  return `${label} ${Math.round(value)} (${bound})`;
}

/** Posture and product suggestions for one company, from the rules in `config/offers.ts`. */
export function guidanceFor(company: Guidable): OfferGuidance {
  const byScore = POSTURES.find((posture) => company.score >= posture.minScore) ?? POSTURES[POSTURES.length - 1];
  let id: PostureId = byScore.id;

  if (company.guard === "dark") {
    id = ADJUSTMENTS.dark;
  } else if (company.guard === "fading") {
    id = cappedAt(id, ADJUSTMENTS.fading);
  }
  if (company.confidence === "low") id = cappedAt(id, ADJUSTMENTS.lowConfidenceMax);
  const { trajectories, steps } = ADJUSTMENTS.worsening;
  if (steps > 0 && trajectories.includes(company.trajectory) && id !== "protect") {
    id = ORDER[Math.min(ORDER.length - 1, ORDER.indexOf(id) + steps)];
  }

  const categoryScore = new Map<CategoryId, number | null>(company.categories.map((row) => [row.id, row.score]));
  const suggestions: Suggestion[] = [];
  for (const product of PRODUCTS) {
    if (!product.postures.includes(id)) continue;
    const { score, delta3m, categories, trajectories: allowed } = product.when;
    if (!inRange(company.score, score) || !inRange(company.delta3m, delta3m)) continue;
    if (allowed && !allowed.includes(company.trajectory)) continue;
    const because: string[] = [];
    if (score) because.push(describe("Índice", company.score, score));
    if (delta3m && company.delta3m !== null) because.push(describe("Cambio a 3 meses", company.delta3m, delta3m));
    let holds = true;
    for (const [category, range] of Object.entries(categories ?? {}) as [CategoryId, Range][]) {
      const value = categoryScore.get(category) ?? null;
      if (!inRange(value, range)) {
        holds = false;
        break;
      }
      because.push(describe(CATEGORY_LABELS[category], value as number, range));
    }
    if (holds) suggestions.push({ product, because });
  }

  return { posture: postureById(id), suggestions };
}
