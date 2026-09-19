import {
  ADJUSTMENTS,
  MAGNITUDES,
  POSTURES,
  PRODUCTS,
  type Basis,
  type Magnitude,
  type Posture,
  type PostureId,
  type ProductRule,
  type Range,
} from "@/config/offers";
import { CATEGORY_LABELS } from "@/lib/data/plain-language";
import type { CategoryId, DashboardCompany } from "@/lib/data/types";
import type { CompanySize } from "@/lib/data/size";

/** An indicative amount range, in the company's currency, and what it is a multiple of. */
export interface Amount {
  low: number;
  high: number;
  /** e.g. "1–3 meses de ingresos". */
  basis: string;
}

export interface Suggestion {
  product: ProductRule;
  /** The conditions that held, in words, e.g. "Liquidez y deuda 82 (mínimo 65)". Shown as a tooltip. */
  because: string[];
  /** Only when the company's size is known and the product has a ticket. */
  amount?: Amount;
}

export interface OfferGuidance {
  posture: Posture;
  suggestions: Suggestion[];
  /** Size band of the company. Null when the size is unknown, or when there is no inflow to size it by. */
  magnitude: Magnitude | null;
  /** Products that fit the company's numbers but are too big for its size, so were left out. */
  tooSmallFor: number;
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

/** The size band of a company by its monthly inflow. Null when nothing came in. */
export function magnitudeFor(size: CompanySize): Magnitude | null {
  if (size.monthlyInflow <= 0) return null;
  return [...MAGNITUDES].reverse().find((band) => size.monthlyInflow >= band.minMonthlyInflow) ?? MAGNITUDES[0];
}

/** Two significant figures: an indicative amount, not a quote. */
function roundAmount(value: number): number {
  if (value <= 0) return 0;
  const step = 10 ** (Math.floor(Math.log10(value)) - 1);
  return Math.round(value / step) * step;
}

const BASIS_UNIT: Record<Basis, (low: string, high: string) => string> = {
  inflow: (low, high) => `${low}–${high} meses de ingresos`,
  outflow: (low, high) => `${low}–${high} meses de pagos`,
  cash: (low, high) => `${low}–${high} % de la caja`,
};

function amountFor(product: ProductRule, size: CompanySize): Amount | undefined {
  const { ticket } = product;
  if (!ticket) return undefined;
  const base = ticket.basis === "inflow" ? size.monthlyInflow : ticket.basis === "outflow" ? size.monthlyOutflow : (size.cash ?? 0);
  const low = roundAmount(base * ticket.low);
  const high = roundAmount(base * ticket.high);
  if (high <= 0) return undefined;
  const scale = ticket.basis === "cash" ? 100 : 1;
  const text = (n: number) => String(Number((n * scale).toFixed(1))).replace(".", ",");
  return { low, high, basis: BASIS_UNIT[ticket.basis](text(ticket.low), text(ticket.high)) };
}

function fitsSize(product: ProductRule, size: CompanySize): boolean {
  const floor = product.size;
  if (!floor) return true;
  if (floor.minMonthlyInflow !== undefined && size.monthlyInflow < floor.minMonthlyInflow) return false;
  if (floor.minCash !== undefined && (size.cash ?? 0) < floor.minCash) return false;
  return true;
}

/**
 * Posture and product suggestions for one company, from the rules in `config/offers.ts`. With the company's `size`,
 * products too big for it are dropped and the rest carry an indicative amount.
 */
export function guidanceFor(company: Guidable, size?: CompanySize | null): OfferGuidance {
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
  let tooSmallFor = 0;
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
    if (!holds) continue;
    if (size && !fitsSize(product, size)) tooSmallFor += 1;
    else suggestions.push({ product, because, amount: size ? amountFor(product, size) : undefined });
  }

  return { posture: postureById(id), suggestions, magnitude: size ? magnitudeFor(size) : null, tooSmallFor };
}
