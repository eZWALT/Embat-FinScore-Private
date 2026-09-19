/**
 * Commercial guidance rules. Edit this file to change what the platform suggests: no other code needs to change.
 *
 * How it works, for one company and its latest month:
 *  1. The score picks a POSTURE (the first one whose `minScore` the score reaches).
 *  2. ADJUSTMENTS can then push the posture towards caution (safety guard, thin history, deteriorating trend).
 *  3. Each PRODUCT is suggested when its posture list contains the final posture and all its `when` conditions hold.
 *     A category with no score never satisfies a condition on it.
 *
 * Scores are 0–100, 100 healthiest. With the current run, the defaults split companies about 24 % / 51 % / 21 % / 4 %
 * across the four postures. This is a monitoring aid, not a credit decision: keep the wording on the cards modest.
 */

import type { CategoryId, Trajectory } from "@/lib/data/types";

export type PostureId = "grow" | "selective" | "careful" | "protect";
export type Tone = "positive" | "neutral" | "caution" | "negative";

export interface Posture {
  id: PostureId;
  /** Score at which this posture starts. The list is read from the strongest down. */
  minScore: number;
  /** Short badge text. */
  label: string;
  tone: Tone;
  headline: string;
}

export const POSTURES: Posture[] = [
  {
    id: "grow",
    minScore: 75,
    label: "Ofrecer",
    tone: "positive",
    headline: "Buen momento para crecer con esta empresa",
  },
  {
    id: "selective",
    minScore: 55,
    label: "Selectivo",
    tone: "neutral",
    headline: "Ofrecer de forma selectiva",
  },
  {
    id: "careful",
    minScore: 35,
    label: "Prudente",
    tone: "caution",
    headline: "Ofrecer solo con garantías",
  },
  {
    id: "protect",
    minScore: 0,
    label: "No ofrecer",
    tone: "negative",
    headline: "No ofrecer crédito nuevo",
  },
];

export const ADJUSTMENTS = {
  /** A company with no bank bookings for 60 days. Forces this posture. */
  dark: "protect" as PostureId,
  /** Inflows have collapsed against its own history. The posture cannot be better than this. */
  fading: "careful" as PostureId,
  /** Low confidence (thin or incomplete data). The posture cannot be better than this. */
  lowConfidenceMax: "selective" as PostureId,
  /** These trajectories move the posture this many steps towards caution (0 disables). */
  worsening: { trajectories: ["deteriorating"] as Trajectory[], steps: 1 },
};

export type Range = { min?: number; max?: number };

/**
 * SIZE. Where the company's bank records are available, its size sets which products make sense and roughly how much to
 * offer: 1,000 and 1,000,000 are not the same conversation. Size is the mean monthly operating inflow over the last
 * 3 months, in the company's own currency (the bands below read as euros; other currencies are not converted).
 */
export interface Magnitude {
  id: string;
  label: string;
  /** Monthly inflow from which this band starts. The list is read from the largest down. */
  minMonthlyInflow: number;
}

export const MAGNITUDES: Magnitude[] = [
  { id: "micro", label: "Micro", minMonthlyInflow: 0 },
  { id: "small", label: "Pequeña", minMonthlyInflow: 10_000 },
  { id: "medium", label: "Mediana", minMonthlyInflow: 100_000 },
  { id: "large", label: "Grande", minMonthlyInflow: 1_000_000 },
  { id: "xlarge", label: "Muy grande", minMonthlyInflow: 10_000_000 },
];

/** What an indicative amount is a multiple of. */
export type Basis = "inflow" | "outflow" | "cash";

/** A product is only suggested to companies at least this big (when the size is known). */
export type SizeFloor = { minMonthlyInflow?: number; minCash?: number };

/** Indicative amount: `low` (prudent) to `high` (ambitious) times the basis. Months of inflow or outflow, a share of cash. */
export interface Ticket {
  basis: Basis;
  low: number;
  high: number;
}

export interface ProductRule {
  id: string;
  name: string;
  /** One line: what it is and who it is for. */
  pitch: string;
  /** Postures in which it may be suggested. */
  postures: PostureId[];
  when: {
    score?: Range;
    /** Change over 3 months, in points. */
    delta3m?: Range;
    categories?: Partial<Record<CategoryId, Range>>;
    /** Suggest only for these trajectories. */
    trajectories?: Trajectory[];
  };
  /** Smallest company this makes sense for. Ignored when the size is unknown. */
  size?: SizeFloor;
  /** How much to offer, roughly. Leave out for products that are priced case by case. */
  ticket?: Ticket;
}

export const PRODUCTS: ProductRule[] = [
  {
    id: "term_loan",
    name: "Préstamo de inversión",
    pitch: "Financiar expansión o maquinaria a plazo largo.",
    postures: ["grow"],
    when: { score: { min: 75 }, categories: { stability: { min: 70 } }, trajectories: ["improving", "stable"] },
    size: { minMonthlyInflow: 20_000 },
    ticket: { basis: "inflow", low: 1, high: 3 },
  },
  {
    id: "credit_line",
    name: "Línea de crédito de circulante",
    pitch: "Cubrir desfases de tesorería con una póliza revolving.",
    postures: ["grow", "selective"],
    when: { score: { min: 60 }, categories: { amounts_owed: { min: 65 } } },
    size: { minMonthlyInflow: 5_000 },
    ticket: { basis: "outflow", low: 0.5, high: 1.5 },
  },
  {
    id: "confirming",
    name: "Confirming a proveedores",
    pitch: "Pagar a proveedores en plazo y ofrecerles anticipo.",
    postures: ["grow", "selective"],
    when: { categories: { payment_history: { min: 60 } } },
    size: { minMonthlyInflow: 20_000 },
    ticket: { basis: "outflow", low: 1, high: 2 },
  },
  {
    id: "deposits",
    name: "Depósitos y cuentas remuneradas",
    pitch: "Rentabilizar la liquidez sobrante.",
    postures: ["grow", "selective"],
    when: { categories: { amounts_owed: { min: 85 } } },
    size: { minCash: 20_000 },
    ticket: { basis: "cash", low: 0.3, high: 0.7 },
  },
  {
    id: "credit_insurance",
    name: "Seguro de crédito comercial",
    pitch: "Protegerse del impago de su cliente principal.",
    postures: ["grow", "selective", "careful"],
    when: { categories: { mix: { max: 55 } } },
    size: { minMonthlyInflow: 10_000 },
    ticket: { basis: "inflow", low: 1, high: 3 },
  },
  {
    id: "factoring",
    name: "Factoring o anticipo de facturas",
    pitch: "Liquidez con las facturas como garantía.",
    postures: ["selective", "careful"],
    when: { score: { min: 35 }, categories: { amounts_owed: { max: 65 } } },
    size: { minMonthlyInflow: 5_000 },
    ticket: { basis: "inflow", low: 0.5, high: 1.5 },
  },
  {
    id: "refinancing",
    name: "Refinanciación de deuda",
    pitch: "Reordenar vencimientos con garantías nuevas.",
    postures: ["careful"],
    when: { score: { min: 35 }, categories: { amounts_owed: { max: 50 } } },
    size: { minMonthlyInflow: 10_000 },
  },
];
