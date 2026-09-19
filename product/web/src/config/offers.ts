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
}

export const PRODUCTS: ProductRule[] = [
  {
    id: "term_loan",
    name: "Préstamo de inversión",
    pitch: "Financiar expansión o maquinaria a plazo largo.",
    postures: ["grow"],
    when: { score: { min: 75 }, categories: { stability: { min: 70 } }, trajectories: ["improving", "stable"] },
  },
  {
    id: "credit_line",
    name: "Línea de crédito de circulante",
    pitch: "Cubrir desfases de tesorería con una póliza revolving.",
    postures: ["grow", "selective"],
    when: { score: { min: 60 }, categories: { amounts_owed: { min: 65 } } },
  },
  {
    id: "confirming",
    name: "Confirming a proveedores",
    pitch: "Pagar a proveedores en plazo y ofrecerles anticipo.",
    postures: ["grow", "selective"],
    when: { categories: { payment_history: { min: 60 } } },
  },
  {
    id: "deposits",
    name: "Depósitos y cuentas remuneradas",
    pitch: "Rentabilizar la liquidez sobrante.",
    postures: ["grow", "selective"],
    when: { categories: { amounts_owed: { min: 85 } } },
  },
  {
    id: "credit_insurance",
    name: "Seguro de crédito comercial",
    pitch: "Protegerse del impago de su cliente principal.",
    postures: ["grow", "selective", "careful"],
    when: { categories: { mix: { max: 55 } } },
  },
  {
    id: "factoring",
    name: "Factoring o anticipo de facturas",
    pitch: "Liquidez con las facturas como garantía.",
    postures: ["selective", "careful"],
    when: { score: { min: 35 }, categories: { amounts_owed: { max: 65 } } },
  },
  {
    id: "refinancing",
    name: "Refinanciación de deuda",
    pitch: "Reordenar vencimientos con garantías nuevas.",
    postures: ["careful"],
    when: { score: { min: 35 }, categories: { amounts_owed: { max: 50 } } },
  },
];
