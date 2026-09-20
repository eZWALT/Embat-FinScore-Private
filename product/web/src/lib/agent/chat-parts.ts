import { getToolName, isTextUIPart, isToolUIPart, type UIMessage } from "ai";

import { isPlotSpec, type PlotSpec } from "./plot-spec";
import type { DashboardView } from "./view-context";

export type AgentContext = {
  companyId?: string;
  groupId?: string;
  asOf?: string;
  view?: DashboardView;
};

/** Fields the Ask / Watcher reply routes expect besides `messages`. */
export function contextBody(context: AgentContext): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  if (context.companyId) body.companyId = context.companyId;
  if (context.groupId) body.groupId = context.groupId;
  if (context.asOf) body.asOf = context.asOf;
  if (context.view) body.view = context.view;
  return body;
}

export function textFromParts(parts: UIMessage["parts"]): string {
  return parts
    .filter(isTextUIPart)
    .map((part) => part.text)
    .join("");
}

/** One chip per tool call, in stream order. Same tool twice → two chips. */
export function toolChipsFromParts(parts: UIMessage["parts"]): { id: string; name: string }[] {
  const chips: { id: string; name: string }[] = [];
  const seen = new Set<string>();
  for (const part of parts) {
    if (!isToolUIPart(part)) continue;
    const name = getToolName(part);
    const id = "toolCallId" in part && part.toolCallId ? String(part.toolCallId) : name;
    if (seen.has(id)) continue;
    seen.add(id);
    chips.push({ id, name });
  }
  return chips;
}

export function plotFromPart(part: UIMessage["parts"][number]): PlotSpec | null {
  if (!isToolUIPart(part) || getToolName(part) !== "plot_series") return null;
  const output = "output" in part ? part.output : undefined;
  const plot = output && typeof output === "object" && "plot" in output ? output.plot : output;
  return isPlotSpec(plot) ? plot : null;
}

export function reasonQuotesFromPart(part: UIMessage["parts"][number]): string[] {
  if (!isToolUIPart(part)) return [];
  const output = "output" in part ? part.output : undefined;
  if (!output || typeof output !== "object") return [];
  const row = output as { reasons?: { sentence?: string }[]; change_reasons?: { sentence?: string }[] };
  const quotes = [...(row.reasons ?? []), ...(row.change_reasons ?? [])]
    .map((reason) => (typeof reason.sentence === "string" ? reason.sentence.trim() : ""))
    .filter(Boolean);
  return quotes.slice(0, 2);
}

export function plotsFromParts(parts: UIMessage["parts"]): PlotSpec[] {
  return parts.map(plotFromPart).filter((plot): plot is PlotSpec => plot != null);
}
