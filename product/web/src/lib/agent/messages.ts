import type { ModelMessage, UIMessage } from "ai";

import { formatDashboardView, type DashboardView } from "./view-context";

type LooseMessage = {
  id?: string;
  role?: string;
  content?: unknown;
  parts?: UIMessage["parts"];
};

function textFromContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block) =>
        typeof block === "object" && block && "text" in block ? String((block as { text: unknown }).text) : "",
      )
      .join("");
  }
  return "";
}

/** Accept useChat UIMessage[] or a curl-friendly `{ role, content }[]`. */
export function coerceUiMessages(input: unknown): UIMessage[] {
  if (!Array.isArray(input)) return [];
  return input.map((raw, index) => {
    const message = (raw ?? {}) as LooseMessage;
    const role = message.role === "assistant" ? "assistant" : message.role === "system" ? "system" : "user";
    if (Array.isArray(message.parts)) {
      return {
        id: message.id ?? `msg-${index}`,
        role,
        parts: message.parts,
      };
    }
    return {
      id: message.id ?? `msg-${index}`,
      role,
      parts: [{ type: "text", text: textFromContent(message.content) }],
    };
  });
}

export function lastUserText(messages: UIMessage[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "user") continue;
    return (message.parts ?? [])
      .map((part) => (part.type === "text" && "text" in part ? String(part.text) : ""))
      .join("");
  }
  return "";
}

/** Invoice / movement / debt questions. Score, alerts and period plots do not need the records schema. */
export function wantsRecordTools(text: string): boolean {
  return /factura|contrapart|movimient|saldo|deuda|invoice|sql|registro|proveedor|cobro pendiente|pr[eé]stamo|banking/i.test(
    text,
  );
}

export function wantsPlotCatalog(text: string): boolean {
  return /gr[aá]fico|plot|dibuja|pinta|abanico|control chart/i.test(text);
}

/** Owner / action / feed. Period and “why this score” use get_company; this mounts the feed. */
export function wantsAlertTools(text: string): boolean {
  return /alerta|avisos?|qui[eé]n debe actuar|qui[eé]n act[uú]a|dueño/i.test(text);
}

/** Lift / false-alarm numbers. “qué alertas hay” is not this. */
export function wantsAlertStats(text: string): boolean {
  return /fiabil|lift|tasa base|acierto|estad[ií]st|75\s*%/i.test(text);
}

/** Dragged period or a named month range. “este mes” is not this. */
export function wantsPeriodHistory(text: string): boolean {
  return /periodo seleccionado|periodo:|\d{4}-\d{2}\s*→|desde .+hasta|[a-záéíóú]+ \d{4} →/i.test(text);
}

/** COMP_ / GROUP_ ids and «Empresa 0011» / «Grupo 0234» mentions. */
export function entitiesFromText(text: string): string[] {
  const found = new Set<string>();
  for (const match of text.matchAll(/\bCOMP_(\d{4})\b/g)) found.add(`COMP_${match[1]}`);
  for (const match of text.matchAll(/\bGROUP_(\d{4})\b/g)) found.add(`GROUP_${match[1]}`);
  for (const match of text.matchAll(/\bEmpresa\s+(\d{1,4})\b/gi)) {
    found.add(`COMP_${match[1].padStart(4, "0")}`);
  }
  for (const match of text.matchAll(/\bGrupo\s+(\d{1,4})\b/gi)) {
    found.add(`GROUP_${match[1].padStart(4, "0")}`);
  }
  return [...found];
}

export function sessionExtra(input: {
  companyId?: string;
  groupId?: string;
  asOf?: string;
  view?: DashboardView;
}): string | undefined {
  const lines: string[] = [];
  if (input.companyId) lines.push(`company_id=${input.companyId}`);
  if (input.groupId) lines.push(`group_id=${input.groupId}`);
  if (input.asOf) lines.push(`as_of=${input.asOf}`);
  const ids = lines.length ? `session\n${lines.join("\n")}` : undefined;
  const screen = input.view ? formatDashboardView(input.view) : undefined;
  if (ids && screen) return `${ids}\n\n${screen}`;
  return screen ?? ids;
}

export type { ModelMessage, UIMessage };
