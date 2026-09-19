"use client";

import { type Chat, useChat } from "@ai-sdk/react";
import { getToolName, isToolUIPart, type UIMessage } from "ai";
import { RotateCcw, X } from "lucide-react";

import { AgentMarkdown } from "@/components/agent-markdown";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ToolCallChip } from "@/components/watcher-post-view";
import { formatPoints } from "@/components/group/labels";
import type { MonthRange } from "@/components/quick/quick-chart";
import { textFromParts } from "@/lib/agent/chat-parts";
import { formatMonth } from "@/lib/format-month";
import type { DashboardCompany } from "@/lib/data/types";

/** The question sent to the agent: the numbers are ours, the reasons are the agent's. */
export function buildPrompt(companies: DashboardCompany[], range: MonthRange) {
  const lines = companies.map((company) => {
    const inRange = company.scoreHistory.filter((point) => point.month >= range.from && point.month <= range.to);
    const first = inRange[0];
    const last = inRange.at(-1);
    if (!first || !last) return `- ${company.companyId}: sin puntuación en el periodo`;
    const delta = last.score - first.score;
    return `- ${company.companyId}${company.groupId ? ` (${company.groupId})` : ""}: ${first.score.toFixed(0)} en ${first.month} → ${last.score.toFixed(0)} en ${last.month} (${formatPoints(delta)} pts), hoy ${company.trajectory}`;
  });
  return `Periodo seleccionado: ${range.from} → ${range.to}.\nEmpresas:\n${lines.join("\n")}\n\nExplica qué pasó en ese periodo y por qué.`;
}

function friendlyError(message: string) {
  if (message.includes("HELMCODE_API_KEY")) return "El agente no está configurado en este entorno (falta la clave del modelo).";
  if (message.includes("DATABASE_URL")) return "El agente no puede leer los datos en este entorno.";
  return "No se pudo generar la explicación.";
}

/**
 * Shows the explanation streaming into `chat`. The parent owns the chat and sends the request from the drag
 * event: `useChat` would stop a chat it created itself when React remounts the component in dev.
 */
export function QuickExplain({
  chat,
  companies,
  range,
  onClose,
}: {
  chat: Chat<UIMessage>;
  companies: DashboardCompany[];
  range: MonthRange;
  onClose: () => void;
}) {
  const { messages, status, error, regenerate } = useChat({ chat });

  const answer = textFromParts(messages.findLast((message) => message.role === "assistant")?.parts ?? []);
  const tools = (messages.findLast((message) => message.role === "assistant")?.parts ?? []).filter(isToolUIPart);
  const busy = status === "submitted" || status === "streaming";

  return (
    <section aria-label="Explicación del periodo" aria-live="polite" className="rounded-xl border bg-card px-4 py-3 animate-in fade-in-0 slide-in-from-bottom-2 duration-300 motion-reduce:animate-none">
      <div className="mb-2 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-medium text-muted-foreground">
            {formatMonth(range.from)} → {formatMonth(range.to)}
          </p>
          <p className="truncate font-mono text-xs text-muted-foreground">{companies.map((c) => c.companyId).join(" · ")}</p>
        </div>
        <Button type="button" variant="ghost" size="icon-sm" onClick={onClose} aria-label="Cerrar explicación">
          <X />
        </Button>
      </div>

      {error ? (
        <div className="flex items-center justify-between gap-3">
          <p role="alert" className="text-sm text-destructive">
            {friendlyError(error.message)}
          </p>
          <Button type="button" variant="outline" size="sm" onClick={() => void regenerate()}>
            <RotateCcw data-icon="inline-start" />
            Reintentar
          </Button>
        </div>
      ) : answer ? (
        <AgentMarkdown text={answer} streaming={busy} />
      ) : (
        <div className="space-y-2" aria-label="Generando explicación">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      )}

      {busy && tools.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {tools.map((part, index) => (
            <ToolCallChip key={"toolCallId" in part ? String(part.toolCallId) : index} name={getToolName(part)} />
          ))}
        </div>
      ) : null}
    </section>
  );
}
