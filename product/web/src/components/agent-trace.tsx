"use client";

import { getToolName, isToolUIPart, type UIMessage } from "ai";
import { ChevronRight, Wrench } from "lucide-react";

import {
  toolInputSummary,
  toolLabel,
  toolOutputSummary,
  toolTimingMs,
} from "@/lib/agent/tool-catalog";

type ToolPart = Extract<UIMessage["parts"][number], { type: string }>;

function toolState(part: ToolPart): "running" | "done" | "error" {
  const state = "state" in part ? String(part.state) : "";
  if (state === "output-error") return "error";
  if (state === "output-available") return "done";
  return "running";
}

function preview(value: unknown, max = 1800): string {
  if (value == null) return "";
  try {
    const text = JSON.stringify(value, null, 2);
    return text.length > max ? `${text.slice(0, max)}\n…` : text;
  } catch {
    return String(value);
  }
}

export function AgentTrace({ part, index }: { part: ToolPart; index?: number }) {
  if (!isToolUIPart(part)) return null;
  const name = getToolName(part);
  const status = toolState(part);
  const input = "input" in part ? part.input : undefined;
  const output = "output" in part ? part.output : undefined;
  const errorText = "errorText" in part && typeof part.errorText === "string" ? part.errorText : "";
  const summary =
    status === "error"
      ? errorText || toolOutputSummary(name, output)
      : toolOutputSummary(name, output) || toolInputSummary(name, input);
  const ms = toolTimingMs(output);
  const statusLabel = status === "running" ? "Buscando…" : status === "error" ? "Error" : "Listo";

  return (
    <details className="group min-w-0 rounded-lg border bg-muted/30 text-[12px] leading-snug">
      <summary className="flex min-w-0 cursor-pointer list-none items-center gap-2 overflow-hidden px-2.5 py-1.5">
        {index != null ? (
          <span
            className="w-3.5 shrink-0 text-center font-mono text-[10px] tabular-nums text-muted-foreground"
            aria-hidden="true"
          >
            {index}
          </span>
        ) : null}
        <ChevronRight className="size-3 shrink-0 text-muted-foreground transition-transform group-open:rotate-90" />
        <Wrench className="size-3 shrink-0 text-muted-foreground" aria-hidden="true" />
        <span className="min-w-0 truncate font-mono text-[11px] text-muted-foreground">
          tools ({name})
        </span>
        <span className="min-w-0 flex-1 truncate font-medium">{toolLabel(name)}</span>
        {summary ? <span className="hidden max-w-[40%] truncate text-muted-foreground sm:inline">{summary}</span> : null}
        {ms != null ? <span className="shrink-0 font-mono text-[10px] text-muted-foreground">{ms} ms</span> : null}
        <span
          className={
            status === "error"
              ? "shrink-0 text-destructive"
              : status === "running"
                ? "shrink-0 text-muted-foreground"
                : "shrink-0 text-muted-foreground"
          }
        >
          {statusLabel}
        </span>
      </summary>
      <div className="space-y-2 border-t px-2.5 py-2">
        {summary ? <p className="text-muted-foreground sm:hidden">{summary}</p> : null}
        {input != null ? (
          <div>
            <p className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Entrada</p>
            <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-muted-foreground">
              {name === "query_clean_db" && input && typeof input === "object" && "sql" in input
                ? String((input as { sql: unknown }).sql)
                : preview(input, 800)}
            </pre>
          </div>
        ) : null}
        {output != null || errorText ? (
          <div>
            <p className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Salida</p>
            <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-muted-foreground">
              {errorText || preview(output)}
            </pre>
          </div>
        ) : null}
      </div>
    </details>
  );
}
