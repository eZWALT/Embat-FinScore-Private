"use client";

import { getToolName, isToolUIPart, type UIMessage } from "ai";
import { ChevronRight, Wrench } from "lucide-react";

import { AgentBusy } from "@/components/agent-busy";
import { toolInputSummary, toolLabel, toolOutputSummary, toolTimingMs } from "@/lib/agent/tool-catalog";
import { formatDecimal } from "@/lib/display";

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

function formatSeconds(ms: number): string {
  if (ms < 50) return "<0.1 s";
  return `${formatDecimal(ms / 1000, 1)} s`;
}

export function AgentTrace({
  part,
  index,
  keepBusy = false,
}: {
  part: ToolPart;
  index?: number;
  /** Stay spinning after the tool finishes, until the first reply token. */
  keepBusy?: boolean;
}) {
  if (!isToolUIPart(part)) return null;
  const name = getToolName(part);
  const status = toolState(part);
  const showSpinner = status === "running" || keepBusy;
  const input = "input" in part ? part.input : undefined;
  const output = "output" in part ? part.output : undefined;
  const errorText = "errorText" in part && typeof part.errorText === "string" ? part.errorText : "";
  const summary =
    status === "error"
      ? errorText || toolOutputSummary(name, output)
      : toolOutputSummary(name, output) || toolInputSummary(name, input);
  const ms = toolTimingMs(output);

  return (
    <details className="group min-w-0 text-[12px] leading-snug">
      <summary className="flex min-w-0 cursor-pointer list-none items-center gap-1.5 overflow-hidden py-0.5 text-muted-foreground">
        {index != null ? (
          <span className="w-3 shrink-0 text-center font-mono text-[10px] tabular-nums" aria-hidden="true">
            {index}
          </span>
        ) : null}
        <ChevronRight className="size-3 shrink-0 transition-transform group-open:rotate-90" />
        <Wrench className="size-3 shrink-0" aria-hidden="true" />
        <span className="min-w-0 flex-1 truncate font-medium text-foreground">{toolLabel(name)}</span>
        {showSpinner ? <AgentBusy /> : null}
        {status === "error" && !keepBusy ? <span className="text-destructive">Error</span> : null}
        {status === "done" && !keepBusy && ms != null ? (
          <span className="shrink-0 font-mono text-[10px] tabular-nums">{formatSeconds(ms)}</span>
        ) : null}
      </summary>
      {summary || input != null || output != null || errorText ? (
        <div className="space-y-2 border-l pl-3 ml-4 py-1.5">
          {summary ? <p className="text-muted-foreground">{summary}</p> : null}
          {input != null ? (
            <pre className="max-h-32 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-muted-foreground">
              {name === "query_clean_db" && input && typeof input === "object" && "sql" in input
                ? String((input as { sql: unknown }).sql)
                : preview(input, 800)}
            </pre>
          ) : null}
          {output != null || errorText ? (
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-muted-foreground">
              {errorText || preview(output)}
            </pre>
          ) : null}
        </div>
      ) : null}
    </details>
  );
}
