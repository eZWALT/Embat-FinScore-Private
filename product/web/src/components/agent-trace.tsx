"use client";

import { getToolName, isToolUIPart, type UIMessage } from "ai";
import { Brain, ChevronRight } from "lucide-react";
import { useEffect, useState } from "react";

import { AgentBusy } from "@/components/agent-busy";
import { toolIcon, toolInputSummary, toolLabel, toolOutputSummary, toolTimingMs } from "@/lib/agent/tool-catalog";
import { formatDecimal } from "@/lib/display";

const COUNT_TICK_MS = 140;

/** Parallel same-kind calls often land in one parts update. Tick ×1 → ×n so the row does not jump. */
function useTickingCount(target: number): number {
  const [shown, setShown] = useState(target > 0 ? 1 : 0);

  useEffect(() => {
    if (target <= 0) {
      setShown(0);
      return;
    }
    setShown((n) => (n < 1 ? 1 : Math.min(n, target)));
    if (target <= 1) return;
    const id = window.setInterval(() => {
      setShown((n) => {
        if (n >= target) {
          window.clearInterval(id);
          return target;
        }
        return n + 1;
      });
    }, COUNT_TICK_MS);
    return () => window.clearInterval(id);
  }, [target]);

  if (target <= 0) return 0;
  return Math.min(Math.max(shown, 1), target);
}

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

/** Same row as a tool chip. Do not dump the chain-of-thought. */
export function AgentThinking({ streaming = false }: { streaming?: boolean }) {
  return (
    <div
      className="flex min-w-0 items-center gap-1.5 py-0.5 text-[12px] leading-snug text-muted-foreground"
      role="status"
      aria-label={streaming ? "Pensando" : "Pensó"}
    >
      <Brain className="size-3 shrink-0 text-violet-600" strokeWidth={2.25} aria-hidden="true" />
      <span className="min-w-0 truncate font-medium text-foreground">Pensando</span>
      {streaming ? <AgentBusy label="Pensando" /> : null}
    </div>
  );
}

function shouldShowInput(name: string, input: unknown): boolean {
  if (name === "query_clean_db") return true;
  if (!input || typeof input !== "object") return false;
  const keys = Object.keys(input as object);
  return !keys.every((key) => ["company_id", "group_id", "entity_id", "month", "kind"].includes(key));
}

function callPreview(name: string, part: ToolPart): { summary: string; input: unknown; output: unknown; errorText: string } {
  const input = "input" in part ? part.input : undefined;
  const output = "output" in part ? part.output : undefined;
  const errorText = "errorText" in part && typeof part.errorText === "string" ? part.errorText : "";
  const status = toolState(part);
  const summary =
    status === "error"
      ? errorText || toolOutputSummary(name, output)
      : toolOutputSummary(name, output) || toolInputSummary(name, input);
  return { summary, input, output, errorText };
}

/** Consecutive same-kind calls share one row: «Leer índice ×2». Count ticks in place. */
export function AgentTrace({
  parts,
  keepBusy = false,
}: {
  parts: ToolPart[];
  /** Stay spinning after the run finishes, until the first reply token. */
  keepBusy?: boolean;
}) {
  const calls = parts.filter(isToolUIPart);
  const count = useTickingCount(calls.length);
  if (!calls.length) return null;
  const name = getToolName(calls[0]);
  const Icon = toolIcon(name);
  const states = calls.map(toolState);
  const ticking = count < calls.length;
  const showSpinner = states.some((state) => state === "running") || keepBusy || ticking;
  const errored = !keepBusy && states.some((state) => state === "error");
  const allDone = states.every((state) => state === "done");
  const ms = calls.reduce((sum, part) => {
    const value = toolTimingMs("output" in part ? part.output : undefined);
    return value != null ? sum + value : sum;
  }, 0);
  const previews = calls.map((part) => callPreview(name, part));
  const hasBody = previews.some((row) => row.summary || row.input != null || row.output != null || row.errorText);

  return (
    <details className="group min-w-0 text-[12px] leading-snug">
      <summary className="flex min-w-0 cursor-pointer list-none items-center gap-1.5 overflow-hidden py-0.5 text-muted-foreground">
        <Icon className="size-3 shrink-0" aria-hidden="true" />
        <span className="min-w-0 truncate font-medium text-foreground">
          {toolLabel(name)}
          {count > 1 ? <span className="tabular-nums text-muted-foreground"> ×{count}</span> : null}
        </span>
        <ChevronRight className="size-3 shrink-0 transition-transform group-open:rotate-90" />
        {showSpinner ? <AgentBusy /> : null}
        {errored ? <span className="text-destructive">Error</span> : null}
        {allDone && !keepBusy && ms > 0 ? (
          <span className="shrink-0 font-mono text-[10px] tabular-nums">{formatSeconds(ms)}</span>
        ) : null}
      </summary>
      {hasBody ? (
        <div className="space-y-2 border-l pl-3 ml-4 py-1.5">
          {previews.map((row, index) => (
            <div key={index} className="space-y-1">
              {calls.length > 1 ? (
                <p className="font-mono text-[10px] text-muted-foreground">×{index + 1}</p>
              ) : null}
              {row.summary ? <p className="text-muted-foreground">{row.summary}</p> : null}
              {row.input != null && shouldShowInput(name, row.input) ? (
                <pre className="max-h-32 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-muted-foreground">
                  {name === "query_clean_db" && row.input && typeof row.input === "object" && "sql" in row.input
                    ? String((row.input as { sql: unknown }).sql)
                    : preview(row.input, 800)}
                </pre>
              ) : null}
              {row.output != null || row.errorText ? (
                <pre className="max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-muted-foreground">
                  {row.errorText || preview(row.output)}
                </pre>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
    </details>
  );
}
