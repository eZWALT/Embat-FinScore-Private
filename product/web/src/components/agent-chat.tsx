"use client";

import { useChat } from "@ai-sdk/react";
import {
  DefaultChatTransport,
  getToolName,
  isReasoningUIPart,
  isTextUIPart,
  isToolUIPart,
  type DynamicToolUIPart,
  type ToolUIPart,
  type UIMessage,
} from "ai";
import { ArrowUp, Brain, Square } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { cn } from "cn";

import { AgentBusy, AgentPulse } from "@/components/agent-busy";
import { AgentMarkdown } from "@/components/agent-markdown";
import { AgentPlot } from "@/components/agent-plot";
import { AgentThinking, AgentTrace } from "@/components/agent-trace";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { contextBody, PLOT_TOOLS, plotFromPart, textFromParts, type AgentContext } from "@/lib/agent/chat-parts";
import { streamFollowupChips } from "@/lib/agent/followup-client";
import { fallbackFollowups } from "@/lib/agent/suggestions";

type AgentToolPart = ToolUIPart | DynamicToolUIPart;

type ChatSegment =
  | { kind: "text"; key: string; text: string }
  | { kind: "thinking"; key: string }
  | { kind: "tools"; key: string; items: { part: AgentToolPart; key: string }[] };

/** Consecutive tools in place; a text part starts a new run. Never regroups tools to the top. */
function segmentsInStreamOrder(messageId: string, parts: UIMessage["parts"]): ChatSegment[] {
  const segments: ChatSegment[] = [];
  parts.forEach((part, partIndex) => {
    if (isReasoningUIPart(part)) {
      const prev = segments.at(-1);
      if (prev?.kind === "thinking") return;
      segments.push({ kind: "thinking", key: `${messageId}-think-${partIndex}` });
      return;
    }
    if (isToolUIPart(part)) {
      const prev = segments.at(-1);
      const item = {
        part,
        key: `${messageId}-tool-${part.toolCallId || partIndex}`,
      };
      if (prev?.kind === "tools") {
        prev.items.push(item);
        return;
      }
      segments.push({ kind: "tools", key: `${messageId}-tools-${partIndex}`, items: [item] });
      return;
    }
    if (isTextUIPart(part) && part.text) {
      segments.push({ kind: "text", key: `${messageId}-t-${partIndex}`, text: part.text });
    }
  });
  return segments;
}

function ChipRow({
  items,
  disabled,
  onPick,
}: {
  items: string[];
  disabled: boolean;
  onPick: (question: string) => void;
}) {
  if (items.length < 1) return null;
  return (
    <div className="grid shrink-0 grid-cols-2 gap-2" aria-label="Siguientes preguntas">
      {items.slice(0, 2).map((question) => (
        <button
          key={question}
          type="button"
          disabled={disabled}
          onClick={() => onPick(question)}
          className="min-h-14 min-w-0 rounded-xl border bg-muted/50 px-3 py-2.5 text-left text-[13px] leading-snug wrap-break-word hover:bg-muted disabled:opacity-50"
        >
          {question}
        </button>
      ))}
    </div>
  );
}

export function AgentChat({
  api,
  companyId,
  groupId,
  asOf,
  placeholder,
  emptyHint,
  suggestions,
  seedPrompt,
  seedKey,
  onSeedSent,
  view,
  layout = "page",
}: {
  api: "/api/ask" | "/api/watcher/reply";
  companyId?: string;
  groupId?: string;
  asOf?: string;
  placeholder: string;
  emptyHint?: string;
  suggestions?: string[];
  seedPrompt?: string;
  seedKey?: number;
  /** Called once the seed question has been sent, so its owner can drop it and not replay it on a remount. */
  onSeedSent?: () => void;
  view?: AgentContext["view"];
  layout?: "page" | "sheet";
}) {
  const ctxRef = useRef<AgentContext>({ companyId, groupId, asOf, view });
  ctxRef.current = { companyId, groupId, asOf, view };

  const [thinking, setThinking] = useState(false);
  const thinkingRef = useRef(false);
  thinkingRef.current = thinking;
  const [turnThinking, setTurnThinking] = useState(false);

  function requestBody() {
    return { ...contextBody(ctxRef.current), thinking: thinkingRef.current };
  }

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api,
        body: () => requestBody(),
        prepareSendMessagesRequest: ({ messages, body }) => ({
          body: {
            messages,
            ...body,
          },
        }),
      }),
    [api],
  );

  const { messages, sendMessage, setMessages, stop, status, error, clearError } = useChat({
    id: `${api}:${companyId ?? ""}:${groupId ?? ""}:${asOf ?? ""}`,
    transport,
  });

  const [input, setInput] = useState("");
  const [followups, setFollowups] = useState<string[]>([]);
  const busy = status === "submitted" || status === "streaming";
  const lastSeed = useRef<string | null>(null);
  const followupFor = useRef<string | null>(null);
  const followupAbort = useRef<AbortController | null>(null);

  function resetFollowups() {
    followupAbort.current?.abort();
    followupAbort.current = null;
    followupFor.current = null;
    setFollowups([]);
  }

  useEffect(() => () => followupAbort.current?.abort(), []);

  useEffect(() => {
    const text = seedPrompt?.trim();
    if (!text || busy) return;
    const token = `${seedKey ?? 0}:${text}`;
    if (lastSeed.current === token) return;
    lastSeed.current = token;
    resetFollowups();
    setTurnThinking(thinkingRef.current);
    void sendMessage({ text }, { body: requestBody() });
    onSeedSent?.();
  }, [seedPrompt, seedKey, busy, sendMessage, onSeedSent]);

  function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    resetFollowups();
    setTurnThinking(thinkingRef.current);
    void sendMessage({ text: trimmed }, { body: requestBody() });
    setInput("");
  }

  function cancelTurn() {
    if (!busy) return;
    stop();
    resetFollowups();
    clearError();
    setMessages((prev) => {
      const next = [...prev];
      if (next.at(-1)?.role === "assistant") next.pop();
      return next;
    });
  }

  const last = messages.at(-1);
  const lastAssistant = last?.role === "assistant" ? last : null;
  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  useEffect(() => {
    if (api !== "/api/ask" || busy) return;
    const id = lastAssistant?.id;
    if (!id || followupFor.current === id) return;
    if (!textFromParts(lastAssistant.parts).trim()) return;
    followupFor.current = id;
    followupAbort.current?.abort();
    const ac = new AbortController();
    followupAbort.current = ac;
    const fallback = fallbackFollowups();
    setFollowups(fallback);
    void streamFollowupChips({
      messages: messagesRef.current,
      context: ctxRef.current,
      signal: ac.signal,
      onChip: (chips) => {
        if (!ac.signal.aborted && chips.length) setFollowups(chips.length === 1 ? [chips[0], fallback[1]] : chips);
      },
    })
      .then((chips) => {
        if (ac.signal.aborted) return;
        if (chips.length >= 2) setFollowups(chips.slice(0, 2));
        else if (chips.length === 1) setFollowups([chips[0], fallback[1]]);
      })
      .catch((caught) => {
        if (caught instanceof DOMException && caught.name === "AbortError") return;
      });
  }, [api, lastAssistant?.id, busy]);

  const answerText = lastAssistant ? textFromParts(lastAssistant.parts).trim() : "";
  const chips =
    api !== "/api/ask" || busy
      ? []
      : lastAssistant && answerText
        ? (followups.length ? followups : fallbackFollowups()).slice(0, 2)
        : lastAssistant
          ? []
          : (suggestions ?? []).slice(0, 2);
  const sheet = layout === "sheet";

  return (
    <div className={sheet ? "flex min-h-0 min-w-0 flex-1 flex-col gap-3 overflow-hidden" : "flex min-w-0 flex-col gap-3"}>
      <ol className={sheet ? "min-h-0 min-w-0 flex-1 space-y-3 overflow-y-auto" : "min-w-0 space-y-3"} aria-live="polite">
        {messages.length === 0 && emptyHint ? (
          <li className="text-sm text-muted-foreground">{emptyHint}</li>
        ) : null}
        {messages.map((message) => {
          const streamingMessage = busy && message.role === "assistant" && message.id === messages.at(-1)?.id;
          const segments = segmentsInStreamOrder(message.id, message.parts);
          if (
            message.role === "assistant" &&
            turnThinking &&
            streamingMessage &&
            !segments.some((segment) => segment.kind === "thinking")
          ) {
            segments.unshift({ kind: "thinking", key: `${message.id}-think-wait` });
          }
          const lastSegment = segments.at(-1);
          const waitingForReply = Boolean(streamingMessage && lastSegment?.kind !== "text");

          return (
            <li
              key={message.id}
              className={
                message.role === "user"
                  ? "ml-auto max-w-[85%] min-w-0 space-y-1.5 rounded-xl bg-muted px-4 py-3"
                  : "min-w-0 max-w-full space-y-1.5 overflow-hidden rounded-xl border bg-background px-4 py-3"
              }
            >
              <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                {message.role === "user" ? "Tú" : "Sentinel"}
              </p>
              {message.role === "user"
                ? segments.map((segment) =>
                    segment.kind === "text" ? (
                      <p key={segment.key} className="whitespace-pre-wrap text-sm leading-snug wrap-break-word">
                        {segment.text}
                      </p>
                    ) : null,
                  )
                : segments.length === 0 && streamingMessage ? (
                    <>
                      {turnThinking ? <AgentThinking streaming /> : <AgentBusy />}
                      <AgentPulse divided={false} />
                    </>
                  )
                : segments.map((segment) => {
                    if (segment.kind === "text") {
                      return (
                        <AgentMarkdown
                          key={segment.key}
                          text={segment.text}
                          streaming={streamingMessage && lastSegment?.kind === "text" && lastSegment.key === segment.key}
                        />
                      );
                    }
                    const showPulse = waitingForReply && lastSegment?.key === segment.key;
                    if (segment.kind === "thinking") {
                      return (
                        <div key={segment.key} className="flex min-w-0 flex-col gap-0.5">
                          <AgentThinking streaming={Boolean(showPulse)} />
                          {showPulse ? <AgentPulse /> : null}
                        </div>
                      );
                    }
                    const runs: { name: string; items: typeof segment.items }[] = [];
                    for (const item of segment.items) {
                      const name = getToolName(item.part);
                      const last = runs.at(-1);
                      if (last?.name === name) last.items.push(item);
                      else runs.push({ name, items: [item] });
                    }
                    return (
                      <div key={segment.key} className="flex min-w-0 flex-col gap-0.5">
                        {runs.map((run, runIndex) => {
                          const lastRun = runIndex === runs.length - 1;
                          return (
                            <div key={run.items[0]?.key ?? `${segment.key}-${run.name}`} className="min-w-0 space-y-1">
                              <AgentTrace
                                parts={run.items.map((item) => item.part)}
                                keepBusy={showPulse && lastRun}
                              />
                              {PLOT_TOOLS.has(run.name)
                                ? run.items.map((item) => {
                                    const plot = plotFromPart(item.part);
                                    return plot ? <AgentPlot key={item.key} spec={plot} /> : null;
                                  })
                                : null}
                            </div>
                          );
                        })}
                        {showPulse ? <AgentPulse /> : null}
                      </div>
                    );
                  })}
            </li>
          );
        })}
        {busy && messages.at(-1)?.role !== "assistant" ? (
          <li className="min-w-0 max-w-full space-y-1.5 overflow-hidden rounded-xl border bg-background px-4 py-3">
            <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Sentinel</p>
            {turnThinking ? <AgentThinking streaming /> : <AgentBusy />}
            <AgentPulse divided={false} />
          </li>
        ) : null}
      </ol>

      <ChipRow items={chips} disabled={busy} onPick={submit} />

      {error ? (
        <p className="text-sm text-destructive" role="alert">
          {error.message}
        </p>
      ) : null}

      <form
        className="flex shrink-0 items-center gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (busy) {
            cancelTurn();
            return;
          }
          submit(input);
        }}
      >
        <button
          type="button"
          aria-pressed={thinking}
          aria-label={thinking ? "Pensamiento profundo activado" : "Activar pensamiento profundo"}
          onClick={() => setThinking((on) => !on)}
          className={cn(
            "inline-flex size-8 shrink-0 items-center justify-center rounded-lg transition-colors",
            thinking ? "bg-violet-50 text-violet-600" : "text-neutral-400 hover:text-neutral-700",
          )}
        >
          <Brain className="size-4" strokeWidth={thinking ? 2.25 : 1.75} />
        </button>
        <Input
          value={input}
          onChange={(event) => setInput(event.currentTarget.value)}
          placeholder={placeholder}
          autoComplete="off"
          aria-label={placeholder}
        />
        <Button
          type="submit"
          size="sm"
          disabled={!busy && !input.trim()}
          aria-label={busy ? "Detener" : "Enviar"}
          className="min-w-[5.75rem]"
        >
          {busy ? (
            <Square data-icon="inline-start" className="size-3 fill-current" />
          ) : (
            <ArrowUp data-icon="inline-start" />
          )}
          {busy ? "Detener" : "Enviar"}
        </Button>
      </form>
    </div>
  );
}
