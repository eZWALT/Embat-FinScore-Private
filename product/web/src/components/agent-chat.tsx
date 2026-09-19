"use client";

import { useChat } from "@ai-sdk/react";
import {
  DefaultChatTransport,
  getToolName,
  isTextUIPart,
  isToolUIPart,
  type DynamicToolUIPart,
  type ToolUIPart,
  type UIMessage,
} from "ai";
import { ArrowUp } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { AgentBusy, AgentPulse } from "@/components/agent-busy";
import { AgentMarkdown } from "@/components/agent-markdown";
import { AgentPlot } from "@/components/agent-plot";
import { AgentTrace } from "@/components/agent-trace";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { contextBody, plotFromPart, type AgentContext } from "@/lib/agent/chat-parts";
import { streamFollowupChips } from "@/lib/agent/followup-client";
import { fallbackFollowups } from "@/lib/agent/suggestions";

type AgentToolPart = ToolUIPart | DynamicToolUIPart;

type ChatSegment =
  | { kind: "text"; key: string; text: string }
  | { kind: "tools"; key: string; items: { part: AgentToolPart; runIndex: number; key: string }[] };

/** Consecutive tools in place; a text part starts a new run. Never regroups tools to the top. */
function segmentsInStreamOrder(messageId: string, parts: UIMessage["parts"]): ChatSegment[] {
  const segments: ChatSegment[] = [];
  parts.forEach((part, partIndex) => {
    if (isToolUIPart(part)) {
      const prev = segments.at(-1);
      const item = {
        part,
        runIndex: prev?.kind === "tools" ? prev.items.length + 1 : 1,
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
  view?: AgentContext["view"];
  layout?: "page" | "sheet";
}) {
  const ctxRef = useRef<AgentContext>({ companyId, groupId, asOf, view });
  ctxRef.current = { companyId, groupId, asOf, view };

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api,
        body: () => contextBody(ctxRef.current),
        prepareSendMessagesRequest: ({ messages, body }) => ({
          body: {
            messages,
            ...body,
          },
        }),
      }),
    [api],
  );

  const { messages, sendMessage, status, error } = useChat({
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
    void sendMessage({ text }, { body: contextBody(ctxRef.current) });
  }, [seedPrompt, seedKey, busy, sendMessage]);

  function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    resetFollowups();
    void sendMessage({ text: trimmed }, { body: contextBody(ctxRef.current) });
    setInput("");
  }

  const last = messages.at(-1);
  const lastAssistant = last?.role === "assistant" ? last : null;
  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  useEffect(() => {
    if (api !== "/api/ask") return;
    const id = lastAssistant?.id;
    if (!id || followupFor.current === id) return;
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
  }, [api, lastAssistant?.id]);

  const waitingOnSend = busy && messages.at(-1)?.role !== "assistant";
  const chips =
    api !== "/api/ask" || waitingOnSend
      ? []
      : lastAssistant
        ? (followups.length ? followups : fallbackFollowups()).slice(0, 2)
        : (suggestions ?? []).slice(0, 2);
  const sheet = layout === "sheet";

  return (
    <div className={sheet ? "flex min-h-0 min-w-0 flex-1 flex-col gap-3 overflow-hidden" : "flex min-w-0 flex-col gap-3"}>
      <ol className={sheet ? "min-h-0 min-w-0 flex-1 space-y-3 overflow-y-auto" : "min-w-0 space-y-3"} aria-live="polite">
        {messages.length === 0 && emptyHint ? (
          <li className="text-sm text-muted-foreground">{emptyHint}</li>
        ) : null}
        {messages.map((message) => {
          const segments = segmentsInStreamOrder(message.id, message.parts);
          const lastSegment = segments.at(-1);
          const streamingMessage = busy && message.role === "assistant" && message.id === messages.at(-1)?.id;
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
                      <AgentBusy />
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
                    return (
                      <div key={segment.key} className="flex min-w-0 flex-col gap-0.5">
                        {segment.items.map((item, itemIndex) => {
                          const plot = getToolName(item.part) === "plot_series" ? plotFromPart(item.part) : null;
                          const lastTool = itemIndex === segment.items.length - 1;
                          return (
                            <div key={item.key} className="min-w-0 space-y-1">
                              <AgentTrace
                                part={item.part}
                                index={item.runIndex}
                                keepBusy={showPulse && lastTool}
                              />
                              {plot ? <AgentPlot spec={plot} /> : null}
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
            <AgentBusy />
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
          submit(input);
        }}
      >
        <Input
          value={input}
          onChange={(event) => setInput(event.currentTarget.value)}
          placeholder={placeholder}
          disabled={busy}
          autoComplete="off"
          aria-label={placeholder}
        />
        <Button type="submit" size="sm" disabled={busy || !input.trim()}>
          {busy ? <AgentBusy className="size-3.5" /> : <ArrowUp data-icon="inline-start" />}
          Enviar
        </Button>
      </form>
    </div>
  );
}
