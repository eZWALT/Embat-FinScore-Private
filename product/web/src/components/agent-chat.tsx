"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, getToolName, isToolUIPart } from "ai";
import { ArrowUp } from "lucide-react";
import { useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { AgentPlot } from "@/components/agent-plot";
import { AgentTrace } from "@/components/agent-trace";
import { contextBody, plotFromPart, type AgentContext } from "@/lib/agent/chat-parts";

export function AgentChat({
  api,
  companyId,
  groupId,
  asOf,
  placeholder,
  emptyHint,
  suggestions,
  layout = "page",
}: {
  api: "/api/ask" | "/api/watcher/reply";
  companyId?: string;
  groupId?: string;
  asOf?: string;
  placeholder: string;
  emptyHint?: string;
  suggestions?: string[];
  layout?: "page" | "sheet";
}) {
  const ctxRef = useRef<AgentContext>({ companyId, groupId, asOf });
  ctxRef.current = { companyId, groupId, asOf };

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
  const busy = status === "submitted" || status === "streaming";

  function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    void sendMessage({ text: trimmed }, { body: contextBody(ctxRef.current) });
    setInput("");
  }

  const showSuggestions = Boolean(suggestions?.length) && messages.length === 0 && !busy;

  const sheet = layout === "sheet";

  return (
    <div className={sheet ? "flex h-full min-h-0 flex-col gap-3" : "flex flex-col gap-3"}>
      <ol className={sheet ? "min-h-0 flex-1 space-y-3 overflow-y-auto" : "space-y-3"} aria-live="polite">
        {messages.length === 0 && emptyHint ? (
          <li className="text-sm text-muted-foreground">{emptyHint}</li>
        ) : null}
        {messages.map((message) => (
          <li
            key={message.id}
            className={
              message.role === "user"
                ? "ml-auto max-w-[85%] space-y-1.5 rounded-xl bg-muted px-4 py-3"
                : "space-y-1.5 rounded-xl border bg-background px-4 py-3"
            }
          >
            <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              {message.role === "user" ? "Tú" : "Centinela"}
            </p>
            {message.parts.map((part, index) => {
              if (message.role === "assistant" && isToolUIPart(part)) {
                const plot = getToolName(part) === "plot_series" ? plotFromPart(part) : null;
                return (
                  <div key={`${message.id}-tool-${part.toolCallId || index}`} className="space-y-2">
                    <AgentTrace part={part} />
                    {plot ? <AgentPlot spec={plot} /> : null}
                  </div>
                );
              }
              if (part.type === "text" && part.text) {
                return (
                  <p key={`${message.id}-t-${index}`} className="whitespace-pre-wrap text-sm leading-snug">
                    {part.text}
                  </p>
                );
              }
              return null;
            })}
          </li>
        ))}
        {busy && messages.at(-1)?.role !== "assistant" ? (
          <li className="text-xs text-muted-foreground">Consultando…</li>
        ) : null}
      </ol>

      {showSuggestions ? (
        <div className="flex flex-col gap-2">
          <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            Sugeridas
          </p>
          <div className="flex flex-col gap-2 sm:flex-row">
            {suggestions!.map((question) => (
              <button
                key={question}
                type="button"
                disabled={busy}
                onClick={() => submit(question)}
                className="rounded-lg border bg-background px-3 py-2 text-left text-sm leading-snug hover:bg-muted disabled:opacity-50"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      {error ? (
        <p className="text-sm text-destructive" role="alert">
          {error.message}
        </p>
      ) : null}

      <form
        className="flex items-center gap-2"
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
          <ArrowUp data-icon="inline-start" />
          Enviar
        </Button>
      </form>
    </div>
  );
}
