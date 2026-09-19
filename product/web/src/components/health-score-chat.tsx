"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { ArrowUp } from "lucide-react";
import { useMemo, useState } from "react";

import { Bubble, BubbleContent } from "@/components/ui/bubble";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Message, MessageContent } from "@/components/ui/message";

function messageText(parts: { type: string; text?: string }[]): string {
  return parts
    .filter((part) => part.type === "text" && part.text)
    .map((part) => part.text)
    .join("");
}

export function HealthScoreChat() {
  const transport = useMemo(() => new DefaultChatTransport({ api: "/api/chat" }), []);
  const { messages, sendMessage, status, error } = useChat({ transport });
  const [input, setInput] = useState("");
  const busy = status === "submitted" || status === "streaming";

  function submit() {
    const text = input.trim();
    if (!text || busy) return;
    void sendMessage({ text });
    setInput("");
  }

  return (
    <Card className="flex h-[min(70vh,560px)] flex-col xl:sticky xl:top-20">
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Preguntar</CardTitle>
        <p className="text-sm text-muted-foreground">Respuestas cortas. Sin datos de la empresa en este chat.</p>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 overflow-y-auto">
        <div className="flex flex-col gap-3" aria-live="polite">
          {messages.length === 0 ? (
            <p className="text-sm text-muted-foreground">Pregunta sobre el índice; las respuestas son breves.</p>
          ) : null}
          {messages.map((message) => {
            const fromUser = message.role === "user";
            const text = messageText(message.parts);
            if (!text && !fromUser && busy) return null;
            return (
              <Message key={message.id} align={fromUser ? "end" : "start"}>
                <MessageContent>
                  <Bubble variant={fromUser ? "default" : "muted"} align={fromUser ? "end" : "start"}>
                    <BubbleContent>{text}</BubbleContent>
                  </Bubble>
                </MessageContent>
              </Message>
            );
          })}
          {busy && messages.at(-1)?.role !== "assistant" ? (
            <p className="text-xs text-muted-foreground">Escribiendo…</p>
          ) : null}
          {error ? (
            <p className="text-sm text-destructive" role="alert">
              {error.message}
            </p>
          ) : null}
        </div>
      </CardContent>
      <CardFooter>
        <form
          className="flex w-full items-center gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            submit();
          }}
        >
          <Input
            value={input}
            onChange={(event) => setInput(event.currentTarget.value)}
            placeholder="Escribe una pregunta"
            disabled={busy}
            autoComplete="off"
            aria-label="Mensaje"
          />
          <Button type="submit" size="icon" disabled={busy || !input.trim()} aria-label="Enviar">
            <ArrowUp />
          </Button>
        </form>
      </CardFooter>
    </Card>
  );
}
