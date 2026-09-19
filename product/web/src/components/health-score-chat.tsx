"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { ArrowUp, MessageCircle, X } from "lucide-react";
import { useMemo, useState } from "react";

import { Bubble, BubbleContent } from "@/components/ui/bubble";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const busy = status === "submitted" || status === "streaming";

  function submit() {
    const text = input.trim();
    if (!text || busy) return;
    void sendMessage({ text });
    setInput("");
  }

  return (
    <div className="pointer-events-none fixed right-4 bottom-4 z-50 flex flex-col items-end gap-3 sm:right-6 sm:bottom-6">
      {open ? (
        <Card className="pointer-events-auto flex h-[min(82vh,720px)] w-[min(calc(100vw-2rem),36rem)] flex-col shadow-lg">
          <CardHeader className="border-b pb-3">
            <CardTitle>Preguntar</CardTitle>
            <CardDescription>Respuestas cortas. Sin datos de la empresa.</CardDescription>
            <CardAction>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                onClick={() => setOpen(false)}
                aria-label="Cerrar chat"
              >
                <X />
              </Button>
            </CardAction>
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
      ) : null}

      <Button
        type="button"
        size="icon-lg"
        className="pointer-events-auto size-12 rounded-full shadow-lg"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        aria-label={open ? "Cerrar chat" : "Abrir chat"}
      >
        {open ? <X /> : <MessageCircle />}
      </Button>
    </div>
  );
}
