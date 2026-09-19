"use client";

import { MessageCircle, X } from "lucide-react";
import { useEffect, useState } from "react";

import { AgentChat } from "@/components/agent-chat";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { formatMonth } from "@/lib/format-month";

const COMPANY_QUESTIONS = [
  "¿Por qué este índice este mes?",
  "¿Qué cambió y quién tiene que actuar?",
  "¿Qué alertas hay y qué importe hay detrás?",
] as const;

const INDEX_QUESTIONS = [
  "¿Qué empresas necesitan atención este mes?",
  "¿Quién se alejó de su propia normalidad?",
] as const;

export function HealthScoreChat({
  companyId,
  groupId,
  asOf,
  open,
  onOpenChange,
  defaultOpen = false,
  seedPrompt,
  seedKey,
  suggestions,
  description,
}: {
  companyId?: string;
  groupId?: string;
  asOf?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  defaultOpen?: boolean;
  seedPrompt?: string;
  seedKey?: number;
  suggestions?: readonly string[];
  description?: string;
}) {
  const [internal, setInternal] = useState(defaultOpen);
  const isOpen = open ?? internal;
  const setOpen = onOpenChange ?? setInternal;

  useEffect(() => {
    if (defaultOpen) setOpen(true);
  }, [defaultOpen, setOpen]);

  useEffect(() => {
    if (seedPrompt) setOpen(true);
  }, [seedPrompt, setOpen]);

  const chips = suggestions
    ? [...suggestions]
    : companyId
      ? [...COMPANY_QUESTIONS]
      : [...INDEX_QUESTIONS];
  const when = asOf ? formatMonth(asOf) : null;
  const subtitle =
    description ??
    (companyId
      ? `Por qué este índice${when ? ` en ${when}` : ""}, qué ha cambiado y qué hay que revisar.`
      : "Qué empresas se han alejado de su normalidad.");

  return (
    <div className="pointer-events-none fixed right-4 bottom-4 z-50 flex flex-col items-end gap-3 sm:right-6 sm:bottom-6">
      {isOpen ? (
        <Card className="pointer-events-auto flex h-[min(82vh,720px)] w-[min(calc(100vw-2rem),36rem)] flex-col shadow-lg">
          <CardHeader className="border-b pb-3">
            <CardTitle>Pregunta</CardTitle>
            <CardDescription>{subtitle}</CardDescription>
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
          <CardContent className="flex min-h-0 flex-1 flex-col pt-4">
            <AgentChat
              key={`${companyId ?? ""}:${groupId ?? ""}:${asOf ?? ""}`}
              api="/api/ask"
              companyId={companyId}
              groupId={groupId}
              asOf={asOf}
              layout="sheet"
              seedPrompt={seedPrompt}
              seedKey={seedKey}
              placeholder={
                companyId
                  ? "Ej. ¿Por qué bajó el índice este mes?"
                  : "Ej. ¿Quién necesita atención este mes?"
              }
              suggestions={chips}
            />
          </CardContent>
        </Card>
      ) : null}

      <Button
        type="button"
        size="icon-lg"
        className="pointer-events-auto size-12 rounded-full shadow-lg"
        onClick={() => setOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-label={isOpen ? "Cerrar chat" : "Abrir chat"}
      >
        {isOpen ? <X /> : <MessageCircle />}
      </Button>
    </div>
  );
}
