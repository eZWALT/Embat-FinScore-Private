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

const COMPANY_QUESTIONS = [
  "¿Por qué este índice este mes?",
  "¿Qué cambió en los últimos tres meses y quién actúa?",
] as const;

const INDEX_QUESTIONS = ["¿Qué empresas necesitan atención este mes?"] as const;

export function HealthScoreChat({
  companyId,
  groupId,
  asOf,
  open,
  onOpenChange,
  defaultOpen = false,
}: {
  companyId?: string;
  groupId?: string;
  asOf?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  defaultOpen?: boolean;
}) {
  const [internal, setInternal] = useState(defaultOpen);
  const isOpen = open ?? internal;
  const setOpen = onOpenChange ?? setInternal;

  useEffect(() => {
    if (defaultOpen) setOpen(true);
  }, [defaultOpen, setOpen]);

  const suggestions = companyId ? [...COMPANY_QUESTIONS] : [...INDEX_QUESTIONS];
  const hint = companyId
    ? `Sobre ${companyId}${groupId ? ` · ${groupId}` : ""}. Índice y alertas de Neon; registros si core está montado.`
    : "Pregunta por el índice o nombra una empresa (COMP_xxxx).";

  return (
    <div className="pointer-events-none fixed right-4 bottom-4 z-50 flex flex-col items-end gap-3 sm:right-6 sm:bottom-6">
      {isOpen ? (
        <Card className="pointer-events-auto flex h-[min(82vh,720px)] w-[min(calc(100vw-2rem),36rem)] flex-col shadow-lg">
          <CardHeader className="border-b pb-3">
            <CardTitle>Consultas</CardTitle>
            <CardDescription>
              {companyId ? `${companyId} · con herramientas` : "Índice de salud · con herramientas"}
            </CardDescription>
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
              placeholder="Pregunta por esta empresa o el índice"
              emptyHint={hint}
              suggestions={suggestions}
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
        aria-label={isOpen ? "Cerrar chat" : "Abrir consultas"}
      >
        {isOpen ? <X /> : <MessageCircle />}
      </Button>
    </div>
  );
}
