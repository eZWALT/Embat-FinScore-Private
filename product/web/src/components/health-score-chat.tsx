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
import type { DashboardView } from "@/lib/agent/view-context";

const QUESTIONS = [
  "¿Qué muestra este gráfico?",
  "¿Qué cambió y quién tiene que actuar?",
  "¿Hay alguna alerta que revisar?",
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
  view,
}: {
  companyId?: string;
  groupId?: string;
  asOf?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  defaultOpen?: boolean;
  seedPrompt?: string;
  seedKey?: number;
  view?: DashboardView;
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

  return (
    <div className="pointer-events-none fixed right-4 bottom-4 z-50 flex flex-col items-end gap-3 sm:right-6 sm:bottom-6">
      {isOpen ? (
        <Card className="pointer-events-auto flex h-[min(82vh,720px)] w-[min(calc(100vw-2rem),36rem)] flex-col shadow-lg">
          <CardHeader className="border-b pb-3">
            <CardTitle>Pregunta</CardTitle>
            <CardDescription>Sobre los paneles, las tendencias o un periodo.</CardDescription>
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
              view={view}
              layout="sheet"
              seedPrompt={seedPrompt}
              seedKey={seedKey}
              placeholder="Ej. ¿Qué tendencia ves en este gráfico?"
              suggestions={[...QUESTIONS]}
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
