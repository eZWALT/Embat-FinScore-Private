"use client";

import { ChevronDown, Maximize2, MessageCircle, Minimize2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { cn } from "cn";

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
import { openingSuggestions } from "@/lib/agent/suggestions";
import type { DashboardView } from "@/lib/agent/view-context";

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
  const [kept, setKept] = useState(isOpen);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (isOpen) setKept(true);
  }, [isOpen]);

  useEffect(() => {
    if (defaultOpen) setOpen(true);
  }, [defaultOpen, setOpen]);

  useEffect(() => {
    if (seedPrompt) setOpen(true);
  }, [seedPrompt, setOpen]);

  const warmed = useRef(false);
  useEffect(() => {
    if (!isOpen || warmed.current) return;
    warmed.current = true;
    void fetch("/api/ask/warmup", { method: "POST", keepalive: true }).catch(() => {});
  }, [isOpen]);

  return (
    <div className="pointer-events-none fixed right-4 bottom-4 z-50 flex flex-col items-end gap-3 sm:right-6 sm:bottom-6">
      {kept ? (
        <Card
          className={cn(
            "pointer-events-auto flex flex-col shadow-lg",
            expanded
              ? "h-[min(96vh,calc(100dvh-5.5rem))] w-[min(calc(100vw-2rem),72rem)]"
              : "h-[min(74vh,620px)] w-[min(calc(100vw-2rem),30rem)]",
            isOpen ? "flex" : "hidden",
          )}
        >
          <CardHeader className="border-b pb-3">
            <CardTitle>Pregunta</CardTitle>
            <CardDescription>Sobre los paneles, las tendencias o un periodo.</CardDescription>
            <CardAction className="flex items-center gap-0.5">
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                onClick={() => setExpanded((value) => !value)}
                aria-pressed={expanded}
                aria-label={expanded ? "Reducir chat" : "Ampliar chat"}
              >
                {expanded ? <Minimize2 /> : <Maximize2 />}
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                onClick={() => setOpen(false)}
                aria-label="Minimizar chat"
              >
                <ChevronDown />
              </Button>
            </CardAction>
          </CardHeader>
          <CardContent className="flex min-h-0 flex-1 flex-col overflow-hidden pt-4">
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
              suggestions={openingSuggestions(view)}
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
        aria-label={isOpen ? "Minimizar chat" : "Abrir chat"}
      >
        {isOpen ? <Minimize2 /> : <MessageCircle />}
      </Button>
    </div>
  );
}
