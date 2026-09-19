"use client";

import { Maximize2, X } from "lucide-react";
import { useEffect, useId, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { Button } from "@/components/ui/button";
import { cn } from "cn";

/**
 * Same plot, two sizes. Inline keeps a maximize control; fullscreen is a portal
 * above the Pregunta sheet (z-50) so chat charts can grow too.
 */
export function PlotExpand({
  title,
  chrome = "overlay",
  className,
  children,
}: {
  title: string;
  chrome?: "overlay" | "row";
  className?: string;
  children: (frame: "inline" | "full") => ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const headingId = useId();

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const expand = (
    <Button
      type="button"
      variant="ghost"
      size="icon-sm"
      aria-label={`Pantalla completa: ${title}`}
      onMouseDown={(event) => event.stopPropagation()}
      onClick={() => setOpen(true)}
    >
      <Maximize2 />
    </Button>
  );

  return (
    <>
      <div className={cn("relative min-w-0", className)}>
        {chrome === "row" ? (
          <div className="mb-1 flex items-center gap-2">
            <p className="min-w-0 flex-1 text-xs font-medium">{title}</p>
            {expand}
          </div>
        ) : (
          <div className="absolute top-1 right-1 z-10 rounded-md bg-background/80 shadow-sm">
            {expand}
          </div>
        )}
        {children("inline")}
      </div>
      {open && mounted
        ? createPortal(
            <div
              role="dialog"
              aria-modal="true"
              aria-labelledby={headingId}
              className="fixed inset-0 z-[80] flex flex-col bg-background"
            >
              <div className="flex items-center gap-3 border-b px-4 py-3">
                <h2 id={headingId} className="min-w-0 flex-1 truncate text-sm font-medium">
                  {title}
                </h2>
                <Button type="button" variant="ghost" size="icon-sm" onClick={() => setOpen(false)} aria-label="Salir de pantalla completa">
                  <X />
                </Button>
              </div>
              <div className="min-h-0 flex-1 p-4">{children("full")}</div>
            </div>,
            document.body,
          )
        : null}
    </>
  );
}

export function plotHeight(frame: "inline" | "full", inline: string) {
  return frame === "full" ? "h-full min-h-0 w-full aspect-auto" : inline;
}
