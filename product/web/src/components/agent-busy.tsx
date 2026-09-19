import { LoaderCircle } from "lucide-react";

import { cn } from "cn";

/** Quiet ring. Same weight as the tool-row icons; no rotating copy. */
export function AgentBusy({ className, label = "Cargando" }: { className?: string; label?: string }) {
  return (
    <span role="status" aria-label={label} className="inline-flex shrink-0">
      <LoaderCircle
        className={cn(
          "size-3 animate-spin text-muted-foreground motion-reduce:animate-none motion-reduce:opacity-70",
          className,
        )}
        aria-hidden="true"
      />
    </span>
  );
}

/** Three-dot step under the tool block. Goes away when the first reply token arrives. */
export function AgentPulse({ className, divided = true }: { className?: string; divided?: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "flex items-center gap-1.5",
        divided ? "mt-2 border-t border-border pt-2" : "mt-1.5",
        className,
      )}
    >
      <span className="agent-dot agent-dot-1" />
      <span className="agent-dot agent-dot-2" />
      <span className="agent-dot agent-dot-3" />
    </span>
  );
}
