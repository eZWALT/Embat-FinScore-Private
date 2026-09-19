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
