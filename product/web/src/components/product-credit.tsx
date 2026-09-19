import { cn } from "cn";

/** Quiet mark. Not the method lecture. */
export function ProductCredit({ className }: { className?: string }) {
  return (
    <p className={cn("text-xs leading-5 text-muted-foreground", className)}>
      © 2026 Centinela · Walter J.T.V · Javier Boix · Rubén Godoy
    </p>
  );
}
