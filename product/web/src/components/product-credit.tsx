import { cn } from "cn";

const AUTHORS = [
  { name: "Walter J.T.V", href: "https://ezwalt.github.io/" },
  { name: "Javier Boix", href: "https://jboixcampos.github.io" },
  { name: "Rubén Godoy", href: "https://github.com/rubengpr" },
];

/** Quiet mark. Not the method lecture. */
export function ProductCredit({ className }: { className?: string }) {
  return (
    <p className={cn("text-xs leading-5 text-muted-foreground", className)}>
      © 2026 Sentinel
      {AUTHORS.map((author) => (
        <span key={author.name}>
          {" · "}
          <a
            href={author.href}
            target="_blank"
            rel="noopener noreferrer"
            className="underline-offset-2 transition-colors hover:text-foreground hover:underline"
          >
            {author.name}
          </a>
        </span>
      ))}
    </p>
  );
}
