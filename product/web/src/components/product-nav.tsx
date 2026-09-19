import Link from "next/link";

const LINKS = [
  { href: "/empresa", label: "Empresa" },
  { href: "/grupos", label: "Grupos" },
] as const;

export function ProductNav({ current }: { current: string }) {
  return (
    <nav className="flex flex-wrap items-center gap-1 text-sm">
      {LINKS.map((link) => (
        <Link
          key={link.href}
          href={link.href}
          aria-current={link.href === current ? "page" : undefined}
          className={
            link.href === current
              ? "rounded-md bg-foreground px-2.5 py-1 font-medium text-background"
              : "rounded-md px-2.5 py-1 text-muted-foreground hover:bg-muted hover:text-foreground"
          }
        >
          {link.label}
        </Link>
      ))}
    </nav>
  );
}
