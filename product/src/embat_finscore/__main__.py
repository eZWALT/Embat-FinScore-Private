"""CLI entry for the product scaffold."""

from embat_finscore import __version__


def main() -> None:
    print("Embat FinScore")
    print(f"version {__version__}")
    print("HackSpain 2026 — Embat Track")
    print("Scaffold only. No scoring model is implemented yet.")


if __name__ == "__main__":
    main()
