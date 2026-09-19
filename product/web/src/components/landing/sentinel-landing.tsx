"use client";

import { Bell, CircleDollarSign, Gauge } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useEffect, type MouseEvent, type ReactNode } from "react";

import { LandingTeam } from "@/components/landing/landing-team";

const POINTS = [
  {
    icon: Gauge,
    title: "Health Score",
    text: "Un 0–100 que pone a cada cliente en el mapa y te dice a quién cuidar primero.",
  },
  {
    icon: CircleDollarSign,
    title: "Explicabilidad",
    text: "Una razón y un importe que convierten cada número en una conversación comercial.",
  },
  {
    icon: Bell,
    title: "Alertas",
    text: "Una señal que llega con dueño, siguiente paso y la oportunidad de actuar a tiempo.",
  },
] as const;

function Corner({ className }: { className: string }) {
  return <span aria-hidden className={`pointer-events-none absolute h-7 w-7 border-neutral-300 ${className}`} />;
}

function goToScreen(screen: "hero" | "team", event?: MouseEvent<HTMLAnchorElement>) {
  event?.preventDefault();
  const nextHash = screen === "team" ? "#team" : "#hero";
  document.documentElement.dataset.landing = screen;
  if (window.location.hash !== nextHash) {
    history.pushState({ landing: screen }, "", nextHash);
  }
}

function syncLandingFromHash() {
  document.documentElement.dataset.landing = window.location.hash === "#team" ? "team" : "hero";
}

export function SentinelLanding({
  figure,
  reflection,
}: {
  figure: ReactNode;
  reflection?: ReactNode;
}) {
  useEffect(() => {
    syncLandingFromHash();
    window.addEventListener("popstate", syncLandingFromHash);
    window.addEventListener("hashchange", syncLandingFromHash);
    return () => {
      window.removeEventListener("popstate", syncLandingFromHash);
      window.removeEventListener("hashchange", syncLandingFromHash);
      delete document.documentElement.dataset.landing;
    };
  }, []);

  return (
    <div className="landing-root relative h-dvh overflow-hidden bg-white text-black">
      <header className="absolute inset-x-0 top-0 z-30">
        <nav className="flex h-16 items-center justify-between px-8 sm:px-12 lg:px-16 xl:px-20">
          <a href="#hero" className="flex items-center" aria-label="Sentinel" onClick={(event) => goToScreen("hero", event)}>
            <Image
              src="/landing/sentinel-logo.jpg"
              alt="Sentinel"
              width={1024}
              height={341}
              className="h-10 w-auto sm:h-11"
              priority
            />
          </a>
          <ul className="flex items-center gap-6">
            <li>
              <a
                href="#team"
                className="landing-team-nav text-sm text-neutral-400 transition-colors hover:text-black"
                onClick={(event) => goToScreen("team", event)}
              >
                Team
              </a>
            </li>
            <li>
              <Link href="/empresa" className="inline-flex items-center bg-black px-3.5 py-2 text-sm text-white">
                Demo
              </Link>
            </li>
          </ul>
        </nav>
      </header>

      <div id="hero" className="landing-hero absolute inset-0 flex flex-col pt-16">
        <main className="grid min-h-0 flex-1 grid-cols-1 grid-rows-[auto_minmax(0,1fr)] lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:grid-rows-1">
          <section className="flex min-h-0 min-w-0 flex-col justify-center px-8 py-10 sm:px-12 lg:px-16 xl:px-20 lg:py-0">
            <div className="pl-6 sm:pl-8 lg:pl-10">
              <h1 className="font-heading text-[clamp(2.75rem,6.2vw,5.25rem)] font-semibold leading-[0.95] tracking-tight text-black">
                Sentinel
              </h1>
              <p className="mt-6 max-w-md text-[clamp(0.95rem,1.35vw,1.125rem)] leading-relaxed text-neutral-600">
                Construimos un sistema de vigilancia financiera que convierte el rastro de tesorería en una
                cartera que se gestiona proactivamente.
              </p>
              <ul className="mt-8 flex max-w-md flex-col gap-4">
                {POINTS.map((point) => (
                  <li key={point.title} className="flex gap-2.5">
                    <point.icon className="mt-0.5 size-4 shrink-0 text-neutral-400" strokeWidth={1.5} aria-hidden />
                    <div>
                      <p className="text-sm font-medium tracking-tight text-black">{point.title}</p>
                      <p className="mt-0.5 text-sm leading-snug text-neutral-500">{point.text}</p>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </section>

          <section className="flex min-h-0 min-w-0 items-center justify-center overflow-hidden px-8 py-8 sm:px-12 lg:pr-16 xl:pr-20">
            <figure className="flex w-full max-w-[min(100%,34rem)] flex-col">
              <div className="relative overflow-hidden p-4">
                <Corner className="top-0 left-0 border-t border-l" />
                <Corner className="top-0 right-0 border-t border-r" />
                <Corner className="bottom-0 left-0 border-b border-l" />
                <Corner className="right-0 bottom-0 border-b border-r" />
                <div className="flex aspect-square items-center justify-center">{figure}</div>
              </div>
              <figcaption className="mt-2 flex items-center justify-between border-t border-neutral-200 pt-2 text-[10px] tracking-[0.2em] text-neutral-400 uppercase">
                <span>Fig. 01.A</span>
                <span>Health Score 0–100</span>
              </figcaption>
              {reflection ? (
                <div
                  className="pointer-events-none mt-1 h-14 overflow-hidden opacity-[0.2] [mask-image:linear-gradient(to_bottom,black,transparent_75%)]"
                  aria-hidden="true"
                >
                  <div className="origin-top scale-y-[-1]">{reflection}</div>
                </div>
              ) : null}
            </figure>
          </section>
        </main>
      </div>

      <div id="team" className="landing-team-screen absolute inset-0 flex flex-col pt-16">
        <LandingTeam />
      </div>
    </div>
  );
}
