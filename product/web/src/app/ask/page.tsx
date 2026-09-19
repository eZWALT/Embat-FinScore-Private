import type { Metadata } from "next";

import { ProductNav } from "@/components/product-nav";

export const metadata: Metadata = {
  title: "Ask · Health Sentinel",
  description: "Questions about a company or group, from the bundle and clean records.",
};

export default function AskPage() {
  return (
    <div className="mx-auto flex min-h-svh max-w-2xl flex-col gap-6 px-4 py-6">
      <ProductNav current="/ask" />
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Ask</h1>
        <p className="text-sm text-muted-foreground">
          Chat with tools lands next. Prompts are already in <code>src/lib/agent/prompts/</code>.
        </p>
      </div>
    </div>
  );
}
