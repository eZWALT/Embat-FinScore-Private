import type { Metadata } from "next";

import { VideoLandingHero } from "@/components/landing/video-hero";

export const metadata: Metadata = {
  title: "Sentinel",
  description: "Vigilancia de la salud financiera. Un índice 0–100, razones en euros y alertas para tesorería.",
};

export default function Home() {
  return <VideoLandingHero />;
}
