"use client";

import { LandingVideo } from "@/components/landing/landing-video";
import { SentinelLanding } from "@/components/landing/sentinel-landing";

export function VideoLandingHero() {
  return (
    <SentinelLanding
      figure={<LandingVideo />}
      reflection={<LandingVideo />}
    />
  );
}
