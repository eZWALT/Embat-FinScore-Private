"use client";

import { useEffect, useRef } from "react";

const SRC = "/landing/sentinel";

export function LandingVideo() {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      video.pause();
      video.currentTime = 0;
    }
  }, []);

  return (
    <video
      ref={ref}
      className="h-full w-full object-contain"
      autoPlay
      muted
      loop
      playsInline
      preload="metadata"
      aria-hidden="true"
    >
      <source src={`${SRC}.webm`} type="video/webm" />
      <source src={`${SRC}.mp4`} type="video/mp4" />
    </video>
  );
}
