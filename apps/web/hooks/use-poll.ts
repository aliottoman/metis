"use client";

// The one way to poll: runs while `active` and the tab is visible, and stops
// the moment it is not. A hidden window polls nothing; coming back reads once
// straight away so the page is current before the next tick.

import { useEffect, useRef } from "react";

export function usePoll(tick: () => void | Promise<unknown>, ms: number, active = true): void {
  const latest = useRef(tick);
  useEffect(() => {
    latest.current = tick;
  }, [tick]);

  useEffect(() => {
    if (!active) return;
    let timer = 0;
    const stop = () => {
      if (timer) window.clearInterval(timer);
      timer = 0;
    };
    const start = () => {
      stop();
      timer = window.setInterval(() => void latest.current(), ms);
    };
    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        void latest.current();
        start();
      } else {
        stop();
      }
    };
    if (document.visibilityState === "visible") start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      stop();
    };
  }, [active, ms]);
}
