"use client";

// Route changes as one motion instead of a cut: the rail stays where it is
// and the page crossfades and rises. Built on document.startViewTransition,
// so a browser without it (or a reader who asked for less motion) gets the
// plain navigation it always had.

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useLayoutEffect } from "react";

// Resolves the in-flight transition once the new route has committed.
let settle: (() => void) | null = null;

// A route that never changes must not hold the page frozen.
const SETTLE_TIMEOUT_MS = 800;

function supported(): boolean {
  return typeof document !== "undefined"
    && typeof document.startViewTransition === "function"
    && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function internalTarget(event: MouseEvent): URL | null {
  if (event.defaultPrevented || event.button !== 0) return null;
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return null;
  const anchor = (event.target as Element | null)?.closest?.("a[href]");
  if (!(anchor instanceof HTMLAnchorElement)) return null;
  if (anchor.target && anchor.target !== "_self") return null;
  if (anchor.hasAttribute("download") || anchor.origin !== window.location.origin) return null;
  const url = new URL(anchor.href);
  // Same page: a search or hash change; leave it to the router.
  if (url.pathname === window.location.pathname) return null;
  return url;
}

/** Wrap a client navigation in a view transition. */
export function transitionTo(navigate: () => void): void {
  if (!supported()) {
    navigate();
    return;
  }
  const root = document.documentElement;
  root.classList.add("is-routing");
  const transition = document.startViewTransition(
    () =>
      new Promise<void>((resolve) => {
        settle = resolve;
        navigate();
        window.setTimeout(resolve, SETTLE_TIMEOUT_MS);
      }),
  );
  void transition.finished.finally(() => root.classList.remove("is-routing"));
}

/**
 * Mount once in the shell. Every internal link on the page then navigates
 * through a view transition; the router's own prefetching is untouched.
 */
export function useRouteTransitions(): void {
  const router = useRouter();
  const pathname = usePathname();

  // The new route has committed: let the transition capture it.
  useLayoutEffect(() => {
    settle?.();
    settle = null;
  }, [pathname]);

  useEffect(() => {
    const onClick = (event: MouseEvent) => {
      if (!supported()) return;
      const url = internalTarget(event);
      if (!url) return;
      event.preventDefault();
      transitionTo(() => router.push(url.pathname + url.search + url.hash));
    };
    document.addEventListener("click", onClick, true);
    return () => document.removeEventListener("click", onClick, true);
  }, [router]);
}
