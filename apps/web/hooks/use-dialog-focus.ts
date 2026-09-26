"use client";

import { useEffect, useRef, type RefObject } from "react";

// Modal surfaces can be nested (for example, a picker opened inside a sheet).
// Only the most recently opened surface owns keyboard focus and Escape.
const layers: HTMLElement[] = [];
const FOCUSABLE = "a[href], area[href], button, input, textarea, select, iframe, [tabindex], [contenteditable='true'], audio[controls], video[controls], summary";

function focusableWithin(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
    (element) => element.tabIndex >= 0
      && !element.matches(":disabled")
      && !element.closest("[hidden], [inert], [aria-hidden='true']")
      && element.getClientRects().length > 0,
  );
}

/** Keep focus in an open modal, then return it to the control that opened it. */
export function useDialogFocus<T extends HTMLElement>(
  ref: RefObject<T | null>,
  open: boolean,
  onClose?: () => void,
): void {
  const close = useRef(onClose);
  useEffect(() => { close.current = onClose; }, [onClose]);

  useEffect(() => {
    const dialog = ref.current;
    if (!open || !dialog) return;
    const document = dialog.ownerDocument;
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousTabIndex = dialog.getAttribute("tabindex");
    if (previousTabIndex === null) dialog.tabIndex = -1;
    layers.push(dialog);

    const isTopmost = () => layers[layers.length - 1] === dialog;
    const focusFirst = () => {
      const options = focusableWithin(dialog);
      const preferred = options.find((element) => element.matches("[autofocus], [data-dialog-autofocus]"));
      (preferred ?? options[0] ?? dialog).focus({ preventScroll: true });
    };
    // A trigger inside the surface can disappear when it opens (the mobile
    // conversation rail does this). Containment alone does not mean it can
    // retain focus; move into a visible control in that case too.
    if (!focusableWithin(dialog).includes(document.activeElement as HTMLElement)) focusFirst();

    const onKeyDown = (event: KeyboardEvent) => {
      if (!isTopmost() || event.defaultPrevented) return;
      if (event.key === "Escape" && close.current) {
        event.preventDefault();
        event.stopPropagation();
        close.current();
        return;
      }
      if (event.key !== "Tab") return;
      const options = focusableWithin(dialog);
      const first = options[0];
      const last = options[options.length - 1];
      if (!first || !last) {
        event.preventDefault();
        dialog.focus({ preventScroll: true });
      } else if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) {
        event.preventDefault();
        last.focus({ preventScroll: true });
      } else if (!event.shiftKey && (document.activeElement === last || !dialog.contains(document.activeElement))) {
        event.preventDefault();
        first.focus({ preventScroll: true });
      } else if (document.activeElement === dialog) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus({ preventScroll: true });
      }
    };
    const onFocusIn = (event: FocusEvent) => {
      if (isTopmost() && event.target instanceof Node && !dialog.contains(event.target)) focusFirst();
    };
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("focusin", onFocusIn);

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("focusin", onFocusIn);
      const wasTopmost = isTopmost();
      const index = layers.lastIndexOf(dialog);
      if (index >= 0) layers.splice(index, 1);
      if (previousTabIndex === null) dialog.removeAttribute("tabindex");
      const currentLayer = layers[layers.length - 1];
      if (wasTopmost && previous?.isConnected && (!currentLayer || currentLayer.contains(previous))) {
        previous.focus({ preventScroll: true });
      }
    };
  }, [open, ref]);
}
