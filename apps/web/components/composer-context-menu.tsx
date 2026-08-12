"use client";

import { useEffect, useMemo, useRef } from "react";

export type ComposerContextTarget = "customer" | "project";

export function ComposerContextMenu({
  customerCount,
  projectCount,
  query,
  onSelect,
  onDismiss,
}: {
  customerCount: number;
  projectCount: number;
  query: string;
  onSelect: (target: ComposerContextTarget) => void;
  onDismiss: () => void;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const options = useMemo(
    () => [
      {
        id: "customer" as const,
        glyph: "◎",
        label: "Customer context",
        meta: "Use one account’s reviewed facts, people, notes, and open work.",
        count: customerCount,
      },
      {
        id: "project" as const,
        glyph: "◇",
        label: "Project workspace",
        meta: "Open a local codebase for mapped search and approval-gated edits.",
        count: projectCount,
      },
    ].filter((option) => !query || option.id.startsWith(query.toLowerCase())),
    [customerCount, projectCount, query],
  );

  useEffect(() => {
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Element | null;
      if (target?.closest("#composer-context-trigger")) return;
      if (!rootRef.current?.contains(event.target as Node)) onDismiss();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onDismiss();
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key.toLowerCase() === "c") {
        event.preventDefault();
        onSelect("customer");
      }
      if (event.key.toLowerCase() === "p") {
        event.preventDefault();
        onSelect("project");
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [onDismiss, onSelect]);

  return (
    <div className="composerContextMenu" ref={rootRef} role="menu" aria-label="Add context">
      <header>
        <span className="eyebrow">Add context</span>
        <strong>What is this message about?</strong>
        <p>Choose a clear boundary. You can remove it before sending.</p>
      </header>
      <div className="composerContextChoices">
        {options.map((option) => (
          <button
            key={option.id}
            type="button"
            role="menuitem"
            data-context-target={option.id}
            onClick={() => onSelect(option.id)}
          >
            <i aria-hidden="true">{option.glyph}</i>
            <span>
              <strong>{option.label}</strong>
              <small>{option.meta}</small>
            </span>
            <b>{option.count}</b>
            <em aria-hidden="true">→</em>
          </button>
        ))}
        {!options.length ? <p className="composerContextEmpty">No context type matches “{query}”.</p> : null}
      </div>
      <footer>Press C for customer · P for project · Esc to close</footer>
    </div>
  );
}
