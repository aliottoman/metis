"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { searchCustomerRecords } from "@/lib/api";
import type { CustomerSearchHit } from "@/lib/types";

/**
 * Search across every account at once.
 *
 * The account list filter answers "which customer is this?"; it cannot answer
 * "who was it that asked about Cohere reranking?" — the question you actually
 * have once a book of business runs past a hundred accounts and the knowledge
 * lives in notes, facts, wins and captured sources rather than in names. This
 * asks the host, which is the only place that can see all of it at once.
 */

const KIND_LABEL: Record<CustomerSearchHit["kind"], string> = {
  account: "Account",
  note: "Note",
  win: "Win",
  fact: "Fact",
  action: "Action",
  source: "Captured note",
};

const DEBOUNCE_MS = 160;

type CustomerSearchProps = {
  onOpen: (hit: CustomerSearchHit) => void;
  onDismiss: () => void;
};

export function CustomerSearch({ onOpen, onDismiss }: CustomerSearchProps) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<CustomerSearchHit[]>([]);
  const [truncated, setTruncated] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  // Only the newest query may write results: a slow early request must not
  // overwrite the answer to what the user is typing now.
  const generationRef = useRef(0);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    const needle = query.trim();
    const generation = ++generationRef.current;
    if (needle.length < 2) {
      setHits([]);
      setTruncated(false);
      setSearching(false);
      return;
    }
    setSearching(true);
    const timer = window.setTimeout(() => {
      void searchCustomerRecords(needle)
        .then((result) => {
          if (generation !== generationRef.current) return;
          setHits(result.hits);
          setTruncated(result.truncated);
          setActiveIndex(0);
          setError(null);
        })
        .catch((searchError) => {
          if (generation !== generationRef.current) return;
          setError(searchError instanceof Error ? searchError.message : "The search could not be run.");
          setHits([]);
        })
        .finally(() => {
          if (generation === generationRef.current) setSearching(false);
        });
    }, DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    listRef.current
      ?.querySelector(`[data-index="${activeIndex}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [activeIndex, hits]);

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onDismiss();
        return;
      }
      if (!hits.length) return;
      if (event.key === "ArrowDown") {
        event.preventDefault();
        setActiveIndex((current) => (current + 1) % hits.length);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        setActiveIndex((current) => (current - 1 + hits.length) % hits.length);
      } else if (event.key === "Enter") {
        event.preventDefault();
        const hit = hits[activeIndex];
        if (hit) onOpen(hit);
      }
    },
    [activeIndex, hits, onDismiss, onOpen],
  );

  const needle = query.trim();

  return (
    <div
      className="customerModalBackdrop"
      role="presentation"
      onMouseDown={(event) => { if (event.target === event.currentTarget) onDismiss(); }}
    >
      <section
        className="customerSearchPanel"
        role="dialog"
        aria-modal="true"
        aria-label="Search all customer records"
        onKeyDown={onKeyDown}
      >
        <header>
          <span aria-hidden="true">⌕</span>
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search accounts, notes, facts, actions, wins…"
            aria-label="Search all customer records"
            spellCheck={false}
            autoComplete="off"
          />
          <button type="button" onClick={onDismiss} aria-label="Close search">Esc</button>
        </header>

        <div className="customerSearchResults" ref={listRef}>
          {error ? <p className="customerSearchNote" role="alert">{error}</p> : null}
          {!error && needle.length < 2 ? (
            <p className="customerSearchNote">Type at least two characters. Every account, note, fact, action, win, and captured note is searched.</p>
          ) : null}
          {!error && needle.length >= 2 && !hits.length && !searching ? (
            <p className="customerSearchNote">Nothing anywhere mentions &ldquo;{needle}&rdquo;.</p>
          ) : null}
          {hits.map((hit, index) => (
            <button
              key={`${hit.kind}-${hit.id}`}
              type="button"
              data-index={index}
              className={index === activeIndex ? "isActive" : ""}
              onPointerEnter={() => setActiveIndex(index)}
              onClick={() => onOpen(hit)}
            >
              <i>{KIND_LABEL[hit.kind]}</i>
              <span>
                <strong>{hit.title}</strong>
                {hit.snippet ? <small>{hit.snippet}</small> : null}
              </span>
              <b>{hit.account_name}</b>
            </button>
          ))}
        </div>

        <footer>
          <span>
            {searching
              ? "Searching…"
              : hits.length
                ? `${hits.length} result${hits.length === 1 ? "" : "s"}${truncated ? " (first page — narrow the search for more)" : ""}`
                : "↑↓ to move · Enter to open"}
          </span>
        </footer>
      </section>
    </div>
  );
}
