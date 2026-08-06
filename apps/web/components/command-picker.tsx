"use client";

import { ReactNode, useCallback, useEffect, useId, useMemo, useRef, useState } from "react";

/**
 * A search-first picker for choosing one record out of many.
 *
 * A dropdown is a fine control for six options and the wrong one for a hundred:
 * the customer list runs past a hundred accounts and the project catalog grows
 * with the disk, so scanning is hopeless and the only usable interaction is to
 * type the first few letters of what you already have in mind. This opens with
 * the cursor in a filter box, ranks matches instead of preserving alphabetical
 * accident, and keeps everything reachable from the keyboard.
 *
 * The caller owns the trigger button and the open state, and positions this
 * panel; that keeps one component serving triggers that look nothing alike.
 */

export type PickerOption = {
  id: string;
  label: string;
  /** Secondary line under the label. */
  meta?: string;
  /** Right-aligned chip, e.g. a win count or "Active". */
  badge?: string;
  /** Leading glyph. */
  glyph?: string;
  /** Extra terms that should match — aliases, frameworks, regions. */
  keywords?: string[];
  /** Section heading, emitted in first-seen order when no query is typed. */
  group?: string;
  disabled?: boolean;
};

type CommandPickerProps = {
  /** Accessible name for the list. */
  label: string;
  options: PickerOption[];
  value: string | null;
  onSelect: (id: string) => void;
  onDismiss: () => void;
  placeholder?: string;
  emptyMessage?: string;
  /** A pinned first row for "none of them" — clearing the scope, say. */
  clearOption?: { id: string; label: string; meta?: string };
  /** Rendered above the list: context or a switch that qualifies the choice. */
  header?: ReactNode;
  footer?: ReactNode;
  busy?: boolean;
  /** When set, a typed name that matches no option's label exactly grows a
   *  final "Create …" row; choosing it calls back with the trimmed query.
   *  The picker's search box is the name field — no second input to manage. */
  onCreate?: (name: string) => void;
  createMeta?: string;
};

const CREATE_ROW_ID = "__commandPickerCreate__";

/** Ranks a match so an exact prefix beats a word start, which beats a mention
 *  buried mid-string. Returns -1 when the option does not match at all. */
function score(option: PickerOption, needle: string): number {
  if (!needle) return 0;
  const haystacks = [option.label, ...(option.keywords ?? []), option.meta ?? ""];
  let best = -1;
  for (const [index, raw] of haystacks.entries()) {
    const value = raw.toLowerCase();
    const at = value.indexOf(needle);
    if (at < 0) continue;
    // The label is worth more than an alias, and an alias more than the meta.
    const field = index === 0 ? 0 : index < haystacks.length - 1 ? 1 : 2;
    const position = at === 0 ? 0 : /\s|[-/·,.]/.test(value[at - 1] ?? "") ? 1 : 2;
    const rank = 100 - (position * 10 + field * 3);
    if (rank > best) best = rank;
  }
  return best;
}

export function CommandPicker({
  label,
  options,
  value,
  onSelect,
  onDismiss,
  placeholder = "Search…",
  emptyMessage = "Nothing matches",
  clearOption,
  header,
  footer,
  busy = false,
  onCreate,
  createMeta,
}: CommandPickerProps) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const needle = query.trim().toLowerCase();

  const visible = useMemo(() => {
    const rows: PickerOption[] = clearOption
      ? [{ id: clearOption.id, label: clearOption.label, meta: clearOption.meta }, ...options]
      : options;
    const ranked = !needle
      ? rows
      : rows
          .map((option, index) => ({ option, index, rank: score(option, needle) }))
          .filter((item) => item.rank >= 0)
          // Stable: equal ranks keep the order the caller chose (usually recency).
          .sort((a, b) => b.rank - a.rank || a.index - b.index)
          .map((item) => item.option);
    // The create row rides last, after every real match: a typo one letter
    // away from an existing record should surface the record first, not a
    // near-duplicate folder. An exact label match removes it entirely.
    const name = query.trim();
    if (
      onCreate &&
      name &&
      !options.some((option) => option.label.toLowerCase() === name.toLowerCase())
    ) {
      return [
        ...ranked,
        { id: CREATE_ROW_ID, label: `Create “${name}”`, meta: createMeta, glyph: "＋" },
      ];
    }
    return ranked;
  }, [clearOption, createMeta, needle, onCreate, options, query]);

  // Sections only make sense on the unfiltered list; once results are ranked by
  // relevance, grouping them would fight the ranking.
  const sections = useMemo(() => {
    if (needle) return [{ group: "", items: visible }];
    const order: string[] = [];
    const byGroup = new Map<string, PickerOption[]>();
    for (const option of visible) {
      const key = option.group ?? "";
      if (!byGroup.has(key)) {
        byGroup.set(key, []);
        order.push(key);
      }
      byGroup.get(key)!.push(option);
    }
    return order.map((group) => ({ group, items: byGroup.get(group)! }));
  }, [needle, visible]);

  const selectable = useMemo(
    () => visible.reduce<number[]>((result, option, index) => {
      if (!option.disabled) result.push(index);
      return result;
    }, []),
    [visible],
  );

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // A new query invalidates the old cursor: land on the best match so Enter
  // always picks what the ranking put first.
  useEffect(() => {
    setActiveIndex(selectable[0] ?? -1);
  }, [needle, selectable]);

  useEffect(() => {
    if (activeIndex < 0) return;
    listRef.current
      ?.querySelector(`[data-index="${activeIndex}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);

  useEffect(() => {
    const onPointerDown = (event: PointerEvent) => {
      if (!wrapperRef.current?.contains(event.target as Node)) onDismiss();
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [onDismiss]);

  const step = useCallback(
    (direction: 1 | -1) => {
      if (!selectable.length) return;
      const position = selectable.indexOf(activeIndex);
      const next = position < 0
        ? direction > 0 ? 0 : selectable.length - 1
        : (position + direction + selectable.length) % selectable.length;
      setActiveIndex(selectable[next]!);
    },
    [activeIndex, selectable],
  );

  function onKeyDown(event: React.KeyboardEvent) {
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        step(1);
        break;
      case "ArrowUp":
        event.preventDefault();
        step(-1);
        break;
      case "Home":
        event.preventDefault();
        setActiveIndex(selectable[0] ?? -1);
        break;
      case "End":
        event.preventDefault();
        setActiveIndex(selectable[selectable.length - 1] ?? -1);
        break;
      case "Enter": {
        event.preventDefault();
        const option = visible[activeIndex];
        if (option && !option.disabled) choose(option);
        break;
      }
      case "Escape":
        event.preventDefault();
        onDismiss();
        break;
      default:
        break;
    }
  }

  function choose(option: PickerOption) {
    if (option.id === CREATE_ROW_ID) {
      if (onCreate) onCreate(query.trim());
      return;
    }
    onSelect(option.id);
  }

  let flatIndex = -1;

  return (
    <div className="commandPicker" ref={wrapperRef} onKeyDown={onKeyDown}>
      <div className="commandPickerSearch">
        <span aria-hidden="true">⌕</span>
        <input
          ref={inputRef}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={placeholder}
          aria-label={placeholder}
          aria-controls={`${id}-list`}
          spellCheck={false}
          autoComplete="off"
        />
        <b>{visible.length}</b>
      </div>

      {header ? <div className="commandPickerHeader">{header}</div> : null}

      <div className="commandPickerList" id={`${id}-list`} role="listbox" aria-label={label} ref={listRef}>
        {sections.map((section) => (
          <div key={section.group || "__ungrouped"}>
            {section.group ? <div className="commandPickerGroup">{section.group}</div> : null}
            {section.items.map((option) => {
              flatIndex += 1;
              const index = flatIndex;
              const selected = option.id === value;
              return (
                <div
                  key={option.id}
                  data-index={index}
                  role="option"
                  aria-selected={selected}
                  aria-disabled={option.disabled || undefined}
                  className={[
                    "commandPickerOption",
                    selected ? "isSelected" : "",
                    index === activeIndex ? "isActive" : "",
                    option.disabled ? "isDisabled" : "",
                  ].filter(Boolean).join(" ")}
                  onPointerEnter={() => !option.disabled && setActiveIndex(index)}
                  onClick={() => !option.disabled && !busy && choose(option)}
                >
                  {option.glyph ? <i className="commandPickerGlyph" aria-hidden="true">{option.glyph}</i> : null}
                  <span>
                    <strong>{option.label}</strong>
                    {option.meta ? <small>{option.meta}</small> : null}
                  </span>
                  {option.badge ? <b>{option.badge}</b> : null}
                </div>
              );
            })}
          </div>
        ))}
        {!visible.length ? <p className="commandPickerEmpty">{emptyMessage}</p> : null}
      </div>

      <footer className="commandPickerFooter">
        {footer ?? <span>↑↓ to move · Enter to choose · Esc to close</span>}
      </footer>
    </div>
  );
}
