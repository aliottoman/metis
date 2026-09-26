"use client";

import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { enabledPickerIndexes, groupPickerOptions } from "@/lib/picker-options";

/**
 * A styled, keyboard-accessible replacement for a native `<select>`.
 *
 * Browsers render the option popup themselves and ignore almost all CSS applied
 * to it, so a native select cannot be made to match the rest of the workspace —
 * the control looks right and the menu it opens does not. This implements the
 * ARIA listbox pattern instead, which puts the popup in the page where it can
 * be styled, grouped and filtered.
 *
 * Everything a native select gives away for free is reimplemented deliberately:
 * roving focus, type-ahead, Home/End, Escape, click-away, and scrolling the
 * active option into view. A filter box appears once a list is long enough that
 * type-ahead stops being enough on its own — the model picker carries ~100
 * entries across a dozen families.
 */

export type SelectOption = {
  value: string;
  label: string;
  /** Secondary text shown right-aligned, e.g. a size or a parameter count. */
  hint?: string;
  /** Options sharing a group are rendered under one heading, in first-seen order. */
  group?: string;
  disabled?: boolean;
};

type SelectMenuProps = {
  label: string;
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  /** Shown when nothing matches the filter. */
  emptyMessage?: string;
  /** Force the filter box on or off; defaults to on for long lists. */
  searchable?: boolean;
  disabled?: boolean;
  /** Reuse the field in dense forms that already provide their own layout. */
  className?: string;
  /** Keep the accessible label while visually hiding it. */
  hideLabel?: boolean;
};

const SEARCHABLE_THRESHOLD = 12;
/** Keep in step with `.selectList { max-height }` in globals.css. */
const MAX_POPUP_HEIGHT = 292;
const TYPEAHEAD_RESET_MS = 700;
/**
 * Keep in step with the longest `.selectMenu.isClosing .selectPopup` exit.
 * The timeout is a fallback for reduced motion and browsers that do not emit
 * a transition/animation end event.
 */
export const SELECT_MENU_EXIT_MS = 160;

export function SelectMenu({
  label,
  value,
  options,
  onChange,
  emptyMessage = "No matches",
  searchable,
  disabled = false,
  className = "",
  hideLabel = false,
}: SelectMenuProps) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [closing, setClosing] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(-1);
  const [dropUp, setDropUp] = useState(false);

  const wrapperRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const typeahead = useRef({ buffer: "", at: 0 });

  const showSearch = searchable ?? options.length > SEARCHABLE_THRESHOLD;

  const selected = useMemo(
    () => options.find((option) => option.value === value) ?? null,
    [options, value],
  );

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return options;
    return options.filter((option) =>
      `${option.label} ${option.hint ?? ""} ${option.group ?? ""}`
        .toLowerCase()
        .includes(needle),
    );
  }, [options, query]);

  // Options keep their given order; groups are emitted the first time they
  // appear so the caller controls ranking without also controlling layout.
  const sections = useMemo(() => groupPickerOptions(visible), [visible]);
  const displayed = useMemo(() => sections.flatMap((section) => section.items), [sections]);

  const selectableIndexes = useMemo(
    () => enabledPickerIndexes(displayed),
    [displayed],
  );

  const clearCloseTimer = useCallback(() => {
    if (closeTimerRef.current !== null) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  }, []);

  const finishClose = useCallback(() => {
    clearCloseTimer();
    setClosing(false);
    setQuery("");
    setActiveIndex(-1);
    setDropUp(false);
  }, [clearCloseTimer]);

  const close = useCallback(
    (returnFocus = true) => {
      clearCloseTimer();
      setOpen(false);
      setClosing(true);
      closeTimerRef.current = setTimeout(finishClose, SELECT_MENU_EXIT_MS);
      if (returnFocus) buttonRef.current?.focus();
    },
    [clearCloseTimer, finishClose],
  );

  const commit = useCallback(
    (option: SelectOption | undefined) => {
      if (!option || option.disabled) return;
      onChange(option.value);
      close();
    },
    [close, onChange],
  );

  const openMenu = useCallback(() => {
    if (disabled) return;
    clearCloseTimer();
    setClosing(false);
    setOpen(true);
    typeahead.current = { buffer: "", at: 0 };
    const current = displayed.findIndex((option) => option.value === value && !option.disabled);
    setActiveIndex(current >= 0 ? current : (selectableIndexes[0] ?? -1));
  }, [clearCloseTimer, disabled, selectableIndexes, value, displayed]);

  useEffect(() => clearCloseTimer, [clearCloseTimer]);

  useEffect(() => {
    if (disabled && open) close(false);
  }, [close, disabled, open]);

  useEffect(() => {
    if (open) setActiveIndex((current) => selectableIndexes.includes(current) ? current : selectableIndexes[0] ?? -1);
  }, [open, selectableIndexes]);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!wrapperRef.current?.contains(event.target as Node)) close(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [close, open]);

  // The workspace scrolls inside its own container, which clips the popup. A
  // control near the bottom of the viewport would open into that clip and show
  // a couple of truncated rows, so the menu flips above the trigger when there
  // is more room there. Re-measured on scroll and resize because either can
  // change the answer while the menu is still open.
  useLayoutEffect(() => {
    if (!open) return;
    const measure = () => {
      const trigger = buttonRef.current?.getBoundingClientRect();
      if (!trigger) return;
      const wanted = Math.min(listRef.current?.scrollHeight ?? 0, MAX_POPUP_HEIGHT) + (showSearch ? 60 : 16);
      const below = window.innerHeight - trigger.bottom;
      setDropUp(below < wanted && trigger.top > below);
    };
    measure();
    window.addEventListener("scroll", measure, true);
    window.addEventListener("resize", measure);
    return () => {
      window.removeEventListener("scroll", measure, true);
      window.removeEventListener("resize", measure);
    };
  }, [open, showSearch, visible.length]);

  useEffect(() => {
    if (open && showSearch) searchRef.current?.focus();
  }, [open, showSearch]);

  // Keep the active option visible when moving through a scrolled list.
  useEffect(() => {
    if (!open || activeIndex < 0) return;
    listRef.current
      ?.querySelector(`[data-index="${activeIndex}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [activeIndex, open]);

  const step = useCallback(
    (direction: 1 | -1) => {
      if (!selectableIndexes.length) return;
      const position = selectableIndexes.indexOf(activeIndex);
      const next =
        position < 0
          ? direction > 0
            ? 0
            : selectableIndexes.length - 1
          : (position + direction + selectableIndexes.length) % selectableIndexes.length;
      setActiveIndex(selectableIndexes[next]!);
    },
    [activeIndex, selectableIndexes],
  );

  const runTypeahead = useCallback(
    (character: string) => {
      const now = Date.now();
      const state = typeahead.current;
      state.buffer = now - state.at > TYPEAHEAD_RESET_MS ? character : state.buffer + character;
      state.at = now;
      const match = displayed.findIndex(
        (option) => !option.disabled && option.label.toLowerCase().startsWith(state.buffer),
      );
      if (match >= 0) setActiveIndex(match);
    },
    [displayed],
  );

  function onKeyDown(event: React.KeyboardEvent) {
    if (disabled || event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229) return;
    if (!open) {
      if (["Enter", " ", "ArrowDown", "ArrowUp"].includes(event.key)) {
        event.preventDefault();
        openMenu();
      }
      return;
    }
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
        if (event.target === searchRef.current && !event.ctrlKey && !event.metaKey) break;
        event.preventDefault();
        setActiveIndex(selectableIndexes[0] ?? -1);
        break;
      case "End":
        if (event.target === searchRef.current && !event.ctrlKey && !event.metaKey) break;
        event.preventDefault();
        setActiveIndex(selectableIndexes[selectableIndexes.length - 1] ?? -1);
        break;
      case "Enter":
        event.preventDefault();
        commit(displayed[activeIndex]);
        break;
      case " ":
        if (event.target === searchRef.current) break;
        event.preventDefault();
        commit(displayed[activeIndex]);
        break;
      case "Tab":
        close(false);
        break;
      case "Escape":
        event.preventDefault();
        event.stopPropagation();
        close();
        break;
      default:
        // Type-ahead only when the filter box is not already taking the keys.
        if (!showSearch && event.key.length === 1 && !event.metaKey && !event.ctrlKey) {
          runTypeahead(event.key.toLowerCase());
        }
    }
  }

  let flatIndex = -1;
  const menuState = open ? "open" : closing ? "closing" : "closed";
  const activeOptionId = open && activeIndex >= 0 && displayed[activeIndex] && !displayed[activeIndex].disabled
    ? `${id}-option-${activeIndex}` : undefined;

  return (
    <div className={`selectField ${className}`.trim()}>
      <span className={`selectLabel ${hideLabel ? "visuallyHidden" : ""}`} id={`${id}-label`}>
        {label}
      </span>
      <div
        className={`selectMenu ${open ? "isOpen" : ""} ${closing ? "isClosing" : ""} ${dropUp ? "dropUp" : ""}`}
        data-state={menuState}
        data-placement={dropUp ? "top" : "bottom"}
        ref={wrapperRef}
        onBlur={(event) => {
          if (open && event.relatedTarget && !event.currentTarget.contains(event.relatedTarget as Node)) close(false);
        }}
      >
        <button
          type="button"
          ref={buttonRef}
          className="selectTrigger"
          role="combobox"
          aria-controls={`${id}-list`}
          aria-expanded={open}
          aria-haspopup="listbox"
          aria-activedescendant={!showSearch ? activeOptionId : undefined}
          aria-labelledby={`${id}-label`}
          disabled={disabled}
          onClick={() => (open ? close() : openMenu())}
          onKeyDown={onKeyDown}
        >
          <span className="selectValue">{selected?.label ?? "Select…"}</span>
          {selected?.hint ? <span className="selectTriggerHint">{selected.hint}</span> : null}
          <svg className="selectChevron" viewBox="0 0 12 12" aria-hidden="true" focusable="false">
            <path
              d="M3 4.5 6 7.5 9 4.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>

        {open || closing ? (
          <div
            className={`selectPopup ${open ? "isOpen" : "isClosing"}`}
            data-state={menuState}
            aria-hidden={closing || undefined}
            inert={closing ? true : undefined}
            onAnimationEnd={(event) => {
              if (
                closing
                && event.target === event.currentTarget
                && ["mMenuFold", "mMenuFoldUp"].includes(event.animationName)
              ) {
                finishClose();
              }
            }}
          >
            {showSearch ? (
              <div className="selectSearch">
                <input
                  ref={searchRef}
                  value={query}
                  onChange={(event) => {
                    setQuery(event.target.value);
                    setActiveIndex(-1);
                  }}
                  onKeyDown={onKeyDown}
                  placeholder="Filter…"
                  role="combobox"
                  aria-expanded={open}
                  aria-autocomplete="list"
                  aria-activedescendant={activeOptionId}
                  aria-label={`Filter ${label}`}
                  aria-controls={`${id}-list`}
                  autoComplete="off"
                  spellCheck={false}
                />
              </div>
            ) : null}
            <div
              className="selectList"
              id={`${id}-list`}
              role="listbox"
              aria-labelledby={`${id}-label`}
              ref={listRef}
              tabIndex={-1}
              onKeyDown={showSearch ? undefined : onKeyDown}
            >
              {sections.map((section) => (
                <div className="selectSection" key={section.group || "__ungrouped"}>
                  {section.group ? (
                    <div className="selectGroupLabel" role="presentation">
                      {section.group}
                    </div>
                  ) : null}
                  {section.items.map((option) => {
                    flatIndex += 1;
                    const index = flatIndex;
                    const isSelected = option.value === value;
                    return (
                      <div
                        key={option.value}
                        id={`${id}-option-${index}`}
                        data-index={index}
                        role="option"
                        aria-selected={isSelected}
                        aria-disabled={option.disabled || undefined}
                        className={[
                          "selectOption",
                          isSelected ? "isSelected" : "",
                          index === activeIndex ? "isActive" : "",
                          option.disabled ? "isDisabled" : "",
                        ]
                          .filter(Boolean)
                          .join(" ")}
                        onPointerEnter={() => !option.disabled && setActiveIndex(index)}
                        onClick={() => commit(option)}
                      >
                        <span className="selectOptionLabel">{option.label}</span>
                        {option.hint ? (
                          <span className="selectOptionHint">{option.hint}</span>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              ))}
              {!visible.length ? <p className="selectEmpty">{emptyMessage}</p> : null}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
