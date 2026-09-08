"use client";

// Every key the app answers to, on one sheet. Opens with "?" from anywhere
// you are not typing.

import { useEffect } from "react";

const GROUPS: { title: string; keys: [string, string][] }[] = [
  {
    title: "Everywhere",
    keys: [
      ["⌘ K", "Search conversations"],
      ["⌘ N", "New conversation"],
      ["?", "This sheet"],
      ["Esc", "Close a sheet, drawer or menu"],
    ],
  },
  {
    title: "Chat",
    keys: [
      ["Enter", "Send"],
      ["Shift Enter", "New line"],
      ["/", "Commands, context and sources"],
      ["Esc", "Cancel an edit"],
    ],
  },
  {
    title: "Today",
    keys: [
      ["J / K", "Move down and up the list"],
      ["Enter", "Open what is selected"],
      ["L", "Later"],
      ["R", "Restore something deferred"],
      ["X", "Select for a batch"],
    ],
  },
  {
    title: "Customers",
    keys: [
      ["⌘ ⇧ K", "Search every record"],
      ["↑ ↓", "Move through accounts"],
    ],
  },
  {
    title: "Assets and recordings",
    keys: [
      ["↑ ↓", "Move through the list"],
      ["Enter", "Open"],
    ],
  },
];

export function ShortcutsSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal shortcuts" role="dialog" aria-modal="true" aria-labelledby="shortcuts-title" onClick={(event) => event.stopPropagation()}>
        <header className="shortcuts-head">
          <h2 id="shortcuts-title">Keyboard shortcuts</h2>
          <button type="button" className="ui-btn is-quiet is-sm" onClick={onClose} aria-label="Close">×</button>
        </header>
        <div className="shortcuts-body">
          {GROUPS.map((group) => (
            <section key={group.title}>
              <h3>{group.title}</h3>
              <dl>
                {group.keys.map(([key, what]) => (
                  <div key={key + what}>
                    <dt><kbd>{key}</kbd></dt>
                    <dd>{what}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}

/** True when the key press came from somewhere that takes text. */
export function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}
