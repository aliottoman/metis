"use client";

// Transient confirmations ("Saved", "Deployed") in one rail at the bottom.
// A toast is for something that already happened and needs no answer; a
// Notice is for something that needs one.

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

interface Toast {
  id: number;
  text: string;
  kind: "default" | "error";
}

const ToastContext = createContext<(text: string, kind?: Toast["kind"]) => void>(() => undefined);

export const TOAST_MS = 3200;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const push = useCallback((text: string, kind: Toast["kind"] = "default") => {
    const id = Date.now() + Math.random();
    setToasts((current) => [...current, { id, text, kind }]);
    window.setTimeout(() => setToasts((current) => current.filter((item) => item.id !== id)), TOAST_MS);
  }, []);
  const value = useMemo(() => push, [push]);
  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="ui-toast-rail" aria-live="polite">
        {toasts.map((toast) => (
          <div key={toast.id} className={`ui-toast${toast.kind === "error" ? " is-error" : ""}`}>
            {toast.text}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

/** `const toast = useToast(); toast("Saved")` */
export function useToast() {
  return useContext(ToastContext);
}
