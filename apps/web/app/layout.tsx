import type { Metadata, Viewport } from "next";
import { Suspense } from "react";

import { AppShell } from "@/components/app-shell";
import { ToastProvider } from "@/components/ui/toast";
import { THEME_BOOT_SCRIPT } from "@/lib/theme";
import { PRODUCT_NAME } from "@/lib/product";
// Tokens first: every colour, size, radius and duration the other two read.
import "./tokens.css";
import "./globals.css";
// Loaded after globals so it wins on equal specificity.
import "./matured.css";
// The primitives, last: a primitive wins on equal specificity.
import "./primitives.css";
// The shell: frame, rail, page header, chat layout, breakpoints, motion.
import "./shell.css";

export const metadata: Metadata = {
  applicationName: PRODUCT_NAME,
  title: {
    default: `${PRODUCT_NAME} — Local intelligence`,
    template: `%s · ${PRODUCT_NAME}`,
  },
  description: "A local-first, self-improving agent with governed local and OCI reasoning.",
  // Safari "Add to Dock" reads these: a chrome-free standalone window titled
  // "Metis" with the generated apple-icon. Next auto-links app/manifest.ts.
  appleWebApp: {
    capable: true,
    title: PRODUCT_NAME,
    statusBarStyle: "default",
  },
};

export const viewport: Viewport = {
  colorScheme: "light dark",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#1b1720" },
    { media: "(prefers-color-scheme: dark)", color: "#0f0d12" },
  ],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Applies a saved light/dark choice before first paint. */}
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT_SCRIPT }} />
      </head>
      <body>
        <Suspense fallback={<div className="appLoading">Preparing your private workspace…</div>}>
          <ToastProvider>
            <AppShell>{children}</AppShell>
          </ToastProvider>
        </Suspense>
      </body>
    </html>
  );
}
