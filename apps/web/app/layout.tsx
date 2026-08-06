import type { Metadata, Viewport } from "next";
import { Suspense } from "react";

import { AppShell } from "@/components/app-shell";
import { PRODUCT_NAME } from "@/lib/product";
import "./globals.css";
// Loaded after globals so it wins on equal specificity. Remove this one
// import to drop the whole treatment.
import "./matured.css";

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
  colorScheme: "light",
  themeColor: "#f2f4f3",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <Suspense fallback={<div className="appLoading">Preparing your private workspace…</div>}>
          <AppShell>{children}</AppShell>
        </Suspense>
      </body>
    </html>
  );
}
