import type { MetadataRoute } from "next";

import { PRODUCT_NAME } from "@/lib/product";

// A standalone-display manifest so the dockable web app opens chrome-free.
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: PRODUCT_NAME,
    short_name: PRODUCT_NAME,
    description: "A local-first, self-improving agent with governed local and OCI reasoning.",
    start_url: "/",
    display: "standalone",
    background_color: "#f3f4ef",
    theme_color: "#f3f4ef",
    icons: [
      { src: "/icon", sizes: "64x64", type: "image/png" },
      { src: "/apple-icon", sizes: "180x180", type: "image/png" },
    ],
  };
}
