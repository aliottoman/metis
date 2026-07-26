import { ImageResponse } from "next/og";

import { brandCluster } from "@/lib/brand-mark";

// Used by Safari's "Add to Dock" and iOS "Add to Home Screen" as the app icon.
export const size = { width: 180, height: 180 };
export const contentType = "image/png";

export default function AppleIcon() {
  return new ImageResponse(brandCluster(180), { ...size });
}
