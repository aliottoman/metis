import { ImageResponse } from "next/og";

import { brandCluster } from "@/lib/brand-mark";

// Browser-tab favicon.
export const size = { width: 64, height: 64 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(brandCluster(64), { ...size });
}
