import type { Metadata } from "next";

import { DacSizing } from "@/components/dac-sizing";

export const metadata: Metadata = { title: "Sizing" };

export default function SizingPage() {
  return <DacSizing />;
}
