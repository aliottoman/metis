import type { Metadata } from "next";

import { MemoryCenter } from "@/components/memory-center";

export const metadata: Metadata = { title: "Memory" };

export default function MemoryPage() {
  return <MemoryCenter />;
}
