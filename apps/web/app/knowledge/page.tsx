import type { Metadata } from "next";

import { KnowledgeCenter } from "@/components/knowledge-center";

export const metadata: Metadata = { title: "Knowledge" };

export default function KnowledgePage() {
  return <KnowledgeCenter />;
}
