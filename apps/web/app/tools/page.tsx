import type { Metadata } from "next";

import { ToolWorkshop } from "@/components/tool-workshop";

export const metadata: Metadata = { title: "Tool Workshop" };

export default function ToolsPage() {
  return <ToolWorkshop />;
}
