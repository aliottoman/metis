import type { Metadata } from "next";

import { AgentFactory } from "@/components/agent-factory";

export const metadata: Metadata = { title: "Agents" };

export default function AgentsPage() {
  return <AgentFactory />;
}
