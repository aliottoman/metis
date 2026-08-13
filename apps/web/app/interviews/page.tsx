import type { Metadata } from "next";

import { InterviewsWorkbench } from "@/components/interviews-workbench";

export const metadata: Metadata = { title: "Interviews" };

export default function InterviewsPage() {
  return <InterviewsWorkbench />;
}
