import type { Metadata } from "next";

import { MeetingsWorkbench } from "@/components/meetings-workbench";

export const metadata: Metadata = { title: "Meetings" };

export default function MeetingsPage() {
  return <MeetingsWorkbench />;
}
