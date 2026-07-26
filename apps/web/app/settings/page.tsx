import type { Metadata } from "next";

import { SettingsPanel } from "@/components/settings-panel";

export const metadata: Metadata = { title: "Settings" };

export default function SettingsPage() {
  return <SettingsPanel />;
}
