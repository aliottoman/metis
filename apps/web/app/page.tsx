import { Suspense } from "react";

import { ChatWorkspace } from "@/components/chat-workspace";

export default function HomePage() {
  return (
    <Suspense fallback={<div className="contentLoading">Opening conversation…</div>}>
      <ChatWorkspace />
    </Suspense>
  );
}
