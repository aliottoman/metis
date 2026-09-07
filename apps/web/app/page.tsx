import { Suspense } from "react";

import { ChatWorkspace } from "@/components/chat-workspace";
import { ConversationList } from "@/components/conversation-list";

export default function HomePage() {
  return (
    <Suspense fallback={<div className="chat" aria-busy="true">Opening conversation…</div>}>
      <div className="chatLayout">
        <ConversationList />
        <ChatWorkspace />
      </div>
    </Suspense>
  );
}
