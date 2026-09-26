"use client";

import { Button } from "@/components/ui/button";
import { Notice } from "@/components/ui/notice";
import { PageHeader } from "@/components/ui/page-header";

export default function PageError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <div className="workspacePage">
      <PageHeader
        eyebrow="Something went wrong"
        title="Let’s get you back to work"
        lede="This page could not finish opening. Try it again, or return to your workspace."
        actions={<><Button href="/">Open chat</Button><Button href="/settings" emphasis="quiet">Check settings</Button></>}
      />
      <Notice kind="error" title="The page needs another try" action="Try again" onAction={reset}>
        If the problem continues, check the local service in Settings.
      </Notice>
    </div>
  );
}
