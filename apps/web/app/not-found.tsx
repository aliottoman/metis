import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/ui/page-header";

export default function NotFound() {
  return (
    <div className="workspacePage">
      <PageHeader
        eyebrow="Page not found"
        title="That page isn’t here"
        lede="The address may have changed. Your workspace is still a click away."
        actions={<><Button href="/" emphasis="primary">Open chat</Button><Button href="/today">Go to Today</Button></>}
      />
    </div>
  );
}
