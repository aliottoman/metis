import { Skeleton } from "@/components/ui/skeleton";

export default function Loading() {
  return (
    <div className="workspacePage" aria-busy="true">
      <p className="ui-eyebrow" role="status">Opening your workspace…</p>
      <Skeleton rows={1} height={36} width="42%" />
      <Skeleton rows={4} height={20} />
    </div>
  );
}
