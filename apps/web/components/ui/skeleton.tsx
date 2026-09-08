// Placeholder rows sized like the rows they stand in for, so nothing shifts
// when the real ones arrive.

export function Skeleton({ rows = 3, height = 14, width = "100%" }: { rows?: number; height?: number; width?: string }) {
  return (
    <div style={{ display: "grid", gap: 8 }} aria-hidden="true">
      {Array.from({ length: rows }, (_, index) => (
        <span key={index} className="ui-skeleton" style={{ height, width: rows > 1 && index === rows - 1 ? "70%" : width }} />
      ))}
    </div>
  );
}
