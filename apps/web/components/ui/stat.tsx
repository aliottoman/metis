// A number that changes the next action, with its label under it.

export function Stat({
  value,
  label,
  attention = false,
}: {
  value: string | number;
  label: string;
  attention?: boolean;
}) {
  return (
    <div className={`ui-stat${attention ? " is-attention" : ""}`}>
      <b>{value}</b>
      <span>{label}</span>
    </div>
  );
}
