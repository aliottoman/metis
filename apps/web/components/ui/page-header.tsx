// The command strip every page opens with: a name, one line of context, and
// the primary action on the right. Shared framing keeps every workspace familiar.

import type { ReactNode } from "react";

export function PageHeader({
  eyebrow,
  title,
  lede,
  actions,
  icon,
}: {
  eyebrow?: string;
  title: string;
  lede?: string;
  actions?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <header className="ui-page-header">
      <div className="ui-page-heading">
        {icon ? <span className="ui-page-icon" aria-hidden="true">{icon}</span> : null}
        <div className="ui-page-copy">
        {eyebrow ? <span className="ui-eyebrow">{eyebrow}</span> : null}
        <h1>{title}</h1>
        {lede ? <p>{lede}</p> : null}
        </div>
      </div>
      {actions ? <div className="ui-page-actions">{actions}</div> : null}
    </header>
  );
}
