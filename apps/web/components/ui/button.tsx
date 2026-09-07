// The one button. Three emphases, two extra sizes, and it can be a link.

import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from "react";

type Emphasis = "primary" | "secondary" | "quiet" | "danger";
type Size = "sm" | "md" | "lg";

type Shared = { emphasis?: Emphasis; size?: Size; children: ReactNode; className?: string };
type AsButton = Shared & ButtonHTMLAttributes<HTMLButtonElement> & { href?: undefined };
type AsLink = Shared & AnchorHTMLAttributes<HTMLAnchorElement> & { href: string };

function classes(emphasis: Emphasis, size: Size, extra?: string): string {
  return ["ui-btn", emphasis !== "secondary" ? `is-${emphasis}` : "", size !== "md" ? `is-${size}` : "", extra ?? ""]
    .filter(Boolean)
    .join(" ");
}

export function Button(props: AsButton | AsLink) {
  const { emphasis = "secondary", size = "md", className, children, ...rest } = props;
  if ("href" in rest && typeof rest.href === "string") {
    return (
      <a className={classes(emphasis, size, className)} {...(rest as AnchorHTMLAttributes<HTMLAnchorElement>)}>
        {children}
      </a>
    );
  }
  const { type = "button", ...buttonProps } = rest as ButtonHTMLAttributes<HTMLButtonElement>;
  return (
    <button type={type} className={classes(emphasis, size, className)} {...buttonProps}>
      {children}
    </button>
  );
}
