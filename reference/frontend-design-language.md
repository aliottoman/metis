# Metis frontend design language

The visual system generated apps should ship with, distilled from the app's
own matured stylesheet (values measured, not eyeballed). A build that follows
this produces a UI that sits beside Metis instead of beside Bootstrap. Plain
CSS — no framework, no build step, one stylesheet.

## Tokens — copy this block into the app's style.css

```css
:root {
  /* paper & ink — warm greige field, near-black volcanic text */
  --paper: #f7f4ef;         /* page background */
  --paper-lift: #fcfaf6;    /* raised cards */
  --surface: #ede9e1;       /* inset panels, table headers, code */
  --ink: #211f1d;           /* primary text */
  --ink-2: #5e5a54;         /* secondary text */
  --ink-3: #7b7789;         /* eyebrows, hints — lavender-tinted grey */
  --line: #dfdacf;          /* hairlines; 1px solid, never box-shadow borders */

  /* accent — one purple, one bloom gradient, used sparingly */
  --purple: #c29ce0;
  --bloom: linear-gradient(270deg, #d18ee2, #ff7759);

  /* status */
  --ok: #2e7d4f;
  --risk-high: #b3402e;
  --risk-medium: #a06a1f;

  --radius-sm: 8px; --radius-md: 14px; --radius-lg: 22px;
  --shadow-sm: 0 1px 2px rgba(24,27,22,.05), 0 6px 18px rgba(24,27,22,.07);
  --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif;
  --font-display: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", system-ui, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
```

## Rules that make it read as this system

1. **Greige field, white cards.** `body { background: var(--paper); }`,
   content in cards of `--paper-lift` with `--shadow-sm` and `--radius-lg`.
   Never pure-white pages, never grey-on-grey.
2. **Eyebrow labels.** Section headings get a small mono eyebrow above them:
   uppercase, `letter-spacing: .08em`, `font-family: var(--font-mono)`,
   `color: var(--ink-3)`, ~11px. Then the display heading in `--ink`.
3. **One accent moment per view.** The bloom gradient appears once — the
   hero strip or the primary action — everything else stays paper and ink.
4. **Hairlines, not boxes.** Dividers are `1px solid var(--line)`. Tables:
   generous row height (≥44px), header in `--surface`, no zebra stripes.
5. **Status is text plus a dot, not a pill rainbow.** Risk verdicts:
   `● high` in `--risk-high`, `● low` in `--ok` — dot, word, nothing else.
6. **Type scale.** Display headings 28–40px `--font-display`; body 15–16px
   `--font-sans` at line-height 1.6; numbers and IDs in `--font-mono`.
7. **Buttons.** Primary: `--ink` background, paper text, `--radius-sm`,
   no gradient. Quiet actions are text buttons in `--ink-2`. Disabled is
   reduced opacity, never grey fills.
8. **Motion.** One transition: `transition: opacity .15s ease, transform
   .15s ease`. Nothing bounces.
9. **Empty and loading states are written, not spinner-only.** "No documents
   yet — drop an invoice above." beats an orphaned spinner.

## Layout skeleton for a typical tool

```html
<main class="shell">            <!-- max-width: 1080px; margin: 0 auto; padding: 48px 24px -->
  <header class="hero">         <!-- eyebrow + display title + one-line purpose -->
  <section class="card">        <!-- the primary workflow (upload, form) -->
  <section class="card">        <!-- results table / detail -->
</main>
```

Serve it from `app/static/` with FastAPI's `StaticFiles`; keep every style in
one `style.css` using the tokens above. If the app needs a chart, a plain
`<canvas>` sparkline in `--purple` on `--surface` — no chart library.
