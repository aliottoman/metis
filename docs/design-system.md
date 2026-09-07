# The design system

One token file, five scales, ten primitives, one icon set. This is the
foundation the interface audit asked for, and the rule that keeps it one
system: **a colour, size, radius, duration or shadow is defined in
`tokens.css` and nowhere else.**

## Files

| File | Role |
| --- | --- |
| `apps/web/app/tokens.css` | Every token. Canonical names first, then every legacy name as an alias. Loaded first. |
| `apps/web/app/globals.css` | The older component styles. Reads tokens; defines none. |
| `apps/web/app/matured.css` | The newer component styles, including the folded chrome revision (§12) and conversation studio (§13). Reads tokens; defines none. |
| `apps/web/app/primitives.css` | The ten primitives. Canonical tokens only. Loaded last. |
| `apps/web/components/ui/` | The primitives as React components. |

The two older files are the migration backlog, not the system. Every
selector in them that a primitive covers is a selector to delete.

## Tokens

Canonical names, and what they are for:

- **Ground** `--color-ground` (the frame), `--color-pane` (the main pane),
  `--color-paper` / `--color-paper-lift` (cards), `--color-surface` (insets).
- **Ink** `--color-ink`, `--color-ink-2` (secondary text), `--color-ink-3`
  (labels; passes 4.5:1 on paper). `--color-line` for hairlines.
- **Accent** `--color-accent` (lavender, the one accent), `--color-accent-ink`
  for accent text, `--color-accent-surface` for a selected fill.
- **Meaning** `--color-coral` (live), `--color-green-ink` (success),
  `--color-danger` (error), `--color-mint` / `--color-gold` / `--color-violet`
  (identity, sparingly).
- **Rail** `--color-rail`, `--color-rail-ink`.
- **Type** `--font-sans`, `--font-display`, `--font-mono`, `--font-serif`;
  the scale `--text-xs` 11 · `--text-sm` 12 · `--text-md` 13 · `--text-base` 14 ·
  `--text-body` 15 · `--text-lg` 16 · `--text-xl` 18 · `--text-2xl` 20 ·
  `--text-3xl` 24 · `--text-4xl` 28 · `--text-5xl` 40 · `--text-6xl` 56.
  Nothing below 11px.
- **Shape** `--radius-control` 8 · `--radius-card` 14 · `--radius-pane` 24 ·
  `--radius-pill`.
- **Space** `--space-1` 4 through `--space-7` 48.
- **Motion** `--motion-fast` 120ms · `--motion-base` 200ms · `--motion-slow`
  320ms; `--ease-standard`, `--ease-out`. Longer durations are ambient
  animation and stay literal.
- **Elevation** `--shadow-1`, `--shadow-2`, `--shadow-3`; `--focus-ring`.

The dark set is defined under `:root[data-theme="dark"]` and switched on by
the theme toggle in a later phase. It covers the canonical names only, which
is the point: a component that reads canonical tokens is dark-ready, and one
that reads a legacy name is not.

### Legacy names

Everything the four old stylesheets defined — `--m-*`, `--ui-*`, `--audio-*`,
`--glass-*`, `--metis-*`, and the unprefixed set — is still defined, in
`tokens.css`, with the value that was winning on screen when they were
merged. Where that value equals a canonical one, the legacy name is
`var(--canonical)`; where it does not, it is a literal, and retiring it means
pointing its users at the nearest canonical token and deleting the line.

## Primitives

| Primitive | Class | Component | Use it for |
| --- | --- | --- | --- |
| Button | `.ui-btn` + `is-primary` / `is-quiet` / `is-danger`, `is-sm` / `is-lg` | `Button` | Every button and button-shaped link |
| Chip | `.ui-chip` + `is-accent` / `is-outline` | `Chip` | A scope, a tag, a count; never a button |
| Status | `.ui-dot` + `is-live` / `is-waiting` / `is-ready` / `is-needs-review` / `is-stopped`, `.ui-status` | `StatusDot`, `Status` | Every status, in five words |
| Stat | `.ui-stat` + `is-attention` | `Stat` | A number that changes the next action |
| Notice | `.ui-notice` + `is-info` / `is-error` / `is-success` | `Notice` | What happened and what to do; carries its action |
| Toast | `.ui-toast-rail`, `.ui-toast` | `ToastProvider`, `useToast()` | A confirmation that needs no answer |
| Skeleton | `.ui-skeleton` | `Skeleton` | Rows the real rows will replace |
| Page header | `.ui-page-header` | `PageHeader` | The command strip every page opens with |
| Field | `.ui-field` | `Field` | A label over a control, with its one problem |
| Audio player | `.ui-audio` | `AudioPlayer` | Every recording |

The Agents page is the reference implementation: it uses nothing but these
and its own layout classes.

## Icons

`lucide-react`, 18px, stroke 1.6, through `NavIcon` in the shell. New icons
come from the same set at the same size; no hand-drawn paths.

## Verifying a style change

A computed-style snapshot of all thirteen pages was used to verify this
foundation: each page's elements were recorded (colour, type, spacing,
radius, shadow, transition, box) before a change and diffed after it. The
token fold diffed to zero on every page; the scale snap changed only
font-size, line-height, radius and transition properties. The script is in
the session that did it; the approach is the thing to keep: **measure the
rendered result before and after, never trust the files.**

## What Phase 0 did not do

Selectors that are defined more than once across the two older files (263
of them; `.chatHeader` fourteen times) were not resolved. Every duplicate is
a place where a primitive or a rebuilt component should replace the whole
family, which is what the shell, chat, and workbench phases do. Resolving
them in place would spend the effort twice.
