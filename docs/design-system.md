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

The dark set is defined twice with the same values: under
`prefers-color-scheme: dark` for the un-stamped "system" state, and under
`:root[data-theme="dark"]` for a pinned choice. It covers the canonical names
and the legacy names the older stylesheets still read, so the whole app
follows it; a component that reads canonical tokens is dark-ready by
construction.

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

## The shell

`apps/web/app/shell.css` and `components/app-shell.tsx` are the frame every
page sits in, built from Phase 1 of the interface audit:

- **One rail, labelled at rest.** Three groups by the job: Work (Today,
  Chat, Customers), Voice (Meetings, Interviews, Agents), Library (Assets,
  Knowledge, Answers, Memory, Tool Workshop, Sizing). Library is collapsed
  by default and remembered; a page inside it opens it. Settings and the
  connection status sit at the foot. The active pill slides between entries.
- **Conversations live inside Chat** (`components/conversation-list.tsx`),
  as a collapsible column beside the workspace. ⌘K focuses its search from
  anywhere; ⌘N starts a chat.
- **The command strip.** `.pageHeader` and `.ui-page-header` are the same
  64px strip: eyebrow, name, one line of context, the primary action on the
  right, a hairline beneath. Today keeps its hero; nothing else has one.
- **Three breakpoints.** 720 (phone: the rail is a drawer, headers and
  toolbars wrap), 1080 (the rail folds to icons), 1440. Every old media
  query was mapped onto these.
- **Theme.** `lib/theme.ts` stamps `data-theme` on the root; the tokens'
  `prefers-color-scheme` rule handles "system". A saved choice is applied by
  an inline script before first paint. The control is in Settings.
- **Motion, once.** `--motion-*` and `--ease-*` drive every transition; the
  page enters with one short rise. Character lives in the primitives, not
  in per-page rules: every `.ui-btn` lifts a pixel on hover and settles on
  press, a secondary turns to ink, a primary carries the lilac→coral bloom
  (`--color-bloom`) behind it, and rail links nudge their icon.
- **The pearl.** `components/metis-companion.tsx` with `app/companion.css`
  is the mark: it sits in the rail brand, breathes, and its oil film turns.
  The fold toggle sits beside it and shows on hover. In the native window
  (`html.metisNativeWindow`, set by the Mac app) the rail leaves 40px for
  the traffic lights in both states.

## The chat page

`apps/web/app/chat.css`, `hooks/use-chat.ts` and `components/chat/*` are
Phase 2 of the audit. The old 2,400-line workspace became one engine hook
and four small views:

- **`useChat()` owns the state**: conversation, composer contents, the run
  and its events, message actions, and scope (customer, project, sources,
  model). `chat-workspace.tsx` only arranges the pieces.
- **Composer first.** A new conversation opens on the composer, with what is
  waiting on you phrased as things to say and the last three conversations
  (`chat/welcome.tsx`). No wordmark, no marketing copy, no disclaimer.
- **One "+" menu.** Customer, Project, Attach files, Dictate, and the
  Auto / Notion / Web sources live behind one button; typing `/` opens the
  same menu. Send becomes a queue button while a run is live, and Stop sits
  beside it.
- **A quiet thread.** Your messages are right-aligned blocks, replies are
  the reading column, the time shows on hover, and actions (Copy, Edit,
  Retry, Make a tool, Save to account, feedback) appear on hover. A reply
  that used retrieval carries "Grounded in N sources".
- **A quiet timeline.** `run-timeline.tsx` folds routine bookkeeping behind
  "Show N routine events", shows elapsed time once in the header and a
  duration per step, and goes red only on the failed step. It opens as a
  360px drawer, a slide-over under 1080.
- **Conversations are titled by their first message.** Chat did this
  already; voice sessions now create their conversation on the first thing
  said, titled by it, so an unused session leaves nothing behind. The list
  hides conversations nobody said anything in.

## The workbenches

`apps/web/app/workbenches.css` styles Today, Customers and Assets, Phase 3
of the audit. Each page is a header strip, then the working surface:

- **Today is one ranked list** (`components/today-view.tsx`): Start here,
  then each kind of work in the queue's own order, then Deferred, then
  what changed since yesterday. Arrow keys or j/k walk it, Enter opens,
  L defers, X selects a batchable row, R brings a deferred one back. The
  brief is a paragraph and a Listen button that becomes the shared player.
- **Customers is master-detail** (`hooks/use-customers.ts`,
  `components/customers/*`): accounts on the left with "Everything" pinned
  at the top, the dashboard or the open account on the right. Every edit
  goes through one `run()` (busy, error, toast, re-read). Sections keep
  their own drafts and the detail remounts per account, so nothing carries
  over. Dialogs share one `Modal`; the URL follows the selection.
- **Assets is list-first** (`lib/assets.ts`, `components/assets/*`):
  running first, then ready, then what needs something. Filters are chips
  with counts, the settings drawer is fixed-width with its own env and log
  state, and a running asset opens full-width.

## The audio family

`apps/web/app/audio.css`, `components/audio/stage.tsx` and `lib/audio.ts`
are Phase 4 of the audit. Voice, Interviews and Meetings share one language:

- **One stage.** `Stage` is the orb (the only expressive element), a mono
  state label under it, one line of hint, a caption line, and the
  controls. `MicButton`, `VolumeControl` and `Transcript` (copy, export)
  go with it. Voice mode and the live interview both stand on it.
- **Sentences as they are spoken.** The backend publishes `voice.spoken`
  to the browser for each sentence the moment it clears the claim gate;
  the voice hook collects them into `spoken` and shows them under the orb,
  then the finished `voice.turn` replaces them with the same words and
  their sources.
- **One player.** `AudioPlayer` gained ten-second skips, an `onTime`
  callback and a `seek` handle, so Meetings highlights words and seeks
  from the transcript through the same player Today uses for the brief.
- **Meetings and Interviews** split into small views (`meetings/overview`,
  `meetings/transcript`, `interviews/setup`, `interviews/live`,
  `interviews/scorecard`) on primitives; the audio section of Settings
  is on primitives too. 868 old rules left matured.css.

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
