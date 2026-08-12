# Metis interface opportunity catalog

This is a product-wide collection of UI elements worth adopting or evolving.
It is deliberately not a redesign brief: every item should retain Metis's warm
paper field, volcanic ink, hairline dividers, sparse purple/bloom accent, and
status-as-dot-plus-text language.

## The guardrails

- **One focus per view.** The primary workflow gets the strongest surface;
  everything else becomes supporting context.
- **Operational, not analytical.** A metric only earns space when it changes
  the next action. Avoid a top row of generic KPI cards.
- **Quiet by default.** Strong contrast, colour and motion are reserved for
  work that is live, blocked, risky, or needs a decision.
- **Explain the why.** A status should say what happened and what the person
  can do next, not merely display a state.

## Cross-product primitives

| Priority | Element | Where it belongs | The Metis version |
|---|---|---|---|
| P0 | **Decision card** | Today, chat approvals, tool and memory reviews | A wide paper card with an eyebrow (`Needs a decision`), one plain-language recommendation, one-line reason, and 1 primary + 1–2 quiet actions. |
| P0 | **Reason line** | Every ranked item | Small muted copy such as `Blocks export run · due today` or `Based on 4 recent notes`. It makes ranking trustworthy. |
| P0 | **Status dot** | Assets, runs, source health, tools | Keep the existing dot + word convention, but standardise the vocabulary: `live`, `waiting`, `ready`, `needs review`, `stopped`. |
| P0 | **Disclosure row** | All settings and detail drawers | A hairline row with a short summary at rest, plus progressive detail on open. Avoid dumping configuration fields into the default page. |
| P1 | **Context chip** | Search, chat composer, customer / project scope | A single textual chip (`Acme · Customer`, `metis/ · Project`) with an icon and clear affordance. No multicolour tag clouds. |
| P1 | **Activity receipt** | Chat, Today, assets | A compact, timestamped line: `Metis indexed 12 files · 2m ago`. This is more reassuring than a loading spinner after the fact. |
| P1 | **Command strip** | Page headers on dense workspaces | A compact header row: search / filter / one primary action. Commands should never be scattered across card footers. |
| P2 | **Soft selection surface** | Lists and multi-select modes | Use the mint inset already in the system for selection, paired with a bottom action bar; never rely on a thin outline alone. |
| P2 | **Intentional empty scene** | Every empty state | One small bloom mark, a clear sentence, and one next action. Preserve the writing-first approach already established in the system. |

## Navigation and search

| Priority | Element | Recommendation |
|---|---|---|
| P0 | **Global command/search trigger** | Replace the small sidebar search's role with a visible `Search Metis` trigger in the sidebar footer or header (`⌘K`). It should search conversations, customers, knowledge, assets, and commands from one place. |
| P1 | **Search result anatomy** | Every result gets: type eyebrow, title, one matched-context line, and a right-aligned destination. Group results by `Conversations`, `Customers`, `Knowledge`, and `Commands`. |
| P1 | **Recent + suggested state** | Before a query, show recent destinations and 3 useful commands—not an empty input. Suggested entries can be `New chat`, `Open Today`, and the active project. |
| P1 | **Nav attention badges** | Today and Assets can carry a small mono count only when attention is required. Avoid persistent notification dots across all navigation. |
| P2 | **Collapsed-nav hover card** | When the sidebar is collapsed, show the full section name plus a one-line purpose on hover, not just a tooltip. |

## Chat and composer

| Priority | Element | Recommendation |
|---|---|---|
| P0 | **Composer scope shelf** | Above the input, show the active customer/project and attached knowledge as one compact shelf. Make scope removable and addable without opening a modal. |
| P0 | **Run-state capsule** | In the chat header, unify `Local`, `Cloud`, and `Project` with the current run state: `Local · ready`, `Project · indexing`, `Cloud · connected`. The label should tell a complete story. |
| P0 | **Inline approval card** | Replace generic action treatment with a clear decision block: recommendation, requested capability, impact, evidence link, then `Approve` / `Not now`. |
| P1 | **Answer provenance footer** | Under consequential responses, add a quiet footer: `Grounded in 3 sources · Review sources`. It visually distinguishes supported work from generative guidance. |
| P1 | **Live activity as a trace** | In the activity panel, use a vertical rail with distinct completed/current/waiting nodes. Keep only the current step expanded; history remains scannable. |
| P1 | **Attachment cards with intent** | Show file type, filename, size and one chosen action (`Use in chat` / `Add to knowledge`) rather than a generic tray of file pills. |
| P2 | **Conversation checkpoint** | A thin, readable divider after a run finishes: `Run completed · 10:42`. It provides orientation in long threads without turning chat into a feed. |

## Today and work queues

| Priority | Element | Recommendation |
|---|---|---|
| P0 | **Now / Up next stack** | Evolve `Start here` into one featured `Now` decision and a 2–3 item `Up next` list. Categories become secondary batching views rather than peers competing for attention. |
| P0 | **Action-specific controls** | In queue rows, show the action that resolves the item (`Review`, `Approve`, `Trust`, `Analyze`) instead of the catch-all `Open`; preserve `Later` as a quiet escape hatch. |
| P1 | **Active queue rail** | The chip rail should visibly track the selected/visible pile and filter or focus the content. It already has styling support, but needs a genuine interaction model. |
| P1 | **Capacity whisper** | If calendar data exists, add one muted line in the header: `2h 20m clear before 15:00`. Never add a dashboard chart solely to display capacity. |
| P2 | **Completion receipt** | When a batch action is applied, surface a small temporary receipt near the queue: `3 memories approved`. It should be durable enough to read, then fade quietly. |

## Projects and assets

| Priority | Element | Recommendation |
|---|---|---|
| P0 | **Project identity card** | Upgrade the asset monogram into a restrained project tile: monogram, name, type, runtime state, one-line purpose, and one primary next step. No decorative image is needed. |
| P0 | **Launch readiness strip** | In the detail drawer, replace scattered trust/setup notices with a simple ordered strip: `Configured → Reviewed → Ready → Running`. Each stage opens its own evidence or control. |
| P1 | **Project quick-switcher** | Give Assets a `Open project…` search control that works like a compact command palette; it is faster than scanning a full card grid once project count grows. |
| P1 | **Runtime heartbeat** | On running assets, show a quiet green dot plus a human time signal (`responded 12s ago`) instead of visual pulse animation that competes with work. |
| P1 | **Drawer as a workstation** | Keep the drawer, but give it a sticky identity/action header and section anchors: `Overview`, `Readiness`, `Configuration`, `Logs`. |
| P2 | **Grid/list switch** | Preserve the visual grid for browsing; add a dense list view for operational use when many assets exist. Save the preference locally. |

## Knowledge, answers, and memory

| Priority | Element | Recommendation |
|---|---|---|
| P0 | **Trust ladder** | Use the same visual model across knowledge sources, answers, memories and tools: `Draft → Proposed → Reviewed → Active`. It should use text and a dot, not four different pill systems. |
| P0 | **Source trace mini-card** | Where an answer or memory cites evidence, show a compact source row with filename, relevant excerpt, freshness, and `Open source`. This makes governance feel tangible. |
| P1 | **Knowledge explore canvas** | Make Explore a reading/search destination: large search input, suggested questions, and source-aware results. Keep library management visually separate. |
| P1 | **Memory proposal diff** | On a memory card, differentiate `What Metis will remember` from `Why it inferred this`, plus a source link. The decision becomes much safer at a glance. |
| P1 | **Answer confidence treatment** | Confidence should be human language (`well supported`, `mixed evidence`, `needs confirmation`) with the same dot treatment—never an opaque percentage. |
| P2 | **Conflict comparison** | When answers conflict, use a two-column evidence comparison with a concluding neutral question, rather than a warning banner alone. |

## Customers and tools

| Priority | Element | Recommendation |
|---|---|---|
| P0 | **Account pulse** | In the customer header, replace a broad stat-heavy summary with three decision signals: relationship health, next commitment, and last meaningful interaction. |
| P0 | **Customer timeline** | Consolidate notes, actions, wins and updates into an optional chronological activity view. Each event needs a strong type mark and one line of context. |
| P1 | **Research / action split** | Make customer notes a reading surface and actions a decision surface. Their controls, density and headers should feel different. |
| P1 | **Tool capability card** | In Tool Workshop, show capability requests as a small readable permission map—what it can read, write, or execute—before technical hashes and metadata. |
| P1 | **Review inbox** | Combine pending tool definitions, build activation and regression proposals into a single `Needs review` list, then let filters reveal each type. |
| P2 | **Evidence drawer** | `View evidence` should open a side-by-side, source-first drawer rather than expanding a dense card in place. |

## Style changes that will make the whole app feel more finished

1. **Use bloom once, with intent.** Reserve it for the active new-chat action,
   Today’s `Now` card, or an empty-state mark—never several gradients on one
   screen.
2. **Retire rainbow status pills.** Standardise to a dot, plain-language word,
   and optionally a subtle tinted surface only for selected/live states.
3. **Make numbers quieter.** Counts, IDs and timestamps should use mono and
   muted ink; titles and decisions should win the visual hierarchy.
4. **Prefer a real reason over an icon.** A short written reason is more useful
   than decorative metrics, activity waves, or an extra status glyph.
5. **Use motion only for cause and effect.** A card can lift on hover; a newly
   completed action can settle into place. Ambient looping animation should be
   rare.

## Reference shelf

These are source references for individual patterns, not templates to copy.

- [Tasks — Today’s Focus](https://tasks.everway.com/) — focused daily work,
  task context and visible progress.
- [Botrix AI Command Center](https://dribbble.com/shots/27308451-Botrix-AI-Command-Center-Dashboard-Design)
  — agent state and direct operational controls.
- [AI Agent Control Center](https://dribbble.com/shots/27006447-AI-Agent-Control-Center-Dashboard)
  — recommendation and activity modules.
- [Productivity Dashboard UI](https://dribbble.com/shots/27142026-Productivity-Dashboard-UI-Modern-SaaS-Experience)
  — My Day framing and secondary update rail.
- [Linear Intake](https://linear.app/intake) — decision-oriented triage and
  AI-supported routing.
- [Superhuman Split Inbox](https://help.superhuman.com/hc/en-us/articles/46005636204941-Custom-Split-Inbox)
  — intentional grouping and lightweight counts.
