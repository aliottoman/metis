# UI refinement — September 10, 2026

This pass keeps Metis’s warm paper, plum, lilac, and coral identity while making
the interface consistent across its workspaces and fixing everyday interaction
failures. It includes the web interface, contextual source links, the Today
briefing, asset workflows, task execution visibility, and native macOS framing.

## Design and interaction rules

- `tokens.css` owns theme colors. Older neutral, status, glass, and audio tokens
  now reference the canonical palette, so they follow both explicit and system
  appearance choices. Knowledge panels, secondary buttons, status badges, and
  picker surfaces no longer assume a light background.
- Workspaces share page headers, spacing, surfaces, and readable descriptions.
  Appearance is a compact three-choice control and synchronizes across windows.
- Chat fills the available height. The conversation scrolls independently while
  the composer remains visible, including with a tall draft on a small screen.
- Phone navigation and conversation history open as dismissible dialogs, contain
  keyboard focus, and restore it when closed. Resizing into the phone layout
  closes history without changing the saved desktop preference.
- Closed controls stay outside keyboard navigation. Dialogs share focus handling;
  tabs and grouped pickers keep the selected record aligned with the visible row.
- Background refreshes do not overlap. Failed requests show recovery paths rather
  than misleading empty or all-clear states.

## Workflow repairs

- Queued messages retain their attachments. Queue editing and failed sends preserve
  newer drafts. IME composition cannot accidentally send a message; file inputs can
  reattach the same file after removal.
- Knowledge refreshes preserve unsaved profile and connection edits. Settings
  reflects confirmed saves and makes failed preference loads visible.
- Customer navigation follows the account and tab in the URL; late responses cannot
  display another account’s actions. Deleting an account returns to the dashboard
  before refreshing.
- Meeting switches do not expose stale recording details. Failed title and transcript
  edits remain available to retry; processing continues to refresh across recordings.
- Audio resets on source changes and defers transcript seeking until metadata is ready.
- Loading, missing-page, and recoverable error screens use the same interface.

## Expanded workflows

- Assets has a searchable, filterable library with saved pins and dedicated asset
  workspaces. Preview, settings, and logs have durable URLs. Starting or stopping
  an asset remains visible while navigating, stale catalog requests cannot erase
  newer launch results, and failed launches surface their actual error.
- Contextual follow-ups can recover an unambiguous account from the conversation’s
  user messages. Comparisons, unrelated topics, and mutation requests do not inherit
  an account implicitly. Citations link to recorded customer sources, uploaded
  documents, and persisted project conversations without inventing source locations.
- Today shows recorded priorities, neglected commitments, and reviewed opportunity
  signals with evidence, reasons, and prepared next steps. Deferrals and completed
  commitments update the queue. Opening or refreshing the factual briefing does not
  invoke a model; generated narration remains an explicit action.
- Substantial tasks have one overview containing the published plan, real progress,
  verification findings, available outputs, pending questions and approvals, and
  recorded decisions. Progress does not use invented percentages or treat a closed
  connection as proof of completion.
- Task steering focuses the composer and explains when the follow-up will send.
  Stopping restores any queued follow-up to the draft so cancellation does not
  automatically launch more work. The task drawer uses the shared focus handling on
  narrow screens.
- Bounded milestone retention keeps plans, decisions, checks, and outputs available
  beyond the latest 300 activity events. Actionable decision selectors use the
  retained current-run events and reconcile later answers and approvals, including
  those made in another window.
- The macOS window leaves room for native traffic lights and uses the compact rail
  consistently. Chat and asset workspaces have an explicit height chain. Native page
  animations and route view transitions are disabled to address a WKWebView rendering
  issue. Chat and the task overview now pass native visual checks after reloading.

## Verification

- 185 web tests pass with no failures or skips. Coverage includes queued drafts,
  grouped pickers, polling, appearance, asset lifecycle races, daily brief actions,
  safe source links, task progress, and requests surviving long event replays.
- Focused backend checks pass: 29 context/source-link tests, 7 Today briefing tests,
  22 attention tests, and 19 asset launcher tests, for 77 focused checks in total. This is not a claim
  that the complete backend suite was rerun.
- The final production build and TypeScript checks pass, including the retained
  pending-decision fix and responsive Assets adjustments. Whitespace checks pass.
- Browser render checks covered Chat, Today, Customers, Assets, Knowledge, Memory,
  Tool Workshop, Answers, Sizing, Settings, Meetings, Interviews, and Agents.
- Visual review covered light and dark appearance. Responsive checks covered a
  390px phone viewport and a 390 × 500 short viewport with a 25-line draft.
- Live browser checks confirmed that Knowledge refresh preserves an unsaved draft,
  customer and Knowledge layouts fit the phone viewport, the composer remains
  visible, and navigation/history dialogs close with Escape and restore focus.
- Final Assets checks confirmed search survives workspace navigation and Back,
  preview/settings/logs retain the asset title, focus mode restores navigation, and
  cards and launch controls fit desktop, intermediate, and phone widths.

## Installed state and remaining verification

- The backend was restarted with the updated code. The production web build was
  updated, and the native bundle was installed with the prior bundle backed up at
  `/tmp/Metis-before-refinement.app`. The final production build includes all
  web changes and is served on port 3000; the development preview remains on 3001.
- Native QA found Chat content in the accessibility tree while its pane appeared
  blank in WKWebView. The height and native transition fixes are installed. After
  selecting the installed bundle and reloading, Chat, conversation history, the
  composer, and an existing task's execution overview rendered correctly. Assets
  and the collapsed rail also passed native visual checks.
- A live asset launch surfaced a pre-existing demo dependency conflict. The launch
  failure is visible in the interface, but that demo’s dependencies remain a
  separate unresolved limitation; passing launcher tests does not prove every
  project can start successfully.
- Verification used local service reads, reversible interface interactions, and
  isolated automated tests. It did not exercise new paid model runs, live microphone
  sessions, production deployments, approval decisions, or destructive changes to
  user records.

## Unified workspace refinement

- Extended the Assets design throughout Customers, account notes, Meetings,
  Interviews, Knowledge, Memory, Answers, Agents, Tools, Sizing, Today, and Settings.
  Pages share typography, headers, search fields, cards, spacing, and action hierarchy.
  Warm paper and plum remain the base; soft mint and amber accents and a subtle
  dotted header texture provide variation without separate visual themes.
- Customers opens on a searchable, sortable account catalog with portfolio metrics
  and filters. An account opens as a focused workspace with clear return navigation,
  keyboard-accessible section tabs, and direct links to specific records.
- Account notes now have an index, a reading pane, search, pinned filters, and an
  editor. Unsaved drafts survive section changes and leaving/reopening an account
  during the visit. Revision-aware saves protect newer edits and avoid duplicate
  submissions when a successful save is followed by a failed refresh.
- Meetings has a recording library with status filters and a focused recording
  workspace. Overview/transcript switching preserves edits, recording and turn
  links are reflected in the URL, and speaker edits have explicit Save/Cancel.
- Knowledge separates Sources, Personal profile, Connections, and Explore. Source
  discovery is searchable; profile edits survive tab switches and refreshes.
  Memory uses a searchable note collection with a dedicated review pane and an
  editor that keeps drafts when closed.
- Answers and Agents have search; Answer decisions guard against overlapping
  requests, and copying handles unavailable clipboard access. Agent briefs stay
  mounted while browsing the studio. Settings offers direct section navigation.
- Native review replaced the platform-styled account sorter with the shared
  keyboard-accessible menu. Account tabs scroll cleanly on phones, and mixed-height
  Agent fields align at the top.
- A final native check found residual Chat entrance animations retaining their
  transparent first frame. Native Chat content and history now appear immediately;
  loading indicators and interactive state feedback keep their existing motion.

### Verification of this pass

- All 188 web tests pass with no failures or skips, including three additional
  customer draft revision/race checks. Production compilation and TypeScript pass.
- Browser checks cover the revised pages in light and dark appearance, with 390px
  phone layouts checked for horizontal overflow. Account sorting was exercised.
- Live draft checks confirmed account note recovery across tabs and account
  navigation, plus profile recovery across Knowledge tabs and refresh. Temporary
  test drafts were removed without saving user records. Appearance was restored
  to System after visual checks.
- Customers and Meetings were visually checked in the installed Mac app with
  its collapsed sidebar and native traffic lights. After the final production
  reload, Chat's welcome, history, and composer also passed native visual review.
- The final production build, including the account sorter and native Chat fix,
  is served on port 3000 and was reloaded in the installed app. The development
  preview on port 3001 remains available on the refreshed Customers catalog.
- The recording library and Agent studio contain no records in the current local
  data, so their detail workflows were reviewed in code rather than exercised
  against a live recording or deployed agent. No new recording, microphone session,
  model run, memory/answer decision, or external deployment was initiated.
