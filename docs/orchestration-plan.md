# Orchestration plan — strong orchestrator, narrow coder, real repo map

Written 2026-08-10, after (a) the four open items from the previous session and
(b) external research into what people actually run successfully for multi-file
builds. This is the design; nothing here is built yet.

## The finding, in one line

Nobody has a verified account of an open-weight model solo-orchestrating a large
multi-file project. What people ship instead is: **a deterministic controller in
ordinary code, a strong model deciding scope, a cheaper model writing one narrow
thing at a time, and a symbol-level repo map so nobody has to hold the tree in
their head.** Three of those four we have or half-have. The fourth (repo map) we
do not have at all, and it is the cheapest of the four.

## Where Metis already stands against each lever

| Lever (from the research) | Metis today | Verdict |
|---|---|---|
| Deterministic controller, not an LLM, wrapping the loop | `control_plane._project_step` is Python. Steps, budgets, refusals, verification rungs and approval gates are all host-side | **Already done.** This is the part people migrate *to*. Keep it; do not adopt a framework |
| Strong planner + cheap executor | `role_chains` has planner/coder/quality with live fallback, but the build's plan call runs as `role="coder"` and the *same* model then writes every file | **Half.** The seat exists and is empty |
| Hard gate between deciding scope and writing code | `explore→act` phases, focused reads refused past an allowance, plan written to `.metis/plan.json` | **Half.** The phases are gated; the *roles* are not — one model does both |
| Narrow, tightly-scoped sub-tasks | `project_focus_path` narrows a drifting turn to ONE owed file, with the exact wording that worked live | **Half.** Reactive (fires after drift) and the instruction is a fixed string, not authored per file |
| Repo map / context management | Flat `file_tree` (≤500 paths), `METIS.md` prose, priority-file excerpts. `code_graph.py` extracts defs/imports/calls but is wired only to the knowledge corpus | **Missing.** Biggest single gap |
| Skip heavyweight multi-agent frameworks | Never adopted one | **Already done** |

Two consequences worth stating plainly:

1. The measured authoring ceiling ("1–2 files reliable, 4+ interdependent files
   it will not commit") was measured with **no repo map and no orchestrator**.
   It is a real ceiling for that configuration; it is not evidence about a
   configuration we have not tried.
2. Every failure mode on record — 22 reads and no write, 36-repeat loops, the
   revamp that died having read 11 files productively — is a *context* failure,
   not an authoring failure. The model was looking for something the host could
   have simply told it.

---

## O1 — Repo map (do first; model-agnostic, no new spend)

**Goal.** Replace the flat file tree in project context with a ranked,
symbol-level map, budgeted in characters, so a model arrives at step one already
knowing what exists and where.

**Why first.** It is the only item that improves every lane at once — hosted,
local, orchestrator, coder — and it costs no new provider and no new gate. It
also directly attacks the dominant failure (reading to the ceiling).

**Seams.**
- `code_graph.py` — already produces `GraphNode` (kind/name/qualname/lines) and
  `GraphEdge` (contains/imports/calls) from stdlib `ast`. Pure, dependency-free,
  unit-tested without a model. This is the extractor for Python, unchanged.
- `project_workspace._snapshot` / `context()` — where `file_tree` is built and
  where the map must be emitted.
- `control_plane._project_step_request` → `project_context` — the consumer.

**Steps.**
1. **Extend extraction beyond Python.** A `symbols.py` with one function per
   language returning the same `GraphNode`/`GraphEdge` shapes:
   - Python → reuse `code_graph` as-is.
   - TS/JS → exported/`function`/`class`/`const fn =` declarations + `import`
     specifiers. Line-oriented, no parser dependency.
   - HTML → `id=` anchors, `<script src>`/`<link href>` references, form actions.
   - CSS → top-level selectors and custom-property definitions.
   The bar is "names and where they live", not a compiler. Wrong-but-cheap beats
   absent; every entry carries its file and line so the model can verify by
   reading.
2. **Rank.** PageRank over the import/call graph, personalised toward
   identifiers mentioned in the request (and, on later steps, in the trace).
   Falls back to degree-ranking when the graph is trivial. This is Aider's
   mechanism and the reason a weaker model can reason about a large repo.
3. **Render under a budget.** `repo_map(root, focus_terms, max_chars)` emits
   grouped-by-file signature lines, highest-ranked first, truncated honestly
   ("… 41 more files not shown"). Budget scales by lane: generous on cloud,
   tight on local, same knobs as `project_reference_max_chars`.
4. **Cache.** Key on the per-file content hash the corpus already tracks;
   invalidate per file, never rebuild the repo.
5. **Wire.** `context()` returns `repo_map` alongside `manifest`; the step
   request sends it; `file_tree` stays but shrinks to a count plus the top
   directories, because the map supersedes it.

**Measurement (this is the point).** Re-run the four benchmark cases per model
(question / 1-line edit / small build / reskin) and record **steps-before-first-
write** and **read count**. GLM's 22-reads-no-write on the perfectly-posed
4-file conversion is the control case. If the map does not move that number, it
is not earning its budget and we say so.

**Risks.** A wrong map is worse than none — so every line is file+line
attributable and the map never claims completeness. Budget creep: the map
competes with `reference_notes` and the trace for the same context, so the
budget is explicit and measured, not "as much as fits".

### O1 — built 2026-08-10

`repo_map.py` (extract → rank → render, pure and dependency-free),
`ProjectWorkspaceService.repo_map` (walks the tree with the manifest's own
ignore rules, caches extraction per file by `(mtime_ns, size)`),
`ControlPlane._project_repo_map` (ranked against the request, the compiled spec,
the focused path and the planned files; rebuilt each step; never fatal), and the
map named in both `PROJECT_AGENT_SYSTEM` and `PROJECT_PLAN_SYSTEM`. Budgets:
`project_repo_map_max_chars` 12k cloud / 5k local, and **0 disables it**, which
is deliberately the before-side of the measurement.

**Four ranking bugs, all found by running it against this repository rather
than against the tests.** Each one produced a plausible-looking map that was
ranked wrongly, which is exactly the failure mode a map must not have:

1. **Locals extracted as declarations.** The TS patterns matched indented
   `const`, so `body`, `preview` and `router` inside components became symbols.
   They drowned the real declarations *and* corrupted the ranking, because a
   hundred files referencing `value` all voted for whichever component declared
   a local by that name. Every pattern is now anchored to column zero or
   `export`.
2. **Every call site counted as a separate vote.** "A depends on B" is one
   fact; counting occurrences gave one module an in-degree of **1,335** and
   first place in every ranking regardless of the request. References are now
   distinct per file.
3. **Attribute calls treated as edges.** `self.db.get(...)` says nothing about
   which class was meant. Because methods were (correctly) excluded from
   attracting rank, `get` became *unambiguous* — and all 47 files calling
   `.get()` on anything voted for the single module with a top-level `get`.
   Only imports and bare-name calls are edges now.
4. **Imports resolved to nothing.** `from . import repo_map` names the *module*,
   which was not a symbol of any file, so the strongest and cleanest signal in
   the graph was silently discarded. Files are now owned by their own filename
   as well as their symbols.

Also: closures are excluded from a file's surface (a method is kept, a function
inside a function is not); no single file may take more than a third of the
budget; private helpers are dropped before public ones when a file is over its
cap; personalisation weights were raised from 4 to 25 for a path the request
names, because at 4 the file the user *named* did not reach the top ten.

**Result on this repository** (94 files, extract 189 ms cold then cached,
rank+render 2 ms): with no request the top of the map is `contracts.py`,
`config.py`, `api.py`, `database.py` — correct for "what does everything depend
on". For *"fix the role ladder in the model control panel"* it is
`contracts.py`, **`model-control.tsx`**, `model_preference.py`,
`model_provider.py`, `control_plane.py` — the correct working set, and none of
it read from disk by a model.

### O1 — measured live, 2026-08-10

Two cases against `zz-chunktest-logivity` (18 files) on the Ollama lane, coder
`deepseek-v4-pro:cloud`, map on vs `project_repo_map_enabled=false` as control.

| Case | Steps | Reads | First write at step | Outcome |
|---|---|---|---|---|
| 4-file conversion, **map off** | 20 | 20 | — | read to the ceiling, nothing staged |
| 4-file conversion, **map on** | 21 | 21 | — | read to the ceiling, nothing staged |

The one-file edit was then run **four times per arm**, because the first pair
suggested the map halved it and a third run did not reproduce that. Run-to-run
variance on this loop is large and already on record; one pair is an anecdote.

| 1-file edit (n=4 each) | Steps | Reads | First write at step |
|---|---|---|---|
| **map off** | 9.75 (8–11) | 7.75 (6–9) | **6.50 (5–8)** |
| **map on** | 8.75 (7–10) | 7.00 (5–8) | **4.50 (2–6)** |

**The map brings the first write about two steps earlier on average, and the
distributions overlap** (map-on's worst first-write is 6; map-off's best is 5).
Every run in both arms reached an approval with the sandbox clean, so the map
changes how fast the model gets there and not whether it does.

That is a real but modest effect, and it is *not* the halving the first pair
appeared to show. Worth stating plainly because the first pair was written down
before it was checked.

**On the four-file conversion the map changes nothing at all** (21/21/0 vs
20/20/0). The map was confirmed present in the request — 1,803 characters of it
— so this is a real negative result, not a wiring failure. The model planned it
*correctly* both times (`intent=edit`, the exact four files, sensible acceptance
scenarios), was narrowed by the host to ONE file, and still would not write it.

**What that separates, which is the useful part.** The map fixes a *context*
problem and this case does not have one: the model is not short of information,
it is free to keep choosing `read` and nothing compels it to write. That is a
**control-flow** failure, and it is exactly what O2 exists to remove — a coder
that is handed one file, a brief and the pre-fetched contents, and whose tool
roster does not contain a read. So the ordering in this document was right for
cost and wrong for the headline: O1 is real and it is not sufficient, and the
four-file conversion stays open until O2.

**A stale-context bug the map found on the way.** Three files in the manifest's
`file_tree` — `app/main.py`, `app/static/index.html`, `app/static/app.js` — do
not exist on disk. The manifest snapshot had gone stale, so every build step was
being told three files exist that do not. `repo_map` walks the tree live and
does not inherit it. Worth fixing in the manifest too.

---

## O2 — Orchestrator / coder split (the highest-value structural change)

**Goal.** One model decides scope and never writes; another writes exactly one
file at a time against an explicit instruction and never decides scope. The
controller between them stays plain Python.

**Why.** Independently cited as the biggest structural lever (one framework
reports +12.2% from the split alone; Aider's architect/editor is the documented
SOTA version). And we already proved the *effect* by hand: the four-file
conversion GLM would not commit to, split into one-file messages, produced all
of it with every verification rung clean. O2 is that, automated.

**Seams.**
- `role_chains["planner"]` — the empty seat.
- `model_provider.project_plan_files(..., role="coder")` — becomes `planner`.
- `project_focus_path` + the `attention` string in `_project_step_request` —
  becomes the orchestrator's authored instruction.
- `_verify_staged_changeset` — already the per-file feedback the orchestrator
  needs.

**The loop (deterministic, host-side).**
```
plan            = orchestrator.plan(request, repo_map, trace)      # files + intent + scope + strategy
for path in plan.files:                                            # controller owns this for-loop
    brief       = orchestrator.brief(path, plan, staged, findings) # what to write, what to import, what to reuse
    context     = host.gather(path, repo_map, brief.reads)         # host pre-fetches; coder should need 0 reads
    result      = coder.write_one(path, brief, context)            # ONE file, no scope decisions
    findings    = host.verify(result)                              # existing rungs: parse → wiring → sandbox
    verdict     = orchestrator.review(path, result, findings)      # accept | retry-with-note | revise plan
```

**Hard gates, enforced by the host and not by prompting.**
- The orchestrator is given **no write tools** — it cannot stage a file even if
  it tries.
- The coder is given **no `revise_plan`** and a write target pinned to the single
  owed path (`project_write_pin` already does exactly this).
- The coder's reads are pre-empted: the host sends the file's current contents
  and the ranked map slice, so a read is an exception, not the default.

**Strategy is the orchestrator's call, not a threshold.** The plan gains
`strategy: one_at_a_time | together`. A model that can write six files in a turn
(deepseek did) should not be forced through six round-trips; a model that stalls
on two should never be asked for four. Today that decision is made by a
behavioural trigger after failure; under O2 a strong model makes it up front and
the behavioural trigger stays as the safety net.

**Cost shape.** The orchestrator makes few, small calls (a plan, one brief per
file, one review per file — all short). A premium model in that seat is cheap;
the tokens are in the coder, which stays on the cheap lane. This is the entire
economic argument for the split and it should be verified with real numbers on
the first run.

### O2 + O3 — built and measured live, 2026-08-10

**O2.** `ProjectDirectionV1` (flat, locally decodable), `project_direction` on
every transport, `ControlPlane._project_direct` (the controller),
`_prefetch_for_coder`, `_directed_attention`, `_planner_aliases`.
`GRAPH_SCHEMA_VERSION` 6 → 7.

The gate, as enforced rather than prompted:
- The **orchestrator** answers a structured call with **no tools at all**, so it
  cannot write. It runs in the `planner` seat — a role the ladders always had
  and builds had never used, because every project call ran as `coder`.
  `_planner_aliases` routes that one call to the planner rung; without it
  `_provider` is global to the run and the planner ladder was decorative.
- The **coder** gets one path (the existing focus/write-pin machinery, reused
  rather than duplicated), the orchestrator's instruction in place of the fixed
  sentence, and **a read allowance of zero** instead of two.
- The **host** pre-fetches the target file plus whatever the orchestrator named,
  so closing reads is fair rather than merely strict.
- The controller validates: a path outside the plan is refused, a per-file
  attempt cap turns repair into a bounded loop, and a lost direction degrades to
  the undirected loop. `done` returns the turn to the ordinary finish path — an
  orchestrator that could end a turn would be deciding scope *and* completion.

**O3.** `ClineModelProvider`, a fourth lane key throughout (`RoutedModelProvider`,
the preference store, contracts, the API, the web panel). Three gateway quirks,
each found by calling the real endpoint: the reply is **wrapped in `data`**, the
budget is **`max_completion_tokens`** (with `max_tokens` the reasoning trace eats
it and the content returns empty), and there is **no `/models` endpoint**. 403 is
an unsubscribed model and 401 a bad key, and neither is retried.

**The result — the case that had never once written a file.** Orchestrator
`anthropic/claude-opus-4.5`, coder `cline-pass/deepseek-v4-pro`:

| 4-file conversion | Steps | Reads | Writes | First write |
|---|---|---|---|---|
| map off, no split | 20 | 20 | **0** | never |
| map on, no split | 21 | 21 | **0** | never |
| **map on + split** | **9** | **4** | **3** | **step 4** |

Three files staged (`app/main.py`, `app/static/index.html`, `app/static/app.js`),
**13 sandbox checks ran, 0 errors**, and an approval card.

**Two honest limits.** It staged **three of four** files — `requirements.txt` was
never directed. And the orchestrator said `done` once *before* the plan was
satisfied; the loop correctly returned to the ordinary path rather than ending,
the coder called `revise_plan`, and directions resumed — so the degrade worked,
but a premature `done` is a real behaviour to watch. The card is also honest that
the sandbox could not import the app (`langchain_core`, `openpyxl`, `pandas` are
declared but not baked into the verify image), so **0 errors means "nothing I
could check is wrong", not "it runs"**.

**De-risking the spend.** Prove the mechanism with **GLM-5.2 as orchestrator**
(fast, cheap, already keyed, and the best planner on record here) and
deepseek-v4-pro / qwen3-coder as coder, on the exact 4-file conversion. If the
mechanism works with GLM in the seat, a stronger model can only improve it — and
we will know what we are buying before we buy it.

---

## O3 — A strong-model lane (only after O2 shows the mechanism works)

**Goal.** Make a premium orchestrator available as a first-class lane.

**Seams.** `RoutedModelProvider._selected` dispatches on `model_aliases["_provider"]`
(`local` | `oci` | `cohere`). A fourth key is a contained job: one provider
class, one config block, one capability record, one entry in the lane picker.

**Recommendation.** **OpenRouter first** — one key, every model including
Claude/GPT/Gemini, which makes the orchestrator seat an A/B rather than a
commitment. Move to a direct API later if one model wins clearly. Note that a
strong closed lane already exists (OCI/Grok), so a first orchestrator experiment
needs no new account at all.

**Non-goal.** Replacing the coder lane. The research is unanimous that the flow
runs one direction: strong model plans, cheap model executes. The coder stays on
Ollama/local.

---

## O4 — Intent without a regex

**Status: half-shipped already.** The narrow build predicate was removed from the
plan gate; the prefilter is now the broad "is this about code at all" question,
and the model's declared `intent`/`scope` decides the rest.

**Finish it, and do not replace it with a classifier.** Embeddings-plus-threshold
is still a fixed classifier with hand-picked anchors — it fails *differently*,
not less, and adds a dependency and latency for a judgment the orchestrator is
already making. The correct end state is "do not classify before you ask".

**Steps.**
1. Audit every remaining regex call site (`is_new_application_request`,
   `wants_web_ui`, `_writes_files`'s fallback, the scaffold trigger) and, for
   each, either feed it from the declared plan or demote it to a fallback used
   only by providers that cannot declare (scripted/deterministic).
2. Under O2 the orchestrator declares intent, scope and strategy in one object;
   everything downstream reads that object.
3. Keep one broad prefilter, documented as a cost gate and nothing more.

---

## O5 — Model panel redesign (independent, small, owed)

The panel grew a "Roles & backups" section onto an already-tall column and now
runs off the bottom.

- One calm summary line per role, collapsed: `Coder · deepseek-v4-pro → qwen3-coder`.
- Click to expand exactly one row at a time; plain text rows, no boxed radio grid.
- Lane choice as a single segmented control, not four cards.
- Bounded height with an internal scroll and a sticky action row, so it can never
  clip its own save button again.
- One accent colour, generous whitespace, reads like a sentence.

Independent of everything else; can ship first because it unblocks nothing and
is visible immediately.

---

## O6 — Extend the tool-call repair layer

**Status: partially shipped** — argument-key synonyms (`args`/`params`/`inputs`)
are coerced in `contracts.py` before validation, and an argument-shape refusal
pins the next step's grammar to the same tool.

**Extend, deterministically, before a malformed strike is counted:**
- Strip code fences and prose wrappers around JSON.
- Repair trailing commas, single quotes, unescaped newlines in string values.
- Normalise paths (leading `./`, absolute paths inside the project, `\` → `/`).
- Fuzzy-match argument names against the tool's real schema (edit distance 1–2).
- Auto-correct `create_file` on an existing path → `apply_patch` shape and back,
  which is the single most common shape error on record.

Cheap, entirely host-side, and it raises every weak lane at once. Each repair
emits an event so we can *measure* which repairs fire — a repair that never fires
should be deleted, and one that fires constantly is telling us about a prompt bug.

### O6 — built and measured 2026-08-10

`tool_repair.py`, called from the two places every lane decodes through:
`step_from_function_call` (OCI, hosted Ollama, Cohere) and the local grammar's
`to_step` conversion. Repairs: code-fence and prose unwrapping, trailing commas,
Python literals, key renaming against the tool's *own* accepted keys, and type
coercion (`"12"` → `12`, `"true"` → `True`, a bare string where a list is
declared, a number where a string is declared) plus path tidying (backticks,
quotes, `\`, `//`, leading `./`).

Three rules kept it a shape fix rather than a licence, and each is pinned by a
test: an unsupported **tool** is still refused; a key is only renamed toward a
key *that tool actually accepts* (so `create_file` never grows an `original`); and
genuinely ambiguous keys — `patch`, `code`, `line` — are deliberately **not**
mapped, because a repair that is right most of the time is worse than a refusal
that is right every time. Unknown keys are kept rather than dropped, so the
workspace still refuses them by name.

Every repair is emitted on `project.agent_step` as `repaired`.

**Measured: across seven live build turns on the hosted lane
(deepseek-v4-pro:cloud), zero repairs fired.** That lane emits clean tool calls.
So O6 is currently insurance, not a fix — its value is for the weak local
grammar lanes and Cohere, which this measurement did **not** exercise. Worth
re-checking the `repaired` counter after any run on those lanes before deciding
whether the layer earns its keep.

---

## Parked: CodeAct

Emitting executable code instead of JSON tool calls has real evidence behind it
for weak tool-callers, but it is a second execution path and a new sandbox
surface, and we already have a working decode contract plus a capability record
that gates models that cannot do tool calling. Revisit only if the coder lane is
still malformed-heavy *after* O6 — at which point the repair events will tell us
exactly which failures CodeAct would have avoided.

---

## Sequence

1. **O5** — model panel. Small, visible, owed, blocks nothing.
2. **O1** — repo map. Model-agnostic, improves every lane, benchmark before/after.
3. **O6** — repair layer. Cheap, measurable, raises the coder floor before O2 leans on it.
4. **O2** — orchestrator/coder split, proved with GLM in the orchestrator seat.
5. **O3** — premium lane, bought only after O2 shows what it would buy.
6. **Re-benchmark** the four cases across lanes and write down the new ceiling.

## What "done" means

Not full autonomy. The honest target, and the one the evidence supports, is: a
deterministic controller, a strong model owning scope, a cheap model writing one
narrow file at a time, a real map so neither has to guess, and the human
checkpoints Metis already has at the approval card. The measurable claim is that
the **4-file interdependent conversion completes in one turn** — the exact task
that is on record as failing on every lane today.
