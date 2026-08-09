# Project agent unification — understanding-first builds on Grok + Command A+

Status: proposed · Owner: Ali · Scope: the scoped **project (build) loop**, plus the
context it runs against. Target build providers: **Grok (OCI Responses)** and
**Cohere Command A+**. Local MLX is gated out of project builds.

---

## 1. Problem

Metis has **two agent architectures**, and only one of them understands intent.

- **Planner path (unscoped chat)** — a genuine understanding-first design: semantic
  routing (`PlanEnvelopeV1`), typed steps (`PlanStepV1.kind = respond|tool|build_tool|validate`),
  `ask_user` elicitation, risk levels. This is where recent investment went.
- **Project path (scoped)** — the *older* design. `_route_after_retrieve`
  (`control_plane.py:1767`) sends any turn with a `_project_id` straight into the
  bounded step-loop, **bypassing the planner entirely**. Inside the loop, whether a
  turn is a "build turn" is decided by a keyword regex, `is_project_build_instruction`.

The regex is the visible symptom; the two-architecture split is the disease. A real
request — *"completely revamp this asset to use the OCI Command A Vision model with
our FastAPI structure"* — contains none of the regex's build verbs (`build|create|
make|scaffold|…`), so it was treated as conversation. The model (Grok) did the
reasonable thing: it asked to clarify. But the project loop has **no channel for
asking** — the only non-tool outcome is "finish" — so the clarification surfaced as
a terminal finish, and the host stapled on the "*No file changes were staged*" honesty
footer. A sensible clarify-first read looked like a crash.

Three structural gaps, independent of the regex:

1. **No semantic intent step on the project path.** The one understanding component
   (the planner) never runs once a project is scoped.
2. **Asking is not a first-class action.** `list_files, search_code, read_file,
   create_file, apply_patch, replace_lines, run_check, inspect_api` — no `respond`,
   no `ask_user`. "Talk to the user" and "finish the build" share one channel.
3. **Finish is judged by the user's wording, not the agent's claim.** The
   empty-finish pushback and the footer key off `is_project_build_instruction`, not
   off whether the model claimed work it didn't do.

---

## 2. The insight that makes this small

Committing builds to **Grok + Command A+** collapses most of the feared complexity.

The entire grammar-crutch apparatus — `ProjectBuildStepWireV1` (a schema that cannot
express `status=complete`), the flat wire schema, `write_pin`/`retry_tool` grammar
pinning, the `build_turn` narrowing — exists **only for the MLX local decoder** (the
`else` branch in `model_provider._project_step`). It never touched the hosted path.

The hosted path is *already* an understanding-first tool-calling loop:

```
_project_step_hosted (model_provider.py:2082)
    → provider emits a real function call
        Grok:   OCI Responses "function_call" item        (speaker="Grok")
        Cohere: _cohere_tool_calls(message)                (Command A+)
    → step_from_function_call(name, arguments, …) (:1237)  ← single normalizer
    → host executes one tool, verifies, loops
```

Both providers already converge on one step type. So "unify onto the planner spine"
for these two models is **not a new loop** — it is *completing the loop we have*:

- give it the planner's `respond` + `ask_user` actions and the elicitation
  suspend/resume machinery,
- delete the regex from control flow and make **finish claim-based**,
- surface **environment + a focus hint** into its context,
- keep the project **safety envelope** (staged overlay → verification rungs → batch
  approval) exactly as-is.

The MLX grammar pipeline stays put, quarantined behind the build-capability gate — it
is simply not part of the build path anymore.

---

## 3. Target architecture

One understanding-first agentic loop. The model, given project context and a complete
tool set, chooses each step. The host owns execution, verification, approval, and a
provider-capability profile. Build / edit / answer / ask / run are **emergent from
tool choice**, never pre-classified.

```
                    ┌─────────────────────────── project turn ───────────────────────────┐
 scoped request ──▶ │  understand (context: manifest, METIS.md, env, focus, file tree)    │
                    │        │                                                             │
                    │        ▼   model picks ONE typed action per step ───────────────┐    │
                    │   explore: list_files · search_code · read_file · inspect_api   │    │
                    │   mutate : create_file · apply_patch · replace_lines            │    │
                    │   run    : run_check                                            │    │
                    │   talk   : respond · ask_user   ◀── NEW, first-class            │    │
                    │        │                                        (loop)──────────┘    │
                    │        ▼                                                             │
                    │   finish  ── claim-based contract (§5) ──────────────────────────▶  │
                    └────────────────────────────┬────────────────────────────────────────┘
                        acted → staged overlay → verification rungs → batch approval → apply
                        responded → publish answer          asked → suspend (elicitation) → resume
```

**Shared with the planner spine:** the typed-step vocabulary, `respond`/`ask_user`,
and the elicitation interrupt/resume nodes. **Project-specific:** the code tool set,
the staged-overlay terminal, the verification rungs, and R-level risk gating. *Same
brain, code-tool hands, its own commit path.*

---

## 4. Provider scope & the build gate

**In scope for builds:** Grok via OCI Responses (`allow_oci_responses=true`,
`oci_responses_project_id`), and Cohere **Command A+** (`cohere_api_key`,
`cohere_model="command-a-plus-05-2026"`). The three project modes map to these:
`grok_continuous` (Grok leads each step), `cohere_continuous` (Command A+ leads),
`grok_bootstrap_local` — **retired for builds** (or redefined as Grok-led; it must not
put MLX in the driver's seat of a build).

**The gate — refuse at selection, reusing the capability record.** Extend the pattern
in `model_preference.py` (`HOSTED_MODEL_TOOL_CALLING` / `hosted_model_capability_error`)
with a **build-capability** predicate: *a project build requires a strong
tool-calling provider (Grok or Command A+).* When a build turn resolves to any other
model, return a clean, deterministic message — same shape as the existing "no project
open" guidance — pointing the user at the two supported providers. Enforced
server-side; surfaced in the web mode picker (disable build modes for incapable
selections) so it never gets that far in the UI.

MLX keeps the old grammar loop for whatever non-build project chat we still allow, but
never drives a build.

---

## 5. The finish contract (claim-based)

A turn ends in exactly one of three ways. The regex plays no part.

| Outcome | Trigger | Terminal | Footer? |
|---|---|---|---|
| **Acted** | the turn staged files and/or ran checks | staged overlay → verification rungs → batch approval card | no |
| **Responded / Asked** | the turn's final action was `respond` or `ask_user` | publish the answer, or **suspend** on the question via the elicitation node, resume on reply | **no** |
| **Claimed-empty** | the model declares work done / files written, but nothing was staged | reject the finish, hand the fact back once, bounded retry | the honesty note, now correctly scoped |

The footer stops being a blunt "nothing staged" stamp. It fires only on the genuine
fabrication case — *the model said it wrote files and it did not* — which is detected
by comparing the model's own declared action to the overlay, not by scanning the
user's prompt for verbs. `respond` and `ask_user` are legitimate non-empty outcomes.

---

## 6. Context pass (pull-based + focus + environment)

Keep the pull-based model: the agent **pulls** context via tools against the working dir
(already true — `list_files`/`search_code`/`read_file`; it is why Grok could name the
current Streamlit + Gemma stack). Two additions:

- **Environment surface.** `projects.context()` today returns `project_name`,
  `manifest`, `metis_md` (≤40k), and the verification recipe. Add the run/build
  environment from the asset runner — `entrypoint`, `buildCommand`, `launchCommand`,
  `envKeys`, `url` (`AssetV1`) — so the agent builds something that actually runs
  under the harness that will run it. Values of secrets never cross; **keys only**.
- **Optional focus hint.** Let the user point at a file or subdir as a *bias*, not a
  hard scope — "this is the part I mean." Carried as a new optional field on the chat
  submit payload alongside `project_id`/`project_mode`; injected into the opening
  context as "focus:", never as a wall that hides the rest of the tree.

Explicitly **out of scope (stretch):** feeding the *building* model an image as
multimodal design input (e.g. "rebuild to match this screenshot"). That requires the
project loop to accept image inputs end-to-end; note it, defer it. The built apps are
multimodal; the building models write code.

---

## 7. Change map by seam

- **`contracts.py`** — add `respond` and `ask_user` to the project tool catalog
  (`PROJECT_TOOL_REQUIRED_ARGUMENTS` / notes / argument properties). `respond`:
  `{message}`. `ask_user`: `{question, options?}` — reuse the `question`/`options`
  shape already on `PlanEnvelopeV1`.
- **`model_provider.py`** — `step_from_function_call` maps the two new tools to step
  kinds for **both** Grok and Cohere; `_project_step_hosted` returns them. Hosted path
  no longer consults `build_turn`. MLX branch unchanged, behind the gate.
- **`control_plane.py`** — generalize the elicitation nodes (`ask_user_prepare`,
  `ask_user_interrupt`, the `awaiting_kind` suspend/resume) so the **project loop** can
  suspend on a question, not just the planner. Remove the `is_project_build_instruction`
  gates at the file-planning and premature-finish sites; replace with the §5 contract.
  `_route_after_retrieve` is unchanged — scoping still enters the loop; the loop just
  stops pre-classifying.
- **`model_preference.py`** — the build-capability gate (§4).
- **`project_workspace.py`** — `context()` gains the environment surface + focus (§6).
- **`apps/web`** — mode picker disables build modes for incapable models; a focus
  picker (file/subdir) on the composer; optional environment display. Submit payload
  gains `project_focus`.

---

## 8. Migration sequence

1. **Baseline** — run the §9 project as a build on the **current** system, once on
   Grok and once on Command A+. Capture: did intent route correctly, did it ask when
   it should, how many steps, malformed rate, did it finish, did verification pass,
   did it run. This is the reference the change is measured against.
2. **Action space** — add `respond`/`ask_user`; wire the project loop into the
   elicitation suspend/resume. Tests: a clarify turn suspends and resumes; a `respond`
   turn publishes with no footer.
3. **Claim-based finish** — replace the regex gates with the §5 contract. Tests: a
   "revamp"-phrased build acts; a claimed-empty finish is rejected once.
4. **Build gate** — capability predicate + web mode gating. Tests: a weak model
   selected for a build is refused with the guidance message.
5. **Context pass** — environment surface + focus hint. Tests: env keys present in
   context; focus biases without hiding the tree.
6. **A/B** — re-run the §9 project on both providers; compare to the baseline on the
   same rubric.

---

## 9. Acceptance test — a real multimodal build

**Project: "Screenshot-to-Spec Studio"** — the closest thing to how this gets used.

Upload a screenshot of any app UI → the built app calls the **OCI on-demand Command A
Vision** model to decompose it into a component tree, extract the color palette and
type scale, and lift the copy → it generates a structured design spec **and** a
side-by-side "rebuilt in our design language" preview. FastAPI backend, a small web
UI, real vision integration, persistence of past analyses.

Why it is a proper test: the image has **many elements** (multi-region vision
reasoning), the build spans **multiple components** (upload route, vision client,
extraction/normalization, spec generator, preview renderer, storage, UI), and it
mirrors the user's actual workflow (vision + FastAPI + Cohere design language). It is
large enough to exercise plan → multi-file build → wiring → run, and bounded enough to
finish inside the step budget (~12–18 files).

> Playful alternative if a change of subject is wanted: **"Larder"** — snap your
> fridge/pantry, Command A Vision detects every ingredient in one frame, and the app
> proposes recipes. Same multimodal-multi-object + multi-component shape.

**Acceptance criteria (both Grok and Command A+):**

- The request routes to the build path **without** relying on build-verb keywords.
- The agent **asks** at least one good clarifying question through `ask_user` (not a
  terminal finish), and resumes cleanly on the answer.
- It plans, then stages a coherent multi-file changeset; verification rungs
  (parses → wires → conforms → runs) pass, or fail with the errors on the approval card.
- The built app **runs** under the harness and the vision call returns structured
  output.
- No spurious "nothing staged" footer on any conversational/clarify turn.
- Per-provider notes: step count, malformed-reply rate, where Command A+ diverges
  from Grok (it needs more tool-call massaging — `_cohere_tool_calls`, citation
  stripping, thinking-text; watch the two new tools there first).

---

## 10. Risks / open questions

- **Cohere tool-calling robustness.** Grok's cleanliness will not transfer for free.
  Validate `respond`/`ask_user` on Command A+ early; it is the likelier fumble.
- **Long builds.** Multi-file vision apps can approach the step budget and the
  hosted-timeout wall. Keep the batch-approval-on-step-limit escape hatch.
- **`grok_bootstrap_local`'s fate.** Confirm it is retired-for-builds vs.
  redefined-as-Grok-led; it must never seat MLX in a build.
- **Multimodal build input** (screenshot → building model) is deferred; note if the
  test makes its absence painful.
