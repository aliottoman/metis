# Architecture and invariants

## Trust boundaries

The FastAPI process is the trusted control plane. Ollama is reachable only by the model broker. Generated implementations execute only through the sandbox adapter, which deliberately has no route to Ollama. A model-backed skill is therefore a declarative workflow: the host performs typed LLM steps and passes only their validated output to deterministic sandbox steps.

Uploaded text is evidence, never authority. It cannot approve an action, widen a filesystem grant, enable networking, add a dependency, or directly enter long-term memory.

The planner receives at most 12,000 characters of attachment text, labelled as an
untrusted excerpt, plus a fixed vocabulary of host-derived document-shape signals
such as `project_documentation`, `software_components`, and
`component_relationships`. A head/tail excerpt preserves both introductory and
late configuration context, and its truncation is recorded as a run event. These
signals may disambiguate an explicit user instruction such as “build what this
README describes”; they cannot initiate work on their own. Capability availability
comes only from the registry, and the trusted control plane recalculates route and
risk after every model plan.

## Durable run protocol

Each conversation has a stable domain `thread_id`, and each user message creates a
distinct `run_id`. Internally, the LangGraph SQLite checkpointer uses the
composite storage key `conversation_id:run_id`; LangGraph normalizes the root
checkpoint namespace, so the composite key prevents concurrent runs in one
conversation from sharing checkpoints. Domain events still expose the
conversation ID as `thread_id`. Events are committed to the outbox before they
are offered over SSE. Clients resume with `after=<sequence>` and deduplicate by
`(run_id, sequence)`.

Approval creation, graph interruption, and the approved side effect remain separate operations. Side effects use stable action IDs and persist their result so process restarts cannot repeat them.

## Capability lifecycle

Capabilities move through `draft`, `quarantined`, `evaluated`, `approved`, `active`, and `deprecated`. A version is content-addressed and immutable. Activation only updates the registry pointer, and rollback selects a previously approved hash.

Every portable bundle includes AgentSkills instructions, a typed Metis manifest, a declarative workflow, implementation source, tests, and evaluation fixtures.

## Provider boundary

Model calls use versioned request/result contracts. The local implementation targets Ollama today. A later OCI planner may return a typed plan, but the local control plane will still resolve tools, recompute risk, request approvals, and execute every action.

Only one heavyweight call enters the broker at a time. Structured planner,
architecture, and code responses have schema-specific output budgets; every
call has a wall-clock limit and malformed output gets one repair attempt. The
advisory Deep Agents worker has a shorter independent limit, so a local
tool-calling loop cannot block the reviewed typed factory. Ollama is asked to
unload each model after use; if another local process prevents a model switch,
Metis surfaces a typed failure rather than leaving a durable run in progress
forever.
