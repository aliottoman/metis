"""The customer-agent tool catalog.

Flat Responses-API shape (``{"type": "function", "name": ...}``), matching
:mod:`project_tools`. This is the ONLY menu the customer agent may pick from when
a chat is scoped to a customer account. It is deliberately small and orthogonal:
every tool names what it does and, explicitly, what it does NOT — so the model
routes a message to exactly one tool, and no capability is left unreachable. The
routing is the model's (any phrasing works); this catalog is its contract.

The evidence line — the DynaAI lesson — is enforced *here*, in the schemas, not
left to the model's goodwill. A tool that writes to the record never takes the
user's words through a model-authored argument, because a model filling a free
``note`` field can quietly paraphrase the very evidence it is recording. So the
write tools carry only non-evidence metadata (a short ``title``, which
``proposal`` to apply); the note body and an activity report are read from the
user's actual message by the host. What the model decides is *which* tool and
*what to call it* — never what the user said.
"""

from typing import Any

# One source of truth for the names, shared by the loop that dispatches them and
# the tests that prove each one is reachable and none overlap.
ANSWER = "answer_about_account"
FILE_NOTE = "file_note"
APPLY_EXTRACTION = "apply_extraction"
RECORD_ACTIVITY = "record_activity"
RECORD_WIN = "record_win"
GENERATE_TRACKER = "generate_tracker"

# Ordered, so a rendered catalog and a test matrix read the same way every time.
CUSTOMER_TOOL_NAMES: tuple[str, ...] = (
    ANSWER,
    FILE_NOTE,
    APPLY_EXTRACTION,
    RECORD_ACTIVITY,
    RECORD_WIN,
    GENERATE_TRACKER,
)

# Tools that mutate the account record. The loop routes these through the same
# guard-and-confirm path as any other write; naming them here keeps that policy
# in one place rather than scattered across dispatch.
CUSTOMER_WRITE_TOOLS: frozenset[str] = frozenset(
    {FILE_NOTE, APPLY_EXTRACTION, RECORD_ACTIVITY, RECORD_WIN}
)


def customer_tools() -> list[dict[str, Any]]:
    """The customer agent's function catalog, in the flat Responses-API shape.

    Kept as a function (not a module constant) so a caller can narrow it later
    — e.g. dropping ``apply_extraction`` when no proposal is pending — without
    the whole app sharing one mutable list.
    """
    return [
        {
            "type": "function",
            "name": ANSWER,
            "description": (
                "Answer a QUESTION about this account from its reviewed record, with "
                "citations. Use when the user is asking something — 'what's the "
                "status', 'who is the contact', 'did we agree X', 'summarise where we "
                "are'. Reads only; it does NOT write anything, and it does NOT file "
                "the user's own statements as a note (that is file_note)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "minLength": 1,
                        "description": "The user's question about the account, as asked.",
                    }
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": FILE_NOTE,
            "description": (
                "File a NOTE — new information the user is stating about the account: "
                "an update, a meeting outcome, something to record. Use when the user "
                "says 'update <account> with…', 'note that…', 'add this…', or pastes a "
                "block of account information. The note body is taken from the user's "
                "own words by the host — you supply only a short title. Filing also "
                "kicks off analysis, which proposes structured facts for review. It "
                "does NOT answer a question, and it does NOT write facts straight to "
                "the profile — applying reviewed findings is apply_extraction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 120,
                        "description": (
                            "A short descriptive title for the note, drawn from its "
                            "content (e.g. 'Throttling finding on the DAC'). Metadata "
                            "only — never the note body, which the host stores verbatim."
                        ),
                    }
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": APPLY_EXTRACTION,
            "description": (
                "Apply a PENDING extraction to the account profile — commit the facts, "
                "actions, and people a prior analysis already proposed. Use only when "
                "the user confirms applying reviewed findings (taps Apply, or says "
                "'yes, add those to the profile'). It does NOT extract or infer facts "
                "of its own; it only commits what analysis proposed, and the write "
                "still surfaces for the user's confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "proposal_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "The id of the pending review proposal to apply.",
                    }
                },
                "required": ["proposal_id"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": RECORD_ACTIVITY,
            "description": (
                "Record ACTIVITY on the account's action list: mark a commitment done, "
                "or add a new follow-up. Use when the user reports finished work ('I "
                "sent the pricing', 'met them, demo done') or names a new to-do ('need "
                "to follow up Friday'). The host matches the report to the real action "
                "list and confirms any closure — it can close a commitment but never "
                "invent one. It does NOT file free-form notes (file_note) or answer "
                "questions (answer_about_account)."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": RECORD_WIN,
            "description": (
                "Record a WIN on the account — a closed deal, signed commitment, or "
                "landed outcome worth tracking as revenue. Use when the user reports a "
                "success worth booking ('BAPCO signed the DAC', 'we won the renewal', "
                "'they committed to 2xH100'). You supply a short title; the host keeps "
                "the user's own words as the win's evidence, and its value can be "
                "estimated afterward. It does NOT file a general note (file_note) or "
                "close a to-do (record_activity)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 160,
                        "description": (
                            "A short title for the win, drawn from what the user said "
                            "(e.g. 'Signed 2xH100 DAC'). Metadata only — the host keeps "
                            "the user's own words as the win's evidence."
                        ),
                    }
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": GENERATE_TRACKER,
            "description": (
                "Produce the account's activity-tracker update — a Markdown summary of "
                "its interactions — into the thread. Use when the user asks for 'the "
                "tracker', 'an activity update', or 'a summary I can paste into the "
                "tracker'. Read-only: it does NOT change the record."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    ]
