"""What voice refuses, decided before anything expensive happens.

This runs first — before retrieval, before the account is resolved, before a
model is called at all. That ordering is the security property, not an
optimization: a refusal that happened *after* a model saw the transcript would
mean the model had already been asked to consider building, approving or
deleting, and the only thing standing between the request and the action would
be that it declined. Refusing at the door means there was never a decision to
make.

The build classifiers are the ones the rest of Metis already uses, imported
rather than reimplemented. A second opinion about what counts as a build
request is exactly how a gate develops a gap: the day someone tightens
`_BUILD_PATTERNS` for the chat path, a private copy here would quietly keep
the old behavior on the one surface that cannot show what it is about to do.

The remaining intents — approvals, deletes, filesystem, tool activation,
sandbox execution, model selection — have no existing classifier because no
other surface needs to refuse them wholesale. Their patterns are deliberately
narrow. A false positive costs one spoken sentence and a composer handoff; the
question is still answerable in the chat window two seconds later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .model_provider import (
    describes_a_new_tool,
    is_explicit_build_request,
    is_explicit_toolify_request,
    is_new_application_request,
    is_project_build_instruction,
)

# Stored once, as a module constant, because it is quoted in the composer
# handoff card, spoken by the voice, and asserted in tests — three places that
# must never drift into three slightly different promises.
BUILD_REFUSAL = (
    "I can't build from voice — that needs the chat window. I've dropped it in "
    "your composer, so it's ready when you switch over."
)

APPROVAL_REFUSAL = (
    "Approvals aren't something I can take by voice. It's waiting in the chat "
    "window, where you can see exactly what you'd be approving."
)

DESTRUCTIVE_REFUSAL = (
    "I won't delete or overwrite anything from voice. That one needs the chat "
    "window, where you can see the record first."
)

FILESYSTEM_REFUSAL = (
    "I can't reach files from voice. Ask me in the chat window and I'll work "
    "from what you've given me there."
)

TOOL_REFUSAL = (
    "Turning a tool on isn't something voice can do. It needs the chat window "
    "and its two approvals."
)

EXECUTION_REFUSAL = (
    "I can't run code from voice. That happens in the sandbox, from the chat window."
)

MODEL_REFUSAL = (
    "I can't switch models from voice. That's in Settings, where you can see "
    "what each one costs you."
)


@dataclass(frozen=True, slots=True)
class VoiceRefusal:
    """One fixed answer, and what the surface should do about it.

    `hand_off` says whether the composer gets the verbatim transcript. It is
    true for the intents where the user asked for something real that another
    surface can do — a build, an approval, a delete — and false for the ones
    where handing over a prefilled sentence would be theatre.
    """

    intent: str
    spoken: str
    hand_off: bool = True


# Each pattern set is narrow on purpose; see the module docstring. They read
# the finalized utterance, which is one person speaking one sentence — not a
# pasted document — so there is no quoted material to be misled by.
_APPROVAL = (
    re.compile(r"\bapprove\b"),
    re.compile(
        r"\b(accept|allow|authorize|authorise)\b[^.?!\n]{0,30}\b"
        r"(run|request|proposal|tool|build|change|approval)\b"
    ),
    re.compile(r"\b(go ahead|sign off)\b[^.?!\n]{0,30}\b(with|on)\b"),
    re.compile(r"\bgrant\b[^.?!\n]{0,30}\b(access|permission|approval)\b"),
)

_DESTRUCTIVE = (
    re.compile(r"\b(delete|remove|erase|wipe|drop|purge)\b"),
    re.compile(
        r"\b(overwrite|replace|rewrite)\b[^.?!\n]{0,30}\b"
        r"(record|note|fact|action|file|answer|memory|account)\b"
    ),
    re.compile(
        r"\b(clear|reset)\b[^.?!\n]{0,30}\b(history|memory|records?|database)\b"
    ),
    # Status changes are a mutation voice does not have: marking an action done
    # closes a commitment, and the receipt for that belongs on screen.
    re.compile(r"\bmark\b[^.?!\n]{0,30}\b(done|complete|closed|finished)\b"),
)

_FILESYSTEM = (
    re.compile(r"\b(open|read|show me|cat|print)\b[^.?!\n]{0,30}\bfile\b"),
    re.compile(
        r"\b(my|the)\s+(desktop|documents|downloads|home)\s+(folder|directory)\b"
    ),
    re.compile(r"\b(read|open|list)\b[^.?!\n]{0,20}\b(folder|directory|path)\b"),
    re.compile(r"[~/][\w.\-/]*/[\w.\-/]+"),
)

_TOOL_ACTIVATION = (
    re.compile(
        r"\b(activate|enable|turn on|switch on|install|publish)\b"
        r"[^.?!\n]{0,30}\btools?\b"
    ),
    re.compile(r"\btools?\b[^.?!\n]{0,20}\b(live|active|available)\b"),
)

_EXECUTION = (
    re.compile(
        r"\b(run|execute|launch)\b[^.?!\n]{0,30}\b"
        r"(code|script|command|sandbox|container|podman|shell|terminal)\b"
    ),
    re.compile(r"\b(pip|npm|pnpm|brew|git)\s+\w+"),
)

_MODEL_SELECTION = (
    re.compile(r"\b(switch|change|use|pin|set)\b[^.?!\n]{0,30}\bmodel\b"),
    re.compile(r"\bmodel\b[^.?!\n]{0,20}\b(to|instead)\b"),
    re.compile(r"\b(switch|change)\b[^.?!\n]{0,20}\b(provider|lane)\b"),
)


def _matches(patterns: tuple[re.Pattern[str], ...], lowered: str) -> bool:
    return any(pattern.search(lowered) for pattern in patterns)


def classify_refusal(transcript: str) -> VoiceRefusal | None:
    """The fixed refusal this utterance earns, or None to let it through.

    Order matters where two intents overlap. "Build it and approve it" is a
    build first, because the build handoff is the one that prefills the
    composer with something the chat window can actually finish.
    """
    said = (transcript or "").strip()
    if not said:
        return None
    lowered = said.lower()

    if (
        is_explicit_build_request(said)
        or is_explicit_toolify_request(said)
        or is_new_application_request(said)
        or is_project_build_instruction(said)
        or describes_a_new_tool(said)
    ):
        return VoiceRefusal("refuse_build", BUILD_REFUSAL)
    if _matches(_APPROVAL, lowered):
        return VoiceRefusal("refuse_protected", APPROVAL_REFUSAL)
    if _matches(_DESTRUCTIVE, lowered):
        return VoiceRefusal("refuse_protected", DESTRUCTIVE_REFUSAL)
    if _matches(_TOOL_ACTIVATION, lowered):
        return VoiceRefusal("refuse_protected", TOOL_REFUSAL)
    if _matches(_EXECUTION, lowered):
        return VoiceRefusal("refuse_protected", EXECUTION_REFUSAL)
    if _matches(_FILESYSTEM, lowered):
        return VoiceRefusal("refuse_protected", FILESYSTEM_REFUSAL)
    if _matches(_MODEL_SELECTION, lowered):
        # Nothing to hand off: Settings is a page, not a composer message, and
        # prefilling "switch to Grok" would be a sentence nobody can send.
        return VoiceRefusal("refuse_protected", MODEL_REFUSAL, hand_off=False)
    return None
