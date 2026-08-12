"""Provider-neutral project function schemas for tool-calling decode.

Two transports reach the same workspace tools, the talk channel (ask_user /
respond) and the completion channel. The OCI provider has always sent these definitions as real function
schemas; the Ollama provider sends them to hosted models, where the platform
enforces tool calling but ignores ``format`` grammars. Lifting them here keeps
one set of definitions for both — the drift this prevents is not hypothetical:
inspect_api spent months advertised in one provider's copy of a tool list and
impossible to call, because a second hand-copied list lagged one tool behind.

The canonical roster stays ``PROJECT_TOOL_REQUIRED_ARGUMENTS`` in contracts.py;
a module-level check below refuses to import a tool list that disagrees with
it, and the parity tests pin every other enumeration to the same table.
"""

from __future__ import annotations

import json
from typing import Any

from .contracts import PROJECT_TOOL_REQUIRED_ARGUMENTS, WHOLE_FILE_END_LINE

# Every project path is relative to the project root. Models default to an
# invented workspace prefix otherwise, and every such call is refused.
_PROJECT_PATH_HELP = (
    'Path relative to the project root, e.g. "app/agents/planner.py". '
    'Absolute paths and ".." are refused.'
)

# The completion channel: a function call rather than a workspace tool, so a
# tool-calling decode has a way to say "done" that the host can still refuse.
FINISH_TOOL_NAME = "finish_project_task"


def unrestricted_project_tools() -> list[dict[str, Any]]:
    """Every project function a tool-calling model may be offered.

    Flat Responses-API shape (``{"type": "function", "name": ...}``), exactly
    as the OCI provider has always sent it; ``chat_tool_format`` below nests
    the same definitions for the Ollama chat API. Built fresh on every call so
    a caller narrowing one tool never mutates a shared copy.
    """
    return [
        {
            "type": "function",
            "name": "list_files",
            "description": (
                "List bounded project-relative file paths. Use before guessing "
                'structure. Omit path, or send "", to list the whole project.'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional directory prefix relative to the project "
                            'root, e.g. "app/agents". Never absolute.'
                        ),
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "search_code",
            "description": "Search readable project text for an exact string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 300},
                    "case_sensitive": {"type": "boolean"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a bounded line range from one UTF-8 project file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": _PROJECT_PATH_HELP,
                    },
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "apply_patch",
            "description": "Propose replacing one unique exact text block in an existing file. The user must approve before it runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": _PROJECT_PATH_HELP,
                    },
                    "original": {"type": "string", "minLength": 1},
                    "replacement": {"type": "string"},
                },
                "required": ["path", "original", "replacement"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "replace_lines",
            "description": (
                "Replace lines start_line through end_line (1-indexed, "
                "inclusive) of an existing or staged file with replacement — "
                "no exact quoting needed. read_file the range first to confirm "
                "coordinates; pass a short distinctive substring of the doomed "
                "block as expect so a mis-aimed range refuses instead of "
                "landing. Replacing lines 1 through the last line rewrites the "
                "whole file. The user must approve before it runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": _PROJECT_PATH_HELP,
                    },
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                    "replacement": {"type": "string"},
                    "expect": {
                        "type": "string",
                        "description": (
                            "Optional guard: a substring the doomed lines must "
                            "contain, or the call is refused."
                        ),
                    },
                },
                "required": ["path", "start_line", "end_line", "replacement"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "create_file",
            "description": "Propose creating a new UTF-8 file without overwriting. The user must approve before it runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": _PROJECT_PATH_HELP,
                    },
                    "content": {"type": "string", "minLength": 1},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "inspect_api",
            "description": (
                "Look up an INSTALLED library before writing against it: its "
                "real exported names, and the real signature of one function "
                "or class. Use it whenever you are about to call an API you "
                "have not verified — a keyword argument that does not exist "
                "parses perfectly and fails at runtime. Reads libraries, "
                "never this project's files; use read_file for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "minLength": 1,
                        "description": 'Import path, e.g. "openai" or "oci_genai_auth".',
                    },
                    "symbol": {
                        "type": "string",
                        "description": "One name in that module. Omit to list its exports.",
                    },
                },
                "required": ["module"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "run_check",
            "description": (
                "Run one verification check this project declared and the user "
                "approved, by name. Use it to prove a change works instead of "
                "asserting it. You cannot supply or modify a command; only the "
                "names in project_context.verification.checks are accepted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 32},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "ask_user",
            "description": (
                "Pause this turn to ask the user ONE clarifying question. Their "
                "answer returns as this call's result and the same turn "
                "continues with it. Use only when the request is genuinely "
                "ambiguous in a way that changes what you would build, or when "
                "the user invited questions — otherwise choose a sensible "
                "default and disclose it in your summary. Never ask more than "
                "once per turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2000,
                        "description": "The one question that unblocks the work.",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 200},
                        "maxItems": 6,
                        "description": (
                            "Up to 6 short answer choices. Omit for free text."
                        ),
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "respond",
            "description": (
                "Answer the user directly, without claiming any build work "
                "happened. Use it for questions about the project, "
                "explanations, and status. A respond turn is complete in "
                "itself; do not follow it with finish_project_task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "minLength": 1,
                        "description": "The complete user-facing answer.",
                    },
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "revise_plan",
            "description": (
                "Correct this turn's file manifest when what you have read "
                "contradicts it — a planned path this project does not have, a "
                "framework that makes the plan wrong, work that needs different "
                "files than were named. Omitted prior paths remain committed. "
                "Send the corrected order and additions in files; name any "
                "proven-invalid unstaged paths explicitly in remove_files and "
                "explain the repository evidence in reason. Use this rather "
                "than writing a file you believe is wrong or finishing to "
                "escape the plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 24,
                        "description": (
                            "The proposed corrected order and any new project-"
                            "relative paths. Omitted prior paths are retained."
                        ),
                    },
                    "remove_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 24,
                        "description": (
                            "Prior, unstaged plan paths repository evidence proves "
                            "invalid or unwritable. Omit this field to remove none."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 600,
                        "description": (
                            "What you found that makes the current plan wrong."
                        ),
                    },
                },
                "required": ["files", "reason"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": FINISH_TOOL_NAME,
            "description": "Finish the project turn with a user-facing response and stable non-secret learnings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {"type": "string", "minLength": 1},
                    "learnings": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 600},
                        "maxItems": 16,
                    },
                },
                "required": ["response", "learnings"],
                "additionalProperties": False,
            },
        },
    ]


def narrowed_project_tools(owed: list[str] | None = None) -> list[dict[str, Any]]:
    """The project functions a model may call this step.

    ``owed`` is the files the turn planned and has not written. When it is
    non-empty, create_file's path is narrowed to an enum of exactly those
    files, which points the model at the work instead of describing it.

    finish_project_task is deliberately **kept**. Withholding it was tried
    and measured: an owed path the host refuses for its own reasons then
    leaves no legal move at all, and a build that had been taking 19 steps
    took the full 48 without producing the file. The host already refuses a
    premature finish; a model that cannot make progress has to be able to
    say so, or the loop only fails more slowly.
    """
    tools = unrestricted_project_tools()
    if not owed:
        return tools
    narrowed: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("name") == "create_file":
            tool = json.loads(json.dumps(tool))
            tool["parameters"]["properties"]["path"] = {
                "type": "string",
                "enum": list(owed),
                "description": (
                    "The next file this build still owes. Only these paths "
                    "remain unwritten."
                ),
            }
            tool["description"] = (
                "Propose creating a new UTF-8 file without overwriting. "
                f"{len(owed)} planned file(s) are still unwritten and the "
                "turn cannot finish until they exist."
            )
        narrowed.append(tool)
    return narrowed


# What a coder may do on a step the orchestrator directed. Everything else is
# either refused by the host anyway (the reads) or belongs to a seat this one
# does not hold (run_check, ask_user, respond).
DIRECTED_TOOLS = frozenset(
    {"create_file", "apply_patch", "replace_lines", "revise_plan", FINISH_TOOL_NAME}
)


def directed_project_tools(owed: list[str] | None = None) -> list[dict[str, Any]]:
    """The roster for a directed step: write the one file, or say you cannot.

    On a directed step the host refuses reads outright — the orchestrator has
    already said what this file needs and the host has already fetched it — but
    the model was still being *shown* read_file, list_files, search_code and
    inspect_api. Telling it in prose that reads are closed while handing it four
    read tools is not a narrow subtask; it is an invitation, and it was taken:
    in a live revamp the coder answered a directed step with read_file and then
    spent twenty-three more steps reading.

    So the illegal tools are removed rather than merely refused. Twelve becomes
    five, and every one of the five is something the step can actually do.

    finish_project_task stays, for the same reason it stays in the narrowed
    roster: a model with no legal move does not stop, it stalls. revise_plan
    stays because "this instruction cannot be carried out" is a real answer and
    the only alternative is writing something wrong.
    """
    return [
        tool
        for tool in narrowed_project_tools(owed)
        if tool.get("name") in DIRECTED_TOOLS
    ]


def whole_file_repair_tools(path: str) -> list[dict[str, Any]]:
    """The bounded strategy after an exact patch or line range was refused.

    Keep revise_plan as the honest escape, but make the only edit a complete
    replace_lines call whose path and range are host-owned. Tool-calling lanes
    therefore receive the same structural recovery the local grammar gets.
    """
    tools: list[dict[str, Any]] = []
    for tool in directed_project_tools([path]):
        name = tool.get("name")
        if name == "revise_plan":
            tools.append(tool)
            continue
        if name != "replace_lines":
            continue
        replacement = json.loads(json.dumps(tool))
        parameters = replacement["parameters"]
        parameters["properties"]["path"] = {
            "type": "string",
            "enum": [path],
            "description": _PROJECT_PATH_HELP,
        }
        parameters["properties"]["start_line"] = {
            "type": "integer",
            "enum": [1],
        }
        parameters["properties"]["end_line"] = {
            "type": "integer",
            "enum": [WHOLE_FILE_END_LINE],
        }
        replacement["description"] = (
            f"Rewrite all of {path}. Send its complete corrected contents as "
            f"replacement with start_line=1 and end_line={WHOLE_FILE_END_LINE}."
        )
        tools.append(replacement)
    return tools


def chat_tool_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same definitions in the nested shape the Ollama chat API expects.

    The Responses API takes ``{"type": "function", "name": ...}`` flat; the
    chat API wants the definition wrapped under a ``function`` key. One
    converter rather than a second hand-written list, so the two dialects
    cannot drift.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]


# Advertising a tool the contract will refuse — or the reverse — is the drift
# this module exists to end, so it refuses to import in that state rather than
# waiting for the parity tests to run.
_advertised = {tool["name"] for tool in unrestricted_project_tools()}
if _advertised != set(PROJECT_TOOL_REQUIRED_ARGUMENTS) | {FINISH_TOOL_NAME}:
    raise RuntimeError(
        "project_tools drifted from PROJECT_TOOL_REQUIRED_ARGUMENTS: "
        f"advertised {sorted(_advertised)}"
    )
