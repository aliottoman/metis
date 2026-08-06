"""Provider-neutral project function schemas for tool-calling decode.

Two transports reach the same seven workspace tools plus the completion
channel. The OCI provider has always sent these definitions as real function
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

from .contracts import PROJECT_TOOL_REQUIRED_ARGUMENTS

# Every project path is relative to the project root. Models default to an
# invented workspace prefix otherwise, and every such call is refused.
_PROJECT_PATH_HELP = (
    "Path relative to the project root, e.g. \"app/agents/planner.py\". "
    "Absolute paths and \"..\" are refused."
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
                "structure. Omit path, or send \"\", to list the whole project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional directory prefix relative to the project "
                            "root, e.g. \"app/agents\". Never absolute."
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
                    "path": {"type": "string", "minLength": 1, "description": _PROJECT_PATH_HELP},
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
                    "path": {"type": "string", "minLength": 1, "description": _PROJECT_PATH_HELP},
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
                    "path": {"type": "string", "minLength": 1, "description": _PROJECT_PATH_HELP},
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
                    "path": {"type": "string", "minLength": 1, "description": _PROJECT_PATH_HELP},
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
