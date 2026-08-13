"""Interview assistant — a standalone page that runs the ElevenLabs interview agent.

Serves the static frontend and mints short-lived conversation tokens. The API key
lives here and never reaches the browser; the browser gets a token that is good
for exactly one conversation.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).parent / "static"
ELEVENLABS_TOKEN_URL = "https://api.elevenlabs.io/v1/convai/conversation/token"

app = FastAPI(title="Interview assistant")


class SessionRequest(BaseModel):
    """The three things the form collects, passed to the agent as dynamic variables."""

    job_title: str = Field(min_length=1, max_length=200)
    company_name: str = Field(min_length=1, max_length=200)
    job_description: str = Field(min_length=1, max_length=20000)


class SessionResponse(BaseModel):
    """A one-conversation token plus the variables the browser should send with it."""

    conversation_token: str
    dynamic_variables: dict[str, str]


# Reads the ElevenLabs API key from the environment, or fails loudly.
def _api_key() -> str:
    key = os.environ.get("INTERVIEW_ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise HTTPException(500, "INTERVIEW_ELEVENLABS_API_KEY is not set.")
    return key


# Reads the interview agent's id from the environment, or fails loudly.
def _agent_id() -> str:
    agent = os.environ.get("INTERVIEW_ELEVENLABS_AGENT_ID", "").strip()
    if not agent:
        raise HTTPException(500, "INTERVIEW_ELEVENLABS_AGENT_ID is not set.")
    return agent


# Asks ElevenLabs for a WebRTC conversation token for our agent.
async def _mint_token() -> str:
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            ELEVENLABS_TOKEN_URL,
            params={"agent_id": _agent_id()},
            headers={"xi-api-key": _api_key()},
        )
    if response.status_code >= 400:
        raise HTTPException(502, f"ElevenLabs refused the conversation: HTTP {response.status_code}")

    token = str(response.json().get("token") or "").strip()
    if not token:
        raise HTTPException(502, "ElevenLabs returned no conversation token.")
    return token


# Starts a session: mints a token and echoes back the variables to send with it.
@app.post("/api/session", response_model=SessionResponse)
async def create_session(request: SessionRequest) -> SessionResponse:
    token = await _mint_token()
    return SessionResponse(
        conversation_token=token,
        dynamic_variables={
            "job_title": request.job_title.strip(),
            "company_name": request.company_name.strip(),
            "job_description": request.job_description.strip(),
        },
    )


# Reports whether the two required environment variables are present.
@app.get("/api/health")
async def health() -> dict[str, bool]:
    return {
        "api_key_set": bool(os.environ.get("INTERVIEW_ELEVENLABS_API_KEY", "").strip()),
        "agent_id_set": bool(os.environ.get("INTERVIEW_ELEVENLABS_AGENT_ID", "").strip()),
    }


# Serves the page itself.
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
