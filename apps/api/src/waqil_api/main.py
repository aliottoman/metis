from __future__ import annotations

from contextlib import asynccontextmanager
import os

# Set before importing orchestration/runtime modules. Metis is local-only and
# never opts into LangChain/LangSmith tracing implicitly.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from .config import Settings
from .runtime import AppRuntime


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app_runtime = AppRuntime(configured)
        await app_runtime.start()
        app.state.runtime = app_runtime
        try:
            yield
        finally:
            await app_runtime.close()

    application = FastAPI(
        title="Metis Local Agent API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=configured.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )
    application.include_router(router)

    @application.get("/health", include_in_schema=False)
    async def root_health():
        return {"status": "ok", "api": "/api/v1/health"}

    return application


app = create_app()


def run() -> None:
    settings = Settings()
    uvicorn.run(
        "waqil_api.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        access_log=True,
    )


if __name__ == "__main__":
    run()
