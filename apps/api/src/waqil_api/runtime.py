from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .asset_library import AssetManager
from .blob_store import BlobStore
from .config import Settings
from .control_plane import ControlPlane
from .corpus import CorpusService
from .database import Database
from .deep_worker import build_deep_worker_factory
from .embeddings import CohereRetrieval
from .events import EventBus
from .model_preference import ModelPreferenceStore
from .model_provider import RoutedModelProvider, build_model_provider
from .notion import NotionService
from .profile import ProfileStore
from .project_workspace import ProjectWorkspaceService
from .tool_registry import ToolRegistry
from .reference_architecture import ReferenceArchitectureRunner


class AppRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.assets = AssetManager(
            settings.asset_roots,
            approval_path=settings.asset_approval_path,
            catalog_path=settings.asset_catalog_path,
        )
        self.database = Database(settings.database_path)
        self.blobs = BlobStore(settings.blob_dir)
        self.events = EventBus(self.database)
        self.retrieval = CohereRetrieval(settings)
        self.corpus = CorpusService(settings, self.database, self.retrieval)
        self.notion = NotionService(settings, self.database, self.corpus)
        self.profile = ProfileStore(settings)
        self.model_preference = ModelPreferenceStore(settings)
        self.projects: ProjectWorkspaceService | None = None
        self.registry = ToolRegistry(self.database, settings)
        self.model = None
        self.local_model = None
        self.reference_runner = ReferenceArchitectureRunner(settings)
        self.checkpointer: Any = None
        self.control_plane: ControlPlane | None = None
        self.deep_worker_factory: Any = None
        self._checkpointer_context: AbstractAsyncContextManager[Any] | None = None

    async def start(self) -> None:
        self.settings.prepare_directories()
        await self.database.open()
        await self.registry.seed_builtins()
        await self.registry.reconcile_trusted_definition_proposals()
        await self.registry.reconcile_trusted_evaluated_builds()
        self.model = build_model_provider(self.settings)
        self.local_model = (
            self.model.local if isinstance(self.model, RoutedModelProvider) else self.model
        )
        self.deep_worker_factory = build_deep_worker_factory(self.local_model)
        self.projects = ProjectWorkspaceService(self.settings, self.assets, self.model)
        self._checkpointer_context = AsyncSqliteSaver.from_conn_string(
            str(self.settings.checkpoint_path)
        )
        self.checkpointer = await self._checkpointer_context.__aenter__()
        await self.checkpointer.conn.execute("PRAGMA journal_mode=WAL")
        await self.checkpointer.conn.execute("PRAGMA busy_timeout=5000")
        await self.checkpointer.conn.commit()
        await self.checkpointer.setup()
        self.control_plane = ControlPlane(
            self.settings,
            self.database,
            self.blobs,
            self.events,
            self.model,
            self.reference_runner,
            self.checkpointer,
            self.deep_worker_factory,
            corpus=self.corpus,
            profile=self.profile,
            registry=self.registry,
            reviewer=self.retrieval,
            tool_model=self.local_model,
            projects=self.projects,
        )
        await self.control_plane.reconcile_startup()

    async def close(self) -> None:
        await self.assets.shutdown()
        if self.control_plane is not None:
            await self.control_plane.shutdown()
        if self.model is not None and hasattr(self.model, "close"):
            await self.model.close()
        if self._checkpointer_context is not None:
            await self._checkpointer_context.__aexit__(None, None, None)
        await self.database.close()
