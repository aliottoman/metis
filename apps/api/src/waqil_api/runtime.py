from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from contextlib import AbstractAsyncContextManager
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .asset_library import AssetManager
from .blob_store import BlobStore
from .config import Settings
from .control_plane import ControlPlane
from .corpus import CorpusService
from .customer_intelligence import CustomerIntelligenceService
from .dac_catalog import DacCatalog
from .dac_service import DacService
from .sku_catalog import SkuCatalog
from .win_valuation import WinValuationService
from .database import Database
from .deep_worker import build_deep_worker_factory
from .embeddings import CohereRetrieval
from .events import EventBus
from .memory_index import MemoryIndex
from .local_model_session import LocalModelSessionManager
from .model_preference import ModelPreferenceStore
from .model_provider import RoutedModelProvider, build_model_provider
from .notion import NotionService
from .profile import ProfileStore
from .project_sandbox import ProjectSandboxService
from .project_verification import ProjectVerificationService
from .project_workspace import ProjectWorkspaceService
from .run_history import RunHistoryService
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
        self.memory_index = MemoryIndex(settings, self.database, self.retrieval)
        self.run_history = RunHistoryService(settings, self.database, self.corpus)
        self.profile = ProfileStore(settings)
        self.model_preference = ModelPreferenceStore(settings)
        self.model_session = LocalModelSessionManager(
            settings, self.model_preference
        )
        # Sizing reads a vendored catalog and needs no I/O, so it is built here
        # rather than in start(); the model provider is attached later so the
        # recommender can use whichever model the user has selected.
        self.dac_catalog = DacCatalog()
        self.dac = DacService(self.dac_catalog, preference=self.model_preference)
        # Same shape as the sizing catalog: vendored JSON, no I/O, model attached
        # in start() so valuation uses whichever model the user has selected.
        self.sku_catalog = SkuCatalog(rates_path=settings.sku_rates_path)
        self.win_valuation = WinValuationService(
            self.database, self.sku_catalog, preference=self.model_preference
        )
        self.verification = ProjectVerificationService(
            settings, approval_path=settings.project_verify_approval_path
        )
        self.projects: ProjectWorkspaceService | None = None
        self.registry = ToolRegistry(self.database, settings)
        self.model = None
        self.local_model = None
        self.customers: CustomerIntelligenceService | None = None
        self.reference_runner = ReferenceArchitectureRunner(settings)
        self.checkpointer: Any = None
        self.control_plane: ControlPlane | None = None
        self.project_sandbox: ProjectSandboxService | None = None
        self.deep_worker_factory: Any = None
        self._checkpointer_context: AbstractAsyncContextManager[Any] | None = None
        self._background: set[asyncio.Task[None]] = set()

    def spawn(self, work: Coroutine[Any, Any, Any], *, name: str) -> None:
        """Run best-effort maintenance without blocking or breaking the caller.

        These tasks only ever affect retrieval quality, so a failure is
        swallowed. They are tracked so shutdown can cancel them rather than
        leaving a half-finished write behind a closed database.
        """

        async def guarded() -> None:
            try:
                await work
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - maintenance is never load-bearing
                pass

        task = asyncio.create_task(guarded(), name=name)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def start(self) -> None:
        self.settings.prepare_directories()
        await self.database.open()
        await self.registry.seed_builtins()
        await self.registry.reconcile_trusted_definition_proposals()
        await self.registry.reconcile_trusted_evaluated_builds()
        self.model = build_model_provider(
            self.settings, model_session=self.model_session
        )
        self.local_model = (
            self.model.local if isinstance(self.model, RoutedModelProvider) else self.model
        )
        self.customers = CustomerIntelligenceService(
            self.database, self.local_model, self.model_session
        )
        self.deep_worker_factory = build_deep_worker_factory(self.local_model)
        self.dac = DacService(
            self.dac_catalog, model=self.model, preference=self.model_preference
        )
        self.win_valuation = WinValuationService(
            self.database, self.sku_catalog,
            model=self.model, preference=self.model_preference,
        )
        self.project_sandbox = ProjectSandboxService(self.settings)
        self.projects = ProjectWorkspaceService(
            self.settings,
            self.assets,
            self.model,
            verification=self.verification,
            sandbox=self.project_sandbox,
        )
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
            memory_index=self.memory_index,
            run_history=self.run_history,
            registry=self.registry,
            reviewer=self.retrieval,
            tool_model=self.local_model,
            projects=self.projects,
            customers=self.customers,
            model_session=self.model_session,
        )
        await self.control_plane.reconcile_startup()
        self.spawn(self._release_idle_model(), name="model-idle-release")

    async def _release_idle_model(self) -> None:
        """Give the weights back once every Metis window has gone away.

        Nothing else does this: closing the app is a window closing, and Ollama
        only counts idle time since the last *model* call, so a launched model
        outlives the app that launched it for the whole keep_alive window.
        """
        after = self.settings.model_release_after_idle_seconds
        if after <= 0:
            return
        # Four checks per window, so the release lands close to the deadline
        # rather than a whole window late.
        interval = max(1.0, min(60.0, after / 4))
        while True:
            await asyncio.sleep(interval)
            # A run can be in flight without a model call being active, so the
            # busy counter alone is not enough to call the session idle.
            if await self.database.has_active_runs():
                self.model_session.touch()
                continue
            await self.model_session.release_if_idle(after)
            # The verify sandbox holds a VM the same way the session holds
            # weights, on its own clock: a build verifies two or three times in
            # a turn, so stopping it between them would pay the boot repeatedly.
            if self.project_sandbox is not None:
                await self.project_sandbox.release_if_idle(
                    self.settings.project_sandbox_release_after_idle_seconds
                )

    async def close(self) -> None:
        # Before the loop goes away, so the unload request can still be sent.
        await self.model_session.release_owned()
        if self.project_sandbox is not None:
            await self.project_sandbox.release_machine()
        for task in list(self._background):
            task.cancel()
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)
        await self.assets.shutdown()
        if self.control_plane is not None:
            await self.control_plane.shutdown()
        if self.model is not None and hasattr(self.model, "close"):
            await self.model.close()
        if self._checkpointer_context is not None:
            await self._checkpointer_context.__aexit__(None, None, None)
        await self.database.close()
