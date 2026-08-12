import type {
  AbortSliceParams,
  ContinueSliceParams,
  DeleteSessionResult,
  RestartWithModelParams,
  RestoreSliceParams,
  SessionParams,
  SliceResult,
  StartSliceParams,
  Usage,
} from "./protocol.js";
import type { RuntimeEvent } from "./sanitize.js";

export type RuntimeEventListener = (event: RuntimeEvent) => void;

export interface SliceRuntime {
  readonly runtimeMode: "cline" | "fake";
  startSlice(params: StartSliceParams): Promise<SliceResult>;
  continueSlice(params: ContinueSliceParams): Promise<SliceResult>;
  abortSlice(params: AbortSliceParams): Promise<SliceResult>;
  restoreSlice(params: RestoreSliceParams): Promise<SliceResult>;
  restartWithModel(params: RestartWithModelParams): Promise<SliceResult>;
  getUsage(params: SessionParams): Promise<Usage>;
  deleteSession(params: SessionParams): Promise<DeleteSessionResult>;
  subscribe(listener: RuntimeEventListener): () => void;
  shutdown(): Promise<void>;
}

export abstract class EventedRuntime implements SliceRuntime {
  abstract readonly runtimeMode: "cline" | "fake";
  readonly #listeners = new Set<RuntimeEventListener>();

  protected emit(event: RuntimeEvent): void {
    for (const listener of this.#listeners) listener(event);
  }

  subscribe(listener: RuntimeEventListener): () => void {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  abstract startSlice(params: StartSliceParams): Promise<SliceResult>;
  abstract continueSlice(params: ContinueSliceParams): Promise<SliceResult>;
  abstract abortSlice(params: AbortSliceParams): Promise<SliceResult>;
  abstract restoreSlice(params: RestoreSliceParams): Promise<SliceResult>;
  abstract restartWithModel(
    params: RestartWithModelParams,
  ): Promise<SliceResult>;
  abstract getUsage(params: SessionParams): Promise<Usage>;
  abstract deleteSession(params: SessionParams): Promise<DeleteSessionResult>;
  abstract shutdown(): Promise<void>;
}
