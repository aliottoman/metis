import type { Readable, Writable } from "node:stream";

import { HostBridge } from "./host-bridge.js";
import {
  MAX_FRAME_BYTES,
  RPC_VERSION,
  RpcFault,
  eventNotification,
  failure,
  parseHostCallResponse,
  parseRpcRequest,
  success,
  type HostCallRequest,
  type RpcEnvelope,
} from "./protocol.js";
import { type RpcClient, RpcService } from "./service.js";

export interface JsonlServerOptions {
  maxFrameBytes?: number;
  onShutdown?: () => void | Promise<void>;
  /** Receives the bridge once stdio is up, so run_check can reach the host. */
  onHostBridge?: (bridge: HostBridge) => void;
  hostCallTimeoutMs?: number;
}

class FrameDecoder {
  readonly #maxFrameBytes: number;
  #buffer = Buffer.alloc(0);
  #discarding = false;
  readonly #onFrame: (line: string) => void;
  readonly #onOversize: () => void;

  constructor(
    maxFrameBytes: number,
    onFrame: (line: string) => void,
    onOversize: () => void,
  ) {
    this.#maxFrameBytes = maxFrameBytes;
    this.#onFrame = onFrame;
    this.#onOversize = onOversize;
  }

  push(chunk: Buffer | string): void {
    const bytes = typeof chunk === "string" ? Buffer.from(chunk) : chunk;
    let offset = 0;
    while (offset < bytes.length) {
      const newline = bytes.indexOf(0x0a, offset);
      const end = newline === -1 ? bytes.length : newline;
      const segment = bytes.subarray(offset, end);
      if (!this.#discarding) {
        if (this.#buffer.length + segment.length > this.#maxFrameBytes) {
          this.#buffer = Buffer.alloc(0);
          this.#discarding = true;
        } else if (segment.length > 0) {
          this.#buffer = Buffer.concat([this.#buffer, segment]);
        }
      }
      if (newline === -1) return;
      if (this.#discarding) {
        this.#discarding = false;
        this.#onOversize();
      } else {
        const line = this.#buffer.toString("utf8").replace(/\r$/, "");
        this.#buffer = Buffer.alloc(0);
        if (line.length > 0) this.#onFrame(line);
      }
      offset = newline + 1;
    }
  }
}

export class JsonlRpcServer {
  readonly #service: RpcService;
  readonly #input: Readable;
  readonly #output: Writable;
  readonly #client: RpcClient;
  readonly #decoder: FrameDecoder;
  readonly #onShutdown?: () => void | Promise<void>;
  readonly #bridge: HostBridge;
  #closed = false;

  constructor(
    service: RpcService,
    input: Readable,
    output: Writable,
    options: JsonlServerOptions = {},
  ) {
    this.#service = service;
    this.#input = input;
    this.#output = output;
    if (options.onShutdown !== undefined) this.#onShutdown = options.onShutdown;
    this.#bridge = new HostBridge(
      (frame: HostCallRequest) => this.#writeRaw(frame),
      options.hostCallTimeoutMs,
    );
    options.onHostBridge?.(this.#bridge);
    this.#client = {
      subscriptions: new Set<string>(),
      deliver: (event) => this.#write(eventNotification(event)),
    };
    this.#decoder = new FrameDecoder(
      options.maxFrameBytes ?? MAX_FRAME_BYTES,
      (line) => void this.#handleLine(line),
      () =>
        this.#write(
          failure(
            "__frame__",
            new RpcFault(
              "FRAME_TOO_LARGE",
              `RPC frame exceeds ${options.maxFrameBytes ?? MAX_FRAME_BYTES} bytes`,
            ),
          ),
        ),
    );
  }

  start(): void {
    this.#input.on("data", this.#onData);
    this.#input.once("end", this.#onEnd);
    this.#input.once("error", this.#onEnd);
  }

  readonly #onData = (chunk: Buffer | string): void => {
    this.#decoder.push(chunk);
  };

  readonly #onEnd = (): void => {
    this.close();
  };

  async #handleLine(line: string): Promise<void> {
    let raw: unknown;
    try {
      raw = JSON.parse(line);
    } catch {
      this.#write(
        failure(
          "__invalid__",
          new RpcFault("INVALID_REQUEST", "RPC frame is not valid JSON"),
        ),
      );
      return;
    }
    // A host answer to our own outbound call, not a request. Route it before
    // request parsing, which would otherwise reject it as malformed.
    const hostAnswer = parseHostCallResponse(raw);
    if (hostAnswer !== undefined) {
      this.#bridge.settle(
        hostAnswer.id,
        hostAnswer.hostResult,
        hostAnswer.hostError?.message,
      );
      return;
    }
    let id = "__invalid__";
    try {
      if (raw !== null && typeof raw === "object" && !Array.isArray(raw)) {
        const candidate = (raw as Record<string, unknown>).id;
        if (typeof candidate === "string" && candidate.length <= 128)
          id = candidate;
      }
      const request = parseRpcRequest(raw);
      id = request.id;
      const result = await this.#service.dispatch(request, this.#client);
      const response = success(id, result);
      if (request.method === "shutdown") {
        // The CLI deliberately exits after shutdown to defeat leaked SDK
        // timers. Prove the response reached stdout before allowing that exit.
        await this.#writeAndFlush(response);
        await this.#onShutdown?.();
      } else {
        this.#write(response);
      }
    } catch (error) {
      this.#write(failure(id, this.#service.normalizeError(error)));
    }
  }

  #write(envelope: RpcEnvelope): void {
    this.#writeRaw(envelope);
  }

  #writeRaw(frame: unknown): void {
    if (this.#closed || this.#output.destroyed) return;
    this.#output.write(`${JSON.stringify(frame)}\n`);
  }

  async #writeAndFlush(envelope: RpcEnvelope): Promise<void> {
    if (this.#closed || this.#output.destroyed) return;
    const frame = `${JSON.stringify(envelope)}\n`;
    await new Promise<void>((resolve, reject) => {
      this.#output.write(frame, (error: Error | null | undefined) => {
        if (error !== null && error !== undefined) reject(error);
        else resolve();
      });
    });
  }

  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    this.#input.off("data", this.#onData);
    // A model waiting on a check the host can no longer answer must be told,
    // or its turn blocks until the iteration budget runs out.
    this.#bridge.close("the host connection ended");
    this.#service.disconnect(this.#client);
  }
}

export function protocolReadyMessage(): string {
  return JSON.stringify({ version: RPC_VERSION, transport: "ndjson-stdio" });
}
