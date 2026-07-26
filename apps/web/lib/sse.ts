import type { RunEventV1 } from "./types";

export interface SseFrame {
  event?: string;
  id?: string;
  data: string;
  retry?: number;
}

export function parseSseBuffer(buffer: string): { frames: SseFrame[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  const rest = blocks.pop() ?? "";
  const frames: SseFrame[] = [];

  for (const block of blocks) {
    let event: string | undefined;
    let id: string | undefined;
    let retry: number | undefined;
    const data: string[] = [];

    for (const line of block.split("\n")) {
      if (!line || line.startsWith(":")) continue;
      const separator = line.indexOf(":");
      const field = separator === -1 ? line : line.slice(0, separator);
      let value = separator === -1 ? "" : line.slice(separator + 1);
      if (value.startsWith(" ")) value = value.slice(1);

      if (field === "event") event = value;
      if (field === "id") id = value;
      if (field === "data") data.push(value);
      if (field === "retry" && /^\d+$/.test(value)) retry = Number(value);
    }

    if (data.length || event || id) frames.push({ event, id, data: data.join("\n"), retry });
  }

  return { frames, rest };
}

export function normalizeRunEvent(
  raw: unknown,
  frame: Pick<SseFrame, "event" | "id">,
  fallbackRunId: string,
  fallbackSequence: number,
): RunEventV1 {
  const value = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const payloadValue = value.payload;
  const payload =
    payloadValue && typeof payloadValue === "object"
      ? (payloadValue as Record<string, unknown>)
      : typeof raw === "string"
        ? { message: raw }
        : value;
  const sequenceValue = value.sequence ?? value.seq ?? frame.id;
  const parsedSequence = Number(sequenceValue);

  return {
    id: String(value.id ?? frame.id ?? `${fallbackRunId}:${fallbackSequence}`),
    sequence: Number.isFinite(parsedSequence) ? parsedSequence : fallbackSequence,
    run_id: String(value.run_id ?? value.runId ?? fallbackRunId),
    thread_id: value.thread_id ? String(value.thread_id) : undefined,
    checkpoint_id: value.checkpoint_id ? String(value.checkpoint_id) : undefined,
    type: String(value.type ?? value.event ?? frame.event ?? "run.event"),
    timestamp: value.timestamp ? String(value.timestamp) : undefined,
    payload,
  };
}
