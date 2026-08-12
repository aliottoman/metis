import type { LocalModelSession, ModelPreference } from "@/lib/types";

/** Mirrors the API's is_cloud_model: both hosted spellings Ollama uses. */
export function isCloudModel(id: string | null | undefined): boolean {
  if (!id) return false;
  return id.endsWith("-cloud") || id.endsWith(":cloud");
}

/** The short form of a model id, dropping the tag after the first colon. */
export function shortModel(id: string | null | undefined): string {
  if (!id) return "model";
  return id.split(":")[0] ?? id;
}

/**
 * Whether the next message runs in the cloud — an explicit cloud provider
 * (Grok via OCI, Command A+ via Cohere, or ClinePass) or a hosted Ollama model
 * pinned on the "local" lane. Cloud runs keep no on-device weights resident,
 * so nothing about them should demand a local launch. This is the one predicate
 * the whole web layer should branch on instead of inspecting the local session.
 */
export function isCloudActive(preference: ModelPreference | null | undefined): boolean {
  if (!preference) return false;
  if (
    preference.provider === "oci"
    || preference.provider === "cohere"
    || preference.provider === "cline"
  ) return true;
  return preference.provider === "local" && isCloudModel(preference.model);
}

/**
 * The name to show for whatever runs the next message, matched to the header
 * model control: "Command A+" / "Grok" / "ClinePass" for cloud providers,
 * "Hosted · X" for a hosted Ollama model, otherwise the resident local model.
 * Null only when nothing is active — no cloud pin and no local model ready —
 * which is the sole case that should ever read as "off".
 */
export function activeModelLabel(
  preference: ModelPreference | null | undefined,
  session: LocalModelSession | null | undefined,
): string | null {
  if (preference?.provider === "cohere") return "Command A+";
  if (preference?.provider === "oci") return "Grok";
  if (preference?.provider === "cline") return "ClinePass";
  if (preference?.provider === "local" && isCloudModel(preference.model)) {
    return `Hosted · ${shortModel(preference.model)}`;
  }
  if (session?.state === "ready" && session.selected_model) {
    return shortModel(session.selected_model);
  }
  return null;
}
