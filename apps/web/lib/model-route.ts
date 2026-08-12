import type { ModelPreference, ProjectMode } from "@/lib/types";

export type ModelRoute = "local" | "ollama_cloud" | "oci" | "cohere" | "cline";

/** Resolve the saved conversation provider before considering Ollama pinning. */
export function conversationModelRoute(
  provider: ModelPreference["provider"],
  hostedOllamaPinned: boolean,
): ModelRoute {
  if (provider === "oci" || provider === "cohere" || provider === "cline") {
    return provider;
  }
  return hostedOllamaPinned ? "ollama_cloud" : "local";
}

/** A configured key plus the backend-owned catalog is the selectable lane. */
export function clinePassReady(
  available: boolean,
  models: readonly string[],
): boolean {
  return available && models.length > 0;
}

/** Whether an unmapped project has a configured cloud planner for its first map. */
export function projectMappingReady(
  preference: Pick<
    ModelPreference,
    "oci_available" | "cohere_available" | "cline_available" | "cline_models"
  > | null | undefined,
): boolean {
  if (!preference) return false;
  return preference.oci_available
    || preference.cohere_available
    || clinePassReady(preference.cline_available, preference.cline_models);
}

/** Local launch controls only belong to an actual Ollama-routed turn. */
export function shouldShowLocalSession(
  projectScoped: boolean,
  projectMode: ProjectMode,
  provider: ModelPreference["provider"],
  route: ModelRoute,
): boolean {
  if (!projectScoped) return route === "local";
  return projectMode === "grok_bootstrap_local" && provider === "local";
}
