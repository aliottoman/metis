// Internal operation/classifier tokens the backend emits verbatim on
// run.model_fallback / run.model_exhausted (e.g. "clinecore_round",
// "provider_exhausted"). The activity timeline must never surface these
// raw snake_case tokens to an end user — humanize known ones, and
// de-jargon anything unrecognized rather than passing it through as-is.

export const OPERATION_LABELS: Record<string, string> = {
  clinecore_round: "building this vertical slice",
  project_step: "this build step",
  project_plan_files: "planning the file list",
};

export const BACKEND_REASON_LABELS: Record<string, string> = {
  provider_exhausted: "its account-wide limit was hit",
  rate_limited: "it is rate-limiting requests",
  backend_error: "it returned a server error",
  unchanged_verifier_findings: "repeated the same verifier findings",
  repairable_import_rejection: "an in-scope issue needs a repair pass",
  empty_manifest: "no files were produced",
  coder_ladder_exhausted: "every coding model in the fallback chain was tried",
};

export function humanizeToken(value: string, labels: Record<string, string>): string {
  return labels[value] ?? value.replaceAll("_", " ");
}
