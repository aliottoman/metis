// Pure helpers for the agent factory page: the brief form's parsing and
// validation, the embed snippet, and the words for a test run. No fetch, no
// React — everything here is testable with plain assertions.

import type { StatusState } from "@/components/ui/status";
import type { AgentBrief, AgentStatus, AgentTestResult, AgentTestsStage, VoiceAgent } from "@/lib/types";

/** What the brief form holds: the URLs and notes are free text until parsed. */
export interface AgentBriefDraft {
  company: string;
  brief: string;
  source_urls: string;
  notes: string;
}

export const EMPTY_BRIEF: AgentBriefDraft = {
  company: "",
  brief: "",
  source_urls: "",
  notes: "",
};

export const MAX_SOURCE_URLS = 6;

/** One address per line, comma, or space; repeats dropped in order. */
export function parseSourceUrls(text: string): string[] {
  const seen = new Set<string>();
  const urls: string[] = [];
  for (const raw of text.split(/[\s,]+/)) {
    const url = raw.trim();
    if (!url || seen.has(url)) continue;
    seen.add(url);
    urls.push(url);
  }
  return urls;
}

/** Lines of a textarea as a list: one item per line, blanks dropped. */
export function splitLines(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

export function validateBrief(
  draft: AgentBriefDraft,
): Partial<Record<keyof AgentBriefDraft, string>> {
  const problems: Partial<Record<keyof AgentBriefDraft, string>> = {};
  if (!draft.company.trim()) problems.company = "Name the company.";
  if (!draft.brief.trim()) problems.brief = "Say what the agent is for.";
  const urls = parseSourceUrls(draft.source_urls);
  const bad = urls.find((url) => !/^https?:\/\//.test(url));
  if (bad) problems.source_urls = `Not a web address: ${bad}`;
  else if (urls.length > MAX_SOURCE_URLS) {
    problems.source_urls = `At most ${MAX_SOURCE_URLS} pages.`;
  }
  return problems;
}

/** The request the API takes, or null while the form is incomplete. */
export function briefToRequest(draft: AgentBriefDraft): AgentBrief | null {
  if (Object.keys(validateBrief(draft)).length) return null;
  return {
    company: draft.company.trim(),
    brief: draft.brief.trim(),
    source_urls: parseSourceUrls(draft.source_urls),
    notes: draft.notes.trim(),
  };
}

/** The two lines a prospect pastes into any page. Mirrors the API's copy. */
export function embedSnippet(agentId: string): string {
  return (
    `<elevenlabs-convai agent-id="${agentId}"></elevenlabs-convai>\n` +
    '<script src="https://unpkg.com/@elevenlabs/convai-widget-embed" async type="text/javascript"></script>'
  );
}

export function statusLabel(status: AgentStatus): string {
  if (status === "deployed") return "live";
  if (status === "failed") return "needs review";
  return "draft";
}

/** One line about the test run: what it is doing, or how it went. */
export function testsSummary(agent: Pick<VoiceAgent, "tests" | "tests_stage">): string {
  const stage: AgentTestsStage = agent.tests_stage;
  if (stage === "running") return "Running the simulations";
  if (stage === "failed") return "The test run failed";
  if (stage !== "ready") return "Not run yet";
  const passed = agent.tests.filter((item) => item.status === "passed").length;
  return `${passed} of ${agent.tests.length} passed`;
}

/** An agent's status in the app's five-word vocabulary. */
export function agentState(status: AgentStatus): StatusState {
  if (status === "deployed") return "live";
  if (status === "failed") return "needs-review";
  return "ready";
}

/** A test result in the same vocabulary. */
export function testState(status: AgentTestResult["status"]): StatusState {
  if (status === "passed") return "live";
  if (status === "failed") return "needs-review";
  return "waiting";
}
