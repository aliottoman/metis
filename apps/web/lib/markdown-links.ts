// A source URL can contain parentheses, such as a Wikipedia article title.
// Match one balanced pair inside the target so the final ')' still closes
// the Markdown link.
export const MARKDOWN_LINK_TOKEN_SOURCE = String.raw`\[[^\]\n]+\]\((?:[^()\s]|\([^()\s]*\))+(?:\s+"[^"]*")?\)`;

export function markdownLinkParts(token: string): { label: string; target: string } | null {
  const match = /^\[([^\]\n]+)\]\(((?:[^()\s]|\([^()\s]*\))+)(?:\s+"[^"]*")?\)$/.exec(token);
  return match ? { label: match[1], target: match[2] } : null;
}

/** Resolve attached-source downloads without moving app navigation off-origin. */
export function markdownHref(raw: string, apiBase: string): string | null {
  const href = raw.trim();
  if (/^\/api\/v1\/uploads\/upl_[a-f0-9]{20}$/.test(href)) {
    return `${apiBase.replace(/\/$/, "")}${href}`;
  }
  if (/^(https?:|mailto:)/i.test(href) || href.startsWith("#")) return href;
  if (href.startsWith("/") && !href.startsWith("//") && !href.includes("\\")) return href;
  return null;
}

export interface CitedSource {
  number: number;
  content: string;
}

/** The API appends only cited evidence in a final Sources section. Keep it
 * separate from the answer so inline [n] markers can point to its entries. */
export function splitCitedSources(content: string): { body: string; sources: CitedSource[] } {
  const normalized = content.replace(/\r\n?/g, "\n");
  const lines = normalized.split("\n");
  let end = lines.length;
  while (end > 0 && !lines[end - 1].trim()) end -= 1;
  let start = end;
  const sources: CitedSource[] = [];
  while (start > 0) {
    const match = lines[start - 1].match(/^\[(\d+)\]\s+(.+)$/);
    if (!match) break;
    sources.unshift({ number: Number(match[1]), content: match[2] });
    start -= 1;
  }
  if (!sources.length || start === 0 || lines[start - 1].trim() !== "**Sources**") {
    return { body: normalized, sources: [] };
  }
  const numbers = new Set(sources.map((source) => source.number));
  if (numbers.size !== sources.length || sources.some((source) => !Number.isSafeInteger(source.number) || source.number < 1)) {
    return { body: normalized, sources: [] };
  }
  return { body: lines.slice(0, start - 1).join("\n").trimEnd(), sources };
}
