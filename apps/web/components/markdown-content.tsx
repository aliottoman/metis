"use client";

import { useId, type ReactNode } from "react";

import { API_BASE } from "@/lib/api";
import { MARKDOWN_LINK_TOKEN_SOURCE, markdownHref, markdownLinkParts, splitCitedSources, type CitedSource } from "@/lib/markdown-links";

type MarkdownContentProps = {
  content: string;
};

type CitationLinks = { ids: Set<number>; prefix: string };

const INLINE_TOKEN_SOURCE = [
  /`[^`\n]+`/.source,
  /\*\*[^*\n]+\*\*/.source,
  /__[^_\n]+__/.source,
  /~~[^~\n]+~~/.source,
  MARKDOWN_LINK_TOKEN_SOURCE,
  /\[\d+\]/.source,
  /\*[^*\n]+\*/.source,
  /_[^_\n]+_/.source,
].join("|");

function inlineMarkdown(text: string, keyPrefix: string, citations?: CitationLinks): ReactNode[] {
  const pattern = new RegExp(INLINE_TOKEN_SOURCE, "g");
  const nodes: ReactNode[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;
  let index = 0;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) nodes.push(text.slice(cursor, match.index));
    const token = match[0];
    const key = `${keyPrefix}-${index}`;

    if (token.startsWith("`")) {
      nodes.push(<code key={key}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith("**") || token.startsWith("__")) {
      nodes.push(<strong key={key}>{inlineMarkdown(token.slice(2, -2), key, citations)}</strong>);
    } else if (token.startsWith("~~")) {
      nodes.push(<del key={key}>{inlineMarkdown(token.slice(2, -2), key, citations)}</del>);
    } else if (token.startsWith("[")) {
      const link = markdownLinkParts(token);
      const href = link ? markdownHref(link.target, API_BASE) : null;
      if (href && link) {
        nodes.push(<a key={key} href={href} target={href.startsWith("http") ? "_blank" : undefined} rel={href.startsWith("http") ? "noreferrer" : undefined}>{inlineMarkdown(link.label, key)}</a>);
      } else {
        const number = /^\[(\d+)\]$/.exec(token);
        nodes.push(number && citations?.ids.has(Number(number[1]))
          ? <a key={key} className="message-citation" href={`#${citations.prefix}-source-${number[1]}`} aria-label={`Jump to source ${number[1]}`}>{token}</a>
          : token);
      }
    } else {
      nodes.push(<em key={key}>{inlineMarkdown(token.slice(1, -1), key, citations)}</em>);
    }
    cursor = match.index + token.length;
    index += 1;
  }

  if (cursor < text.length) nodes.push(text.slice(cursor));
  return nodes;
}

function splitTableRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isTableDivider(line: string): boolean {
  const cells = splitTableRow(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function renderParagraph(lines: string[], key: string, citations?: CitationLinks): ReactNode {
  return (
    <p key={key}>
      {lines.map((line, index) => (
        <span key={`${key}-${index}`}>
          {index > 0 ? " " : null}
          {inlineMarkdown(line.trim(), `${key}-${index}`, citations)}
        </span>
      ))}
    </p>
  );
}

export function MarkdownContent({ content }: MarkdownContentProps) {
  const prefix = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const { body, sources } = splitCitedSources(content);
  const citations: CitationLinks | undefined = sources.length
    ? { ids: new Set(sources.map((source) => source.number)), prefix }
    : undefined;
  const lines = body.split("\n");
  const blocks: ReactNode[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    const fence = line.match(/^```([\w.+-]*)\s*$/);
    if (fence) {
      const code: string[] = [];
      index += 1;
      while (index < lines.length && !/^```\s*$/.test(lines[index])) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      blocks.push(
        <pre key={`code-${index}`}>
          <span>{fence[1] || "code"}</span>
          <code>{code.join("\n")}</code>
        </pre>,
      );
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      const children = inlineMarkdown(heading[2].trim(), `heading-${index}`, citations);
      if (level === 1) blocks.push(<h1 key={`heading-${index}`}>{children}</h1>);
      else if (level === 2) blocks.push(<h2 key={`heading-${index}`}>{children}</h2>);
      else if (level === 3) blocks.push(<h3 key={`heading-${index}`}>{children}</h3>);
      else if (level === 4) blocks.push(<h4 key={`heading-${index}`}>{children}</h4>);
      else if (level === 5) blocks.push(<h5 key={`heading-${index}`}>{children}</h5>);
      else blocks.push(<h6 key={`heading-${index}`}>{children}</h6>);
      index += 1;
      continue;
    }

    if (/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      blocks.push(<hr key={`rule-${index}`} />);
      index += 1;
      continue;
    }

    if (line.includes("|") && index + 1 < lines.length && isTableDivider(lines[index + 1])) {
      const headers = splitTableRow(line);
      const alignments = splitTableRow(lines[index + 1]).map((cell) => {
        if (cell.startsWith(":") && cell.endsWith(":")) return "center";
        if (cell.endsWith(":")) return "right";
        return "left";
      });
      const rows: string[][] = [];
      index += 2;
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(splitTableRow(lines[index]));
        index += 1;
      }
      blocks.push(
        <div className="messageTableWrap" key={`table-${index}`}>
          <table>
            <thead><tr>{headers.map((cell, cellIndex) => <th key={cellIndex} style={{ textAlign: alignments[cellIndex] as "left" | "center" | "right" }}>{inlineMarkdown(cell, `th-${index}-${cellIndex}`, citations)}</th>)}</tr></thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {headers.map((_, cellIndex) => <td key={cellIndex} style={{ textAlign: alignments[cellIndex] as "left" | "center" | "right" }}>{inlineMarkdown(row[cellIndex] ?? "", `td-${index}-${rowIndex}-${cellIndex}`, citations)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }

    const list = line.match(/^\s*([-+*]|\d+[.)])\s+(.+)$/);
    if (list) {
      const ordered = /^\d/.test(list[1]);
      // Honor the first marker's number so `3. 4. 5.` renders 3,4,5; the rest
      // increment positionally, so a model that writes `1.` on every line still
      // yields 1,2,3 rather than repeating the literal marker.
      const start = ordered ? Number.parseInt(list[1], 10) || 1 : 1;
      const items: string[] = [];
      while (index < lines.length) {
        const current = lines[index];
        if (!current.trim()) {
          // A blank line ends the list ONLY if what follows isn't another item
          // of the same kind. Otherwise it's a "loose" list (blank lines between
          // items — which models emit constantly) and must stay ONE list, or
          // every item becomes its own <ol> and the counter restarts at 1.
          let lookahead = index + 1;
          while (lookahead < lines.length && !lines[lookahead].trim()) lookahead += 1;
          const nextItem = lookahead < lines.length
            ? lines[lookahead].match(/^\s*([-+*]|\d+[.)])\s+(.+)$/)
            : null;
          if (!nextItem || /^\d/.test(nextItem[1]) !== ordered) break;
          index = lookahead;
          continue;
        }
        const item = current.match(/^\s*([-+*]|\d+[.)])\s+(.+)$/);
        if (!item || /^\d/.test(item[1]) !== ordered) break;
        items.push(item[2]);
        index += 1;
      }
      const children = items.map((item, itemIndex) => <li key={itemIndex}>{inlineMarkdown(item, `li-${index}-${itemIndex}`, citations)}</li>);
      blocks.push(
        ordered
          ? <ol key={`list-${index}`} start={start}>{children}</ol>
          : <ul key={`list-${index}`}>{children}</ul>,
      );
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const quote: string[] = [];
      while (index < lines.length && /^\s*>\s?/.test(lines[index])) {
        quote.push(lines[index].replace(/^\s*>\s?/, ""));
        index += 1;
      }
      blocks.push(<blockquote key={`quote-${index}`}>{renderParagraph(quote, `quote-p-${index}`, citations)}</blockquote>);
      continue;
    }

    const paragraph: string[] = [];
    while (index < lines.length && lines[index].trim()) {
      const next = lines[index];
      if (
        paragraph.length > 0
        && (
          /^```/.test(next)
          || /^(#{1,6})\s+/.test(next)
          || /^\s*([-+*]|\d+[.)])\s+/.test(next)
          || /^\s*>\s?/.test(next)
          || (next.includes("|") && index + 1 < lines.length && isTableDivider(lines[index + 1]))
        )
      ) break;
      paragraph.push(next);
      index += 1;
    }
    blocks.push(renderParagraph(paragraph, `paragraph-${index}`, citations));
  }

  return <div className="messageContent">{blocks}{sources.length ? <SourceList sources={sources} prefix={prefix} /> : null}</div>;
}

function SourceList({ sources, prefix }: { sources: CitedSource[]; prefix: string }) {
  return <section className="message-sources" aria-label="Sources">
    <h3>Sources</h3>
    <ol>
      {sources.map((source) => <li key={source.number} value={source.number} id={`${prefix}-source-${source.number}`}>
        {inlineMarkdown(source.content, `${prefix}-source-${source.number}`)}
      </li>)}
    </ol>
  </section>;
}
