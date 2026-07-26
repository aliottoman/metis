"use client";

import type { ReactNode } from "react";

type MarkdownContentProps = {
  content: string;
};

function safeHref(raw: string): string | null {
  const href = raw.trim();
  if (/^(https?:|mailto:)/i.test(href) || href.startsWith("/") || href.startsWith("#")) {
    return href;
  }
  return null;
}

function inlineMarkdown(text: string, keyPrefix: string): ReactNode[] {
  const pattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|~~[^~\n]+~~|\[[^\]\n]+\]\([^) \n]+(?:\s+"[^"]*")?\)|\*[^*\n]+\*|_[^_\n]+_)/g;
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
      nodes.push(<strong key={key}>{inlineMarkdown(token.slice(2, -2), key)}</strong>);
    } else if (token.startsWith("~~")) {
      nodes.push(<del key={key}>{inlineMarkdown(token.slice(2, -2), key)}</del>);
    } else if (token.startsWith("[")) {
      const link = token.match(/^\[([^\]]+)\]\(([^) \n]+)(?:\s+"[^"]*")?\)$/);
      const href = link ? safeHref(link[2]) : null;
      nodes.push(
        href
          ? <a key={key} href={href} target={href.startsWith("http") ? "_blank" : undefined} rel={href.startsWith("http") ? "noreferrer" : undefined}>{inlineMarkdown(link![1], key)}</a>
          : token,
      );
    } else {
      nodes.push(<em key={key}>{inlineMarkdown(token.slice(1, -1), key)}</em>);
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

function renderParagraph(lines: string[], key: string): ReactNode {
  return (
    <p key={key}>
      {lines.map((line, index) => (
        <span key={`${key}-${index}`}>
          {index > 0 ? " " : null}
          {inlineMarkdown(line.trim(), `${key}-${index}`)}
        </span>
      ))}
    </p>
  );
}

export function MarkdownContent({ content }: MarkdownContentProps) {
  const lines = content.replace(/\r\n?/g, "\n").split("\n");
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
      const children = inlineMarkdown(heading[2].trim(), `heading-${index}`);
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
            <thead><tr>{headers.map((cell, cellIndex) => <th key={cellIndex} style={{ textAlign: alignments[cellIndex] as "left" | "center" | "right" }}>{inlineMarkdown(cell, `th-${index}-${cellIndex}`)}</th>)}</tr></thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {headers.map((_, cellIndex) => <td key={cellIndex} style={{ textAlign: alignments[cellIndex] as "left" | "center" | "right" }}>{inlineMarkdown(row[cellIndex] ?? "", `td-${index}-${rowIndex}-${cellIndex}`)}</td>)}
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
      const items: string[] = [];
      while (index < lines.length) {
        const item = lines[index].match(/^\s*([-+*]|\d+[.)])\s+(.+)$/);
        if (!item || /^\d/.test(item[1]) !== ordered) break;
        items.push(item[2]);
        index += 1;
      }
      const children = items.map((item, itemIndex) => <li key={itemIndex}>{inlineMarkdown(item, `li-${index}-${itemIndex}`)}</li>);
      blocks.push(ordered ? <ol key={`list-${index}`}>{children}</ol> : <ul key={`list-${index}`}>{children}</ul>);
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const quote: string[] = [];
      while (index < lines.length && /^\s*>\s?/.test(lines[index])) {
        quote.push(lines[index].replace(/^\s*>\s?/, ""));
        index += 1;
      }
      blocks.push(<blockquote key={`quote-${index}`}>{renderParagraph(quote, `quote-p-${index}`)}</blockquote>);
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
    blocks.push(renderParagraph(paragraph, `paragraph-${index}`));
  }

  return <div className="messageContent">{blocks}</div>;
}
