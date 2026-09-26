import assert from "node:assert/strict";
import test from "node:test";

import { MARKDOWN_LINK_TOKEN_SOURCE, markdownHref, markdownLinkParts, splitCitedSources } from "../lib/markdown-links.ts";

test("attached-document citations open on the configured API origin", () => {
  const path = `/api/v1/uploads/upl_${"a".repeat(32)}`;
  assert.equal(markdownHref(path, "http://127.0.0.1:8000"), `http://127.0.0.1:8000${path}`);
  assert.equal(markdownHref(path, "http://localhost:8001/"), `http://localhost:8001${path}`);
  assert.equal(markdownHref(path, ""), path);
});

test("customer, meeting, and conversation sources remain app navigation", () => {
  for (const href of ["/customers?account=cust_a&tab=facts&fact=cfact_a", "/meetings?meeting=mtg_a&turn=mturn_a", "/?conversation=conv_a&run=run_a", "#source", "https://notion.so/page", "mailto:person@example.com"]) {
    assert.equal(markdownHref(href, "http://127.0.0.1:8000"), href);
  }
});

test("a malformed upload link cannot become another API request", () => {
  for (const href of [
    "/api/v1/uploads/../settings",
    "/api/v1/uploads/upl_wrong",
    `/api/v1/uploads/upl_${"a".repeat(20)}`,
    `/api/v1/uploads/upl_${"a".repeat(31)}`,
    `/api/v1/uploads/upl_${"a".repeat(33)}`,
    `/api/v1/uploads/upl_${"g".repeat(32)}`,
    `/api/v1/uploads/upl_${"A".repeat(32)}`,
    `/api/v1/uploads/upl_${"a".repeat(32)}?download=1`,
  ]) {
    assert.equal(markdownHref(href, "http://127.0.0.1:8000"), null);
  }
  assert.equal(markdownHref("/api/v1/settings", "http://127.0.0.1:8000"), "/api/v1/settings");
  for (const href of ["javascript:alert(1)", "//example.com", "/\\example.com"]) {
    assert.equal(markdownHref(href, "http://127.0.0.1:8000"), null);
  }
});

test("final cited sources are separated for inline citation links", () => {
  const uploadPath = `/api/v1/uploads/upl_${"a".repeat(32)}`;
  assert.deepEqual(splitCitedSources(`A fact [2].\n\n**Sources**\n[2] NASA — [nasa.gov](https://www.nasa.gov/page)\n[4] Notes — [document](${uploadPath})\n`), {
    body: "A fact [2].",
    sources: [
      { number: 2, content: "NASA — [nasa.gov](https://www.nasa.gov/page)" },
      { number: 4, content: `Notes — [document](${uploadPath})` },
    ],
  });
});

test("ordinary text and incomplete source sections stay in the answer", () => {
  for (const value of [
    "The word Sources appears here.\n[1] This is an item.",
    "Answer\n\n**Sources**\nThis is unfinished.",
    "Answer\n\n**Sources**\n[1] First\n[1] Duplicate",
  ]) {
    assert.deepEqual(splitCitedSources(value), { body: value, sources: [] });
  }
});

test("source links keep parentheses inside page URLs", () => {
  const token = "[Wikipedia](https://en.wikipedia.org/wiki/Example_(film))";
  assert.deepEqual(("Source " + token + ".").match(new RegExp(MARKDOWN_LINK_TOKEN_SOURCE, "g")), [token]);
  assert.deepEqual(markdownLinkParts(token), {
    label: "Wikipedia",
    target: "https://en.wikipedia.org/wiki/Example_(film)",
  });
});
