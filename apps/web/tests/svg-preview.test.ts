import assert from "node:assert/strict";
import test from "node:test";

import { isAllowedSvgReference } from "../lib/svg-preview.ts";

test("SVG preview references remain embedded or fragment-local", () => {
  assert.equal(isAllowedSvgReference("#local-gradient"), true);
  assert.equal(isAllowedSvgReference("data:image/png;base64,iVBORw0KGgo="), true);
  assert.equal(isAllowedSvgReference("https://example.invalid/tracker.png"), false);
  assert.equal(isAllowedSvgReference("/usr/local/icon.png"), false);
  assert.equal(isAllowedSvgReference("javascript:alert(1)"), false);
  assert.equal(isAllowedSvgReference("data:image/svg+xml,<svg/>"), false);
});
