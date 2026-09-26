import assert from "node:assert/strict";
import test from "node:test";

import { enabledPickerIndexes, groupPickerOptions } from "../lib/picker-options.ts";

test("keyboard indexes follow the displayed order when groups are interleaved", () => {
  const input = [
    { id: "alpha", group: "Local" },
    { id: "beta", group: "Cloud" },
    { id: "gamma", group: "Local", disabled: true },
    { id: "delta", group: "Local" },
  ];
  const sections = groupPickerOptions(input);
  const displayed = sections.flatMap((section) => section.items);
  assert.deepEqual(sections.map((section) => section.group), ["Local", "Cloud"]);
  assert.deepEqual(displayed.map((option) => option.id), ["alpha", "gamma", "delta", "beta"]);
  assert.deepEqual(enabledPickerIndexes(displayed).map((index) => displayed[index]?.id), ["alpha", "delta", "beta"]);
  assert.deepEqual(input.map((option) => option.id), ["alpha", "beta", "gamma", "delta"]);
});

test("search results retain their ranking without regrouping", () => {
  const ranked = [{ id: "best", group: "A" }, { id: "second", group: "B" }, { id: "third", group: "A" }];
  assert.deepEqual(groupPickerOptions(ranked, false), [{ group: "", items: ranked }]);
  assert.deepEqual(enabledPickerIndexes([{ disabled: true }]), []);
});
