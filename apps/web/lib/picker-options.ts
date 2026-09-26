/** The displayed order is also the keyboard order, including interleaved groups. */
export function groupPickerOptions<T extends { group?: string }>(options: T[], grouped = true): { group: string; items: T[] }[] {
  if (!grouped) return [{ group: "", items: options }];
  const groups = new Map<string, T[]>();
  for (const option of options) {
    const key = option.group ?? "";
    const items = groups.get(key);
    if (items) items.push(option);
    else groups.set(key, [option]);
  }
  return Array.from(groups, ([group, items]) => ({ group, items }));
}

export function enabledPickerIndexes<T extends { disabled?: boolean }>(options: T[]): number[] {
  return options.flatMap((option, index) => option.disabled ? [] : [index]);
}
