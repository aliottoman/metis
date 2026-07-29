/**
 * Folder drops on the composer.
 *
 * A browser hands a dropped directory over as a `FileSystemDirectoryEntry`: we
 * learn its name and can walk its contents, but we never learn its absolute
 * path. Both routes a folder can take are addressed on the host instead — a
 * corpus source by absolute path, a project workspace by catalog id — so the
 * work here is to describe the drop well enough for the user to choose a
 * route, and to resolve the folder against catalogs the browser has already
 * loaded so that a folder Metis already knows needs no typing.
 */

/** Mirrors `corpus._SKIP_DIRS`: the preview should count what indexing counts. */
export const SKIP_DIRECTORIES: ReadonlySet<string> = new Set([
  ".git", ".venv", "venv", "node_modules", "__pycache__", ".wakil", ".data",
  ".idea", ".vscode", "dist", "build", ".next", ".mypy_cache", ".pytest_cache",
  ".uv-cache", ".pnpm-store", ".ruff_cache", ".turbo", "target", ".gradle",
]);

/**
 * A drop is a preview, not an index, so the walk is bounded — dropping a
 * monorepo must not stall the composer. Past the bound the counts are a floor
 * and the scan says so rather than quietly under-reporting.
 */
export const FOLDER_SCAN_LIMITS = { maxFiles: 1_200, maxDepth: 8 } as const;

/**
 * Attaching a folder means one upload request per file, so past this the
 * attach route is withdrawn and indexing is the honest suggestion.
 */
export const MAX_ATTACHABLE_FILES = 25;

/**
 * The `max_text_attachment_bytes` default. Advisory only — the host is the
 * real gate (422 on the aggregate) — but it lets the sheet warn before
 * spending a round trip per file.
 */
export const ATTACHMENT_TEXT_BUDGET_BYTES = 64 * 1024;

export interface DroppedFile {
  /** Path relative to the dropped folder, e.g. `src/api.ts`. */
  path: string;
  name: string;
  size: number;
  file: File;
}

export interface FolderScan {
  name: string;
  files: DroppedFile[];
  /** Build and vendor directories passed over — reported, never hidden. */
  skippedDirectories: number;
  /** True when the walk hit a bound, making every count a floor. */
  truncated: boolean;
}

/**
 * DataTransfer items are neutered as soon as the drop handler returns, so
 * entries must be claimed synchronously — before the caller's first await.
 */
export function droppedDirectories(
  items: DataTransferItemList | null | undefined,
): FileSystemDirectoryEntry[] {
  if (!items) return [];
  const directories: FileSystemDirectoryEntry[] = [];
  for (const item of Array.from(items)) {
    if (item.kind !== "file" || typeof item.webkitGetAsEntry !== "function") continue;
    const entry = item.webkitGetAsEntry();
    if (entry?.isDirectory) directories.push(entry as FileSystemDirectoryEntry);
  }
  return directories;
}

/**
 * A dropped directory also shows up in `dataTransfer.files` as a File that no
 * upload can read, so the loose files are what remains once the directories
 * are named out. Name matching is the only signal a browser offers here.
 */
export function looseFiles(
  files: FileList | null | undefined,
  directories: readonly FileSystemDirectoryEntry[],
): File[] {
  const directoryNames = new Set(directories.map((entry) => entry.name));
  return Array.from(files ?? []).filter((file) => !directoryNames.has(file.name));
}

function readEntries(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
  return new Promise((resolve) => reader.readEntries(resolve, () => resolve([])));
}

/**
 * `readEntries` yields one batch per call (100 in Chromium) and signals the
 * end with an empty batch, so every directory needs a drain loop. The batch
 * ceiling keeps a misbehaving reader from spinning forever.
 */
async function readDirectory(directory: FileSystemDirectoryEntry): Promise<FileSystemEntry[]> {
  const reader = directory.createReader();
  const entries: FileSystemEntry[] = [];
  for (let batch = 0; batch < 400; batch += 1) {
    const next = await readEntries(reader);
    if (!next.length) break;
    entries.push(...next);
  }
  return entries;
}

/** An unreadable file is skipped; one denied entry must not fail the walk. */
function fileOf(entry: FileSystemFileEntry): Promise<File | null> {
  return new Promise((resolve) => entry.file(resolve, () => resolve(null)));
}

export async function scanFolderEntry(
  entry: FileSystemDirectoryEntry,
  limits: { maxFiles: number; maxDepth: number } = FOLDER_SCAN_LIMITS,
): Promise<FolderScan> {
  const files: DroppedFile[] = [];
  let skippedDirectories = 0;
  let truncated = false;
  const queue = [{ directory: entry, prefix: "", depth: 0 }];

  while (queue.length) {
    const current = queue.shift();
    if (!current) break;
    if (files.length >= limits.maxFiles) {
      truncated = true;
      break;
    }
    for (const child of await readDirectory(current.directory)) {
      if (child.isDirectory) {
        if (SKIP_DIRECTORIES.has(child.name)) {
          skippedDirectories += 1;
          continue;
        }
        if (current.depth + 1 > limits.maxDepth) {
          truncated = true;
          continue;
        }
        queue.push({
          directory: child as FileSystemDirectoryEntry,
          prefix: `${current.prefix}${child.name}/`,
          depth: current.depth + 1,
        });
        continue;
      }
      if (files.length >= limits.maxFiles) {
        truncated = true;
        break;
      }
      const file = await fileOf(child as FileSystemFileEntry);
      if (!file) continue;
      files.push({
        path: `${current.prefix}${child.name}`,
        name: child.name,
        size: file.size,
        file,
      });
    }
  }

  return { name: entry.name, files, skippedDirectories, truncated };
}

function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot).toLowerCase() : "";
}

/**
 * The accept list is passed in rather than imported so this module stays free
 * of runtime dependencies and testable on its own.
 */
export function attachableFiles(
  files: readonly DroppedFile[],
  accept: string,
): DroppedFile[] {
  const extensions = new Set(
    accept.split(",").map((value) => value.trim().toLowerCase()).filter((value) => value.startsWith(".")),
  );
  return files.filter((item) => extensions.has(extensionOf(item.name)));
}

export function totalBytes(files: readonly DroppedFile[]): number {
  return files.reduce((total, item) => total + item.size, 0);
}

const CODE_EXTENSIONS = new Set([
  ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs",
  ".java", ".kt", ".rb", ".php", ".c", ".h", ".cpp", ".hpp", ".cc", ".cs",
  ".swift", ".lua", ".sql", ".sh", ".bash", ".zsh",
]);
const DOC_EXTENSIONS = new Set([
  ".md", ".markdown", ".mdx", ".rst", ".txt", ".text", ".pdf", ".docx",
]);

/** Pre-selects the source kind so a dropped repo does not arrive as "mixed". */
export function suggestCorpusKind(files: readonly DroppedFile[]): "code" | "docs" | "mixed" {
  let code = 0;
  let docs = 0;
  for (const item of files) {
    const extension = extensionOf(item.name);
    if (CODE_EXTENSIONS.has(extension)) code += 1;
    else if (DOC_EXTENSIONS.has(extension)) docs += 1;
  }
  const total = code + docs;
  if (!total) return "mixed";
  if (code / total >= 0.6) return "code";
  if (docs / total >= 0.6) return "docs";
  return "mixed";
}

export function basename(path: string): string {
  const trimmed = path.replace(/[/\\]+$/, "");
  const cut = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return cut >= 0 ? trimmed.slice(cut + 1) : trimmed;
}

export function parentDirectory(path: string): string | null {
  const trimmed = path.replace(/[/\\]+$/, "");
  const cut = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  if (cut <= 0) return null;
  return trimmed.slice(0, cut);
}

/** Folder names reach the project catalog with `-`/`_` collapsed to spaces. */
export function normalizeFolderName(value: string): string {
  return value.replace(/[_\-\s]+/g, " ").trim().toLowerCase();
}

export function findSourceForFolder<T extends { root_path: string }>(
  folderName: string,
  sources: readonly T[],
): T | undefined {
  return sources.find((source) => basename(source.root_path) === folderName)
    ?? sources.find((source) => normalizeFolderName(basename(source.root_path)) === normalizeFolderName(folderName));
}

export function findProjectForFolder<T extends { name: string }>(
  folderName: string,
  projects: readonly T[],
): T | undefined {
  return projects.find((project) => project.name === folderName)
    ?? projects.find((project) => normalizeFolderName(project.name) === normalizeFolderName(folderName));
}

/**
 * Existing sources hold the only absolute paths the browser ever sees, so the
 * directory that already holds the most of them is the best available guess
 * for where a newly dropped folder lives. Wrong guesses cost one correction —
 * the host rejects a path that is not a directory.
 *
 * Mirrors Metis manages itself (the Notion and run-history corpora under
 * `.data/`) would otherwise dominate the count and point every drop at the
 * app's own storage, so any path the indexer would refuse to walk is skipped.
 */
export function guessRootPath(
  folderName: string,
  sourcePaths: readonly string[],
): string | null {
  const counts = new Map<string, number>();
  for (const path of sourcePaths) {
    if (path.split(/[/\\]/).some((segment) => SKIP_DIRECTORIES.has(segment))) continue;
    const parent = parentDirectory(path);
    if (!parent) continue;
    counts.set(parent, (counts.get(parent) ?? 0) + 1);
  }
  const ranked = [...counts.entries()].sort(
    (left, right) => right[1] - left[1] || left[0].localeCompare(right[0]),
  );
  const parent = ranked[0]?.[0];
  return parent ? `${parent}/${folderName}` : null;
}

export function formatByteSize(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}
