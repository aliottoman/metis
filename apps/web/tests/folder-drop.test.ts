import assert from "node:assert/strict";
import test from "node:test";

import {
  attachableFiles,
  basename,
  droppedDirectories,
  findProjectForFolder,
  findSourceForFolder,
  formatByteSize,
  guessRootPath,
  looseFiles,
  parentDirectory,
  scanFolderEntry,
  suggestCorpusKind,
  totalBytes,
} from "../lib/folder-drop.ts";

function fileEntry(name: string, size = 8): FileSystemEntry {
  return {
    isFile: true,
    isDirectory: false,
    name,
    file: (resolve: (file: File) => void) => resolve(new File(["x".repeat(size)], name)),
  } as unknown as FileSystemEntry;
}

/** Mimics the batched reader: the children once, then the empty end batch. */
function directoryEntry(name: string, children: FileSystemEntry[]): FileSystemDirectoryEntry {
  return {
    isFile: false,
    isDirectory: true,
    name,
    createReader: () => {
      let drained = false;
      return {
        readEntries: (resolve: (entries: FileSystemEntry[]) => void) => {
          resolve(drained ? [] : children);
          drained = true;
        },
      };
    },
  } as unknown as FileSystemDirectoryEntry;
}

function transferItem(entry: FileSystemEntry | null, kind = "file"): DataTransferItem {
  return { kind, webkitGetAsEntry: () => entry } as unknown as DataTransferItem;
}

test("only directory entries are claimed from a drop", () => {
  const folder = directoryEntry("repo", []);
  const items = [
    transferItem(folder),
    transferItem(fileEntry("notes.md")),
    transferItem(null),
    transferItem(directoryEntry("docs", []), "string"),
  ] as unknown as DataTransferItemList;
  assert.deepEqual(droppedDirectories(items).map((entry) => entry.name), ["repo"]);
  assert.deepEqual(droppedDirectories(null), []);
});

test("the folder placeholder is kept out of the loose files", () => {
  const files = [new File(["a"], "repo"), new File(["b"], "brief.pdf")] as unknown as FileList;
  const kept = looseFiles(files, [directoryEntry("repo", [])]);
  assert.deepEqual(kept.map((file) => file.name), ["brief.pdf"]);
});

test("a scan walks nested files and passes over vendor directories", async () => {
  const tree = directoryEntry("repo", [
    fileEntry("README.md"),
    directoryEntry("src", [fileEntry("api.ts"), fileEntry("ui.tsx")]),
    directoryEntry("node_modules", [fileEntry("left-pad.js")]),
    directoryEntry(".git", [fileEntry("HEAD")]),
  ]);

  const scan = await scanFolderEntry(tree);

  assert.equal(scan.name, "repo");
  assert.deepEqual(scan.files.map((item) => item.path).sort(), ["README.md", "src/api.ts", "src/ui.tsx"]);
  assert.equal(scan.skippedDirectories, 2);
  assert.equal(scan.truncated, false);
});

test("a scan that hits its file bound reports the counts as a floor", async () => {
  const tree = directoryEntry("repo", [fileEntry("a.ts"), fileEntry("b.ts"), fileEntry("c.ts")]);

  const scan = await scanFolderEntry(tree, { maxFiles: 2, maxDepth: 8 });

  assert.equal(scan.files.length, 2);
  assert.equal(scan.truncated, true);
});

test("a scan that hits its depth bound keeps the shallower files", async () => {
  const tree = directoryEntry("repo", [
    fileEntry("top.ts"),
    directoryEntry("one", [fileEntry("mid.ts"), directoryEntry("two", [fileEntry("deep.ts")])]),
  ]);

  const scan = await scanFolderEntry(tree, { maxFiles: 100, maxDepth: 1 });

  assert.deepEqual(scan.files.map((item) => item.path).sort(), ["one/mid.ts", "top.ts"]);
  assert.equal(scan.truncated, true);
});

test("an unreadable file is skipped rather than failing the walk", async () => {
  const denied = {
    isFile: true,
    isDirectory: false,
    name: "locked.ts",
    file: (_resolve: unknown, reject: (error: Error) => void) => reject(new Error("denied")),
  } as unknown as FileSystemEntry;

  const scan = await scanFolderEntry(directoryEntry("repo", [denied, fileEntry("open.ts")]));

  assert.deepEqual(scan.files.map((item) => item.name), ["open.ts"]);
});

test("attachable files are the ones the chat picker would have accepted", () => {
  const files = [
    { path: "a.ts", name: "a.ts", size: 10, file: null as unknown as File },
    { path: "b.psd", name: "b.psd", size: 10, file: null as unknown as File },
    { path: "c.md", name: "c.md", size: 5, file: null as unknown as File },
    { path: "Makefile", name: "Makefile", size: 5, file: null as unknown as File },
  ];
  const accept = ".ts,.md,text/*";

  assert.deepEqual(attachableFiles(files, accept).map((item) => item.name), ["a.ts", "c.md"]);
  assert.equal(totalBytes(files), 30);
});

test("the source kind is pre-selected from what the folder mostly holds", () => {
  const of = (names: string[]) =>
    names.map((name) => ({ path: name, name, size: 1, file: null as unknown as File }));

  assert.equal(suggestCorpusKind(of(["a.ts", "b.py", "c.go", "readme.md"])), "code");
  assert.equal(suggestCorpusKind(of(["a.md", "b.md", "c.rst", "setup.py"])), "docs");
  assert.equal(suggestCorpusKind(of(["a.ts", "b.md"])), "mixed");
  assert.equal(suggestCorpusKind(of(["logo.png"])), "mixed");
});

test("a dropped folder resolves against catalogs that renamed it", () => {
  const sources = [
    { root_path: "/Users/me/code/other-repo" },
    { root_path: "/Users/me/code/my_notes" },
  ];
  assert.equal(findSourceForFolder("my-notes", sources)?.root_path, "/Users/me/code/my_notes");
  assert.equal(findSourceForFolder("absent", sources), undefined);

  const projects = [{ name: "Waqil 2.0" }, { name: "Other" }];
  assert.equal(findProjectForFolder("waqil-2.0", projects)?.name, "Waqil 2.0");
  assert.equal(findProjectForFolder("nothing", projects), undefined);
});

test("the path guess follows wherever most known sources already live", () => {
  const guess = guessRootPath("new-repo", [
    "/Users/me/code/one",
    "/Users/me/code/two",
    "/Volumes/backup/three",
  ]);
  assert.equal(guess, "/Users/me/code/new-repo");
  assert.equal(guessRootPath("new-repo", []), null);
});

test("the path guess ignores the mirrors Metis keeps under its own data dir", () => {
  const guess = guessRootPath("new-repo", [
    "/Users/me/app/.data/corpus/runs",
    "/Users/me/app/.data/corpus/notion",
    "/Users/me/documents/personal-brain",
  ]);
  assert.equal(guess, "/Users/me/documents/new-repo");
  assert.equal(guessRootPath("new-repo", ["/Users/me/app/.data/corpus/runs"]), null);
});

test("path splitting survives trailing separators and root paths", () => {
  assert.equal(basename("/Users/me/code/repo/"), "repo");
  assert.equal(basename("repo"), "repo");
  assert.equal(parentDirectory("/Users/me/code/repo"), "/Users/me/code");
  assert.equal(parentDirectory("/repo"), null);
  assert.equal(parentDirectory("repo"), null);
});

test("sizes read the way a person would say them", () => {
  assert.equal(formatByteSize(512), "512 B");
  assert.equal(formatByteSize(2048), "2 KB");
  assert.equal(formatByteSize(5 * 1024 * 1024), "5.0 MB");
});
