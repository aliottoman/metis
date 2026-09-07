"use client";

// A dropped folder is routed, never uploaded: open it as a project, index it
// as knowledge, or attach its files. The scan is local; only the route you
// choose does anything.

import { useEffect, useMemo, useState } from "react";

import { Notice } from "@/components/ui/notice";
import type { Chat } from "@/hooks/use-chat";
import { createCorpusSource, listCorpusSources, listProjectWorkspaces, reindexCorpusSource, scanAssets, setCorpusConsent } from "@/lib/api";
import { CHAT_ATTACHMENT_ACCEPT } from "@/lib/attachments";
import {
  ATTACHMENT_TEXT_BUDGET_BYTES,
  attachableFiles,
  findProjectForFolder,
  findSourceForFolder,
  formatByteSize,
  guessRootPath,
  MAX_ATTACHABLE_FILES,
  scanFolderEntry,
  suggestCorpusKind,
  totalBytes,
  type FolderScan,
} from "@/lib/folder-drop";
import type { CorpusSource } from "@/lib/types";

type Busy = "project" | "knowledge" | "attach" | "rescan" | null;

export function FolderDropSheet({ chat, directories, onClose }: { chat: Chat; directories: FileSystemDirectoryEntry[]; onClose: () => void }) {
  const [scan, setScan] = useState<FolderScan | null>(null);
  const [sources, setSources] = useState<CorpusSource[]>([]);
  const [path, setPath] = useState("");
  const [kind, setKind] = useState<CorpusSource["kind"]>("code");
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState<Busy>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [dropped] = directories;
  const ignored = directories.slice(1).map((entry) => entry.name);

  useEffect(() => {
    let mounted = true;
    void Promise.all([scanFolderEntry(dropped), listCorpusSources().catch(() => [] as CorpusSource[])])
      .then(([found, known]) => {
        if (!mounted) return;
        const existing = findSourceForFolder(found.name, known);
        setScan(found);
        setSources(known);
        setPath(existing?.root_path ?? guessRootPath(found.name, known.map((item) => item.root_path)) ?? "");
        setKind(existing?.kind ?? suggestCorpusKind(found.files));
      })
      .catch((error) => mounted && setNotice(error instanceof Error ? error.message : "That folder could not be read."));
    return () => { mounted = false; };
  }, [dropped]);

  const project = useMemo(() => (scan ? findProjectForFolder(scan.name, chat.projects) : undefined), [chat.projects, scan]);
  const source = useMemo(() => (scan ? findSourceForFolder(scan.name, sources) : undefined), [scan, sources]);
  const attachable = useMemo(() => (scan ? attachableFiles(scan.files, CHAT_ATTACHMENT_ACCEPT) : []), [scan]);
  const attachBytes = totalBytes(attachable);
  const attachBlocked = !attachable.length ? "Nothing in this folder is a format the composer can attach."
    : attachable.length > MAX_ATTACHABLE_FILES ? `${attachable.length} files is past the ${MAX_ATTACHABLE_FILES} the composer uploads at once. Index it instead.`
      : attachBytes > ATTACHMENT_TEXT_BUDGET_BYTES ? `${formatByteSize(attachBytes)} is past the ${formatByteSize(ATTACHMENT_TEXT_BUDGET_BYTES)} context budget. Index it instead.`
        : null;

  const openProject = async () => {
    if (!project) return;
    setBusy("project");
    const opened = await chat.chooseProject(project.id);
    setBusy(null);
    if (opened) onClose();
  };
  const rescan = async () => {
    setBusy("rescan");
    setNotice(null);
    try {
      await scanAssets();
      const found = await listProjectWorkspaces();
      chat.setProjects(found);
      if (scan && !findProjectForFolder(scan.name, found)) setNotice(`Still no project named “${scan.name}”. Projects are discovered under the folders configured in Settings.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The project catalog could not be refreshed.");
    } finally {
      setBusy(null);
    }
  };
  const index = async () => {
    if (!scan) return;
    if (!source && !path.trim()) {
      setNotice("Metis needs the folder's full path on disk before it can index it.");
      return;
    }
    setBusy("knowledge");
    setNotice(null);
    try {
      const created = source ?? await createCorpusSource(path.trim(), scan.name, kind);
      const decided = created.consent || !consent ? created : await setCorpusConsent(created.id, true, "granted from a composer folder drop");
      setSources((current) => (current.some((item) => item.id === decided.id) ? current.map((item) => (item.id === decided.id ? decided : item)) : [...current, decided]));
      if (!decided.consent) {
        setNotice("Added as a source. It stays unindexed until you allow cloud embedding for it.");
        return;
      }
      const result = await reindexCorpusSource(decided.id);
      setNotice(`Indexed ${result.files_indexed} files into ${result.chunks} chunks. Future answers can cite this folder.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "That folder could not be indexed.");
    } finally {
      setBusy(null);
    }
  };
  const attach = async () => {
    if (attachBlocked) return;
    setBusy("attach");
    await chat.addFiles(attachable.map((item) => item.file));
    setBusy(null);
    onClose();
  };

  const indexLabel = busy === "knowledge" ? "Working…" : source ? (source.consent ? "Reindex" : consent ? "Grant consent & index" : "Consent required") : consent ? "Add & index" : "Add source";
  const indexDisabled = Boolean(busy) || (!source && !path.trim()) || (Boolean(source) && !source?.consent && !consent);

  return (
    <section className="folder-sheet" aria-label="Route this folder">
      <header>
        <span className="ui-eyebrow">Folder dropped</span>
        <strong>{scan ? scan.name : "Reading the folder…"}</strong>
        {scan ? (
          <p>
            {scan.files.length}{scan.truncated ? "+" : ""} files · {formatByteSize(totalBytes(scan.files))}
            {scan.skippedDirectories ? ` · ${scan.skippedDirectories} build folders passed over` : ""}
            {ignored.length ? ` · one folder at a time, so ${ignored.join(", ")} was ignored` : ""}
          </p>
        ) : null}
      </header>
      {scan ? (
        <div className="folder-routes">
          <article>
            <strong>Open as project</strong>
            <p>Metis reads, searches and proposes exact edits under approval. Nothing leaves this machine.</p>
            {project ? (
              <button type="button" className="ui-btn is-primary is-sm" disabled={Boolean(busy) || chat.projectOpening || chat.runActive} onClick={() => void openProject()}>{busy === "project" ? "Opening…" : `Open ${project.name}`}</button>
            ) : (
              <button type="button" className="ui-btn is-sm" disabled={Boolean(busy)} onClick={() => void rescan()}>{busy === "rescan" ? "Rescanning…" : "Not in the catalog · Rescan"}</button>
            )}
          </article>
          <article>
            <strong>Index as knowledge</strong>
            <p>Chunked and embedded once, then cited in every conversation.</p>
            {source ? (
              <small className="folder-path">{source.root_path}</small>
            ) : (
              <label className="ui-field">
                <span>Path on disk</span>
                <input value={path} onChange={(event) => setPath(event.target.value)} placeholder="/absolute/path/to/the/folder" spellCheck={false} disabled={Boolean(busy)} />
              </label>
            )}
            {!source ? (
              <label className="ui-field">
                <span>Kind</span>
                <select value={kind} onChange={(event) => setKind(event.target.value as CorpusSource["kind"])} disabled={Boolean(busy)}>
                  {["code", "docs", "notes", "mixed"].map((item) => <option key={item} value={item}>{item}</option>)}
                </select>
              </label>
            ) : null}
            {!source?.consent ? (
              <label className="folder-consent">
                <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} disabled={Boolean(busy)} />
                <span>Allow cloud embedding. The text goes to Cohere to build the index; the vectors stay in local SQLite.</span>
              </label>
            ) : null}
            <button type="button" className="ui-btn is-primary is-sm" disabled={indexDisabled} onClick={() => void index()}>{indexLabel}</button>
          </article>
          <article>
            <strong>Attach the files</strong>
            <p>Their text goes into this one message, within the {formatByteSize(ATTACHMENT_TEXT_BUDGET_BYTES)} budget.</p>
            {attachBlocked ? <small>{attachBlocked}</small> : (
              <button type="button" className="ui-btn is-sm" disabled={Boolean(busy) || chat.uploading || chat.runActive} onClick={() => void attach()}>{busy === "attach" ? "Uploading…" : `Attach ${attachable.length} files · ${formatByteSize(attachBytes)}`}</button>
            )}
          </article>
        </div>
      ) : null}
      {notice ? <Notice kind="info">{notice}</Notice> : null}
      <footer>
        <small>Reading a folder is local. Only the route you choose acts on it.</small>
        <button type="button" className="ui-btn is-quiet is-sm" onClick={onClose} disabled={Boolean(busy)}>Dismiss</button>
      </footer>
    </section>
  );
}
