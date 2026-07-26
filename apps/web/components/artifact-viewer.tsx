"use client";

import { useEffect, useState } from "react";

import { artifactUrl } from "@/lib/api";
import { sanitizeSvgForPreview } from "@/lib/svg-preview";
import type { ArtifactRef } from "@/lib/types";

function formatBytes(size?: number): string {
  if (size == null) return "Generated artifact";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function artifactKind(artifact: ArtifactRef): string {
  const name = artifact.name.toLowerCase();
  if (artifact.media_type?.includes("svg") || name.endsWith(".svg")) return "SVG";
  if (artifact.media_type?.startsWith("image/") || /\.(png|jpe?g|webp)$/.test(name)) return "IMG";
  if (artifact.media_type?.includes("json") || name.endsWith(".json")) return "JSON";
  if (name.endsWith(".py")) return "PY";
  return "FILE";
}

export function ArtifactViewer({ artifacts }: { artifacts: ArtifactRef[] }) {
  const [selected, setSelected] = useState<ArtifactRef | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewText, setPreviewText] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSelected(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    if (!selected) {
      setPreviewUrl(null);
      setPreviewText(null);
      setPreviewError(null);
      return;
    }
    const controller = new AbortController();
    let objectUrl: string | null = null;
    setPreviewUrl(null);
    setPreviewText(null);
    setPreviewError(null);
    void fetch(artifactUrl(selected.id, selected.download_url), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Preview unavailable (${response.status})`);
        if (artifactKind(selected) === "IMG") {
          return { kind: "blob" as const, value: await response.blob() };
        }
        return { kind: "text" as const, value: await response.text() };
      })
      .then((preview) => {
        if (preview.kind === "blob") {
          objectUrl = URL.createObjectURL(preview.value);
          setPreviewUrl(objectUrl);
          return;
        }
        setPreviewText(
          artifactKind(selected) === "SVG"
            ? sanitizeSvgForPreview(preview.value)
            : preview.value,
        );
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setPreviewError(error instanceof Error ? error.message : "Preview unavailable");
      });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [selected]);

  if (!artifacts.length) return null;

  return (
    <>
      <section className="artifactSection" aria-label="Generated artifacts">
        <div className="artifactHeading"><span>Artifacts</span><small>{artifacts.length} ready</small></div>
        <div className="artifactGrid">
          {artifacts.map((artifact) => {
            const url = artifactUrl(artifact.id, artifact.download_url);
            const previewable = ["SVG", "IMG", "JSON", "PY"].includes(artifactKind(artifact));
            return (
              <article className="artifactCard" key={artifact.id}>
                <button type="button" onClick={() => previewable && setSelected(artifact)} disabled={!previewable} aria-label={`Preview ${artifact.name}`}>
                  <span className={`fileBadge file-${artifactKind(artifact).toLowerCase()}`}>{artifactKind(artifact)}</span>
                  <span><strong>{artifact.name}</strong><small>{formatBytes(artifact.size)}</small></span>
                </button>
                <a href={url} download={artifact.name} aria-label={`Download ${artifact.name}`}>↓</a>
              </article>
            );
          })}
        </div>
      </section>

      {selected ? (
        <div className="modalBackdrop artifactBackdrop" role="presentation" onMouseDown={() => setSelected(null)}>
          <section className="artifactModal" role="dialog" aria-modal="true" aria-labelledby="artifact-title" onMouseDown={(event) => event.stopPropagation()}>
            <header>
              <div><span className="eyebrow">Sandboxed preview</span><h2 id="artifact-title">{selected.name}</h2></div>
              <div>
                <a className="secondaryButton" href={artifactUrl(selected.id, selected.download_url)} download={selected.name}>Download</a>
                <button className="iconButton" type="button" aria-label="Close preview" onClick={() => setSelected(null)}>×</button>
              </div>
            </header>
            <div className="artifactPreview">
              {previewError ? <div className="previewUnavailable"><strong>Preview unavailable</strong><span>{previewError}</span></div> : artifactKind(selected) === "IMG" && !previewUrl ? (
                <div className="previewLoading"><span /><span /><span /></div>
              ) : artifactKind(selected) !== "IMG" && previewText == null ? (
                <div className="previewLoading"><span /><span /><span /></div>
              ) : artifactKind(selected) === "IMG" ? (
                // Generated images come from the local artifact endpoint; SVG uses a sandboxed frame below.
                // eslint-disable-next-line @next/next/no-img-element
                <img src={previewUrl ?? ""} alt={`Preview of ${selected.name}`} />
              ) : artifactKind(selected) === "SVG" ? (
                <iframe
                  title={`Preview of ${selected.name}`}
                  srcDoc={previewText ?? ""}
                  sandbox=""
                />
              ) : (
                <pre className="artifactTextPreview">{previewText}</pre>
              )}
            </div>
          </section>
        </div>
      ) : null}
    </>
  );
}
