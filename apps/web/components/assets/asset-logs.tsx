"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Copy, RefreshCw } from "lucide-react";
import { getAssetLogs } from "@/lib/api";
import { isActive } from "@/lib/assets";
import type { AssetV1 } from "@/lib/types";
import { usePoll } from "@/hooks/use-poll";
import { useToast } from "@/components/ui/toast";
import { Notice } from "@/components/ui/notice";

export function AssetLogs({ asset, visible }: { asset: AssetV1; visible: boolean }) {
  const [logs, setLogs] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [follow, setFollow] = useState(true);
  const lock = useRef(false);
  const output = useRef<HTMLPreElement>(null);
  const toast = useToast();
  const load = useCallback(async () => {
    if (lock.current) return;
    lock.current = true; setLoading(true);
    try { const result = await getAssetLogs(asset.id); setLogs(result.logs); setError(null); }
    catch (problem) { setError(problem instanceof Error ? problem.message : "Runtime output is unavailable."); }
    finally { lock.current = false; setLoading(false); }
  }, [asset.id]);
  useEffect(() => { if (visible) void load(); }, [visible, load, asset.status]);
  usePoll(load, 3_000, visible && isActive(asset));
  useEffect(() => { if (follow && visible && output.current) output.current.scrollTop = output.current.scrollHeight; }, [logs, follow, visible]);
  const copy = async () => { try { await navigator.clipboard.writeText(logs ?? ""); toast("Logs copied"); } catch { setError("The logs could not be copied. You can select and copy the output below."); } };
  return <div className="asset-log-view"><div className="asset-log-toolbar"><div><h2>Runtime output</h2><p>{isActive(asset) ? "Updates while this app is running." : "Output from the most recent launch."}</p></div><label><input type="checkbox" checked={follow} onChange={(event) => setFollow(event.target.checked)} />Follow output</label><button type="button" className="ui-btn is-sm" disabled={!logs} onClick={() => void copy()}><Copy size={14} />Copy</button><button type="button" className="ui-btn is-sm" disabled={loading} onClick={() => void load()} aria-label="Refresh logs"><RefreshCw size={14} className={loading ? "asset-spin" : ""} /></button></div>{error ? <Notice kind="error">{error}</Notice> : null}<pre ref={output} className="asset-log-output" tabIndex={0} aria-label="Runtime log output">{logs || (logs === null ? error ? "Output unavailable." : "Loading output…" : "No output yet. Start the app to see its launch activity here.")}</pre></div>;
}
