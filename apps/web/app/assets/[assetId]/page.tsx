import type { Metadata } from "next";
import { AssetWorkspace } from "@/components/assets/asset-workspace";

export const metadata: Metadata = { title: "Asset workspace" };

export default async function AssetPage({ params }: { params: Promise<{ assetId: string }> }) {
  const { assetId } = await params;
  return <AssetWorkspace assetId={assetId} />;
}
