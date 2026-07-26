import type { Metadata } from "next";

import { AssetLibrary } from "@/components/asset-library";

export const metadata: Metadata = {
  title: "Assets",
  description: "Discover, configure, and launch reviewed local project demos in Metis.",
};

export default function AssetsPage() {
  return <AssetLibrary />;
}
