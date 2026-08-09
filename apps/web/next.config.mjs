/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  poweredByHeader: false,
  reactStrictMode: true,
  // Lets a dev server build into a separate dir (NEXT_DIST_DIR=.next-dev) so it
  // never overwrites the manifests the prod server on :3000 is serving from .next.
  distDir: process.env.NEXT_DIST_DIR || ".next",
};

export default nextConfig;
