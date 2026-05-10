/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Produces a self-contained .next/standalone bundle for the Docker image.
  // The bundle includes a minimal Node server and only the code it needs.
  output: "standalone",
};

module.exports = nextConfig;
