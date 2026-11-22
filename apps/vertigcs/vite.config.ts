import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  resolve: {
    // Strip unused WGSL/WebGPU shader payloads for compact WebGL2 GitHub Pages cold load (DR-13)
    conditions: ["visgl:webgl-only", "import", "module", "browser", "default"],
  },
  server: {
    host: "127.0.0.1",
    port: 5180,
    strictPort: true,
  },
  preview: {
    host: "127.0.0.1",
    port: 5180,
    strictPort: true,
  },
  worker: {
    format: "es",
  },
  build: {
    target: "es2022",
    sourcemap: false,
    outDir: "dist",
  },
});
