// Zero-dependency URL resolution helpers shared by main thread and dedicated Web Worker.
// Keeps flight_sim.worker bundle tiny (~3.5 kB) with zero MapLibre/Deck.gl imports.

export function resolvePublicAssetUrl(
  relPath: string,
  baseUrl: string = import.meta.env?.BASE_URL ?? "./"
): string {
  const cleanRel = relPath.replace(/^\/+/, "");
  const normBase = baseUrl || "./";
  if (normBase.endsWith("/")) {
    return `${normBase}${cleanRel}`;
  }
  return `${normBase}/${cleanRel}`;
}

export function resolveWasmModuleUrl(
  baseUrl: string,
  isDev: boolean,
  workerLocationHref: string
): string {
  const normBase = baseUrl || "./";
  const wasmRel = normBase.startsWith("/")
    ? `${normBase.replace(/\/+$/, "")}/wasm/libvertirta_flight_core.mjs`
    : isDev
      ? "/wasm/libvertirta_flight_core.mjs"
      : "../wasm/libvertirta_flight_core.mjs";
  return new URL(wasmRel, workerLocationHref).href;
}
