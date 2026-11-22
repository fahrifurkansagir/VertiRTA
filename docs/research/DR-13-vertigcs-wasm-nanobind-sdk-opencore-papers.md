# VertiRTA Productization and Publication Engineering Blueprint (2024–2026)

Productizing the 43k-LOC VertiRTA repository into a commercial-grade open-core autonomy platform while maximizing peer-reviewed acceptance across 2026–2027 submission cycles requires four coordinated engineering decisions. First, standardize VertiGCS on MapLibre GL JS v6 + @deck.gl/maplibre v9.4 (interleaved WebGL2 with visgl:webgl-only bundle stripping) paired with a single-threaded Emscripten WebAssembly build of libvertirta_flight_core inside a dedicated Web Worker; this single TypeScript frontend serves both a zero-install GitHub Pages 60 fps fault-injection sandbox (requiring zero API tokens and zero coi-serviceworker reload workarounds) and the live :5180/:8891 20-aircraft fleet command console, while isolating map rendering behind an engine-neutral adapter so CesiumJS can plug in for enterprise BVLOS contracts. Second, deliver frictionless developer onboarding via nanobind + scikit-build-core + cibuildwheel v4.x producing manylinux_2_28 (x86_64 and native ubuntu-24.04-arm aarch64) and Windows cp312-abi3 wheels, backed by a zero-copy multi-threaded C++ gymnasium.vector.VectorEnv, a modern FILE_SET HEADERS VertiRTAConfig.cmake, and a digest-pinned non-root multi-stage docker compose stack. [1][2][3][4][5][6][7][8]

Third, safeguard community trust and OEM procurement simultaneously by governing the Apache-2.0 core and the 256-byte .vrta binary black-box format under Developer Certificate of Origin (DCO 1.1)—avoiding the Foxglove Studio closed-source backlash while replicating the Auterion/PX4 and PickNik/MoveIt Pro commercial boundaries. Capture commercial revenue around late-2025/2026 non-discretionary regulatory wedges rather than paywalling core control algorithms: DO-178C / DO-333 / ASTM F3269-21 / EASA SC-VTOL MOC-5 certification artifact packs, EASA Part-IS (enforced October 2025 and February 2026) HSM/PKI fleet security, OEM airframe calibration and GPU reachability toolkits, and managed InterUSS ASTM F3548-21 and Turkey SHGM İHATTYS / SHT-21 Ek-6 connectors. Fourth, split the research output into two tightly scoped, double-anonymous manuscripts: Paper 1 for IEEE RA-L (rolling submission immediately for the IROS 2027 transfer window closing April 30, 2027, or Q1 2027 targeting ICRA 2028) covering coaxial X8 INDI-WLS fault control, 4D HJI-CBVF reachability, Cortex-M7 WCET/ESBMC verification, and U-Space preemption; and Paper 2 for CoRL 2027 (IEEE RA-L/T-RL backup) covering bilingual Z3-GRPO conformal grounding, VertiBench-TR/EN, and Active Visual SLZ. [1][2][3][4][5][6][7][8]

Two operational caveats must be engineered into the blueprint from day one. Copernicus DEM GLO-30 encodes orthometric elevations against the EGM2008 vertical datum (EPSG:3855) rather than the WGS84 ellipsoid—requiring explicit geoid-undulation conversion before rendering or terrain-clearance evaluation so GNSS-referenced aircraft do not float or clip into terrain. Equally critical on the academic track, both IEEE RA-L/ICRA and CoRL now enforce strict double-anonymous peer review and hard 8-page budgets where monolithic seven-subsystem manuscripts fail systematically. [1][2][3][4][5][6][7][8]

1. Unified VertiGCS Architecture: WebGL Engine, Emscripten WASM, and Zero-Token Istanbul Terrain

Maintaining separate web applications for a public interactive demo and a production ground control station creates permanent code divergence and fractures the adoption funnel. A unified Vite + TypeScript frontend (VertiGCS) resolves this by decoupling all telemetry ingress behind a FlightTransport abstraction. In Sandbox Mode (static GitHub Pages hosting), FlightTransport connects to a dedicated browser Web Worker executing libvertirta_flight_core.wasm alongside local drag-and-drop .vrta playback; in Fleet Ops Mode (:5180/:8891), the identical rendering canvas, telemetry HUD, and 3D overlays subscribe to binary WebSocket streams for 20 concurrent aircraft and the Z3-STL UNSAT-core inspection endpoint.

Head-to-Head Engine Verdict: MapLibre GL + Deck.gl vs. CesiumJS

Selecting the single WebGL stack for VertiGCS requires evaluating 2025–2026 release capabilities rather than legacy assumptions. MapLibre GL JS added native 3D globe projection in v5 (December 2025), introduced MapLibre Tiles (mlt) in v5.12.0, and advanced through v6. deck.gl v9.4 (released September 5, 2026) introduced a dedicated @deck.gl/maplibre package supporting MapLibre GL JS v4, v5, and v6 in overlaid and interleaved modes, experimental WebGPU parity across its official layer catalog, instance_index shader picking to cut GPU memory overhead, and the visgl:webgl-only bundler condition for stripping WGSL shaders. CesiumJS (cesium v1.147.0 / @cesium/engine v26.3.0 in September 2026) added KHR_gaussian_splatting hierarchical LOD (April 2026) and a Vector Tiles Technology Preview (September 2, 2026) introducing initial Mapbox Vector Tile draping under 3D Tiles 2.0, while continuing to rely on WebGL2 without a shipped production WebGPU backend. [1][2][3][4][5][6][7][8][9][10][11][12]

Engineering Dimension

MapLibre GL JS v6 + @deck.gl/maplibre v9.4

CesiumJS v1.147 (@cesium/engine v26.3)

Architectural Winner for VertiGCS

JS Bundle Size (Minified / Gzipped)

maplibre-gl v6.19.0: 1.1 MB / 293.1 KB gzip. Deck.gl v9.4.0-alpha.2 (visgl:webgl-only, esbuild gzip -9): core 144.5 KB + GeoJsonLayer 39.5 KB + Tile3DLayer (incl. ScenegraphLayer/glTF) 103.7 KB. Additive estimate ≈ 580.8 KB gzip (Brotli cuts ~20% further).

No 2026 Bundlephobia gzip metric is published for @cesium/engine (23.3 MB unpacked npm, 21 runtime deps). A documented 2020 community webpack release build measured 2.82 MB / 715.0 KB gzip (539 KB Brotli) for the Viewer chunk plus on-demand Workers/, ThirdParty/ WASM decoders, and a 3.65 MB Assets/ tree (2015 asset breakdown).

MapLibre + Deck.gl (smaller cold-load payload and zero CESIUM_BASE_URL static worker/asset copying).

Coordinate System & Altitude Accuracy at Istanbul (41.0082°N)

Web Mercator (EPSG:3857) inflates horizontal length by 1.3252× (+32.52%) and area by 1.7561×; spherical GlobeView uses mean radius 6,371,008.8 m vs. WGS84 geocentric 6,368,973.7 m (−2,035.1 m radial diff). Unscaled metric Z is accurate locally, whereas mistakenly scaling Z by Mercator 1/cos(lat) renders 300 m AGL as 397.6 m (+97.6 m visual error).

True WGS84 ellipsoid (EPSG:4979) with exact local curvature (prime vertical N = 6,387,348.8 m at 41.0082°N). Earth curvature drop is 31.31 m over 20 km and 195.70 m over 50 km (WGS84 vs. Mercator-sphere drop differs by only 0.045–0.283 m, while flat Mercator omits curvature entirely).

CesiumJS for >30 km BVLOS / RF viewsheds; Tie for <20 km Istanbul urban eVTOL corridors when local ENU-to-WGS84 math runs in C++ WASM.

Zero-Token Vector & Terrain Pipeline

Native pmtiles:// protocol for Protomaps/Overture vector tiles and Terrarium-RGB DEM (analytical sizing for Istanbul z0–z12, 184 tiles: 5.03 MB lossless WebP / 8.09 MB PNG); Deck.gl TerrainLayer also consumes quantized-mesh.

Native quantized-mesh-1.0 (analytical sizing for Istanbul EPSG:4326: 5.81–9.69 MB z0–z12; 21.45–35.74 MB z0–z13); no native PMTiles vector protocol (Sept 2026 MVT preview targets 3D Tiles 2.0; multi-engine apps like GeoLibre drape vector PMTiles via a hidden MapLibre renderer).

MapLibre + Deck.gl (single-file static .pmtiles works cleanly on GitHub Pages following the March 13, 2026 Fastly fix).

20-Aircraft + CBVF 60 fps GPU Ingestion

Deck.gl ingests zero-copy binary TypedArray buffers (Float32Array positions, Uint8Array colors), uses instance_index shader picking, and scopes buffer uploads via updateTriggers to avoid JS GC stalls.

ModelInstanceCollection was removed in CesiumJS 1.96 (was never public API); community benchmarks on 5,000 models showed ~60% lower GPU load with instancing vs. PrimitiveCollection. Twenty glTF Model primitives remain lightweight, though dynamic CBVF safety meshes require strict primitive pooling to prevent Chrome OOM crashes.

MapLibre + Deck.gl (cleaner zero-copy binary attribute streaming from WASM linear memory).

3D Terrain Draping & Globe Caveats

Official @deck.gl/maplibre constraints: deck.gl layers are not draped over MapLibre terrain; in GlobeView with default back-face culling, TextLayer and non-billboard IconLayer do not render (disabling culling rotates non-billboard icons 180°); non-default vertical FOV and camera roll do not synchronize; interleaved mode requires WebGL2.

Native terrain clamping for vectors, polylines, 3D Tiles, and glTF models, plus true 3D terrain occlusion, viewsheds, and sun/shadow lighting.

CesiumJS (significantly stronger native 3D terrain clamping and occlusion).

WebGPU Readiness (Late 2026)

Deck.gl v9.4 achieves experimental WebGPU render parity across all official layers while preserving WebGL2 compatibility.

WebGL2 only as of late 2026; WebGPU remains unshipped.

MapLibre + Deck.gl (direct path to WebGPU compute shaders for live 4D HJI-CBVF slice rendering).

Standardize VertiGCS on MapLibre GL JS v6 + @deck.gl/maplibre v9.4 (interleaved WebGL2) as the single production engine. For VertiRTA's dual mandate—a viral zero-install GitHub Pages interactive sandbox and a responsive 20-aircraft urban fleet operations dashboard—MapLibre + Deck.gl wins decisively on zero-token PMTiles ergonomics, cold-load weight, zero-copy TypedArray ingestion from WASM linear memory, 4 GB VRAM efficiency, and WebGPU forward compatibility. [1][2][3][4][5][6][7]

Two strict code invariants neutralize @deck.gl/maplibre's documented limitations. First, lock VertiGCS to Mercator (MapView) across all operational urban zoom levels (z ≥ 8) and permit GlobeView strictly for continental overview (z < 6), sidestepping GlobeView's TextLayer and non-billboard IconLayer culling defect. Second, compute explicit terrain-relative AGL and MSL altitudes inside C++ WASM (querying a lightweight Copernicus GLO-30 height grid or MapLibre's queryTerrainElevation, which returns meters above sea level in v5+) so Deck.gl aircraft and translucent 3D CBVF reachability tubes render at exact MSL heights without depending on automatic MapLibre terrain draping. Finally, isolate all map interactions behind a thin IGeoViewportAdapter TypeScript interface (mirroring GeoLibre's engine-neutral module split) so a commercial @cesium/engine adapter (vite-plugin-cesium-engine) can plug into the identical telemetry Worker and UI shell for >50 km BVLOS or OGC 3D Tiles defense customers. [1][2][3][4][5][6][7]

Exact Emscripten CMake Pipeline, ABI Strategy, and Bit-Exact IEEE-754 Parity

Because libvertirta_flight_core already fits inside 28.8 KB on ARM Cortex-M7, achieving the < 150 KB .wasm target hinges on avoiding three predictable Emscripten bloat and compatibility traps: linking embind + RTTI, dragging in C++ exceptions or iostream, and enabling -pthread on GitHub Pages.

Although embind simplifies exposing C++ classes and typed_memory_view wrappers, its template registration layer adds up to ~11 KB gzipped on small modules, grows binary size with every bound method, and triggers full serialization copies if structs are bound as value_object. Even with -fno-rtti -DEMSCRIPTEN_HAS_UNBOUND_TYPE_NAMES=0, embind adds unnecessary dispatch glue for a flight core requiring only ~8–12 functions (vertirta_init, vertirta_reset, vertirta_step_batch, vertirta_inject_fault, vertirta_get_vrta_ptr, vertirta_eval_cbvf, vertirta_parse_vrta_frame). Export a strict extern "C" flat ABI via a response file (-sEXPORTED_FUNCTIONS=@exported_funcs.txt), pre-allocate a static contiguous telemetry/state arena in linear memory, and wrap the raw exports in a lightweight TypeScript module exposing zero-copy Float64Array and DataView windows over wasm.HEAPU8.buffer. [1][2][3][4][5][6][7][8][9][10][11][12]

Equally important for GitHub Pages UX is eliminating coi-serviceworker. GitHub Pages does not allow custom Cross-Origin-Opener-Policy: same-origin or Cross-Origin-Embedder-Policy: require-corp HTTP headers. Enabling -pthread (SharedArrayBuffer) therefore forces coi-serviceworker, which triggers an ungraceful first-visit page reload, fails on shift-reload, and collides with tile-caching service workers. Instead, compile libvertirta_flight_core single-threaded (-sENVIRONMENT=worker, omitting -pthread) and run it inside a dedicated browser Web Worker. At 20 aircraft × 256 bytes = 5.00 KB per 60 fps tick (256 bytes in single-aircraft sandbox mode), ping-ponging two pre-allocated ArrayBuffer objects via postMessage(buf, [buf]) transfers telemetry with zero copy overhead and instant first-paint execution on GitHub Pages. [1][2][3][4][5][6][7][8][9][10][11][12]

(Compilation and link flags align with Emscripten LLVM 20.1.4 changes and Chrome's October 2026 C++-to-WebAssembly guidance.) [1][2][3][4]

Achieving bit-exact .vrta replay parity between ARM Cortex-M7 (-mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-d16) and WebAssembly 2.0 f64 requires neutralizing four specific IEEE-754 divergence vectors. Core WebAssembly 2.0 (Release 2.0, September 2025) mandates deterministic round-to-nearest-even f32/f64 semantics for +, -, *, /, and sqrt, with no dynamic rounding modes, traps, or status registers. To lock Cortex-M7 and WASM into identical bitstreams, enforce four build rules: [1][2][3][4]

Zero-Token Map and Copernicus GLO-30 Terrain Pipeline

Because GitHub Pages enforces a 100 MB per-file ceiling and defense/enterprise deployments run air-gapped, VertiGCS must operate with zero third-party map API tokens. Package the Istanbul operational bounding box (28.5°E–29.5°E, 40.8°N–41.4°N, ~84 km × 67 km ≈ 5,600 km²) into self-hosted static archives served from gh-pages or GitHub Releases / object storage. [1][2][3][4][5][6]

For the vector basemap (istanbul_base.pmtiles), extract the bounding box from Protomaps daily builds or Overture Maps via pmtiles extract and style it with open @protomaps/basemaps JSON styles. Regional metro extracts typically measure tens of megabytes (e.g., 84 MB in the PyViz Protomaps reference extract) and fit comfortably below GitHub Pages' 100 MB cap when bounded to z0–z14. Crucially, the GitHub Pages Fastly CDN regression that appended Content-Encoding: gzip to Firefox HTTP 206 Partial Content Range requests was resolved by GitHub Pages on March 13, 2026, restoring reliable zero-backend .pmtiles streaming across Chrome, Safari, and Firefox. [1][2][3][4][5][6]

For 3D terrain, ingest Copernicus DEM GLO-30 (30 m global DSM under a free public license requiring the mandatory ESA/Copernicus non-liability notice and citation DOI 10.5270/ESA-c5d3d65, offering <2 m vertical accuracy on slopes ≤20% and <4 m above 20%). Because the AWS Open Data GLO-30 bucket omits CORS headers, always self-host regional derivatives. For the default MapLibre + Deck.gl deployment, build a Terrarium-RGB PMTiles archive—either extracted from Mapterhorn (CC-BY global Terrarium PMTiles built on Copernicus GLO-30 up to z12) or converted from raw GLO-30 COGs via rio-rgbify -> mbtiles -> pmtiles convert. Across Istanbul's 184 Web Mercator tiles (z0–z12), analytical sizing estimates just 5.03 MB in lossless WebP (~28 KB/tile) or 8.09 MB in PNG (~45 KB/tile). Where explicit TIN mesh geometry is required (for Deck.gl TerrainLayer or the optional CesiumJS enterprise adapter), generate quantized-mesh 1.0 tiles using pymartini v0.6.0 (a Cython Martini RTIN generator 2–3× faster than Node Martini with double-precision triangle error) paired with quantized-mesh-encoder (which encodes 100k coordinates / 180k triangles in 20 ms with oct-encoded vertex normals and WGS84 ellipsoid support for both Deck.gl and CesiumJS), or the C++ emodnet_qmgc (qm_tiler) / Dockerized cesium-terrain-builder toolchain. Over Istanbul (EPSG:4326 TMS), analytical sizing puts gzipped quantized-mesh archives at 5.81–9.69 MB (z0–z12, 496 tiles) and 21.45–35.74 MB (z0–z13, 1,830 tiles). [1][2][3][4][5][6]

Every terrain and flight-core pipeline must enforce explicit vertical datum alignment. Copernicus GLO-30 elevations reference the EGM2008 orthometric geoid (EPSG:3855), not the WGS84 ellipsoid. Applying EGM96 instead of EGM2008 introduced a 2.02 m systematic bias in an August 2026 ESA SNAP evaluation, while omitting geoid undulation altogether separates GNSS ellipsoidal altitudes from orthometric terrain by tens of meters. Furthermore, geodetic evaluations across Turkey measured a −88.8 cm mean difference (24.2 cm SD) between raw EGM2008 and Turkish GNSS/leveling benchmarks prior to tilt fitting, motivating support for the official Turkish Geoid Model-2020 (TG-20) in Turkish operational deployments. Standardize one vertical bridge equation across VertiRTA C++, WASM, and VertiGCS: [1][2][3][4][5][6]

Embedding the 5' × 5' EGM2008 geoid undulation grid for the Istanbul corridor (13 × 8 = 104 grid nodes, <1 KB in float32) directly inside VertiGCS guarantees exact vertical agreement between GNSS-referenced .vrta telemetry, Terrarium MSL tiles, and ellipsoidal quantized-mesh tiles. [1][2][3][4][5][6]

Memory, VRAM, and 60 fps Locks for 4 GB GPUs and Browser Tabs

Integrated 4 GB VRAM laptops and browser tabs suffer frame-time spikes or WebGL context loss whenever high-frequency telemetry triggers React reconciliation, unbounded JS arrays, or oversampled HiDPI framebuffers. Lock VertiGCS performance with five mandatory runtime guardrails:

2. Turnkey SDK and Developer Experience: Python Wheels, Gymnasium, Modern CMake, and Docker

Open-source robotics adoption hinges on time-to-first-simulation: if RL or control researchers must manually compile C++20, Eigen, and Z3 before running a single rollout, most abandon the repository. Packaging VertiRTA as prebuilt binary wheels, relocatable CMake targets, and hardened containers removes every installation barrier across Linux (x86_64, Jetson aarch64) and Windows.

2025–2026 C++20-to-Python Wheel Pipeline (nanobind + scikit-build-core + cibuildwheel)

Across 2025–2026, nanobind + scikit-build-core + cibuildwheel v4.x has become the standard C++17/C++20 Python extension stack—illustrated by ONNX migrating from pybind11 to nanobind specifically to unlock Python Stable ABI (abi3) wheels. Compared with pybind11, nanobind compiles faster via a compact pre-compiled core library, enforces stricter compile-time type safety, reduces extension binary size by 2–3× according to published ecosystem comparisons, and provides zero-copy nb::ndarray exchange across NumPy, PyTorch, JAX, TensorFlow, CuPy, and MLX via both the Python Buffer Protocol and DLPack without build-time dependencies on any framework. [1][2][3][4][5][6][7][8][9][10][11]

Three 2025–2026 packaging mechanics streamline VertiRTA's CI matrix. First, pairing STABLE_ABI in nanobind_add_module() with wheel.py-api = "cp312" in pyproject.toml emits a single cp312-abi3 wheel per OS/architecture covering Python 3.12, 3.13, 3.14, and 3.15+ (plus a separate cp311 wheel if supporting Python 3.11; alternatively, nanobind 3.x split mode with nanobind-backend extends single-wheel abi3 coverage down to Python 3.10, noting Py_buffer entered Stable ABI in Python 3.11). Second, because cibuildwheel 3.1.0+ builds free-threaded Python 3.14 (cp314t-*) by default and abi3 does not support 3.14t (abi3t targets 3.15t+), explicitly skip cp31?t-* until free-threaded C++ wrapper audits complete. Third, cibuildwheel v3.0+ defaults to manylinux_2_28, v3.4.0+ adds build-frontend = "uv", and v4.0+ (June 2026) adds an automatic abi3audit verification stage and defaults to delvewheel for Windows DLL repair. Run aarch64 Linux wheel builds natively on GitHub Actions ubuntu-24.04-arm (or ubuntu-26.04-arm) runners rather than QEMU emulation. [1][2][3][4][5][6][7][8][9][10][11]

(Synthesized from nanobind packaging documentation and cibuildwheel v3.4–v4.3 release specifications.) [1][2][3][4]

Wire .github/workflows/wheels.yml across a three-runner matrix (ubuntu-24.04 for manylinux_2_28_x86_64, ubuntu-24.04-arm for manylinux_2_28_aarch64, and windows-latest using MSVC + delvewheel) invoking pypa/cibuildwheel@v4 and publishing to PyPI on release tags. Generate .pyi typing stubs during wheel packaging (following nanobind's stub-generation pattern) so IDE type checkers resolve all C++ bindings and Gymnasium signatures out of the box. [1][2][3][4]

Zero-Copy nb::ndarray Exposure, Deterministic Gymnasium Wrappers, and CLI

Existing aerial safe-RL benchmarks like safe-control-gym step PyBullet at 1000 Hz in Python at 10–20× real-time for single instances, leaving fast learners like SBX (Stable-Baselines3 + JAX, up to 20× faster than PyTorch SB3 via JIT gradient compilation) starved on simulator IPC. By driving vectorized rollouts in native C++ with nb::gil_scoped_release across a fixed thread pool and zero-copy nb::ndarray buffers, VertiRTA scales linearly with CPU cores. Enforce three nanobind memory-ownership rules in the C++ binding layer: [1][2][3][4]

Register two Gymnasium v1.1+ compliant environments under gymnasium.register(...). VertiRTA-CoaxialX8-v0 (gymnasium.Env) wraps a single 6-DOF coaxial X8 vehicle with configurable rotor/IMU/wind fault schedules and selectable RTA safety filters (none, cbf_qp, hj_cbvf_qp), exposing CBVF value, spatial gradient, and WLS saturation flags inside info and validated in CI via stable_baselines3.common.env_checker.check_env. VertiVecEnv (gymnasium.vector.VectorEnv) exposes the parallel C++ batch engine, explicitly populating metadata["autoreset_mode"] and implementing all three Gymnasium v1.1+ modes (AutoresetMode.NEXT_STEP default, AutoresetMode.SAME_STEP populating info["final_obs"]/info["final_info"], and AutoresetMode.DISABLED honoring reset(options={"reset_mask": mask})). Deterministic seeding in reset(seed=s) initializes independent per-environment RNG streams only when an integer seed is passed (leaving RNG state untouched when seed=None). [1][2][3][4][5][6]

Ship five flat subcommands under the vertirta Typer CLI:

[STRIPPED_MATH_OR_ICON] vertirta sim --airframe x8 --fault rotor3@12.0s --rta cbvf --out run.vrta — headless deterministic flight simulation writing a 256-byte-framed .vrta log.

[STRIPPED_MATH_OR_ICON] vertirta inspect run.vrta --summary --export-parquet run.parquet — zero-copy .vrta parser and columnar Parquet exporter.

[STRIPPED_MATH_OR_ICON] vertirta atlas --runs 30000 --workers 16 --seed 42 --out atlas.parquet — deterministic Survival Atlas generator.

[STRIPPED_MATH_OR_ICON] vertirta verify-stl --lang tr --cmd "Boğaz köprüsünden 200 m uzak dur" — bilingual Z3-STL translation, satisfiability check, and minimal UNSAT-core output.

[STRIPPED_MATH_OR_ICON] vertirta gcs --port 5180 --ws-port 8891 — local VertiGCS server and binary WebSocket fleet bridge.

Modern Relocatable CMake (find_package(VertiRTA))

Embedded avionics teams and ROS 2 integrators require clean target-based CMake consumption with strict component isolation so Cortex-M7 firmware links VertiRTA::flight_core without touching Z3 or Python. Adopt CMake 3.25+ FILE_SET HEADERS (eliminating error-prone BUILD_INTERFACE/INSTALL_INTERFACE include expressions) paired with CMakePackageConfigHelpers and GNUInstallDirs: [1][2][3][4][5][6]

In cmake/VertiRTAConfig.cmake.in, place @PACKAGE_INIT@ and include(CMakeFindDependencyMacro), load VertiRTACoreTargets.cmake unconditionally, and guard find_dependency(Z3 REQUIRED) plus VertiRTAStlTargets.cmake behind component checks when consumers call find_package(VertiRTA REQUIRED COMPONENTS flight_core stl_z3), concluding with check_required_components(VertiRTA). Omit build-tree export() entirely, and verify relocatable installation on every PR via a standalone consumer project (tests/cmake/CMakeLists.txt) across Linux, Windows, and macOS. [1][2][3][4][5]

Hardened Multi-Stage docker compose Stack

Deploy the live :5180/:8891 fleet command stack via three decoupled, digest-pinned services (vertirta-sim, vertirta-copilot, vertigcs-web) so build toolchains (CMake, Ninja, Clang, Emscripten, Node) never enter production runtime images. Compile C++20 services in a pinned debian:bookworm@sha256:... builder stage and copy strictly the stripped binaries and libz3.so into gcr.io/distroless/cc-debian12 or debian:bookworm-slim executing under an explicit non-root USER 10001:10001. Build VertiGCS in parallel Node 22 + emsdk stages and serve static JS/WASM/PMTiles from nginxinc/nginx-unprivileged on port :5180. [1][2][3][4][5]

Coordinate startup deterministically with HTTP /healthz probes (interval, timeout, retries, start_period) on :8891 and :8892 coupled to depends_on: condition: service_healthy so vertigcs-web never serves upstream 502 errors during boot. Enforce read_only: true, cap_drop: [ALL], ephemeral tmpfs mounts for Z3 scratch files, read-only bind mounts (:ro) for offline .pmtiles and .vrta log directories, and explicit container CPU/memory limits. [1][2][3][4][5]

3. Apache-2.0 Open-Core Commercial Productization and Ecosystem Strategy

Commercializing safety-critical robotics open source succeeds when the boundary separates commodity adoption infrastructure (permanently open) from compounding regulatory, cryptographic, and fleet-scale operational value (commercial). Three 2024–2026 robotics open-core precedents establish the blueprint for VertiRTA.

First, Auterion / PX4 / Dronecode proves that permissive core stewardship builds category dominance: PX4 remains BSD-3 under neutral Linux Foundation (Dronecode) governance (v1.16 in 2025, v1.17 stable in 2026), with Auterion supplying 39% of all PX4 contributions in 2025 (and 52.7% in 2022), while monetizing AuterionOS, Auterion Suite fleet software, Skynode X/S avionics, and NDAA-compliant defense subscriptions—supporting a $130M Series B at a valuation above $600M (September 2025) and a $50M Pentagon contract for 33,000 kits according to droneintelligence.ai market reporting. Second, PickNik / MoveIt 2 / MoveIt Pro maintains open-source MoveIt 2 as the community motion-planning standard ("MoveIt Pro is to MoveIt as GitHub is to Git"), while monetizing MoveIt Pro 9.0's commercial web UI, 200+ Behavior Tree skills, scan-and-plan toolboxes, and production deployment tooling—culminating in PickNik joining forces with Qualcomm Technologies in September 2026. Third, Foxglove Studio offers both a warning and a strategic template: discontinuing open-source Studio after v1.87.0 (February/March 2024) triggered intense r/robotics and ROS Discourse backlash and community forks such as Flora, yet Foxglove preserved developer trust in data portability by keeping the MCAP binary format, libraries, and CLI permanently Apache-2.0—and later defused commercial seat-cost friction by making Basic viewer seats free and unlimited on Pro and Enterprise plans. [1][2][3][4][5]

Governance Rule Zero: DCO 1.1 and Permanent Apache-2.0 .vrta

Adopt Developer Certificate of Origin (DCO 1.1) (Signed-off-by: enforced via GitHub DCO App) rather than an asymmetric corporate Contributor License Agreement (CLA) on the Apache-2.0 core—mirroring the OpenInfra Foundation's July 1, 2025 migration from CLA to DCO. Corporate CLAs stall external OEM contributions in legal review and signal relicensing risk (the MongoDB and Foxglove concern). Under Apache-2.0 + DCO 1.1 ("inbound = outbound"), defense and eVTOL engineers contribute upstream without corporate copyright assignment friction, while your commercial entity builds proprietary enterprise suites cleanly atop stable C++ plugin ABI and gRPC interfaces. Publish an explicit public commitment that libvertirta_flight_core, .vrta format specs/parsers, and single-operator VertiGCS remain permanently Apache-2.0. [1][2][3][4][5]

Open-Core Feature Boundary and Late-2026 Regulatory Wedges

In aerospace autonomy, algorithms commoditize quickly while certification artifacts, cryptographic compliance workflows, and accredited airspace connectors create durable recurring revenue. Regulatory deadlines across 2025–2026 hand VertiRTA four concrete enterprise monetization pillars:

Module / Capability

License Tier

Commercial Value Driver and 2025–2026 Regulatory Anchor

libvertirta_flight_core (INDI-WLS, 4D HJI-CBVF filter, CORE-MATH), reference X8 model, starter ESBMC harness

Apache-2.0 (DCO 1.1)

Maximizes academic citations, GitHub stars, and bottom-up OEM engineering adoption.

.vrta 256-byte binary specification, C++/Python/TS parsers, Parquet exporter

Apache-2.0 (Permanent)

Replicates Foxglove's MCAP strategy: turns .vrta into the zero-lock-in black-box data standard across MCUs, simulators, and flight-test teams.

VertiGCS single-operator UI, GitHub Pages WASM sandbox, pip install vertirta Gymnasium SDK, VertiBench-TR/EN

Apache-2.0

Zero-friction developer funnel converting every paper reader and visitor into an active user.

1. Certification Artifact Packs (VertiRTA CertPack)

Commercial Enterprise

EASA published final MOC-5 SC-VTOL Issue 2 on November 5, 2026 and AI Concept Paper Proposed Issue 03 (June 3, 2026), capping Large OTS LLMs at IDAL D / AL 5 / TQL-5 and barring them from catastrophic failure conditions—making ASTM F3269-21 deterministic Simplex RTA monitors indispensable. Package DO-178C DAL A/B requirements traceability, MC/DC coverage suites, Software Accomplishment Summary (SAS) templates, DO-333 (ED-216) formal-methods verification dossiers, EUROCAE ED-342 simulation-based handling-qualities compliance evidence, and SHGM SHT-21 Ek-6 (ICAO Annex 8 alignment effective November 26, 2026) UAS/VTOL airworthiness packs.

2. EASA Part-IS Fleet Security and HSM/PKI Suite (VertiRTA Shield)

Commercial Enterprise

EASA Part-IS became legally enforceable October 16, 2025 for Part-21 DOA/POA and aerodrome organizations (Reg. EU 2022/1645) and February 22, 2026 for AOC, CAMO, Part-145, and USSP/ATM entities (Reg. EU 2023/203). Monetize hardware-HSM .vrta signing, C2-link mutual-TLS PKI with enforced key ownership/expiry, air-gapped on-prem VertiGCS Fleet Ops (SSO/OIDC, RBAC, multi-operator replay), and automated IS.I.OR.205–260 safety-linked risk registers and 72-hour external cyber-incident reporting exports.

3. OEM Airframe Calibration and GPU Reachability Kit (VertiRTA Studio)

Commercial Enterprise

Sell multi-rotor, coaxial, tilt-rotor, and lift+cruise system-identification pipelines, multi-GPU 4D–6D HJI-CBVF Survival Atlas synthesis clusters, and production hardware-in-the-loop BSPs (STM32H7, NXP i.MX RT, FMUv6x/Skynode).

4. Managed USSP and National Airspace Connectors (VertiRTA Airspace)

Commercial SaaS / On-Prem

Conformance-tested connectors for InterUSS ASTM F3548-21 strategic coordination & ASTM F3411-22a Remote ID, EU U-Space (Reg. 2021/664 & 2021/666 electronic conspicuity), FAA UTM FIMS/LAANC, and Turkey SHGM İHATTYS real-time flight authorization, Remote ID, and dynamic geofencing mandated under the 2026 SHT-İHA overhaul.

OEM Go-To-Market Positioning in Late 2026

Rapid industry consolidation across 2024–2026 means pitching a generic flight-control stack to Tier-1 eVTOL OEMs will bounce off internal acquisitions. Tailor the enterprise motion sharply by OEM tier:

4. Dual Peer-Reviewed Publication Split, SOTA Baseline Matrix, and Statistical Rigor

Area Chair Diagnosis: Why One Monolithic Paper Fails Under 2025–2026 Venue Rules

Submitting all seven subsystems of VertiRTA—coaxial X8 INDI-WLS fault control, 4D HJI-CBVF reachability, Cortex-M7 WCET/ESBMC formal verification, U-Space preemption, bilingual Z3-GRPO conformal LLM grounding, VertiBench-TR/EN, and Active Visual SLZ—in a single manuscript creates a near-certain rejection at flagship venues for three structural reasons. [1][2][3][4]

First, page ceilings are unforgiving. IEEE RA-L permits 6 standard pages plus at most 2 extra pages ($175/page) for an 8-page absolute maximum including figures, tables, references, and appendices; placing supplemental text or figures inside multimedia attachments triggers editorial desk rejection without review. ICRA enforces an 8-page all-inclusive ceiling (38.0% acceptance rate at ICRA 2026: 1,882 of 4,947 papers per CS Conf Stats). CoRL limits the main text to 8 pages (excluding acknowledgments, references, and optional appendix, which reviewers are not required to read), and in CoRL 2026 the mandatory Limitations section counted inside those 8 pages, with a 9th page allowed only at camera-ready. Compressing seven subsystems into 8 pages leaves less than one page per topic, stripping away proofs, ablations, and failure analyses and triggering "unfocused system-integration report" rejections. [1][2][3][4]

Second, reviewer pools are disjoint. Control-theory and formal-methods reviewers assigned through IEEE RA-L keywords evaluate actuator-saturation feasibility against CBF-QP, HJI viscosity properties, ESBMC verification soundness, and cycle-accurate Cortex-M7 WCET—treating bilingual GRPO fine-tuning and Turkish NLP benchmarks as unverified distractions. CoRL robot-learning reviewers evaluate conformal risk bounds, multi-seed learning curves, KnowNo/ConformalNL2LTL/SELP baselines, and real-world perception/transfer evidence—treating Cortex-M7 register profiling and ASTM F3548 preemption as orthogonal systems plumbing.

Third, 2026–2027 submission calendars align naturally to a staggered split. Since January 1, 2025, IEEE RAS mandates double-anonymous peer review across all fully sponsored journals including RA-L (JCR Impact Factor 5.3 in 2025) as well as ICRA. Accepted RA-L papers (3 months to first decision, 30-day non-extendable revision window, 5 months to final decision) can transfer to one RAS conference within 270 days of acceptance: with today being November 22, 2026, the ICRA 2027 transfer window closes December 31, 2026 (already closed for new submissions), whereas submitting Paper 1 to RA-L immediately in late November 2026 targets the IROS 2027 transfer window (closing April 30, 2027), and a Q1 2027 submission targets ICRA 2028. Because CoRL 2026 concluded on November 9–12, 2026, CoRL 2027 (submissions typically closing May 2027) provides an ideal six-month runway for Paper 2, with IEEE RA-L or IEEE T-RL as the immediate archival backup. [1][2][3][4]

Dual-Paper Scope and Decoupling Contract

Attribute

Paper 1: Deterministic Flight Safety and Verified RTA

Paper 2: Bilingual Neuro-Symbolic Mission and Visual SLZ

Target Venue & Cadence

IEEE RA-L (rolling submission late Nov 2026 targeting IROS 2027 transfer by April 30, 2027, or Q1 2027 targeting ICRA 2028; double-anonymous, 8 pp max all-inclusive; OA APC $2,800 in 2026 if OA elected).

CoRL 2027 (May 2027 deadline; double-blind on OpenReview, 8 pp main + Limitations + unlimited refs/appendix + ≤250 MB / 3 min video; backup: IEEE RA-L / T-RL).

Working Title

Verifiably Safe Fault-Tolerant eVTOL Flight Control: Coaxial X8 INDI-WLS, 4D HJI-CBVF Reachability, and Formally Checked Cortex-M7 Run-Time Assurance under U-Space Preemption

Bilingual Conformal Neuro-Symbolic Assurance for Urban eVTOLs: Graded Z3-GRPO Temporal Logic Grounding, VertiBench-TR/EN, and Active Visual Safe Landing

Core Claim

Demonstrates end-to-end deterministic containment from nonlinear coaxial X8 rotor failures and 4D HJI-CBVF reachability down to a 28.8 KB formally verified Cortex-M7 binary (ESBMC + bounded WCET) enforcing ASTM F3548 U-Space preemption across a 30,000-flight Survival Atlas.

Demonstrates that probabilistic bilingual (Turkish/English) LLM mission planners and active visual landing perception can operate safely above an ASTM F3269-21 Simplex RTA boundary via graded Z3-GRPO process rewards, language-group-conditional conformal abstention, and risk-aware Active Visual SLZ.

Clean Interface Boundary

Models mission guidance purely as untrusted reference commands and U-Space constraint volumes (zero LLM/VLM scope).

References Paper 1's deterministic RTA cage (in neutral third-person phrasing under double-anonymous rules) as a black-box safety filter, focusing strictly on bilingual NL-to-STL grounding, conformal calibration, and Active Visual SLZ.

2024–2026 SOTA Baseline Matrix and Required Reviewer Tables

Both manuscripts must benchmark directly against recognized 2024–2026 community baselines rather than internal ablations alone.

Paper 1 (IEEE RA-L / ICRA) — Required Baselines and Reviewer Tables

Evaluation Axis

Mandatory 2024–2026 SOTA Baselines

Metrics Reviewers Demand

Core Result VertiRTA Must Establish

Fault-Tolerant Allocation & Tracking

Prioritized INDI-WLS (Smeur et al., TU Delft); Pseudo-Inverse Reallocation; px4_mpc (Lim et al. / DISCOWER ROS 2 + acados NMPC); Fault-Tolerant NMPC + INDI (Nan et al., RA-L 2022); safe-control-gym (Yuan et al., RA-L 2022 — contrast its 1D/2D quadrotor models against 6-DOF coaxial X8).

Survival rate with exact 95% Clopper-Pearson CI, attitude/altitude recovery settling time, position RMSE (rliable IQM), actuator saturation duration, compute latency on MCU vs. companion CPU.

Recovers from single and dual coaxial X8 rotor losses inside 28.8 KB flash and hard real-time bare-metal MCU deadlines where companion-computer acados NMPC cannot reside.

Safety Filter (CBF-QP vs. HJ-CBVF)

Analytical CBF-QP; Classical Bang-Bang HJ Reachability (hj_reachability / OptimizedDP); Robust CBVF-QP (Choi et al., CDC 2021); Model-Predictive Safety Filter (MPSC) (safe-control-gym); situate against Viscosity CBFs (ACC 2026) and Deep QP Safety Filter (2026) in related work.

Safety violation rate (%), QP infeasibility rate (%) under rotor-loss thrust shrinkage, control chattering (\int \|\dot{u}\| \, ), mission-time overhead (%).

Analytical CBF-QP loses feasibility when rotor loss shrinks admissible thrust bounds; bang-bang HJ chatters aggressively; VertiRTA's 4D HJI-CBVF-QP maintains persistent feasibility across the maximal viable set with smooth filtering.

Embedded Formal Verification & WCET

ESBMC v7.x (Clang C++17 AST, Z3 4.13+ and Bitwuzla IEEE-754 FP theory, --k-induction, --overflow-check, --nan-check) contrasted against CBMC (in-tree C++ parser lagging on C++17 templates, SAT-primary) and Frama-C / Frama-Clang (mature C Eva/WP, early-stage C++ translation).

Verified property counts, solver runtime (Z3 vs. Bitwuzla), loop unwinding bound k, exact Flash/RAM bytes, Cortex-M7 DWT->CYCCNT (with __DSB() barrier) + GPIO oscilloscope traces separating BCET/ACET/MOET from analytically bounded WCET under cache/DMA contention.

Zero bounds, pointer, overflow, division-by-zero, or NaN faults within verified unwind depth; WCET strictly below the inner-loop deadline (e.g., <2.5 ms at 400 Hz).

Airspace Preemption

ASTM F3548-21 static strategic deconfliction (which explicitly excludes tactical deconfliction) vs. Priority-Based U-Space Dynamic Preemption under EU Reg. 2021/664 & 2021/666 crewed-intruder alerts.

Preemption reaction latency (ms), separation-buffer violations, safe diversion completion under simultaneous rotor degradation + crewed-intruder preemption.

Demonstrates tight coupling of dynamic U-Space volumes into 4D HJI-CBVF viability checks so diversion clearances are guaranteed dynamically reachable.

Paper 2 (CoRL / IEEE RA-L) — Required Baselines and Reviewer Tables

Evaluation Axis

Mandatory 2024–2026 SOTA Baselines

Metrics Reviewers Demand

Core Result VertiRTA Must Establish

NL-to-Formal-Logic Grounding & Verification

LLM+P (Liu et al., PDDL); Lang2LTL (CoRL 2023) & Lang2LTL-2 (multimodal spatiotemporal); NL2TL / AutoTAMP (Chen et al., ICRA 2024, STL translation & iterative replanning); SELP (Wu et al., ICRA 2025: equivalence voting + LTL constrained decoding + fine-tuning); Code-as-Policies and zero-shot frontier LLM.

Semantic equivalence accuracy (TR, EN, code-switched), STL syntax validity (%), UNSAT / infeasible-command rejection accuracy (%), Z3 solver latency (ms), closed-loop safety and completion rates.

Outperforms unconstrained prompting, AutoTAMP, and SELP on continuous spatio-temporal eVTOL STL bounds and surfaces minimal Z3 UNSAT cores to operators.

RL Post-Training & Reward Design

Outcome-Only GRPO vs. Graded Z3 Step-Verified GRPO (analogous to an ICLR 2026 submission, Logic-Verified GRPO, where graded Z3 step rewards improved Unknown-conclusion epistemic calibration by +14 pp over outcome-only GRPO on FOLIO/ProntoQA); plus Eureka (Ma et al., ICLR 2024) and AutoRL (Chiang et al., RA-L 2019) for reward-engineered RL baselines without formal STL guards.

Pass@1 / Pass@K formal validity, epistemic calibration on unfeasible/contradictory commands, KL drift, and closed-loop safety violation rate vs. unconstrained Eureka/RL policies.

Graded Z3-STL rewards train compact on-prem models (3B–8B) to separate satisfiable from UNSAT mission intents, narrowing the Turkish–English performance gap without cloud APIs.

Conformal Uncertainty Alignment

KnowNo (Ren et al., CoRL 2023 Oral); IntroPlan (introspective conformal planning); ConformalNL2LTL (Sundarsingh et al., 2025/2026 sequence-level conformal NL-to-LTL); SCP-NL2TL (Wang et al., Aug 2026 selective conformal risk control with back-translation verification); cite AdaptNC (NeurIPS 2026) on distribution shift.

Marginal AND language-group-conditional (TR vs. EN) empirical coverage across α ∈ {0.05, 0.10, 0.15, 0.20}, human-clarification / abstention rate (%), selective risk-coverage Pareto frontier, OOD shift degradation.

Pooled marginal conformal calibration silently under-covers Turkish when calibrated on mixed TR+EN data; group-conditional Z3-conformal calibration upholds ≥(1 − α) coverage in both languages while curbing operator clarification load.

Multilingual Benchmark & Active Visual SLZ

Benchmark methodology informed by ReLI (2025 cross-lingual HRI) and VLABench (ICCV 2025 primitive vs. composite tracks); Visual SLZ baselines: Risk-Map Semantic Emergency Landing (2024–2026 urban UAV SLZ), OR-SLZNet (2025/2026), and Deep-Regression SLZ benchmarked on UAVid (4K oblique urban video, 300 labeled frames, 8 classes), ICG Drone, and UDD6/VDD plus closed-loop Istanbul active-viewpoint sim.

SLZ mIoU, risk-score MAE/MSE on UAVid and ICG Drone, pedestrian/obstacle false-safe rate (primary safety metric), active-viewpoint information gain vs. passive nadir descent, onboard Jetson Orin inference FPS.

Coupling active viewpoint selection with semantic risk maps and 4D CBVF reachability prevents "false-safe" landings on slanted roofs or dynamic crowds while guaranteeing touchdown reachability under degraded thrust.

Minimum Ablation Sample Sizes and Statistical Protocol

Arbitrary 100-trial tables routinely trigger Revise and Resubmit or rejection decisions at RA-L and CoRL. Lock experimental budgets to exact power and finite-sample conformal laws before running cluster sweeps.

For Paper 1, partition the 30,000-simulation Survival Atlas across 10 orthogonal fault, wind-gust, and U-Space preemption strata (3,000 runs per stratum). At 99.0% survival, 3,000 runs tighten the exact 95% Clopper-Pearson binomial CI to [98.58%, 99.32%] (0.75 pp width)—whereas 100 runs produce a noisy 5.42 pp width [94.55%, 99.97%]; across all 30,000 simulations, 99.0% survival yields a 0.23 pp width [98.88%, 99.11%] and 99.9% survival yields a 0.08 pp width [99.86%, 99.93%]. Because VertiRTA's CORE-MATH simulator is bit-exact deterministic, run every baseline and ablation on identical seeded scenarios and evaluate pairwise survival gains with exact McNemar's test plus Holm-Bonferroni correction across 6 pairwise comparisons (α_eff = 0.00833, 90% power). Under realistic paired seed correlation (0.5% baseline-only wins), detecting a 3.0 pp improvement (98% vs. 95%) drops from 1,151 unpaired runs/arm to 678 paired scenarios (and from 1,504 to 1,148 paired scenarios for a 2.0 pp gain, or 851 to 364 for a 5.0 pp gain). Setting a practical ablation floor of 1,200 paired scenarios per fault class guarantees statistical power even for 2 pp margins. For continuous flight-tracking metrics (RMSE, settling time, chattering), report Interquartile Mean (IQM), 95% stratified percentile bootstrap CIs (1,000–2,000 resamples), Performance Profiles, and Mann-Whitney Probability of Improvement via rliable rather than outlier-sensitive mean ± SD. [1][2][3][4][5][6]

For Paper 2, training-conditional coverage of split conformal prediction follows an exact Beta distribution and observed finite-test-set coverage follows a Beta-Binomial law: [1][2][3][4][5][6]

To guarantee P(ξ ≥ 1 − α − ε) ≥ 1 − δ monotonically for all sample sizes ≥ N_cal, exact Beta inversion sets sharp calibration floors: at α = 0.05 (95% target) and tolerance ε = 0.02, you require N_cal ≥ 360 (95% confidence, δ = 0.05) or 760 (99% confidence, δ = 0.01), rising to 1,380 (δ = 0.05) at ε = 0.01; at α = 0.10 (90% target) and ε = 0.02, you require N_cal ≥ 650 (δ = 0.05) or 1,330 (δ = 0.01). Because standard split conformal guarantees marginal rather than subgroup coverage, allocate N_cal ≥ 750–1,000 per language (1,500–2,000 total calibration instructions across Turkish and English, plus a dedicated code-switched OOD evaluation split) and N_test ≥ 500–1,000 per language. At α = 0.05 and N_cal = 1,000, scaling N_test from 200 to 1,000 narrows the 95% Beta-Binomial predictive interval of observed test coverage from [91.5%, 98.0%] to [93.0%, 96.8%]. Following SCP-NL2TL's validation practice, report mean realized risk and exceedance rates across repeated random calibration/test resplits (e.g., 100 Monte Carlo resplits). Train all Z3-GRPO configurations across 5 independent random seeds (three-seed tables draw routine CoRL pushback). Finally, because CoRL explicitly scrutinizes simulation-only submissions for transfer evidence, pair real-imagery UAVid + Istanbul aerial SLZ evaluation with hardware-in-the-loop Cortex-M7 execution—and ideally 20–30 closed-loop physical flights on a compact coaxial/hexarotor testbed running the bilingual Z3-GRPO co-pilot inside the deterministic RTA cage. [1][2][3][4][5][6]

2026 Reproducibility and Submission Gate Checklist

Enforce every verification gate below prior to uploading to PaperPlaza (IEEE RA-L/ICRA) or OpenReview (CoRL):

1. https://geolibre.app/features/ (Features - GeoLibre)

2. https://github.com/lefticus/cons_expr/issues/18 (web page has to be loaded twice to work · Issue #18 - GitHub)

3. https://deck.gl/next/docs/whats-new (What's New | deck.gl)

4. https://deck.gl/docs/developer-guide/building-apps (Building Apps | deck.gl)

5. https://blog.vito.nyc/posts/cmake-pkg/ (Modern CMake Packaging: A Guide - vito.nyc)

6. https://github.com/actions/runner-images/blob/main/images/ubuntu/Ubuntu2404-Arm64-Readme.md (runner-images/images/ubuntu/Ubuntu2404-Arm64-Readme.md at)

7. https://iscinumpy.dev/tags/cibuildwheel/ (Overview of all pages with the tag #Cibuildwheel - ISciNumPy.dev)

8. https://safeguard.sh/resources/blog/multi-stage-docker-build-security (Multi-Stage Docker Build Security Best Practices (2026))

9. https://www.theflightbrief.com/articles/easa-moc-5-sc-vtol-issue-2-november-2026 (EASA Publishes Final MOC-5 SC-VTOL Issue 2 | The Flight Brief)

10. https://www.corl.org/contributions/instruction-for-authors (Instruction for Authors - CoRL 2026)

11. https://www.ieee-ras.org/publications/ra-l/ (RA-L - IEEE Robotics and Automation Society Website)

12. https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/ (RA-L Information for Authors - IEEE Robotics and Automation Society)

13. https://github.com/actions/runner-images/issues/14226 ([Ubuntu] Ubuntu 26.04 and Ubuntu 26.04 Arm is now available as a)

14. https://foxglove.dev/blog/foxglove-vs-foxglove-studio-two-years-on (Foxglove vs. Foxglove Studio: Two Years On)

15. https://dev.to/flora-suite/continuing-the-journey-flora-a-fork-of-foxglove-3pd3 (Continuing the Journey: Flora—A Fork of Foxglove - DEV Community)

16. https://foxglove.dev/blog/foxglove-vs-foxglove-studio-two-years-on (Foxglove vs. Foxglove Studio: Two Years On)
