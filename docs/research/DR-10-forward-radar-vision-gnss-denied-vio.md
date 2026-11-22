# VertiRTA v2.0 Autonomous Forward Perception, 77 GHz Radar Fusion, and GNSS-Denied Navigation Blueprint

Deploying forward thin-obstacle avoidance, active slow-ramp GNSS spoofing defense, and GNSS-denied corridor navigation on VertiRTA v2.0 inside Istanbul's 603 m coastal pinch west of Sabiha Gökçen (LTFJ) CTR (29.190° E) confronts four hard physical and legal constraints that invalidate conventional robotics stacks. First, a 2024–2026 source-level license audit disqualifies both OpenVINS and VINS-Fusion alongside ORB-SLAM3 because all three carry GNU GPL-3.0 licenses; similarly, BlenderProc (GPL-3.0), DINOv2-SALAD (GPL-3.0), Drone-vs-Bird (non-commercial DUA), InsPLAD / DDOS / UAVScenes (CC BY-NC), Deep Hough Transform (CC BY-NC), SegFormer (NVIDIA NC), Sample4Geo (unlicensed), and Meta's 2025 DINOv3 (custom license containing military/warfare/ITAR restrictions) cannot enter an Apache-2.0 dual-use UAM/defense repository. Second, the onboard GTX 1650 SUPER (Turing TU116, 4 GB GDDR6) ships with zero Tensor Cores, and naive PyTorch or multi-process ONNX Runtime deployments burn 500–990 MB of VRAM on CUDA contexts before loading weights. Staying under the < 300 MB combined VRAM ceiling alongside PIDNet-S is feasible only inside a single C++20 TensorRT Lean Runtime process using CUDA_MODULE_LOADING=LAZY and a mutually exclusive setDeviceMemoryV2() activation scratchpad (244.0 MB core process VRAM; 248.8–259.2 MB with 1 Hz geo-registration resident). [1][2][3][4][5][6][7][8][9][10][11][12][13][14][15][16][17][18]

Third, neither electro-optical vision nor 77 GHz radar alone certifies safe clearance against 10–25 mm cables at 20–25 m/s cruise speeds. Narrow-FOV (45°) horizon-band split-stacking (1280×320 -> 640×640) multiplies sky-backlit detection range by 4.8× over resized 90° wide-angle feeds (92.7–154.5 m vs. 19.2–32.0 m at 0.25 px), yet drops to 38.6–64.4 m against cluttered Istanbul urban backdrops—well short of a 20 m/s comfortable braking envelope (81.1 m). Conversely, 77 GHz radar produces strong normal-incidence specular glints (+3.72 to +5.94 dBsm at 100 m, scaling as 1/R³) and stranded Bragg diffraction lobes out to 15°, but collapses into urban clutter (-45 dBsm) at skew angles > 20°. Only asynchronous Radar–Camera interpretation-plane fusion into WireCapsuleTable closes both blind spots. Fourth, in the GNSS-denied navigation layer, 10 m GSD Sentinel-2 imagery compresses a 50–120 m AGL nadir camera footprint into just 8.4×4.7 to 20.1×11.3 reference pixels (64×–153× scale chasm), causing complete local keypoint matcher failure and restricting global retrieval to ~45 m CEP50. Holding bounded 3σ CTR polygon buffers (8.8–17.2 m steady-state) inside the 603 m pinch requires GCP-controlled 0.5–1.0 m orthophotos, Basalt (BSD-3-Clause) or SuiteSparse-free OKVIS2 (BSD-3-Clause) VIO anchored by a radar altimeter/barometer, XFeat + LighterGlue (Apache-2.0) local registration, and a Lagged-Baseline Bias-Compensated 2D Vector MCUSUM spoofing guard that catches 0.5–2.0 m/s silent ramps in 0.2–1.4 s (P95). [1][2][3][4][5][6][7][8][9][10][11][12][13][14][15][16][17][18]

1. Forward Thin-Obstacle Vision and 77 GHz 4D mmWave Radar Fusion

Thin overhead conductors (10–25 mm ACSR powerlines, tower-crane hoist ropes, and guy wires) pose the sharpest collision hazard along Istanbul's Anatolian coastal corridor. Protecting VertiRTA v2.0 requires jointly engineering sub-pixel optical geometry, license-clean datasets and synthetic rendering, single-context TU116 VRAM budgeting, and physics-faithful 77 GHz radar fusion rather than assuming cameras or automotive radars behave like rangefinders.

1.1 Exact Optical Geometry and Horizon-Band Split-Stacking Envelope

For a forward 1280×720 (16:9) global-shutter RGB camera with square pixels and horizontal field of view [STRIPPED_MATH_OR_ICON], the native pinhole focal length in pixels is [STRIPPED_MATH_OR_ICON]. After applying horizontal resize scale [STRIPPED_MATH_OR_ICON], the network sees effective focal length [STRIPPED_MATH_OR_ICON]. A physical cable of diameter [STRIPPED_MATH_OR_ICON] at slant range [STRIPPED_MATH_OR_ICON] subtends cross-wire width:

Optical modulation losses prevent raw geometric width from translating directly into crisp 1-pixel binary contrast. Combining lens diffraction and off-axis aberrations (MTF ≈ 0.65), Bayer RGGB demosaicing (MTF ≈ 0.70), and residual rotor-vibration smear (MTF ≈ 0.85) yields an effective 1-pixel contrast transfer of roughly 0.387. Because a powerline spans at least 50–100 contiguous pixels across the horizon, topology-aware segmentation with directional line aggregation accumulates coherent signal along the wire length. Against a bright, uniform sky (intrinsic contrast ≈ 0.75), practical detection holds down to roughly 0.25 px geometric width robustly (reaching 0.20 px in clear-sky limits). Against low-contrast coastal urban clutter—window mullions, balconies, dark roofs, and hazy Marmara horizons (intrinsic contrast ≈ 0.20–0.30)—background edge gradients lift the practical reliable detection floor to 0.60–0.75 px.

Evaluating 1280×720 -> 640×640 cropping exposes a major geometric flaw in naive center-cropping. Taking a central 640×640 1:1 native crop from a 45° (50°) lens preserves full pixel density (f_net = 1545.1 px or 1372.5 px), but throws away half the horizontal azimuth field, narrowing horizontal coverage to 23.40° (26.25°) while wasting vertical resolution on empty zenith sky and low ground (23.40° / 26.25° VFOV). During 15°–20° banked turns or crosswind crabbing, that narrow azimuth cone creates dangerous lateral blind spots. The superior geometry is IMU-horizon-stabilized split-stacking (1280×320 -> 640×640): derotate the frame using 200 Hz IMU roll/pitch, crop a full-width 1280×320 horizon band (45.0° × 11.82° at 45° HFOV or 50.0° × 13.30° at 50° HFOV, spanning ±5.9° to ±6.65° vertical corridor), split into Left (640×320) and Right (640×320) native tiles, and stack vertically into one 640×640 tensor at 1:1 native pixel resolution with zero downsampling loss. Adding a 16 px horizontal overlap or explicit coordinate channels prevents seam splitting on wires crossing the image centerline.

Camera Configuration (1280×720 Input)

Effective f_net (px)

IFOV (mrad)

GSD at 100 m (mm)

15 mm Sky 0.25 / 0.20 px (m)

15 mm Urban 0.60 px (m)

25 mm Sky 0.25 / 0.20 px (m)

25 mm Urban 0.60 px (m)

Effective FOV (H × V)

45° Stacked Horizon (1280×320 -> 640×640)

1545.1

0.647

64.7

92.7 / 115.9

38.6

154.5 / 193.1

64.4

45.0° × 11.8°

45° Naive Central Crop (640×640 native)

1545.1

0.647

64.7

92.7 / 115.9

38.6

154.5 / 193.1

64.4

23.4° × 23.4°

50° Stacked Horizon (1280×320 -> 640×640)

1372.5

0.729

72.9

82.3 / 102.9

34.3

137.2 / 171.6

57.2

50.0° × 13.3°

50° Full-Frame Resized (1280 -> 640)

686.2

1.457

145.7

41.2 / 51.5

17.2

68.6 / 85.8

28.6

50.0° × 29.4°

90° Full-Frame Resized (1280 -> 640)

320.0

3.125

312.5

19.2 / 24.0

8.0

32.0 / 40.0

13.3

90.0° × 59.0°

110° Full-Frame Resized (1280 -> 640)

224.1

4.463

446.3

13.4 / 16.8

5.6

22.4 / 28.0

9.3

110.0° × 77.3°

The optical-kinematic coupling dictates two firm operational boundaries. Resizing a standard 90° or 110° wide-angle camera into 640×640 shrinks a 15 mm cable at 50 m to 0.096 px (90°) or 0.067 px (110°), limiting detection to 13.4–24.0 m against open sky and 5.6–8.0 m against buildings—insufficient even for a 10 m/s eVTOL (22.2–28.9 m stopping distance with 0.22 s confirmation/actuator lag and a 10 m safety bubble). Yet even the optimal 45° split-stacked pipeline drops to 38.6 m on a 15 mm cable (and 25.8 m on a 10 mm tower-crane hoist wire) in urban clutter, whereas 20 m/s cruise demands 54.4 m for a 0.5g emergency stop, 81.1 m for a 0.3g nominal stop, or 58.2–71.0 m for a 12 m lateral/vertical CBVF step-evasion. Forward vision therefore cannot certify wire avoidance alone over cluttered urban backdrops: it supplies high-angular-resolution bearing and sky-silhouette cues to 77 GHz radar, while the HJI-CBVF-QP must cap airspeed to ≤ 15 m/s (35.8 m emergency stop) whenever radar is absent or degraded over visually busy terrain.

1.2 2024–2026 Dataset License Audit and Clean Apache-2.0 Synthetic Pipeline

Most widely cited aerial powerline, bird, and counter-UAS datasets carry non-commercial, ShareAlike, GPL, or unverified-provenance conditions incompatible with an Apache-2.0 dual-use stack. Auditing primary upstream repositories and publication terms as of 2026 separates usable assets from legal liabilities.

Dataset

Domain and Scale

Verified License (2024–2026)

VertiRTA v2.0 Verdict and Engineering Caveats

TTPLA (R3ab/ttpla_dataset)

1,100 real 4K (3840×2160) UAV images; transmission towers + powerlines

Apache-2.0 (verbatim LICENSE, Rabab Abdelfattah)

APPROVED — Cleanest real powerline corpus. Caveat: close-up inspection framing near towers rather than long-range horizon approach.

Me[span_33](start_span)[span_33](end_span)[span_35](start_span)[span_35](end_span)ndeley Powerline GT (Yetgin & Gerek, twxp8xccsw/9)

Real aircraft VL + IR captures across 21 regions of Türkiye (TEİAŞ); 512×512 masks

CC BY 4.0 (Mendeley Data v9)

APPROVED (w/ attribution) — Regionally matched Turkish grid backgrounds. Caveat: 512×512 resolution limits long-range sub-pixel fidelity.

DeformX/WireSeg-36K

36,000 synthetic (1024×1024) + 300 real cable masks + float32 depth

CC BY 4.0 (rendered dataset)

CONDITIONAL (Pretraining Only) — Indoor/robotic DLO domains (data_center, wire_on_plane, flying_wire), not aerial horizons; upstream DeformX-Assets USDs carry non-redistributable NVIDIA terms.

Purdue UAV

50 air-to-air UAV videos (70,250 HD frames, up to 8 drones/frame)

BSD-3-Clause (explicit data distribution license)

APPROVED — Premier permissive air-to-air small-drone training and tracking set.

DUT-Anti-UAV (wangdongdut/DUT-Anti-UAV)

10,000 visible RGB drone images + 20 tracking clips (35+ UAV types)

Apache-2.0 (repo LICENSE; subset archives omit separate terms)

APPROVED (Audit Provenance) — Upstream repo is Apache-2.0 (ignore secondary sites mislabeling it MIT), though per-image copyright provenance in zip subsets is undocumented.

Anti-UAV (ZhaoJ9014/Anti-UAV)

Anti-UAV300 RGB+IR HD sequences (410 and 600 subsets are IR-only)

MIT (repo LICENSE)

APPROVED (RGB300 Subset) — Single-target tracking focus; verify downstream commercial clearance on internet-sourced video segments.

PLDU / PLDM (SnorkerHeng/PLD-UAV)

573 urban / 287 mountain 540×360 pixel-wise wire images

AMBIGUOUS / NO UPSTREAM LICENSE

QUARANTINE — Upstream GitHub repository lacks a LICENSE file; aggregator portals listing CC BY 4.0 conflate the MDPI article's CC BY license with dataset distribution rights.

SynthWires (Madaan et al., IROS 2017)

68,000 synthetic wires composited on real 640×480 drone imagery

AMBIGUOUS / UNLICENSED

QUARANTINE — No license terms published on project page and upstream repository unavailable.

WireSegHR (adobe-research/auto-wire-removal)

High-resolution street/consumer wire segmentation test set

AMBIGUOUS / NO LICENSE FILE

QUARANTINE — No license file or usage terms published in repo; associated segmentation model uses NC SegFormer weights.

SPL (george-kalitsios/Synthetic-Power-Lines)

12,850 Unity 1920×1080 synthetic powerline frames

AMBIGUOUS / CONTACT-AUTHOR

QUARANTINE — No OSI/CC license file; distribution requires contacting author directly.

InsPLAD (andreluizbvs/InsPLAD)

10,607 Full HD UAV powerline asset/defect images

CC BY-NC-SA 4.0 / CC BY-NC 3.0

HARD BAN — Explicitly forbids commercial use and mandates ShareAlike.

STN PLAD & UZH-RPG PAMPC

133 5K asset images / IROS 2023 hybrid powerline dataset

GPL-3.0

HARD BAN — Strong copyleft; incompatible with Apache-2.0.

DDOS (Kolbeinsson & Mikolajczyk, 2024)

34,000 synthetic 1280×720 drone frames with wires, depth, flow

CC BY-NC 4.0

HARD BAN — Non-commercial clause bars commercial/defense training.

Drone-vs-Bird (WOSDETC 2023–2025)

Multi-year moving-camera drone + foreground bird videos

Non-Commercial DUA (email agreement)

HARD BAN — Explicitly non-commercial research DUA; official annotations also omit bird bounding boxes.

MVA2023 SOD4SB (Small Bird Detection)

39,070 UAV images (137,121 bird boxes) + drone2021 train1

CC BY-SA 4.0 (paper) / Unclear data license

QUARANTINE — ShareAlike ambiguity; disregard unverified Roboflow re-uploads marked CC BY 4.0.

UAVScenes (ICCV 2025)

120,000 labeled RGB + LiDAR UAV pairs

CC BY-NC-SA 4.0

HARD BAN — Academic non-commercial use only.

Seraphim Drone Dataset (2025)

83,483 YOLO drone images aggregated from 23 open datasets

CC BY 4.0 (claimed aggregator)

QUARANTINE (License Laundering Risk) — Bundles 23 unenumerated upstream sources without auditable per-source commercial clearance.

Public Tower-Crane Sets (Roboflow Universe)

113–697 crane-only (up to 7,111 mixed construction) ground images

Claimed CC BY 4.0; Roboflow 3.0 models AGPL-3.0

INSUFFICIENT / QUARANTINE — Small, ground-up viewpoint, unvetted web copyrights; synthetic generation is mandatory.

Designing a synthetic pipeline that stays genuinely Apache-2.0 requires avoiding a widespread Blender GPL trap. Blender itself is licensed under GNU GPL-3.0+, DLR-RM/BlenderProc is explicitly GPL-3.0, and under Blender Foundation policy, distributed Python scripts calling Blender's bpy API must be shared under a GPL-compatible license. Although .blend files and rendered images/masks are legally program output belonging solely to the user and free of GPL obligations, committing bpy scripts or BlenderProc into the Apache-2.0 VertiRTA v2.0 codebase violates strict permissive licensing. [1][2][3][4][5][6]

The clean-room alternative separates procedural geometry authoring from rendering using strictly permissive backends:

1.3 Lightweight MIT/Apache-2.0 Thin-Structure Architectures Under 150 MB FP16 VRAM on GTX 1650 SUPER

Hardware reality on the local NVIDIA GTX 1650 SUPER dictates the entire inference architecture. Built on the 12 nm Turing TU116 die, the card provides 1,280 CUDA cores (1725 MHz boost), 4 GB GDDR6 on a 128-bit bus (192 GB/s), 4.416 TFLOPS FP32, and no Tensor Cores or RT Cores. According to Turing architectural catalogs, TU116 retains dedicated FP16 shader ALUs running at 2× FP32 throughput (8.832 TFLOPS FP16), enabling FP16 TensorRT inference to halve memory traffic and roughly double math throughput even without Tensor Core MMA acceleration. [1][2][3][4][5][6][7][8]

The decisive memory bottleneck is CUDA framework overhead, not model weight size. Standard PyTorch or ONNX Runtime CUDAExecutionProvider initialization allocates 500–990 MB of VRAM for the CUDA context, cuDNN, and cuBLAS before loading weights. Running nadir PIDNet-S (7.6M params) and forward perception in two separate OS processes duplicates that runtime overhead above 800 MB, breaching the < 300 MB ceiling by > 2.5×. Fitting inside < 300 MB total process VRAM requires four mandatory TensorRT C++20 practices: [1][2][3][4][5][6][7][8]

(Note on benchmark provenance: no published independent TU116 TensorRT latency benchmarks exist across all eight candidate architectures below; the per-model TU116 FP16 latencies and peak activation pools in the chart and narrative are analytical engineering sizing estimates derived from model FLOPs, TU116 memory bandwidth, and published RTX 3090 / Jetson Orin baselines and must be profiled on target hardware with trtexec during CI verification.) [1][2][3][4][5]

Pixel cross-entropy alone fails on 1-pixel cables because a 1-pixel lateral offset yields zero IoU, producing broken dotted masks that derail 3D line triangulation. Several widely used thin-structure architectures also fail our license screen: SegFormer / MiT carries the non-commercial NVIDIA Source Code License, Hanqer/deep-hough-transform is CC BY-NC 3.0, and Ultralytics YOLOv8/v11 is AGPL-3.0. [1][2][3]

The clean MIT/Apache-2.0 forward perception stack pairs two compact heads sharing the 50 MB activation scratchpad:

setting [STRIPPED_MATH_OR_ICON] with 3 differentiable min/max-pooling soft-skeletonization iterations to preserve thin-wire homotopy and connectivity across cluttered urban edges. If higher urban capacity is required, CoinCheung/BiSeNet (BiSeNetV2, MIT, already updated for strongly-typed TensorRT 11 ONNX; ~3.4M slim params, 7.8 MB FP16 weights) or DDRNet-23-Slim (MIT/Apache-2.0; 5.7M params, 13.1 MB FP16 weights) swaps into the same shared scratchpad. [1][2][3][4][5][6][7]

Under a 30 Hz (33.3 ms) master schedule, executing M-LSD-WireSeg (4.2 ms) and D-FINE-N (6.5 ms) every frame consumes 10.7 ms, while interleaving PIDNet-S (7.5 ms) at 10 Hz peaks at 18.2 ms (55% GPU duty cycle). Running 1 Hz visual geo-registration (XFeat + LighterGlue) on non-SLZ frames peaks at 16.2 ms and adds 4.8 MB FP16 weights for a compact 2.1M-parameter distilled matcher (248.8 MB total VRAM), or ~15.2 MB FP16 weights for the full ~6.6M-parameter XFeat (1.6M) + LighterGlue (~5M) pairing reported in a 2026 edge survey (~259.2 MB total VRAM)—still keeping > 40 MB safety margin under the 300 MB cap. [1][2][3][4][5][6][7]

1.4 77 GHz 4D mmWave Radar Physics, Doppler Separation, Gazebo Harmonic Modeling, and Asynchronous Fusion

At 77.0 GHz ([STRIPPED_MATH_OR_ICON]), an overhead cable violates the point-target radar assumption. At slant range R = 100 m, the first Fresnel zone along the wire spans [STRIPPED_MATH_OR_ICON]—much shorter than the ~7.0 m cable span illuminated by a 4° azimuth beam. At exact normal incidence ([STRIPPED_MATH_OR_ICON]), stationary-phase physical optics integration yields a specular RCS that grows linearly with range: [1][2][3][4][5][6][7]

where [STRIPPED_MATH_OR_ICON] is conductor radius. At 100 m, numerical Fresnel integration confirms a bright normal glint of +1.96 dBsm (1.57 m²) for 10 mm wire, +3.72 dBsm (2.36 m²) for 15 mm, +4.97 dBsm (3.14 m²) for 20 mm, and +5.94 dBsm (3.93 m²) for 25 mm—comparable to a pedestrian (0.5–2 m²). Because [STRIPPED_MATH_OR_ICON], received power along the normal glint decays as 1/R³ instead of 1/R⁴. [1][2]

Off normal incidence, backscatter drops sharply. Helically stranded ACSR cables with outer strand pitch period p ≈ 8–20 mm produce discrete Bragg diffraction lobes satisfying: [1][2]

At 77 GHz for p = 12–16 mm, active first- and second-order Bragg modes sit between 7.0° and 14.1°. Comprehensive public 77 GHz dBsm-vs-angle lookup tables for braided conductors remain sparse in open literature, so modeling relies on Sarabandi & Park's classic W-band (94 GHz) experimental phenomenology alongside 77 GHz wire measurements (-18.8 to -32.9 dBsm across incidence sweeps). Three brutal physical cliffs govern detection: (1) Bragg modes cease above 15° incidence because local specular points on cylindrical strands disappear; (2) beyond 20° skew/grazing incidence, RCS plunges toward -45 dBsm and falls completely beneath urban clutter, rendering conventional radar blind (smooth unstranded crane hoist ropes exhibit zero Bragg lobes at all); and (3) continuous heavy rain filling inter-strand grooves substantially degrades Bragg RCS. [1][2]

Link-budget verification across three hardware tiers exposes sharp operational limits. Bare single-chip COTS EVMs (Tier-1A, e.g., AWR1843/AWR2944, 13 dBm TX, 12 dB NF) detect -18 dBsm rogue quadcopters at only 26.3 m, -25 dBsm birds at 17.6 m, and stranded Bragg wire lobes (-20 to -24 dBsm) at 18.6–23.4 m—unacceptable for eVTOL cruise. A 4-chip cascaded 4D imaging radar (Tier-1B, e.g., TI AWR2243 12TX/16RX 86-virtual-element MIMO or Continental ARS540 class) serves as the minimum viable infrastructure baseline (274.9 m tower cranes, 204.0–224.5 m normal-incidence 15–20 mm wires, 34.6–43.6 m Bragg lobes), yet remains short-range against small aerial intruders (32.7–48.9 m). Full all-weather small-drone and Bragg-wire Sense-and-Avoid requires a Tier-2 active-PA/LNA steered front-end: RFbeam's 2026 vendor specification for its active-antenna AWR2E44P demonstrator (30 dBm/ch PA, 6 dB NF, 35.5 dB two-way gain, 20 ms coherent integration, 12 dB SNR, 9.1 dB losses) projects 238 m boresight (107 m at ±25°/±25° corners) on a steady -25 dBsm drone—closely matching our independent radar-equation calculation of 240.1 m—subject to airborne 76–77 GHz spectrum approval under ETSI TR 104 078 (June 2025) / EASA-JARUS frameworks. Note also that Swerling I target fluctuation demands 17.0 dB post-processing SNR versus 8.5 dB for non-fluctuating targets (13.0 dB for Swerling III), cutting single-frame free-space ranges by roughly 39% relative to steady-RCS curves (21% relative to Swerling III). [1]

Separating static wires from birds and rogue drones via Doppler requires compensating for ego-motion. For a stationary object at unit bearing [STRIPPED_MATH_OR_ICON], the ego-compensated Doppler residual is [STRIPPED_MATH_OR_ICON]. Differentiating [STRIPPED_MATH_OR_ICON] reveals substantial off-boresight azimuth-uncertainty leakage:

Even with high-resolution 4D radar ([STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON]) and VIO ([STRIPPED_MATH_OR_ICON]), the 3σ static Doppler gate expands from ±0.64 m/s on boresight to ±1.01 m/s (20 m/s cruise) and ±1.17 m/s (25 m/s cruise) at 30° azimuth (widening further if [STRIPPED_MATH_OR_ICON] degrades to 2.5° at FOV edges). Consequently, hovering quadcopters and tangentially soaring gulls fall inside the static gate. Moreover, W-band experiments show plastic drone rotor micro-Doppler sidebands sit roughly 30 dB below bulk body returns (~-48 dBsm), putting 77 GHz micro-Doppler classification out of reach past 4.7–8.7 m on Tier-1 COTS units and ~64 m even on Tier-2 active-PA radars. VertiRTA v2.0 resolves this via three-layer separation: returns with [STRIPPED_MATH_OR_ICON] route directly to IntruderTrackerKf; returns inside the static gate cross-check against D-FINE-N vision detections to catch hovering drones/birds; and multi-frame track kinematics (speed, acceleration, and path curvature—shown in NATO STO trials to classify drones vs. birds at 97% TPR when RCS and speed overlap) refine target class hypotheses inside IntruderTrackerKf. [1][2]

Gazebo Harmonic Custom 77 GHz Radar Plugin and Asynchronous Fusion into WireCapsule

Gazebo Harmonic's official gz-sensors library (Apache-2.0) includes cameras, depth, GPU LiDAR, IMU, NavSat, and underwater DVL, but no mmWave radar model. Approximating 77 GHz radar with noisy GpuLidarSensor points masks glint/grazing physics and generates dangerously overconfident fusion filters. Build an Apache-2.0 C++20 custom sensor subclassing gz::sensors::Sensor (MmWave77GHzSensor) paired with a gz::sim::System plugin (MmWave77GHzSystem registered via <sensor type="custom" gz:type="mmwave_77ghz"> and updated in PreUpdate/PostUpdate): [1][2]

To fuse asynchronous 30 Hz vision (M-LSD-WireSeg) and 15–20 Hz 4D radar into IForwardHazardProvider (WireCapsuleTable <= 64, IntruderTrackerKf <= 16) for the 100 Hz HJI-CBVF-QP:

1. 200 Hz SE(3) Pose Ring Buffer: Buffer VIO/EKF2 body poses T_W_B(t) over the trailing 2.0 s and interpolate exact camera mid-exposure ([STRIPPED_MATH_OR_ICON]) and radar ([STRIPPED_MATH_OR_ICON]) poses via SE(3) SLERP, eliminating 0.4–0.8 m latency skew at 20 m/s.

2. Interpretation-Plane Association: Back-project each 2D vision segment (u_a, v_a) -> (u_b, v_b) through calibrated intrinsics into a 3D interpretation plane with unit normal [STRIPPED_MATH_OR_ICON] passing through camera center [STRIPPED_MATH_OR_ICON]. Gate static-filtered 4D radar points [STRIPPED_MATH_OR_ICON] (Cartesian covariance [STRIPPED_MATH_OR_ICON]) against active planes via Mahalanobis distance [STRIPPED_MATH_OR_ICON].

3. Wire-Span EKF and Grazing-Angle Safety Fallback: Track each conductor span with a 6-state anchor-tangent-sag filter [STRIPPED_MATH_OR_ICON] plus clipped vision-bearing extents [STRIPPED_MATH_OR_ICON]. Within [STRIPPED_MATH_OR_ICON], a single normal glint locks range onto the interpretation plane immediately and 2+ returns (glint + Bragg lobe or pylon) lock full 3D orientation. In radar-blind grazing geometry ([STRIPPED_MATH_OR_ICON]), accumulate sequential vision planes across VIO motion to triangulate via SVD once baseline parallax exceeds 3°; before convergence, emit a conservative horizon-standoff capsule shrinking with closing speed so the CBVF immediately caps airspeed.

4. Piecewise Catenary Tessellation into WireCapsule (p_a, p_b, R_w): Tessellate spans longer than 30 m into 2–4 linear WireCapsule segments (keeping chord-to-sag error < 0.5 m; 12–16 conductors occupy 24–48 of the 64 slots) with dynamic radius:

combining airframe half-span (~4–5 m), sag chord error (0.5 m), Marmara gust sway (1.5 m), and 3σ cross-wire EKF uncertainty, published double-buffered and lock-free to the 100 Hz safety filter.

2. GNSS Jamming and Stealthy Slow-Ramp Spoofing Detection (core/gnss_denied/)

Within Istanbul's 603 m LTFJ J2 corridor pinch, broadband GNSS jamming is noisy and obvious, whereas stealthy slow-ramp spoofing (0.5–2.0 m/s drift) preserves nominal C/N0 (40–48 dB-Hz), nominal AGC gain, nominal HDOP, and clean fix flags while silently walking the autopilot laterally into LTFJ CTR. [1][2]

2.1 Why Standard PX4 EKF2 Innovation Gates Fail Silently

At 10 Hz (Δt = 0.1 s), a 1.0 m/s spoofing ramp shifts reported GNSS position by only 0.10 m per epoch (0.05 m at 0.5 m/s). Relative to typical EKF2 horizontal innovation standard deviation 1.0–1.5 m, the single-epoch normalized innovation is 0.07–0.10σ—producing a squared test ratio near 0.0004 against PX4's 1.0 rejection threshold (5σ gate). Worse, because EKF2 fuses every accepted measurement into its own position, velocity, and accelerometer-bias states, the filter adapts to the spoofed trajectory and holds future innovations below the gate permanently—an attack explicitly demonstrated in published PX4 spoofing studies. [1][2]

An operational timing pitfall compounds this exposure: an independent fleet audit of public PX4 logs by Munim Technologies found that roughly 30% of vehicles (29.6% initial sample of 164 logs; 31.2% replication of 98 logs) ran with GNSS transport delay misconfigured by > 50 ms, inflating EKF2 GPS velocity test ratios to a median 2.7× steady-flight value during acceleration (versus 1.7× on calibrated logs). Deploying tight spoofing detectors in core/gnss_denied/ therefore mandates PPS time-stamping or calibrated per-receiver delay (SENS_GNSS0_DELAY in PX4 main post-Sept 2026 PR #24399, which renamed vehicle_gps_position to vehicle_gnss; SENS_GPS0_DELAY in PX4 v1.18). [1][2]

2.2 Multi-Hypothesis Detector Architecture (H0 through H3)

core/gnss_denied/ runs a four-hypothesis bank at 10 Hz (H_0: nominal; H_1: RF jamming/interference; H_2: step/meaconing jump; H_3: stealthy slow-ramp spoofing) governed by one strict security boundary: the VIO estimator feeding the detector must remain completely decoupled from GNSS fusion so a spoofer can never poison the reference velocity. [1][2]

Monitor A (H_1, TTD < 0.2 s) inspects multi-band receiver RF health (UBX-MON-RF / Septentrio ISM / PX4 vehicle_gnss jamming and spoofing flags), triggering on broadband noise spikes, anomalous AGC drop paired with falling C/N0, cross-band C/N0 divergence or L1 C/A max-C/N0 oscillation, uniform C/N0 across elevation angles (single-transmitter terrestrial spoofers), or HDOP > 2.5 geometry degradation. Monitor B (H_2, TTD < 0.3 s) sums EKF2 horizontal innovations across a 5-epoch (0.5 s, 10 DoF) window, [STRIPPED_MATH_OR_ICON], paired with consistency checks on reported eph/epv jumps. [1][2]

Monitor C (H_3, stealthy 0.5–2.0 m/s slow-ramp defense) pairs a Lagged-Baseline Bias-Compensated 2D Vector MCUSUM (H_3a) with a Parallel Unbiased Position-Increment Sliding-Window GLRT (H_3b). At 10 Hz, form the ENU velocity-equivalent increment residual:

using dual-band GNSS carrier-Doppler velocity ([STRIPPED_MATH_OR_ICON]) minus decoupled baro/radalt-aided VIO velocity ([STRIPPED_MATH_OR_ICON], giving white noise [STRIPPED_MATH_OR_ICON] per axis), plus slowly drifting VIO scale/heading bias [STRIPPED_MATH_OR_ICON] modeled as a 2D Ornstein–Uhlenbeck process ([STRIPPED_MATH_OR_ICON] over urban terrain; degrading to [STRIPPED_MATH_OR_ICON] over water).

Running naive uncompensated 2D MCUSUM directly on [STRIPPED_MATH_OR_ICON] fails at 0.5 m/s: tolerating raw VIO bias without false alarms forces drift allowance [STRIPPED_MATH_OR_ICON] (h_t = 3.8, ARL0 = 200 hr), leaving zero net drift for a 0.5 m/s ramp (11.4 s median TTD, > 15 s P95). Instead, maintain a 15 s (L_2 = 150) ring buffer and subtract the delayed baseline mean over [t - 15 s, t - 5 s] (L_1 = 50 guard band):

The 5.0 s guard band prevents ramp onset from contaminating [STRIPPED_MATH_OR_ICON] during detection, while subtracting [STRIPPED_MATH_OR_ICON] cuts analytical 1-axis residual bias std from 0.100 m/s down to 0.052 m/s (baro/radalt-aided VIO, τ_b = 60 s). Updating Crosier's direction-invariant 2D Vector MCUSUM:

detects spoofing along any arbitrary heading without Bonferroni loss. To close the loophole where an attacker spoofs pseudorange position while leaving Doppler untouched—or injects an ultra-slow 0.08 m/s² quadratic creep attempting to slip under the 5 s lagged mean—run parallel telescoping position-increment GLRTs, [STRIPPED_MATH_OR_ICON] at [STRIPPED_MATH_OR_ICON]. With dual-band L1/L5 carrier-smoothed code ([STRIPPED_MATH_OR_ICON]), 2D GLRT ([STRIPPED_MATH_OR_ICON]; thresholds 1.70 m at 1.5 s and 2.56 m at 4.0 s) catches any position-only ramp ≥ 1.14 m/s in 1.5 s and ≥ 0.64 m/s in 4.0 s. By contrast, single-band 0.80 m pseudorange noise requires 4.0 s even for 1.32 m/s, confirming that multi-band carrier-velocity + VIO is indispensable for sub-1.5 s detection.

2.3 False-Alarm vs. Time-to-Detect Trade-Offs Against the 603 m LTFJ J2 Pinch

Detector Mode and VIO Regime (10 Hz, σ_w = 0.14 m/s)

k_d / h_t

ARL0 Est. (hr)

P_FA / 30 min

0.5 m/s TTD 50/95/99 (s)

1.0 m/s TTD 50/95/99 (s)

2.0 m/s TTD 50/95/99 (s)

Max 99th-Pct Slip (m)

Lagged MCUSUM (Recommended Nominal) (σ_b = 0.10, τ = 60 s)

0.25 / 2.0

400.0*

0.125%

0.8 / 1.4 / 1.7

0.3 / 0.4 / 0.4

0.2 / 0.2 / 0.2

0.85 m

Lagged MCUSUM (Higher Margin) (σ_b = 0.10, τ = 60 s)

0.25 / 2.4

400.0*

0.125%

1.0 / 1.6 / 2.1

0.4 / 0.4 / 0.5

0.2 / 0.2 / 0.2

1.05 m

Lagged MCUSUM (Hair-Trigger) (σ_b = 0.10, τ = 60 s)

0.22 / 1.8

11.1

4.400%

0.7 / 1.1 / 1.3

0.3 / 0.3 / 0.4

0.2 / 0.2 / 0.2

0.65 m

Naive Uncompensated MCUSUM (σ_b = 0.12, τ = 20 s)

0.50 / 3.8

200.0*

0.250%

11.4 / ≥15 / ≥15

0.8 / 1.3 / 1.8

0.3 / 0.3 / 0.4

> 7.5 m

Static Lagged Over Coastal Water (σ_b = 0.15, τ = 30 s)

0.25 / 2.4

0.04

99.999%

0.9 / 2.4 / 5.3

0.4 / 0.5 / 0.6

0.2 / 0.2 / 0.2

Unusable (FA flood)

Adaptive Coastal Water Mode (σ_b = 0.15, τ = 30 s)

0.35 / 3.4

4.3

11.086%

2.1 / ≥6.0 / ≥6.0

0.6 / 0.8 / 0.9

0.3 / 0.3 / 0.3

≥ 3.00 m

*Continuity-adjusted ARL0 point estimates from zero observed false alarms in 200 simulated hours (Lagged) and 100 simulated hours (Naive); corresponding 95% statistical lower confidence bounds on true ARL0 are > 67 hr and > 33 hr, respectively. Right-censored TTD percentiles hit the 6.0 s (Adaptive) or 15.0 s (Naive) simulation cap.

Inside the 603 m pinch, false alarms and missed detections carry sharply asymmetric operational costs. At (k_d = 0.25, h_t = 2.0) over nominal urban terrain, even a 99th-percentile pre-detection slip across 0.5–2.0 m/s is just 0.40–0.85 m (< 0.15% of the 603 m corridor width). Conversely, a nuisance false alarm needlessly ejecting healthy GNSS forces the aircraft into open-loop dead-reckoning where corridor buffers immediately begin closing at ~1.2–3.6 m/s (combined two-sided 6σ growth) if water occludes geo-registration. Consequently, core/gnss_denied/ must schedule (k_d, h_t) directly off the nadir PIDNet-S water-pixel fraction and VIO feature count: hold (0.25, 2.0) over textured land (water < 20%), widen to (0.35, 3.4) over Marmara shoreline crossings (water > 35%), and require corroboration from Monitor A/B or IMU+airspeed wind-triangle residuals before latching permanent GNSS revocation over water. Once an alarm latches, freeze GNSS fusion into EKF2, roll back the global anchor pose by 2.0 s via the pose ring buffer (stripping the 0.4–1.0 m pre-detection spoof bias), initialize handover uncertainty at σ_init = 3.5 m (2.0 m for clean jamming loss), and transition cleanly to VIO plus visual geo-registration.

3. Permissive VIO, Cross-View Visual Geo-Registration, and Dynamic 3σ CBVF CTR Margin Inflation

3.1 Strictly Permissive VIO Comparison at 50–120 m AGL Over Coastal Urban Terrain

A source-level LICENSE inspection eliminates two widely assumed candidates alongside ORB-SLAM3: both OpenVINS (rpng/open_vins) and VINS-Fusion (HKUST-Aerial-Robotics/VINS-Fusion) are GPL-3.0, carrying the identical strong-copyleft viral requirement. In addition, HybVIO is GPL-3.0, LARVIO is academic-only, KumarRobotics/msckf_vio uses the university-restricted Penn Software License, and rpng/sqrtVINS is LGPL-3.0. Only four mature C++ VIO pipelines offer permissive BSD licensing, and OKVIS2 hides a transitive SuiteSparse GPL/LGPL build dependency that requires deliberate removal. [1][2][3][4][5]

VIO Stack

Verified Upstream License

Transitive License Risks

Camera Modes

EuRoC Avg ATE (m)

Resource Footprint (RAM / CPU)

Coastal 50–120 m AGL Verdict

Basalt (VladyslavUsenko/basalt)

BSD-3-Clause

Core headers use Eigen3/Sophus/Cereal; audit vcpkg third-party ports

Stereo+IMU native (mono only via MIT adaptation)

0.051 (10/11 seqs; V2_03 omitted)

108–137 MB RAM; high multi-core CPU (418–538%)

PRIMARY CHOICE — Square-root sliding-window marginalization, wide double-sphere optics, zero GPU usage. Add IMU-seeded KLT, water masking, baro/radalt scale factor, and covariance output.

OKVIS2 (smartroboticslab/okvis2)

BSD-3-Clause

CRITICAL: Default install links SuiteSparse (CHOLMOD GPL-2.0+/LGPL-2.1+, CXSparse LGPL-2.1+)

Multi-cam / Stereo / Mono + IMU (+ GNSS in OKVIS2-X)

0.071 (VIO) / 0.048 (causal SLAM)

Moderate CPU/RAM (no independent Table 10 figure in audited benchmarks)

STRONG ALTERNATIVE — Only if built with USE_NN=OFF (disabling LibTorch) and Ceres/solver backend stripped of GPL CHOLMOD (pure Eigen Schur). Native mono fallback + multi-sensor factor extensibility.

Kim[span_283](start_span)[span_283](end_span)era-VIO (MIT-SPARK/Kimera-VIO)

BSD-2-Clause

Verify GTSAM/third-party build flags in SBOM

Stereo+IMU & Mono+IMU

0.119

251 MB RAM; low CPU (80.5% avg)

CPU-CONSTRAINED BACKUP — Lowest CPU load in TUM/EuRoC benchmarks, but ~1.7× higher ATE than OKVIS2 and harder to tune outside EuRoC.

ROVIO (ethz-asl/rovio)

BSD License (package.xml)

Legacy ROS1/catkin codebase; fork archived 2024

Mono / Stereo + IMU

0.47–0.75 (TUM-VI); 21.3–21.8 agile

~271 MB RAM; 82.2% CPU

NOT RECOMMENDED — Direct photometric EKF diverges rapidly under coastal sunlight/glint changes and fast eVTOL motion.

OpenVINS (rpng/open_vins)

GPL-3.0

Viral copyleft (sqrtVINS LGPL-3.0)

Mono / Stereo + IMU

0.29–0.38 agile

113 MB short; 1.7–6.[span_258](start_span)[span_258](end_span)5 GB long outdoor

HARD BAN — Violates Apache-2.0 stack licensing.

VINS-Fusion (HKUST)

GPL-3.0

Viral copyleft

Mono / Stereo + IMU + GPS

0.138

High KITTI road drift

HARD BAN — Violates Apache-2.0 stack licensing.

ORB-SLAM3

GPL-3.0

Viral copyleft

All

0.035 (non-causal)

High RAM

HARD BAN (explicitly excluded).

High-altitude (50–120 m [span_250](start_span)[span_250](end_span)AGL) coastal operation degrades standard stereo VIO sharply because stereo depth error scales quadratically with height, [STRIPPED_MATH_OR_ICON]. For a wide-angle nadir VIO camera (f = 600 px, σ_d = 0.25 px), a compact 0.12 m stereo bar drops to 0.60–0.90 px disparity at 80–120 m AGL, blowing depth noise up to 22.2–50.0 m and collapsing effectively to monocular VIO. Even a 0.30 m stereo baseline reaches its practical ceiling around 40–60 m AGL—consistent with an empirical UAV altitude study reporting optimal 2.186 m RMSE (0.27% over an 800 m trajectory; 2–5 m typical) at 40–60 m with a 30 cm baseline before stereo-VO degraded heavily toward 100 m—where 100–120 m AGL yields σ_Z = 13.9–20.0 m. Mounting a rigid 0.80 m fuselage/belly carbon-boom stereo pair (σ_Z = 3.3–7.5 m at 80–120 m, avoiding flexible wings where vibration warps rigid calibration) and tightly fusing a nadir radar altimeter (e.g., PX4-supported Ainstein US-D1; vendor docs note seamless land-to-water operation, and peer-reviewed river trials achieved 5–7 cm accuracy over water) plus barometer locks metric scale continuously across 50–120 m AGL. [1][2][3][4][5][6][7][8][9]

Over multi-kilometer coastal cruise without loop closure, realistic open-loop horizontal drift rates (% of distance flown) for Basalt / OKVIS2 fall into three operational regimes: 0.5%–1.0% (5–10 m/km) over textured Anatolian urban terrain (< 10% water in FOV), governed mainly by gyro yaw-bias random walk; 1.2%–1.8% (12–18 m/km) along mixed shorelines (30–40% Marmara water in FOV) with PIDNet-S water masking active to purge moving wave-glint tracks from Basalt's KLT optical flow; and 2.5%–4.0%+ (25–40 m/km) during hazy or sea-dominated legs (> 60% water) where visual constraints starve.

3.2 Sentinel-2 (10 m GSD) vs. 0.5–1.0 m Aerial Orthophotos and Cross-View Matchers

Evaluating VertiRTA v2.0's nadir 1280×720 (80° HFOV) camera footprint across 50–120 m AGL immediately exposes why 10 m GSD Sentinel-2 fails for meter-level urban geo-registration:

Altitude (AGL)

Nadir Ground Footprint (m)

UAV Native GSD (cm/px)

Sentinel-2 (10 m) Crop Size (px)

Scale Gap vs. Sentinel-2

1.0 m Ortho Crop Size (px)

0.5 m Ortho Crop Size (px)

Scale Gap vs. 0.5 m Ortho

50 m

83.9 × 47.2

6.6

8.4 × 4.7 px

153×

84 × 47 px

168 × 94 px

7.6×

80 m

134.3 × 75.5

10.5

13.4 × 7.6 px

95×

134 × 76 px

269 × 151 px

4.8×

100 m

167.8 × 94.4

13.1

16.8 × 9.4 px

76×

168 × 94 px

336 × 189 px

3.8×

120 m

201.4 × 113.3

15.7

20.1 × 11.3 px

64×

201 × 113 px

403 × 227 px

3.2×

Across 50–120 m AGL, a full nadir camera frame covers only 8×5 to 20×11 Sentinel-2 pixels, collapsing individual buildings and streets into 1–2 mixed pixels. Local keypoint matchers (XFeat/ALIKED + LightGlue) fail 100% on Sentinel-2 at these altitudes. Even stitching sequential UAV frames into a multi-hundred-meter mosaic for global descriptor matching hits Sentinel-2's inherent geodetic floor: ESA's Copernicus Mission Performance Centre validation measures Level-1C absolute geolocation error at 11.6 m (2σ CE95, requirement < 12.5 m). Combined with 10 m pixel quantization and seasonal shift, Sentinel-2 delivers roughly 35–60 m CEP50 (> 80 m CE95)—useful solely as a coarse gross-blunder check, not for tightening CTR buffers. [1][2][3][4]

Conversely, 0.5–1.0 m aerial orthophotos provide 168×94 to 403×227 px per nadir frame (3.2×–7.6× scale ratio, trivially matched by downsampling the UAV frame using radalt/baro AGL), but zero wall-to-wall, commercially open 0.5 m Istanbul orthophotos exist: [1][2][3][4]

Cross-View Model / Pipeline

Verified License (2024–2026)

Params and Edge Compute Cost

Realistic 50–120 m AGL CEP50 (0.5 m GCP Ortho)

Verdict and Caveats

XFeat + LighterGlue (verlab/accelerated_features)

Apache-2.0 (code + weights)

1.6M + ~5M (~60–80 FPS Orin TRT FP16 per 2026 survey; ~5.5 ms TU116 est.)

4–7 m (CE95: 12–18 m); ~20% miss under shadow/clutter

PRIMARY ONBOARD MATCHER — Fastest C++/ONNX/TRT deployment. Requires VIO yaw + radalt scale pre-warping (4.8–15.2 MB FP16 weights depending on matcher size).

ALIKED-n16rot + LightGlue (Shiaoming/ALIKED + cvg/LightGlue)

BSD-3-Clause + Apache-2.0 (aliked_lightglue.pth)

~10M full LightGlue (~80 FPS RTX 4090; ~12 ms TU116 est.)

3–5 m (CE95: 9–14 m; aligns with AnyVisLoc 74.1% @ 5 m & GIM 5.46 m)

HIGH-ACCURACY ALTERNATIVE — Stronger cross-season/rotation robustness than XFeat; training code unpublished (inference+weights BSD-3). Full 9-layer weights add ~25 MB FP16 (~7.4 MB pruned).

MegaLoc (gmberton/MegaLoc, 2025)

MIT (code + weights)

ResNet-50 trunk (~25M; ~14 ms est.)

15–25 m (tile-stride limited retrieval)

APPROVED (Lost-in-Space Re-Init Only) — Supersedes EigenPlaces (MIT); closed training recipe. Swap-in only if σ_pos > 50 m.

DINOv2-ViT-S/14 (facebookresearch/dinov2)

Apache-2.0 (code + weights)

22.1M, 50.8 MB FP16 weights, ~16.5 ms TU116 est.

15–25 m (global descriptor only)

APPROVED (Heavy Fallback) — Clean Apache-2.0, but keeping 50.8 MB weights GPU-resident strains the < 300 MB pool.

DINOv3 (facebookresearch/dinov3, Aug 2025)

Custom Meta DINOv3 License

29M (ConvNeXt-T) to 7B

Strong satellite backbone (SAT-493M)

HARD BAN — Non-OSI custom license with military/warfare/ITAR end-use prohibitions and derivative redistribution friction.

DINOv2-SALAD (serizba/salad)

GPL-3.0

~86M

High VPR recall

HARD BAN — Strong copyleft GPL-3.0.

Sample[span_373](start_span)[span_373](end_span)4Geo (Skyy93/Sample4Geo)

NO LICENSE FILE

87.57M, 90.24 GFLOPs

High benchmark R@1 (92.65% U1652)

HARD BAN — Unlicensed upstream repo + too large (~190 MB FP16 weights) for TU116. (CAMP MIT and DAC Apache-2.0 inherit Sample4Geo code and remain 36–97M params.)

SuperPoint / SuperGlue (incl. superpoint_lightglue.pth)

Restrictive / Non-Commercial

—

—

HARD BAN (explicitly excluded).

EfficientLoFTR (zju3dv/EfficientLoFTR)

Custom Project Registration License v1.0

~10M, 5–10 Hz edge est.

Strong semi-dense matching

QUARANTINE — Upstream repo migrated away from Apache-2.0 to a custom license.

During active GNSS-denied corridor cruise, VIO maintains a tight local position prior, eliminating any need for continuous global retrieval. At 0.5–1.0 Hz (whenever PIDNet-S confirms < 50% water in the nadir frame), derotate the nadir frame to North-up via VIO/compass yaw, rescale to 0.5 m/px via radalt/baro AGL, extract a 3σ-bounded search patch (256×256 to 512×512 px) from an onboard MBTiles 0.5 m orthophoto cache, and run XFeat + LighterGlue (Apache-2.0) (or ALIKED-n16rot + LightGlue at 0.5 Hz for stronger seasonal resilience) with RANSAC homography and DSM-height PnP. Accept a visual fix into the navigation filter only when three gates pass simultaneously: ≥ 25 RANSAC inliers with < 2.5 px reprojection RMSE, inlier spatial convex-hull coverage > 25% of frame area (rejecting degenerate rooftop clusters), and 2D Mahalanobis innovation [STRIPPED_MATH_OR_ICON] against the VIO prior, fusing with measurement covariance [STRIPPED_MATH_OR_ICON] where [STRIPPED_MATH_OR_ICON]. [1][2][3][4]

3.3 Exact Covariance Propagation and Dynamic 3σ CBVF CTR Polygon Buffer Inflation

Per Turkish DHMI AIP records (AD 2 LTFJ and AD 2 LTBX), Istanbul's Anatolian coastal route is squeezed between two overlapping 5 NM (9,260 m) control zones. Sabiha Gökçen (LTFJ) CTR is centered on ARP 40°53'54"N, 029°18'33"E (40.89833°N, 29.30917°E), 5 NM radius, SFC – 2,500 ft AMSL, placing its western 5 NM tangent at 29.19912°E—767.1 m east of 29.190°E at ARP latitude and narrowing to the 603 m operational corridor pinch against coastal/maritime bounds. Samandıra Army Air Base (LTBX) CTR is centered on 40°59'56"N, 029°12'58"E (40.99889°N, 29.21611°E), 5 NM radius, SFC – 2,000 ft AMSL, joined to LTFJ CTR at 40°59'15"N, 029°19'30"E (13.64 km center-to-center). [1][2][3][4]

Between global fixes, VertiRTA v2.0 propagates local ENU horizontal covariance [STRIPPED_MATH_OR_ICON] by augmenting VIO's incremental sliding-window covariance with an explicit conservative systematic drift term so the filter never becomes overconfident on long straight legs:

where [STRIPPED_MATH_OR_ICON] is open-loop distance since the last accepted fix, [STRIPPED_MATH_OR_ICON] (0.015–0.10 m²/m) captures random-walk diffusion, and [STRIPPED_MATH_OR_ICON] (0.5%–4.0%, dynamically scheduled from PIDNet-S water fraction and tracked feature count) captures 1σ fractional scale/yaw drift. Each gated visual geo-registration fix applies a Joseph-form Kalman update with [STRIPPED_MATH_OR_ICON] and resets [STRIPPED_MATH_OR_ICON].

For every LTFJ and LTBX CTR polygon edge [STRIPPED_MATH_OR_ICON] with outward unit normal [STRIPPED_MATH_OR_ICON], inflate signed distance [STRIPPED_MATH_OR_ICON] by the directional 3σ Mahalanobis support function plus fixed map geodetic bias [STRIPPED_MATH_OR_ICON] (2.0 m for GCP-controlled orthophotos):

Because [STRIPPED_MATH_OR_ICON] expands continuously during GNSS-denied legs ([STRIPPED_MATH_OR_ICON]), omitting the time-derivative term breaks forward invariance of the safe set. The 100 Hz HJI-CBVF-QP must enforce the full time-varying Lie-derivative barrier inequality:

where [STRIPPED_MATH_OR_ICON] automatically steers the eVTOL proactively away from encroaching CTR buffers as uncertainty grows.

Under VertiRTA v2.0's conservative two-sided inflation of both the eastern LTFJ CTR arc and the western coastal keep-out wall (consuming 6σ_pos(t) of the 603 m pinch; one-sided eastern-only inflation with a fixed 50 m western coastal setback relaxes the 80 m maneuver minimum to σ_pos = 157.7 m), flight-envelope authority divides into three deterministic states. In Bounded GNSS-Denied Cruise (σ_pos < 25 m, 3σ < 75 m), 0.5–1.0 m orthophotos paired with XFeat/ALIKED + LightGlue hold steady-state peak 3σ to 8.8–17.2 m (568.5–585.3 m usable width) and survive 30 s (600 m) Marmara water blackouts with 3σ ≤ 50.2 m (> 502 m open). In Degraded Caution Mode (25 m ≤ σ_pos < 65 m), caused by prolonged fog, night, or over-water registration outages, the flight controller throttles airspeed to 12–15 m/s (slowing both VIO drift per second and braking distance) and shifts reference tracking westward away from LTFJ/LTBX arcs. At σ_pos ≥ 65.0 m (3σ ≥ 195 m, 213 m flyable width remaining), VertiRTA v2.0 triggers a Deterministic SLZ Abort, descending onto the highest-confidence nadir PIDNet-S Safe Landing Zone well before the 80 m CBVF infeasibility floor (σ_pos = 87.2 m) or zero-width pinch collapse (σ_pos = 100.5 m). Without visual geo-registration at 20 m/s, this abort fires after 318 s (6.37 km) at 1.0% coastal urban drift, dropping to 127 s (2.55 km) under 2.5% coastal sea-glint and 80 s (1.59 km) over open sea.

4. End-to-End License Compliance and Verification Checklist

Before merging into the Apache-2.0 VertiRTA v2.0 mainline, enforce five automated CI/CD and SITL release gates:

1. https://github.com/rpng/open_vins/blob/master/LICENSE (open_vins/LICENSE at master - GitHub)

2. https://github.com/rpng/open_vins (GitHub - rpng/open_vins: An open source platform for visual-inertial)

3. https://github.com/hkust-aerial-robotics/vins-fusion (HKUST-Aerial-Robotics/VINS-Fusion: An optimization ... - GitHub)

4. https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/LICENCE (VINS-Fusion/LICENCE at master - GitHub)

5. https://iris.unisalento.it/bitstream/11587/514549/1/The_Drone-vs-Bird_Detection_Grand_Challenge_at_ICASSP_2023_A_Review_of_Methods_and_Results-compressed.pdf (The Drone-vs-Bird Detection Grand Challenge at ICASSP 2023 - IRIS)

6. https://andreluizbvs.github.io/InsPLAD/ (InsPLAD - André Luiz Buarque Vieira e Silva)

7. https://github.com/serizba/salad (Optimal Transport Aggregation for Visual Place Recognition · GitHub)

8. https://github.com/hanqer/deep-hough-transform (GitHub - Hanqer/deep-hough-transform: Jittor and Pytorch code for)

9. https://huggingface.co/datasets/benediktkol/DDOS (benediktkol/DDOS · Datasets at Hugging Face)

10. https://github.com/NVlabs/SegFormer/blob/master/LICENSE (SegFormer/LICENSE at master - GitHub)

11. https://github.com/facebookresearch/dinov3/issues/348 (on-site commercial demo without redistribution of DINOv3 weights)

12. https://www.tomshardware.com/reviews/nvidia-gtx_1650-super-turing (Nvidia GTX 1650 Super Review: 1080p Gaming for $160)

13. https://www.notebookcheck.net/NVIDIA-GeForce-GTX-1650-Desktop-GPU-Benchmarks-and-Specs.421420.0.html (NVIDIA GeForce GTX 1650 Desktop GPU - Benchmarks and Specs)

14. https://flopper.io/gpu/nvidia-geforce-gtx-1660-ti-6gb (NVIDIA GeForce GTX 1660 Ti 6GB GPU Specifications - Flopper.io)

15. https://www.spheron.network/blog/gpu-memory-requirements-llm/ (LLM VRAM Requirements: How Much GPU Memory You Need (2026))

16. https://forums.developer.nvidia.com/t/gpu-vs-cpu-deep-learning-memory-usage/284630 (GPU vs CPU deep learning memory usage - Jetson Nano)

17. https://docs.nvidia.com/deeplearning/tensorrt/11.5.0/architecture/how-trt-works.html (How TensorRT Works - NVIDIA Documentation Hub)

18. https://egordmitriev.dev/blog/2026-05-23-drone-to-satellite-localization-2026 (Drone-to-Satellite Visual Geo-Localization: State of the Art, Open)
