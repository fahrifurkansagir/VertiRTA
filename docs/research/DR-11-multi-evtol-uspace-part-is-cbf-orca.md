# VertiRTA Multi-Agent U-Space Architecture and Fleet Safety Control Specification (5–20 Coaxial X8 eVTOLs, Anatolian Coastal Corridor)

Expanding VertiRTA from single-aircraft Runtime Assurance (RTA) to a 5–20 aircraft simultaneous fleet along Istanbul's Anatolian Marmara coast (Maltepe, Dragos, Cevizli, Kartal, strictly west of 29.200° E) requires co-design across four coupled layers: a vendor-neutral ASTM F3548-21 / InterUSS Discovery and Synchronization Service (DSS) and ASTM F3411-22a Remote ID plane; an EASA Regulation (EU) 2023/203 (Part-IS, mandatory since 22 February 2026) cryptographic telemetry layer that patches an OMG DDS Security replay gap; harmonized FAA EB-105A / Draft AC 150/5390-4 and EASA PTS-VPT-DSN vertiport operations coupled to pad-side active liquid battery chillers; and a deadlock-free 3D ellipsoidal CBF-ORCA tactical deconfliction layer driving each aircraft's 100 Hz CbvfSafetyFilter (K <= 16 QP rows) and 400 Hz INDI-WLS mixer. Across this 6.08 km coastal corridor, all four vertiport pads and connecting flight lanes maintain at least 2.01 km lateral clearance outside the LTFJ (Sabiha Gökçen) 5 NM CTR, at least 2.32 km outside the LTBX (Samandıra) 5 NM CTR, and at least 2,061 m west of 29.200° E. [1][2][3]

Three system-level findings govern feasibility and certification under the 2024–2026 baseline. First, pad-side active liquid battery cooling—not FATO wake spacing—is the binding fleet capacity bottleneck: under 35 °C Istanbul summer conditions with end-of-life (EoL) 400 V / 22.18 kWh packs, forced-air ground cooling triggers a multi-cycle thermal ratchet above 45 °C that stretches turnaround from 12.13 min to 22.18 min and drives a 20-aircraft fleet with 3 chargers per vertiport into 14.26 min mean airborne holding, whereas 4 actively chilled chargers per vertiport (15 °C coolant, 260 W/K) stabilize 20 aircraft at 62.0–66.2 flights/hr with ≤1.74 min mean hold. Second, OMG DDS Security 1.2 alone does not block encrypted telemetry replay because compliant DDS:Crypto:AES-GCM-GMAC receivers do not verify monotonic IV_s progression; prepending an application-layer 40-byte security header (carrying a 20-byte monotonic nonce, TAI timestamp, and key epoch) checked by a 128-bit sliding window closes this gap, while splitting 20 Hz V2V kinematics (symmetric AES-256-GCM) from 1 Hz Remote ID and all Priority-0 emergency alerts (Ed25519 signed) holds estimated ARM Cortex-A72 cryptographic load to 1.47% of one core across a 20-aircraft fleet. Third, sub-500 ms Priority-0 corridor clearing and FATO vacation are achievable within our worst-case latency budget and 100 Hz closed-loop simulation—48.6 ms p50 / 131.8 ms p99.9 on the V2V tactical path and 95.2 ms p50 / 290.6 ms p99.9 on the USSP/DSS Contingent path—provided emergency alerts bypass Fast DDS RELIABLE transport (which exhibited a reproducible 177.62 ms median latency spike in published ROS 2/Fast DDS evaluations) in favor of BEST_EFFORT triple-burst UDP multicast. [1][2][3]

Three evidence boundaries apply to this specification. Neither FAA EB-105A (December 2024), FAA Draft AC 150/5390-4 (October 2026), EASA PTS-VPT-DSN (March 2022), nor the FAA Orlando MCO HITL report (February 2026) prescribes a statutory wake-vortex separation timer in seconds for 240 kg multirotor eVTOLs; the 60–90 s spacing schedule specified here synthesizes ICAO baseline wake references, NASA fast-time vertiport scheduling studies, and FAA downwash/outwash measurements. Likewise, the 1-RC Thevenin electro-thermal turnaround numbers are calibrated to published Molicel INR-21700-P50B datasheet impedance and NMC heat-capacity values rather than calorimeter tests of a finished VertiRTA pack, and the embedded ARM Cortex-A72 cryptographic, RF link, and QP timing budgets synthesize published component benchmarks that must be profiled experimentally on the target flight computer and local Istanbul radio links. [1][2][3]

Operational Airspace Envelope and Anatolian Coastal Vertiport Network

Physical airspace separation is anchored to official Turkish AIP coordinates (LT_AD_2_LTFJ and LT_AD_2_LTBX). Sabiha Gökçen International Airport (LTFJ) maintains a 5 NM (9.260 km) radius Control Zone (CTR) centered at 40°53'54"N, 029°18'33"E (40.898333° N, 29.309167° E), SFC to 2,500 ft MSL, joined along its northwest arc via 40°59'15"N, 029°19'30"E and 40°54'46.03"N, 029°11'31.64"E (40.912786° N, 29.192122° E). Samandıra Army Air Base (LTBX) maintains a 5 NM (9.260 km) CTR centered at 40°59'56"N, 029°12'58"E (40.998889° N, 29.216111° E), SFC to 2,000 ft AMSL, joined southwest at 40°59'15"N, 029°19'30"E. The overlying Istanbul TMA floor sits at 1,500 ft MSL (457 m MSL). [1][2][3]

Geodesic verification demonstrates that the westernmost tangent of LTFJ's 5 NM circle at 40.8983° N reaches 29.19912° E, and its published northwest join coordinate lies at 29.19212° E. Treating 29.200° E directly as an active flight-planning limit would leave less than 80 m of lateral buffer against LTFJ controlled airspace. VertiRTA therefore hardcodes an eastern geofence barrier at 29.18800° E inside every aircraft's onboard CbvfSafetyFilter and USSP conformance monitor, enforcing at least 340 m of continuous buffer west of the LTFJ northwest join coordinate and at least 1,010 m west of 29.200° E across the entire operating latitude band.

Vertiport ID & Coastal Site

WGS84 Coordinates (Lat, Lon)

Pad Elev. (MSL / W84)

Dist. to LTFJ ARP (Margin > 5 NM)

Dist. to LTBX CTR Center (Margin > 5 NM)

Margin West of 29.200° E

Role & Pad Allocation

VPT-MLT (Maltepe Orhangazi)

40.92380° N, 29.11850° E

3.0 m / 40.35 m

8.79 NM (+7.01 km)

6.32 NM (+2.44 km)

6,857 m

Primary West Hub (2 Op FATOs + 1 FATO-EMG, 4 Chilled Chargers)

VPT-DRG (Dragos Sahil)

40.91420° N, 29.13550° E

4.0 m / 41.35 m

7.94 NM (+5.44 km)

6.26 NM (+2.34 km)

5,426 m

Coastal Spoke (1 Op FATO + 1 FATO-EMG, 4 Chilled Chargers)

VPT-CVZ (Cevizli Sahil)

40.90380° N, 29.15450° E

3.5 m / 40.85 m

7.03 NM (+3.75 km)

6.36 NM (+2.51 km)

3,828 m

Coastal Spoke (1 Op FATO + 1 FATO-EMG, 4 Chilled Chargers)

VPT-KRT (Kartal Kordonboyu)

40.89020° N, 29.17550° E

4.0 m / 41.35 m

6.09 NM (+2.01 km)

6.78 NM (+3.30 km)

2,061 m

Primary East Hub (2 Op FATOs + 1 FATO-EMG, 4 Chilled Chargers)

Geodetic datum note: For Maltepe/Kartal (UTM Zone 35N), the EGM2008 geoid undulation is N = 37.35 m (consistent within 0.8 m of the 120 ft / 36.6 m undulation published at LTFJ) and magnetic declination is 6.09° E (+2.9'/yr). Because ASTM F3548-21 permits only WGS84 ellipsoidal altitudes (reference: "W84", units: "M"), every USSP and onboard altitude transform MUST apply h_W84 = H_MSL + 37.35 m to prevent a 37 m vertical datum mismatch. [1][2][3]

At the national regulatory layer, official Turkish Directorate General of Civil Aviation (SHGM) announcements and the İHATTYS tracking portal (ihatakip.shgm.gov.tr) establish mandatory aircraft registration (≥500 g), flight authorization outside green zones, and İHATTYS telemetry integration under the 2026 SHT-İHA instruction. According to Turkish UAS industry and training-provider summaries of the revised SHT-İHA text (reported as effective 30 July 2026, with a 31 July 2027 compliance transition for pre-existing aircraft), a 240 kg MTOW aircraft falls in class M3 (≥150 kg), requires P2 authorization for BVLOS/autonomous/corridor operations (with Class 3 medical certification), and faces mandatory Remote ID, automated geofencing, İHATTYS relay, and SHGM software certification spanning flight control, GCS, and multi-UAV fleet command software—provisions that VertiRTA's certification team should verify directly against the authoritative SHGM regulatory gazette prior to operational filing. The operational ceiling is fixed at 120 m AGL (400 ft AGL)—123–124 m MSL (160.35–161.35 m W84), maintaining a 333 m vertical buffer beneath the 1,500 ft MSL Istanbul TMA floor—with vertically segregated bidirectional cruise lanes (southeastbound MLT → KRT at 70 m AGL, northwestbound KRT → MLT at 105 m AGL, providing 35 m separation matched to our downwash ellipsoid) and a seaward emergency descent/transit shelf over the Marmara waterline at 40–60 m AGL. [1][2][3][4][5][6][7]

1. ASTM F3548-21 USSP / InterUSS DSS Architecture and ASTM F3411-22a Remote ID

Vendor-Neutral Local Microservice Topology

To avoid cloud vendor lock-in while remaining testable against the Linux Foundation InterUSS uss_qualifier automated test suite, VertiRTA deploys a self-hosted container stack (Docker Compose or bare-metal Kubernetes across edge servers at the Maltepe and Kartal hubs) modeled on interuss/dss and the open-source openutm/flight-blender microservice pattern: [1][2][3][4][5][6][7]

S2 Spatial Indexing and Known DSS Precision/Area Caveats: The InterUSS DSS spatially indexes 4D volumes using Google S2 cells. Because the reviewed DSS architecture excerpt does not hardcode a single S2 cell level, we size local DAR queries using Level-13 S2 cells as the representative reference scale (averaging 1.27 km², spanning 0.76–1.59 km² with 850–1,225 m edge lengths per published S2 cell statistics), under which the ~24 km² Maltepe–Kartal coastal corridor spans roughly 15–25 cells; exact deployed cell-level limits should be confirmed in dss build configuration. Implementers must explicitly handle two documented InterUSS behaviors: (a) DAR false positives, where two disjoint operational intents occupying the same S2 cell are both returned by a DSS query, requiring exact peer-to-peer 4D intersection evaluation inside vertirta-ussp; and (b) InterUSS Issue #1549, where interuss/dss rejected single constraint polygons exceeding 2,500 km² despite ASTM CSTM0025 specifying a 10,000 km² composite ceiling—avoided here by tiling any wide Istanbul NOTAM polygons into sub-100 km² tiles. To satisfy ASTM F3548-21 GEN0500 (geospatial numerical precision), exact 4D intersections execute in local tangent-plane ENU coordinates anchored at (40.9080° N, 29.1470° E) with sub-5 cm geodesic error across the 7 km span and a 1.0 m conservative inflation margin. [1][2][3][4][5][6][7]

Exact ASTM F3548-21 Schemas, Priority Mapping, and State Machine

Every operational intent managed by vertirta-ussp adheres to the ASTM F3548-21 (utm.yaml / scd.yaml) OpenAPI 3.0 data contracts. Implementers must guard against an inversion between ASTM and real-time flight-control nomenclature: in ASTM scd.yaml, priority is an ordinal integer where larger values denote higher priority (a lower-priority intent cannot create a conflict with a higher-priority intent), whereas real-time embedded control designates emergency preemption as "Priority-0". vertirta-ussp enforces a strict bijective translation: [1][2][3][4][5][6][7]

Internal RTA Priority Class

ASTM F3548-21 priority Integer

Typical State(s)

SCD Conflict Rule

Trigger Condition

Priority-0 (Emergency CEL / CSFL)

100

Contingent

May overlap any lower-priority OIV; omits key for zero-block DSS write

Irreversible same-arm CEL spin, single-rotor CSFL, BMS thermal fault, < 20% SoC

Priority-1 (Active Revenue Shuttle)

10

Accepted, Activated, Nonconforming

Zero 4D overlap with priority >= 10; yields to priority = 100

Scheduled passenger/cargo corridor legs across Maltepe–Kartal

Priority-2 (Non-Revenue Repositioning)

5

Accepted, Activated

Zero 4D overlap with priority >= 5; preemptible by revenue demand

Pad-clearing / charger-rebalancing flights (per NASA TM-20250009519)

Schema provenance note: EntityOVN (16–128 chars), Subscription, ConstraintReference, OperationalIntentState, Priority, Volume4D, Volume3D, Altitude (W84, M, -8,000 to 100,000), and Time (RFC3339, 'Z') are verified directly against InterUSS utm.yaml and scd.yaml; the full Operationa[span_56](start_span)[span_56](end_span)lIntentReference field set and REST path templates below reflect standard ASTM F3548-21 / InterUSS conventions and should be code-generated directly from the pinned astm-utm/Protocol/utm.yaml file at build time. [1][2][3]

Operational Intent State Transition Specification (F3548-21)

Current State

Target State

DSS Endpoint & Action

OVN key Required?

Peer Telemetry Endpoint

Conformance / Operational Trigger & Constraints

None

Accepted

PUT /dss/v1/operational_intent_references/{id}

Yes (all overlapping OVNs)

Not active

Pre-flight SCD passes against all priority >= own OIVs, LTFJ/LTBX buffer constraints, and FATO 60–90 s wake slot.

Accepted

Activated

PUT /dss/v1[span_57](start_span)[span_57](end_span)/operational_intent_references/{id}/{ovn}

Yes

Ready

Pre-takeoff FATO lock confirmed, battery SoC ≥ mission + 25% reserve, BMS temperature ≤ 40 °C, Remote ID arming healthy (OPEN_DRONE_ID_ARM_STATUS).

Activated

Activated

PUT /dss/v1/operational_intent_references/{id}/{ovn}

Yes

Ready

In-flight nominal re-routing or holding extension; requires full SCD re-check and 409 Conflict OVN retry loop.

Activated

Nonconforming

PUT /dss/v1/operational_intent_references/{id}/{ovn}

May omit

Mandatory (GET /uss/v1/operational_intents/{id}/telemetry)

Aircraft deviates outside nominal volumes for > 2.0 s (e.g., gust or CBF-ORCA evasion) while retaining full 4-DOF control; USS publishes expanded off_nominal_volumes.

Nonconforming

Activated

PUT /dss/v1/operational_intent_references/{id}/{ovn}

Yes

Ready

Permitted only after UA re-enters a newly strategically deconflicted nominal 4D volume and obtains fresh OVNs.

Activated / Nonconforming

Contingent

PUT /dss/v1/operational_intent_references/{id}/{ovn}

Omitted (zero-block)

Mandatory (≥1 Hz live state)

Irreversible (Contingent cannot return to Activated; must transition to Ended). Triggered on CSFL, irreversible same-arm CEL spin, or unrecoverable C2/USSP loss; sets priority: 100, publishes diversion off_nominal_volumes to nearest FATO-EMG, and pushes immediate POST /uss/v1/operational_intents alerts to all DSS-returned subscribers.

Accepted / Activated / Nonconforming / Contingent

Ended

DELETE /dss/v1/operational_intent_references/{id}/{ovn}

No

Closed

Flight cancelled before launch, or aircraft confirmed weight-on-wheels and clear of FATO/TLOF; implicit subscriptions deleted automatically.

Strategic Conflict Detection & Resolution (SCD/SCR) and Optimistic Concurrency Loop

When planning or mutating an Accepted or Activated operation, vertirta-ussp executes a five-step atomic SCD/SCR loop complying with ASTM F3548-21 SCD0005–0080:

ASTM F3411-22a Broadcast and Network Remote ID Specification

Both Broadcast RID (direct RF over BLE 5 Long Range + Wi-Fi Beacon/NAN, compliant with ASTM F3411-22a and ASD-STAN EN 4709-002) and Network RID (F3411-22a v2 DSS/USS API, fulfilling EASA Regulation (EU) 2021/664 Article 8) carry identical application-layer telemetry. [1][2][3][4][5][6][7][8][9][10]

Broadcast Remote ID Exact 25-Byte Packed Wire Structs (opendroneid-core-c)

Each aircraft's companion computer encodes OpenDroneID blocks following the reference opendroneid-core-c byte layout (ProtoVersion = 2 for F3411-22a) and feeds them over MAVLink (OPEN_DRONE_ID_*) or native SPI/UART to a dual-transport ESP32-S3 / nRF5340 RF beacon transmitting BLE 5 Extended Advertising (LE Coded PHY, S=8) and 2.4 GHz Wi-Fi Beacon Vendor Specific IEs on Channel 6, paired with BLE 4 legacy rotation at ≤260 ms slot intervals so standard smartphones receive Location and System within 1 s and all other blocks within 3 s: [1][2][3][4][5][6][7][8][9][10]

Using RFC 9575 DRIP Wrapper over Extended Transports (Section 4.3.2) achieves exact wire-capacity packing: the 4 mandatory messages (BasicID, Location, System, OperatorID) plus the 89-byte cryptographic wrapper (1 B SAM Type + 4 B VNB + 4 B VNA + 16 B DET + 64 B Ed25519 signature) spanning 5 Authentication pages (17 + 4 × 23 = 109 B capacity) occupy exactly 9 messages (228 bytes)—fitting inside one BLE 5 Extended / Wi-Fi Beacon ODID_MessagePack_encoded frame at 1 Hz with zero fragmentation, meeting typical ASTM F3411 RF compliance targets (1 km minimum range, < 5% loss, ≤1 s latency as summarized by accredited test-lab guidance). An ODID_SelfID_encoded (0x3, 23-byte ASCII mission/emergency descriptor) alternates with OperatorID on odd seconds during routine cruise and preempts immediately on CEL/CSFL ("EMG-CEL-SPIN-VPT-CVZ"). Codepoint verification note: draft-wiethuechter-drip-det-moc-02 maps DRIP SAM Types to 0x01 (DRIP Link), 0x02 (DRIP Wrapper), 0x03 (DRIP Manifest), and 0x04 (DRIP Frame), whereas one secondary summary of RFC 9575 conflates Section 4.3/4.4 numbers as 0x03/0x04; implementations must pin final byte values directly to the authoritative IANA DRIP SAM Types registry. [1][2][3]

Network Remote ID (ASTM F3411-22a v2 / EASA Article 8) Microservice Endpoints

In parallel, vertirta-ussp operates as an ASTM F3411-22a v2 Net-RID Service Provider (SP), publishing an IdentificationServiceArea (ISA) in the local DSS covering the coastal fleet envelope and exposing GET {uss_base_url}/uss/flights (returning RIDAircraftState telemetry with near-real-time trails up to 60 s) and per-flight details endpoints. Mixed v1/v2 interoperability obeys the InterUSS URL-suffix rule: if uss_base_url does not end in /flights, consumers append /uss/flights for F3411-22a (v2), and ISA mutation notifications are posted to {subscriber.url}/uss/identification_service_areas/{isa_id}. Exact request/response JSON schemas and DSS ISA path parameters should be generated directly from the interuss/dss F3411-22a OpenAPI specification during build. [1][2][3]

2. EASA Regulation (EU) 2023/203 (Part-IS) Aviation Cybersecurity in U-Space Telemetry

Regulatory Alignment with Part-IS (Mandatory 22 February 2026)

Commission Implementing Regulation (EU) 2023/203 (Part-IS) became legally mandatory on 22 February 2026 across ATM/ANS providers, air operators, CAMOs, Part-145 organizations, and—via its explicit amendment to Article 15(1)(f) and Article 18(l) of Regulation (EU) 2021/664—all certified U-Space Service Providers (USSPs) and single CISPs. Complementing it, Delegated Regulation (EU) 2022/1645 (Part-IS.D.OR) took effect on 16 October 2025 for Part-21 DOA/POA organizations and aerodromes, and EASA issued updated AMC/GM Issue 1 Amendment 1 via ED Decisions 2025/013/R–2025/015/R (24 July 2025) alongside Regulation (EU) 2025/2293 (11 November 2025). Per official EASA Part-IS FAQs, compliance with the EU NIS or NIS 2 Directive does not grant automatic Part-IS compliance (Part-IS is not treated as NIS 2 Lex Specialis and requires competent-authority equivalence verification, with external safety reporting under IS.I.OR.230 strictly non-substitutable). [1][2][3]

Part-IS Clause

Mandatory Regulatory Duty

VertiRTA Technical & Architectural Control

IS.I.OR.200 / .250

Establish safety-linked ISMS & approved ISMM

Integrated Safety-Security Management System binding fleet USSP, vertiport chargers, and airborne RTA computers under one shared risk severity scale.

IS.I.OR.205 / .210

Risk assessment & treatment across internal assets & external M2M interfaces

Complete interface register covering DDS V2V mesh, USSP–DSS REST, Vertiport WebSocket, Broadcast RID, and SHGM İHATTYS gateway; named key ownership and 30-day rotation.

IS.I.OR.215 / .220

Event detection, containment, and recovery to a safe state within defined RTO

Real-time signature/nonce anomaly counters, Mahalanobis kinematic cross-checks in IntruderTrackerKf, and deterministic cyber fail-safe FSM (RTO ≤ 10 s to autonomous safe recovery).

IS.I.OR.230

External safety-cyber reporting (immediate alert + ≤72 h formal report + follow-up)

Automated ECCAIRS-2-compatible occurrence report generator triggered on any SEC_USSP_QUARANTINE or sustained spoofing event, aligned with Regulation (EU) 376/2014.

IS.I.OR.240 / .245

Verified personnel/service identities; 5-year ISMS/incident record retention (3-year post-exit personnel)

Hardware-rooted PKI (TPM 2.0 / PKCS#11), per-aircraft Ed25519 leaf certs, hash-chained DDS_LogTopic + PostgreSQL WORM audit archive.

Lightweight Cryptographic Architecture Across DDS, REST, WebSocket, and Broadcast RID

Every aircraft, vertiport controller, and USSP microservice carries a hardware-backed Ed25519 keypair (32-byte public key, 64-byte deterministic RFC 8032 signature, non-malleable, eliminating runtime RNG failure modes) signed by an internal VertiRTA Fleet Intermediate CA. Each aircraft's Ed25519 public key binds deterministically to a 128-bit IETF RFC 9374 DRIP Entity Tag (DET) structured as an IPv6 HHIT (2001:30::/28 [28 bits] | RAA [14 bits] | HDA [14 bits] | HHIT Suite ID = 0x05 EdDSA/cSHAKE128 [8 bits] | 64-bit `cSHAKE128(Prefix | [1][2][3][4]

| HID |

| 0x05 |

| PK, 64, "", 0x00B5A69C795DF5D5F0087F56843F2C40)). **Software dependency pinning is mandatory**: pin **libsodium ≥ 1.0.21 / 1.0.22-stable** (remediating the December 2025 Edwards25519 subgroup-validation defect in crypto_core_ed25519_is_valid_point()affecting ≤1.0.20[span_111](start_span)[span_111](end_span)[span_112](start_span)[span_112](end_span)) and **Fast DDS ≥ 2.14.6 or ≥ 3.4.1** (closingCVE-2024-28231, CVE-2025-62799, and CVE-2026-22590` reported in industrial DDS security advisories). [1][2][3][4]

Layer 1: Closing the OMG DDS Security Replay Gap on V2V Telemetry & Preemption

Fleet V2V tactical state (20 Hz) and Priority-0 emergency alerts run over OMG DDS Security 1.2 (Fast DDS 3.4.x / 2.14.6+) built with -DSECURITY=ON and launched with ROS_SECURITY_STRATEGY=Enforce, activating all four implemented built-in plugins: DDS:Auth:PKI-DH, DDS:Access:Permissions, DDS:Crypto:AES-GCM-GMAC (256-bit keys), and DDS:Logging:DDS_LogTopic. The CA-signed S/MIME Domain Governance XML MUST set <allow_unauthenticated_participants>false</allow_unauthenticated_participants>, <discovery_protection_kind>ENCRYPT</discovery_protection_kind>, <rtps_protection_kind>ENCRYPT</rtps_protection_kind>, and <default>DENY</default> in every participant's permissions file—eliminating plaintext SPDP discovery exposure (239.255.0.1:7400) and permissive fallback misconfigurations. [1][2][3][4]

Crucially, however, security analysis of OMG DDS Security by KIT showed that while DDS:Crypto:AES-GCM-GMAC forms a 96-bit `IV = ID_session |

| IV_swith an incrementing 64-bit suffix, **compliant receivers perform zero checks onIV_smonotonicity or reuse**, exposing raw DDS Security links to encrypted packet replay.VertiRTAeliminates this vulnerability at the application wire boundary via a fixed **40-byteVertiRtaSecureHeader` (containing the 16-byte DET, a 20-byte monotonic nonce/timestamp/epoch block, and 4 bytes of framing) encrypted inside every DDS IDL payload and checked by an O(1) 128-bit sliding window:

Layer 2: REST / WebSocket Transport (TLS 1.3 mTLS + RFC 9421) and Broadcast RID (RFC 9575)

USSP–DSS REST endpoints and Vertiport Manager WebSockets run over TLS 1.3 mutual authentication (RFC 8446) using X25519 key exchange, Ed25519 leaf certificates, and TLS_AES_256_GCM_SHA384 (or TLS_CHACHA20_POLY1305_SHA256) over persistent HTTP/2 and WSS keep-alive sessions. On ARM Cortex-A72 (Raspberry Pi 4) LAN benchmarks (10 ms RTT), a full 1-RTT TLS 1.3 handshake takes 11.3–13.3 ms, paid once at connection setup and amortized across ongoing telemetry. All safety-critical JSON mutations (OIV updates, FATO clearances, emergency preemption relays) also carry RFC 9421 HTTP Message Signatures (Ed25519) over @method, @path, content-digest, x-vertirta-nonce, and created so non-repudiation survives reverse proxies and 5-year archival. [1][2]

On the Broadcast RID channel, every 1 Hz BLE 5 / Wi-Fi Beacon Message Pack embeds the 5-page Extended-Transport DRIP Wrapper signing the concatenated BasicID + Location + System + OperatorID blocks, complemented by a 0.2 Hz DRIP Link Broadcast Endorsement (BE: HDA, UA). Receivers enforce all four RFC 9575 Section 6.4.2 trust assessment rules: (1) cryptographic Ed25519 signature verification against the attested DET public key, (2) valid Wrapper or Manifest framing, (3) rapidly changing nonce-like Location/Vector fields within freshness bounds, and (4) non-cryptographic physical correlation against optical/radar sensors or IntruderTrackerKf tracks. [1][2]

Cryptographic Latency, Wire Overhead, and Embedded CPU Budget (N = 20 Fleet)

The table below gives the analytical wire and CPU engineering budget for a 20-aircraft fleet on an ARM64 Cortex-A72 companion computer (assuming conservative libsodium primitives of ~120 µs Ed25519 sign, ~280 µs Ed25519 verify, ~2.0 µs AES-256-GCM per 256-byte frame, and ~0.05 µs ReplayWindow128 check, scaled from published x86_64/ARM/RPi benchmarks—to be confirmed by hardware profiling on the production flight computer). For DDS wire sizing, a 96-byte kinematic payload adds the 40-byte VertiRtaSecureHeader plus approximately 44 bytes of OMG DDS Security AES-GCM CryptoHeader/CryptoFooter submessage framing (exact CDR framing varies slightly by middleware encapsulation), and +64 bytes when an Ed25519 signature is appended: [1][2]

Cryptographic Mode / Transport Channel

Per-Packet Wire Expansion

Approx. Total Packet Size

Ops/sec per Aircraft (N=20, 19 Peers)

Est. Cortex-A72 Single-Core Load

Assessment

Mode A (Baseline: Ed25519 on every 20 Hz V2V packet)

+40 B Header + ~44 B DDS GCM + 64 B Sig = ~148 B

~244 B

423 Ed25519 vrfy/s + 23 sign/s

12.12% (~1.12 ms 100 Hz burst jitter)

Unnecessary asymmetric load on routine kinematics

Mode B (Adopted: AES-256-GCM + Nonce @ 20 Hz; Ed25519 @ 1 Hz & all P0/OIV)

~84 B (20 Hz V2V) / ~148 B (P0 Alert)

~180 B / ~244 B (single 1,500 B MTU)

400 AES-GCM/s + 48 Ed25519 vrfy/s + 4 sign/s

1.47% (< 0.3 ms P0 verify)

Recommended; zero MTU fragmentation, non-repudiable P0

TLS 1.3 mTLS + RFC 9421 (USSP/DSS REST & WSS)

+29 B TLS record + ~110 B RFC 9421 header

~950 B JSON OIV

1–5 req/s (persistent HTTP/2)

< 0.25% (11.3–13.3 ms initial 1-RTT setup)

Negligible steady-state latency

ASTM F3411-22a + RFC 9575 Extended DRIP Wrapper

+125 B (5 × 25 B Auth pages)

228 B (9 msgs exact)

1 sign/s + 19 vrfy/s

0.54% (included in Mode B)

Fits single BLE 5 / Wi-Fi Beacon pack

Fail-Safe Degradation Under Spoofed or Replayed USSP / V2V Messages

To satisfy IS.I.OR.220 (incident detection, containment, and recovery to a safe state within a defined recovery time), every aircraft's RTA supervisor enforces a deterministic cyber-resilience state machine:

1. SEC_NOMINAL → SEC_SUSPECT_PEER (V2V / RID Spoof or Replay): Any packet failing AES-GCM authentication, Ed25519 verification, ReplayWindow128, or ±250 ms TAI skew is dropped before reaching IntruderTrackerKf and logged to DDS_LogTopic. If a cryptographically authenticated (compromised-key) peer broadcasts kinematically impossible states (failing the 3-σ Mahalanobis innovation gate in IntruderTrackerKf or contradicting onboard radar/optical/ADS-B observations, violating RFC 9575 Sec. 6.4.2), that peer is stripped of Priority-0 preemption rights (blocking denial-of-service FATO hijacking) and demoted to a non-cooperative obstacle (β_ij = 1.0, inflated covariance) so ownship avoids any real airframe without obeying fabricated retreat vectors.

2. SEC_NOMINAL → SEC_USSP_QUARANTINE (Compromised or Spoofed USSP): Triggered immediately upon TLS 1.3 mTLS pinning failure, invalid RFC 9421 signature or nonce replay on any USSP/Vertiport command, > 3 consecutive 409 Conflict OVN storms indicating DSS desynchronization, or any semantic safety violation—specifically, any USSP-commanded trajectory or diversion crossing the onboard hardcoded 29.188° E LTFJ/LTBX geofence or 120 m AGL ceiling. The aircraft drops all further USSP trajectory directives, locks onto its last signed onboard geofence and vertiport table, continues decentralized 3D CBF-ORCA V2V avoidance + CbvfSafetyFilter protection, and broadcasts "SEC-DEGRADED-LOCAL" in RID Self-ID.

3. SEC_USSP_QUARANTINE → SEC_AUTONOMOUS_RTB (Recovery Time Objective ≤ 10 s): If authenticated USSP contact is not restored within 10.0 s (or active attack continues), the aircraft diverts autonomously to the nearest coastal FATO-EMG verified clear by onboard optical/LiDAR sensing and authenticated V2V mesh, executes a Contingent landing, inhibits subsequent autonomous dispatch until ground security reset, and triggers the ground ISMS gateway to emit a pre-populated ECCAIRS-2 IS.I.OR.230 package for the mandatory ≤72-hour authority filing.

3. Vertiport FATO/TLOF Wake Separation and 400 V / 22 kWh Thermal-Derated Charging Turnaround

Coaxial X8 Aerodynamic Baseline and Harmonized FAA / EASA Pad Geometry

The VertiRTA 240 kg MTOW (2,354.4 N) coaxial X8 mounts four counter-rotating dual-rotor pods at arm radius L = 1.55 m with 1.65 m (65-in) carbon rotors (0.20 d axial spacing), giving propulsion circle diameter RD = 4.75 m, controlling dimension D = 4.80 m, and effective 4-pod disk loading 275.3 N/m² (28.1 kg/m²). Modeling coaxial wake interference (k_coax = 1.22, reflecting the 10–15% hover power penalty documented for stacked coaxial multirotors), figure of merit FM = 0.72, electrical drivetrain efficiency η_e = 0.88, and hot Istanbul summer air density (ρ = 1.145 kg/m³ at 35 °C), electrical power draw is 49.70 kW in hover (48.05 kW ISA), 56.39 kW in 2.5 m/s vertical climb, 26.66 kW in 20 m/s cruise (25.58 kW ISA), and 40.76 kW in 2.0 m/s descent. Hover induced velocity reaches 10.96 m/s, producing peak near-pad outwash near 15.9 m/s (57.2 km/h / 35.5 mph)—just above the FAA 34.5 mph (55.5 km/h) Downwash Caution Area (DCA) threshold. [1][2][3]

Because FAA Engineering Brief EB-105A (published 27 December 2024) is slated to be superseded in 2027 by Draft AC 150/5390-4 (released October 2026 for public comment through 11 January 2027), while European/Turkish urban guidelines align with EASA PTS-VPT-DSN (March 2022), VertiRTA adopts a Harmonized Square Coastal Pad Envelope satisfying all three frameworks simultaneously: [1][2][3]

Vertiport Design Parameter

FAA EB-105A (Dec 2024)

FAA Draft AC 150/5390-4 (Oct 2026)

EASA PTS-VPT-DSN (Mar 2022)

Adopted VertiRTA Spec (RD = 4.75 m, D = 4.80 m)

TLOF (Load-Bearing)

1.0 RD = 4.75 m ("VTL" + "H" marking)

1.0 RD + 3.0 m = 7.75 m (+20% if elevated non-solid FATO)

0.83 D = 3.98 m surface (1.0 D = 4.80 m elevated)

8.0 m × 8.0 m square concrete, dynamic load-bearing

FATO

2.0 RD = 9.50 m (1.5–5% drain, 3 m shoulder, ≤2% neg. slope)

max(1.5 D, TLOF + 0.68 D) = 11.01 m (-0.5% to -1.0% grade)

1.5 D = 7.20 m (solid slope ≤ 2%)

12.0 m × 12.0 m square (2.5 D; diagonal 16.97 m > 13.58 m OFV base; 13.6 m if circular); 1.5% radial drain + NFPA 415 edge drains

Safety Area (Outer)

2.5 RD = 11.88 m (+0.25 RD per side)

FATO + 2 × max(0.28 D, 6.1 m) = 23.21 m

max(2 D, FATO + 6.0 m) = 13.20 m (≤4% slope; frangible aids ≤5 cm / 25 cm + 5%)

24.2 m × 24.2 m outer (6.1 m clear ring; frangible lighting only)

Obstacle-Free Volume / Approach

Clear VMC ingress/egress; 8:1 slope (AC 150/5390-2D heritage)

8:1 approach/departure; chargers outside 85 m (280 ft) protection zone

OFV Type 1 Funnel: h1 = 3.0 m, h2 = 30.5 m, Ø FATO_omni = 2.83 D (13.58 m), Ø TO_omni = 5.0 D (24.0 m), 12.5% (1:8) slope; obstacle arc 3.5 D day / 5 D night

EASA OFV Type 1 Funnel (12.5% omnidirectional over Marmara sea sector; chargers ≥85 m clear of final approach centerline)

Downwash / Outwash Protection

DCA wherever outwash ≥ 34.5 mph (55.5 km/h)

Default 128 × 238 m DCA (≤3,175 kg class) + 64 m taxi radius

Table C-1: ≤60 km/h pax/public, ≤50 km/h ≥80 km/h roads, ≤80 km/h apron crew, ≤100 km/h buildings

35 m radial DCA fence + blast deflectors toward Sahil promenade (sized to 240 kg 57.2 km/h peak near-pad outwash; requires site-specific outwash survey if filing directly under Draft AC 150/5390-4)

FATO-to-FATO / Parking Stands

Single aircraft per FATO; hover-taxi stands ≥ FATO size

Single aircraft per FATO; NFPA-compliant chargers outside safety/approach zones

≥60 m separation reference between simultaneous independent VFR FATOs

60 m FATO centerline spacing (Maltepe/Kartal dual FATOs & FATO-EMG); 12 m × 12 m dynamic load-bearing charger stands

Temporal Wake Separation and Dedicated Emergency FATO Reserve Policy

Because FAA EB-105A, EASA PTS-VPT-DSN, and the February 2026 FAA Orlando MCO HITL report do not establish a statutory wake-vortex timer for 240 kg multirotor eVTOLs—while IFATCA highlights the "Light Trap" where high-disk-loading distributed electric propulsion creates energetic initial outwash jets—VertiRTA enforces a wind-conditioned Single-In, Single-Out (SISO) temporal spacing rule grounded in ICAO's 60 s baseline wake reference and NASA TM-20250009519's 60 s vertipad reservation findings (where SISO cut predicted Losses of Well Clear by 41%, from 326 to 192): [1][2][3][4][5][6][7]

1. 60 s spacing (60 movements/hr = 30 arrival-departure cycles/hr per active FATO): Authorized in VMC when mandatory FATO Wind Direction Indicators (per EASA PTS-VPT-DSN) measure steady cross-pad wind ≥ 3 kt (1.5 m/s) advecting hover turbulence off the FATO.

2. 75 s spacing (48 movements/hr = 24 cycles/hr): Default nominal separation in calm air (< 3 kt), allowing vortex-ring decay before the trailing arrival enters the 30.5 m OFV funnel.

3. 90 s spacing (40 movements/hr = 20 cycles/hr): Enforced automatically when surface gusts exceed 15 kt or immediately following a high-sink-rate or off-nominal arrival.

In alignment with EASA PTS VPT-DSN.F.900 (En-Route Alternate Vertiports for Continued Safe Flight and Landing—noting that exact Chapter F numerical sizing criteria are not detailed in the published prototype excerpts), every coastal vertiport maintains a Dedicated Unscheduled Emergency FATO (FATO-EMG), separated by ≥60 m from active FATOs at the Maltepe and Kartal hubs and configured at Dragos and Cevizli as a strict zero-dwell, no-parking, Priority-0 preemptible pad. Because the widest adjacent inter-vertiport gap is Cevizli–Kartal (2.324 km), every aircraft on the corridor stays within 1.162 km of an emergency FATO. Even in a worst-case spinning CEL emergency restricted to 6.0 m/s horizontal translation plus a 40 s controlled descent (233.7 s total at 55 kW elevated draw), diversion consumes 3.57 kWh (16.1% BoL SoC / 18.9% EoL SoC)—fully protected by VertiRTA's 25% nominal dispatch reserve and 20% hard emergency floor.

400 V / 22.18 kWh 1-RC Thevenin Electro-Thermal Fast-Charging Turnaround Model

The battery pack uses 112S11P Molicel INR-21700-P50B cells (3.6 V nominal, 5.0 Ah / 18.0 Wh typical, 4.2 V charge cutoff, 12.8 mΩ typical DCIR at 50% SoC, 60 A continuous discharge with 80 °C cutoff, up to 25 A / 5 C datasheet max charge with 70 °C cutoff, 1,400-cycle rating at 100 W discharge). This yields 403.2 V nominal, 55.0 Ah (198,000 C), 22.176 kWh nominal energy, 84.76 kg cell mass (at 68.8 g typical), and 98.0 kg effective pack thermal mass (C_th = 101,920 J/K at NMC specific heat 1,040 J/(kg·K)). Note that while the Molicel datasheet rates the P50B up to 5 C charge (70 °C cutoff), independent cell benchmarks list a more conservative 3 C (15 A) peak charge and lithium-ion fast-charging literature places the continuous NMC SEI/electrolyte health boundary at 45 °C. Although two-RC equivalent circuit models capture slow diffusive recovery more closely, a temperature- and SoC-dependent 1-RC Thevenin model with exact exponential integration matches VertiRTA's onboard BMS and fleet scheduler: [1][2][3][4][5][6][7]

where [STRIPPED_MATH_OR_ICON] denotes discharge (and [STRIPPED_MATH_OR_ICON] denotes charging), [STRIPPED_MATH_OR_ICON], cell resistances (8.2 mΩ ohmic and 4.6 mΩ polarization at 25 °C / 50% SoC, scaled by 112/11 = 10.182) follow Arrhenius thermal scaling ([STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON]) and U-shaped SoC growth, and EoL aging applies 1.35× resistance growth and 0.85× capacity. [1]

To stay safely below the 45 °C NMC degradation threshold, the BMS enforces an 80 kW charger power limit, a conservative 465.9 V (4.16 V/cell) CV clamp, and a four-zone thermal/SoC C-rate derating schedule:

1. T ≤ 40 °C: up to 3.5 C (192.5 A BoL), tapering linearly above 78% SoC down to 1.0 C at 90% SoC.

2. 40 °C < T ≤ 45 °C: linear thermal derate from 3.5 C → 1.0 C.

3. 45 °C < T ≤ 50 °C: steep thermal derate from 1.0 C → 0.25 C.

4. T > 50 °C: 0 C hard charge inhibit until active cooling lowers cell temperature to ≤47 °C.

Simulating 5 consecutive flight-and-charge missions to periodic thermal steady state at 35 °C Istanbul summer ambient (32 W/K airborne cooling, 90 s pre-charge dock + 90 s post-charge boarding/checks = 3.0 min fixed ground ops, recharging to 88% SoC) reveals a decisive operational split between active pad-side liquid chilling and air-only cooling: [1]

Steady-State Regime (35 °C Ambient, 5 Cycles, Target 88% SoC)

Landing SoC / Min Voltage

Landing Temp

Peak Charge Temp

Charge Time (Derated Time)

Avg Charge C-Rate

Total Ground Turnaround (+3.0 min Ops)

BoL + Active Ground Chiller (1-Leg, 6.1 km)

74.9% / 432 V

30.6 °C

30.7 °C

3.72 min (0.0 min)

2.12 C

6.72 min

EoL + Active Ground Chiller (2-Leg, 12.2 km)

56.2% / 411 V

33.9 °C

33.9 °C

7.40 min (0.0 min)

2.58 C

10.40 min

BoL + Active Ground Chiller (3-Leg, 15.0 km)

52.7% / 411 V

34.6 °C

34.6 °C

7.60 min (0.0 min)

2.79 C

10.60 min

EoL + Active Ground Chiller (3-Leg, 15.0 km)

45.8% / 402 V

35.7 °C

35.7 °C

9.13 min (0.0 min)

2.77 C

12.13 min

EoL + Forced-Air Only (2-Leg, 12.2 km)

56.4% / 414 V

45.1 °C

44.8 °C

14.33 min (14.3 min)

1.32 C

17.33 min

EoL + Forced-Air Only (3-Leg, 15.0 km)

46.2% / 405 V

45.4 °C

45.1 °C

19.18 min (19.2 min)

1.31 C

22.18 min

Offloading thermal rejection to a ground chiller (15 °C glycol umbilical into the aircraft's cold-plate quick-disconnect) removes airborne chiller mass while pre-cooling the pack to ~26–30 °C before takeoff, absorbing the subsequent 56.4 kW climb heat pulse without ever crossing the 40 °C derating knee. [1][2][3][4]

Fleet Queueing Stability Across 5–20 Aircraft (4-Vertiport Tandem Network)

Modeling each coastal vertiport as a finite-capacity tandem queueing network—an M/G/1 FATO server (squared coefficient of variation Cv² = 0.15, reflecting tightly controlled 60–90 s spacing) feeding an M/G/c/c charging stand pool (Cv² = 0.20) under BBS-SNO blocking (no arrival enters the FATO without a confirmed open stand, matching VertiSim and CTMC vertiport queueing analyses)—establishes exact stability bounds for N = 5, 10, 15, and 20 aircraft (5.5 min mean leg flight time, 9.5 min blended EoL/BoL chilled turnaround vs. 21.0 min air-only thermal-derated failure, including a 1.25× hub demand skew at Maltepe and Kartal): [1][2][3][4]

Infrastructure Configuration (per Vertiport, 4-Node Network)

N = 5 (flt/hr / Hold)

N = 10 (flt/hr / Hold)

N = 15 (flt/hr / Hold)

N = 20 (flt/hr / Hold / Hub ρ_FATO / Hub ρ_Chg)

Stability Verdict (N = 5–20)

1 Op-FATO (75 s), 3 Chilled Chargers (9.5 m turn)

16.8 / 0.23 m

32.0 / 0.89 m

44.4 / 2.16 m

53.3 / 4.09 m / 0.69 / 0.88 (49.8% stand-full)

Stable to 10; Marginal 15–20

1 Op-FATO (90 s), 3 Chilled Chargers (9.5 m turn)

16.2 / 0.29 m

30.7 / 1.00 m

42.4 / 2.28 m

50.8 / 4.10 m / 0.79 / 0.84 (45.0% stand-full)

Stable to 10; Marginal 15–20

1 Op-FATO (60 s), 4 Chilled Chargers (9.5 m turn)

17.4 / 0.11 m

34.1 / 0.34 m

49.3 / 0.85 m

62.0 / 1.74 m / 0.65 / 0.77 (30.5% stand-full)

STABLE across 5–20 (Spoke baseline)

2 Op-FATO (75 s), 4 Chilled Chargers (9.5 m turn)

18.4 / 0.05 m

36.2 / 0.23 m

52.6 / 0.72 m

66.2 / 1.68 m / 0.43 / 0.82 (36.2% stand-full)

STABLE across 5–20 (Recommended Hub spec)

Thermal Failure: 1 FATO (75 s), 3 Air-Only (21 m turn)

10.1 / 0.49 m

18.7 / 2.85 m

24.4 / 7.65 m

27.6 / 14.26 m / 0.36 / 1.00 (65.3% stand-full)

UNSTABLE / QUEUE COLLAPSE

Thermal Failure: 1 FATO (75 s), 4 Air-Only (21 m turn)

10.3 / 0.14 m

20.0 / 0.85 m

28.1 / 2.78 m

33.7 / 6.23 m / 0.44 / 0.92 (48.8% stand-full)

Marginal/Degraded above 10

Across every actively chilled configuration at N = 20, hub charger utilization (0.77–0.88) strictly exceeds hub FATO utilization (0.43–0.69), confirming NASA TM-20250009519's conclusion that surface charging/stand resources—not airspace or pad wake timers—govern high-demand capacity. Provisioning 4 chilled 80 kW stands per vertiport (16 fleet-wide, 1.28 MW nameplate / ~0.95 MW diversified demand), 2 active FATOs + 1 FATO-EMG at Maltepe and Kartal, and 1 active FATO + 1 FATO-EMG at Dragos and Cevizli guarantees stable queueing across all 5–20 aircraft even under 75–90 s calm/gust spacing. [1]

4. Deadlock-Free Decentralized 3D CBF-ORCA Tactical Deconfliction, Sub-500 ms CEL/CSFL Preemption, and Hybrid Co-Simulation

Mathematical Formulation Feeding IntruderTrackerKf and CbvfSafetyFilter (K <= 16)

Each aircraft [STRIPPED_MATH_OR_ICON] tracks double-integrator translational dynamics at the 100 Hz RTA loop, [STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON], subject to acceleration bounds [STRIPPED_MATH_OR_ICON] (a_xy = 5.0 m/s², a_z = 3.0 m/s², tracked by the 400 Hz INDI-WLS inner loop) and bounded gust/tracking disturbance [STRIPPED_MATH_OR_ICON]. To keep aircraft out of neighboring rotor downwash columns (following 3D ellipsoidal quadrotor ORCA), define the anisotropic scaling matrix S = \operatorname{diag}(1, 1, R_{xy}/R_z) with nominal R_xy = 12.0 m (2.5 D plus tracking margin) and R_z = 36.0 m (3:1 anisotropy), expanding to R_xy_CEL = 25.0 m and R_z_CEL = 75.0 m around any Priority-0 CEL/CSFL aircraft. [1]

For each peer [STRIPPED_MATH_OR_ICON] tracked by IntruderTrackerKf with relative state [STRIPPED_MATH_OR_ICON] and 3-σ position/velocity uncertainty bounds [STRIPPED_MATH_OR_ICON] in [STRIPPED_MATH_OR_ICON]-scaled coordinates ([STRIPPED_MATH_OR_ICON], [STRIPPED_MATH_OR_ICON]), define the covariance-inflated ellipsoidal barrier:

Applying a 2nd-order Exponential Control Barrier Function (ECBF) with Hurwitz gains k_1 = 1.0 s⁻¹ and k_2 = 1.4 s⁻¹ imposes [STRIPPED_MATH_OR_ICON], where [STRIPPED_MATH_OR_ICON] and [STRIPPED_MATH_OR_ICON]. Splitting avoidance responsibility via ORCA reciprocity weights [STRIPPED_MATH_OR_ICON] ([STRIPPED_MATH_OR_ICON] for cooperative pairs; [STRIPPED_MATH_OR_ICON] for Priority-0 or quarantined non-cooperative intruders) yields one radial linear inequality row in [STRIPPED_MATH_OR_ICON]: [1][2]

where [STRIPPED_MATH_OR_ICON] (with [STRIPPED_MATH_OR_ICON]) for Priority-0 CEL/CSFL or non-cooperative intruders—unilaterally absorbing worst-case spinning lateral acceleration a_j_max = 4.0 m/s² regardless of intruder motion—and [STRIPPED_MATH_OR_ICON] (with [STRIPPED_MATH_OR_ICON]) for authenticated cooperative fleet peers whose control accelerations cancel cleanly in the joint pairwise sum.

Risk-Gated Right-Hand-Rule (RHR) Curl for Zero Deadlock and Zero Livelock

Purely radial reciprocal CBFs and symmetric ORCA exhibit two classic failure modes: collinear antipodal approaches stall on an undesired boundary equilibrium set [STRIPPED_MATH_OR_ICON] (deadlock), while noisy relative-velocity estimates across the velocity-obstacle centerline induce side-switching oscillations (reciprocal dance / livelock, observed in ~25% of noisy quadrotor ORCA flight trials). VertiRTA eliminates both mechanisms without weakening radial Nagumo safety invariance: [1][2][3][4]

Stability, Safety, and Liveness Proof Sketch + 16-Row QP Budget

Forward Invariance & Robust Recursive Feasibility: Let [STRIPPED_MATH_OR_ICON] denote the offline Hamilton-Jacobi-Isaacs Control Barrier-Value Function (CBVF) for the static airspace/geofence/actuator-saturation viability kernel solving the CBVF variational inequality [STRIPPED_MATH_OR_ICON]. Because its zero-superlevel set [STRIPPED_MATH_OR_ICON] equals the maximal finite-time viability kernel under [STRIPPED_MATH_OR_ICON] and [STRIPPED_MATH_OR_ICON], static airspace rows remain recursively feasible throughout the envelope. For any cooperative pair [STRIPPED_MATH_OR_ICON], summing their radial ECBF rows with [STRIPPED_MATH_OR_ICON] recovers the exact joint barrier condition [STRIPPED_MATH_OR_ICON] under bounded KF error and disturbances; for any Priority-0 or non-cooperative intruder [STRIPPED_MATH_OR_ICON], setting [STRIPPED_MATH_OR_ICON] and [STRIPPED_MATH_OR_ICON] enforces unilateral invariance against any [STRIPPED_MATH_OR_ICON].

Deadlock Elimination & Merge Liveness: At any undesired boundary equilibrium [STRIPPED_MATH_OR_ICON] where [STRIPPED_MATH_OR_ICON] and a nonempty set [STRIPPED_MATH_OR_ICON] of pairwise barriers is active, KKT stationarity on the QP requires [STRIPPED_MATH_OR_ICON] with [STRIPPED_MATH_OR_ICON]—meaning [STRIPPED_MATH_OR_ICON] must lie inside the positive normal cone of active radial vectors. Because [STRIPPED_MATH_OR_ICON] carries a strictly non-zero coherent tangential component orthogonal to every active radial line of sight, [STRIPPED_MATH_OR_ICON] exits the normal cone for every non-goal state ([STRIPPED_MATH_OR_ICON]), ruling out stalled symmetric equilibria and inducing a smooth counter-clockwise merge vortex. Combined with strategic SCD and 60–90 s FATO admission control—which bounds local terminal merge density well below hard actuator-saturation jamming—every merging aircraft achieves finite-time goal convergence. In closed-loop 50 Hz numerical testing of an 8-agent symmetric antipodal vertiport merge (K <= 16 dual active-set QP), baseline symmetric CBF-ORCA deadlocked at 106.5 m mean remaining distance after 30 s, whereas VertiRTA 3D CBF-ORCA with RHR curl and [STRIPPED_MATH_OR_ICON] resolved completely in 20.56 s (1.0 m mean goal error, minimum ellipsoidal separation 1.32 R, max_slack = 0.000, max_rows = 10 <= 16). [1][2]

To maintain microsecond determinism inside CbvfSafetyFilter (x = [u_x, u_y, u_z, s]^T, 4 decision variables), VertiRTA adopts DAQP-soft (an [STRIPPED_MATH_OR_ICON]-penalized dual active-set C solver benchmarked at 0.4–4.3 µs on small dense robotics QPs even during infeasible constraint loops, compared with ProxQP's 8–29 µs nominal times and multi-millisecond infeasible spikes). The 16-row inequality budget is partitioned strictly: [1][2]

QP Row Indices

Count

Constraint Class

Slack Allowed?

Active-Set Pruning Rule

Rows 1–4

4

HJI-CBVF Airspace Viability Envelope: East 29.188° E wall (LTFJ/LTBX shield), West Marmara boundary, 120 m AGL ceiling, 10 m AGL floor (relaxed inside authorized FATO cylinder)

Hard (Zero slack)

Always active

Rows 5–6

2

Dynamic Vertiport OFV / FATO Lockout: 12.5% EASA funnel cone + instant Priority-0 FATO-EMG exclusion cylinder

Hard (Zero slack)

Active within 300 m of any vertiport or on P0 FATO lockout

Rows 7–16

≤ 10

Top-10 Critical 3D CBF-ORCA Peer Barriers (out of up to 19 tracked fleet peers)

Penalized s \ge 0 (W_s = 1e5)

Priority-0 CEL/CSFL peers unconditionally occupy Rows 7+ (never pruned); remaining slots sorted by lowest ellipsoidal clearance / shortest time-to-violation

Coaxial X8 Same-Arm CEL Spin Physics and < 500 ms Fleet Preemption Analysis

Under a single-rotor failure (7 healthy rotors), a coaxial X8 retains full 4-DOF hover and yaw controllability and executes a non-spinning Continued Safe Flight and Landing (CSFL). Under an irreversible same-arm dual-motor failure (e.g., arm-4 dual-ESC/power-bus loss leaving 6 rotors across arms 1, 2, and 3), opposing arm 2 must throttle down ([STRIPPED_MATH_OR_ICON]) to preserve roll/pitch trim, loading arms 1 and 3 to ~598–626 N/rotor (vs. 750 N max). Because stacked coaxial pairs already exhibit reduced yaw leverage and the primary lift rotors operate near 80–85% saturation during arrest and descent, the 400 Hz INDI-WLS allocator sheds yaw authority ([STRIPPED_MATH_OR_ICON]) to protect roll, pitch, and collective thrust—transitioning the airframe into a controllable Mueller & D'Andrea relaxed-hover spinning equilibrium (CEL): [1][2][3][4]

For the 240 kg coaxial X8 (I_xx = I_yy = 92 kg·m², I_zz = 158 kg·m², aerodynamic yaw damping γ_yaw = 14.5 N·m/(rad/s)), unbalanced yaw torques of 25–65 N·m yield steady spin rates of 1.72–4.48 rad/s (99–257 deg/s), spin periods of 1.32–3.59 s, spin-axis tilts of 9.9–19.9°, and tight horizontal drift radii of 0.16–0.56 m. Because the spinning craft retains translational guidance (≤6.0 m/s horizontal, -2.5 to -3.5 m/s controlled sink) to the nearest FATO-EMG but cannot execute aggressive reciprocal swerves, it triggers dual-path Priority-0 Preemption:

Preemption Stage

Mechanism & Determinism Basis

p50 Budget

p99.9 Budget

Active On Path

1. Fault Detection

400 Hz INDI angular-acceleration residual + ESC RPM drop (matching 45 ms experimental rotor-loss detection)

20.0 ms

45.0 ms

Path A & B

2. Sign & Publish

Ed25519 sign (~120 µs) + VertiRtaSecureHeader + dual publish (DDS + WSS)

0.2 ms

0.6 ms

Path A & B

3. V2V Air Propagation

Fast DDS BEST_EFFORT UDP multicast, 3× burst at 5 ms intervals (explicitly avoiding Fast DDS RELIABLE 177.62 ms median spikes)

8.0 ms

35.0 ms

Path A (Tactical)

4. Peer Verify & KF Inject

Ed25519 verify (~280 µs) + ReplayWindow128 + IntruderTrackerKf Priority-0 latch

0.4 ms

1.2 ms

Path A (Tactical)

5. USSP/DSS Contingent

Zero-block Contingent DSS write (OVN key omitted) + S2 query + HTTP/2 subscriber push

45.0 ms

160.0 ms

Path B (Strategic)

6. Vertiport FATO Abort

Vertiport Manager locks FATO-EMG, issues signed FATO_ABORT go-around ("fly straight ahead on the rail" / brief hover reposition per FAA MCO procedure)

10.0 ms

35.0 ms

Path B (Strategic)

7. CbvfSafetyFilter QP

Next 100 Hz tick (≤10 ms wait) + ~5 µs DAQP-soft solve with β_ij = 1.0, R_xy_CEL = 25 m

5.0 ms

10.0 ms

Path A & B

8. INDI-WLS Actuator Onset

400 Hz INDI-WLS torque allocation + ESC/rotor spin-up (τ ≈ 35 ms) delivering physical avoidance acceleration

15.0 ms

40.0 ms

Path A & B

Total Path A (V2V Clear)

Stages 1 + 2 + 3 + 4 + 7 + 8

48.6 ms

131.8 ms

Margin: 368.2 ms

Total Path B (FATO Vacate)

Stages 1 + 2 + 5 + 6 + 7 + 8

95.2 ms

290.6 ms

Margin: 209.4 ms

In closed-loop 100 Hz simulation of a head-on 16 m/s closing encounter (nominal aircraft at 10 m/s converging on a descending spinning CEL aircraft at 6 m/s, -2.5 m/s sink, 48 m ahead with 82 ms detection+comm delay), the nominal aircraft initiates non-reciprocal lateral-right escape at 90.0 ms and passes with a minimum ellipsoidal separation of 25.88 m against the 25.0 m expanded CEL barrier—zero safety violation. (Note that this closed-loop check modeled a constant-velocity descending CEL intruder; flight-code deployment uses the full [STRIPPED_MATH_OR_ICON] margin above to bound spinning lateral drift accelerations up to 4.0 m/s².)

Hybrid Co-Simulation Architecture (1–2 PX4/Gazebo Heroes + 4–19 C++20 6-DOF Agents at < 2 ms/tick @ 100 Hz)

Running 20 full PX4 SITL + Gazebo Harmonic processes on one workstation severely degrades Real-Time Factor (RTF), and PX4 v1.17–1.18 GZBridge implements unidirectional Gazebo → PX4 simulation-clock pacing rather than a bidirectional multi-vehicle execution barrier. VertiRTA resolves this through a hybrid co-simulation architecture that executes full autopilot firmware on 1–2 Hero aircraft (e.g., the CEL fault aircraft and a merging terminal arrival) while simulating the remaining 4–19 fleet craft in a cache-aligned native C++20 physics/RTA engine sharing the exact same local InterUSS DSS, USSP, and Fast DDS bus: [1][2][3][4][5][6]

Deterministic C++20 Execution and < 2 ms/tick Timing Budget

The 1–2 Hero Aircraft run full px4_sitl inside Gazebo Harmonic (gz_x8_vertirta SDF modeling the 240 kg inertia tensor, 8 coaxial rotor plugins, IMU, GNSS, barometer, and downward landing camera), multiplexed through a single MicroXRCEAgent udp4 -p 8888 instance onto isolated ROS 2 / DDS namespaces (/px4_1/fmu/*, /px4_2/fmu/*). Their production C++20 VertiRTA RTA nodes execute out-of-process directly on the fleet V2V bus. [1][2][3][4][5]

The remaining 4–19 fleet aircraft run inside FleetSimEngine, a native C++20 executable compiled with -O3 -march=native, memory-locked via mlockall(MCL_CURRENT | MCL_FUTURE), and thread-pinned away from Gazebo physics/rendering cores. All agent states reside in 64-byte cache-aligned Structure-of-Arrays (SoA) buffers with zero heap allocation in the hot loop. On each 100 Hz Gazebo /clock tick (10 ms frame), FleetSimEngine steps four 400 Hz inner substeps (2.5 ms each: RK4 6-DOF rigid-body dynamics, coaxial aerodynamics with identical same-arm CEL failure injection, exact-exponential 1-RC electro-thermal battery update, and 8-motor INDI-WLS active-set allocation) followed by one 100 Hz outer RTA pass (IntruderTrackerKf over up to 19 tracks, 3D ellipsoidal CBF-ORCA + RHR curl, Top-10 active-set pruning to K <= 16 rows, warm-started DAQP-soft solve, and zero-copy Fast DDS Shared Memory / FlatBuffers state export). Because Fast DDS security plugins are explicitly incompatible with data-sharing delivery, the co-simulator exposes two test profiles: zero-copy SHM for high-speed Monte Carlo stress runs, and loopback UDP with full DDS:Crypto:AES-GCM-GMAC + ReplayWindow128 for Part-IS security qualification. [1][2][3][4][5]

FleetSimEngine Sub-Task (per Native C++20 Agent / 100 Hz Tick)

Nominal Budget

Worst-Case Budget

Implementation & Benchmark Basis

4 × 400 Hz RK4 6-DOF + Coaxial Aero + 1-RC Battery

10.0 µs

10.0 µs

SIMD AVX2/NEON SoA vectorization, exact exponential V_1 step

4 × 400 Hz INDI-WLS 8-Motor Allocation

24.0 µs

32.0 µs

Fixed 4×8 active-set WLS, stack-allocated Eigen matrices

100 Hz IntruderTrackerKf (19 Tracks)

8.0 µs

14.0 µs

Constant-velocity 6-state block-diagonal KF + Mahalanobis gate

3D Ellipsoidal CBF-ORCA + Top-10 Pruning

12.0 µs

20.0 µs

Scaled vector ops + std::nth_element pruning to K <= 16

16-Row CbvfSafetyFilter QP (DAQP-soft Warm-Start)

5.0 µs

18.0 µs

Anchored to DAQP-soft 0.4–4.3 µs small dense QP benchmarks

Zero-Copy SHM / FlatBuffers State Export

8.0 µs

14.0 µs

Pre-allocated Fast DDS loaned samples / FlatBuffers

Total per Agent (100 Hz Tick)

67.0 µs

108.0 µs

—

19 Native Agents — Single-Threaded

1.27 ms

2.05 ms

Near 2.0 ms cap at worst case

19 Native Agents — 4-Thread Pinned Pool (+35 µs Barrier)

0.37 ms

0.57 ms

71.5% headroom below 2.00 ms/tick target

Implementation Rollout and Verification Checklist

Moving VertiRTA into build and regulatory qualification along the Anatolian coastal corridor proceeds through four mandatory verification gates:

1. https://dhmi.gov.tr/AIPDocuments/LT_AD_2_LTFJ_en.pdf (ad 2 ltfj - DHMİ)

2. https://www.dhmi.gov.tr/AIPDocuments/LT_AD_2_LTBX_en.pdf (LTBX - İSTANBUL / SAMANDIRA - DHMİ)

3. https://www.scribd.com/document/1053048960/400000001019283 (LTFJ Ad 2.1 Aerodrome Location Indicator and Name - Scribd)

4. https://www.batterydesign.net/thermal/ (Battery Thermal Management: Heat Generation, Cooling & TIM)

5. https://monocypher.org/speed (Speed Benchmarks - Monocypher)

6. https://arxiv.org/html/2609.19080v1 (An Always-Feasible QP Solver for Constrained Robot Control - arXiv)

7. https://www.researchgate.net/publication/402556006_Hybrid_ML-KEM_in_TLS_13_Performance_Analysis_on_ARM64_Under_Network_Stress (Hybrid ML-KEM in TLS 1.3: Performance Analysis on ARM64 Under)

8. https://aip.aero/tr/en/vfr/?LTFJ (Istanbul Sabiha Gökçen International Airport LTFJ - AIP:Aero)

9. https://web.shgm.gov.tr/documents/sivilhavacilik/files/mevzuat/SHT-IHA_Rev-05.pdf (İnsansız Hava Aracı Sistemleri Talimatı (SHT-İHA))

10. https://www.fotografline.com/blog/drone-ehliyeti-iha-kaydi-rehberi-2026 (Drone Ehliyeti ve İHA Kaydı 2026: Yeni SHGM Kuralları - Fotografline)
