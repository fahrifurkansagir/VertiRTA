# VertiRTA — SAE ARP4761A / ED-135 Mini-FHA, Requirement Traceability & Illustrative FTA Cut-Sets

> **Scope Note:** All quantitative failure rates (`lambda`) in Section 3 are **illustrative literature ranges** (NASA/CR-2019-220217 & tactical avionics benchmarks) demonstrating architectural cut-set scaling — **NOT** vendor-qualified reliability data. Per `ARP4761A` (20 Dec 2023) and `DO-178C`, software defects are systematic and NEVER enter FTA as exponential random rates; software assurance is addressed via FDAL/IDAL design discipline and deterministic verification (`Sn1..Sn10`).

---

## 1. Functional Hazard Assessment (`SAE ARP4761A` Mini-FHA, `SC-VTOL-02` Cat. Enhanced Reference)

| FHA ID | System Function | Failure Condition | Critical Phase | Operational Effect | Severity (Cat. Enh.) | Target Prob. / FDAL | RTA Mitigation & Verification Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **FHA-01** | Lift & Propulsion | Single motor/ESC total loss (`1 of 8`) | Takeoff (post-TDP), Cruise, Approach | Full 4-DOF retained (WLS offloads surviving partner to `60.0%` / `441.5 N` peak, `1.84x` power; upper-loss unblanks lower `+17.6%`); divert to vertiport (**CSFL**). | **Major** (mitigated) | `<= 10^-5/FH` (FDAL C; RCF **FDAL A**) | 400 Hz dual-channel FDI (`<85 ms`) + `LATCHED_CSFL` INDI+WLS (`Sn4`, `Sn5`, Stratum `S2`). |
| **FHA-02** | Lift & Propulsion | Cross-arm dual motor/ESC loss (`24` pairs) | Cruise, Approach | 4-DOF retained with reduced thrust/yaw envelope (`v_cap,f` enforced); immediate alternate vertiport landing (**CSFL**). | **Hazardous** | `<= 10^-7/FH` (FDAL B; RCF **FDAL A**) | 400 Hz FDI + `LATCHED_CSFL` degraded CBVF LUT + `Part-IAM` divert (`Sn4`, `Sn5`, Stratum `S3`). |
| **FHA-03** | Lift & Propulsion | Same-arm coaxial-pair loss (`4` booms) | All Phases | Loss of 4-DOF static moment balance; yaw abandoned into Option B1b active flat spin (`2.2 rad/s`, `Delta_psi = +/- 10.6 deg`); **CEL ONLY** (exempt from Cat. Enhanced CSFL). | **Catastrophic** (CSFL failure) | `<= 10^-9/FH` (FDAL A) + No Single Failure | `LATCHED_CEL` (`<= 2.5 ms`) + `V_2D` braking LUT (`Sn5`, Stratum `S4`); hardware `<= 1.58e-5/FH` & boom CCF isolation marked `G5 [UNDEVELOPED]`. |
| **FHA-04** | Inertial Sensing | Single IMU fault (freeze, drift, step, noise) | All Phases | 400 Hz 3-IMU MVS + `chi2_24` TMR isolates faulty channel in `<= 20 ms`; negligible flight impact. | **Minor** (with TMR) | `<= 10^-3/FH` (FDAL D; voter **FDAL A**) | 400 Hz 3-IMU TMR + continuous latent-fault check per `MOC 5 VTOL.2300` (`Sn5`, Stratum `S5`). |
| **FHA-05** | Inertial Sensing | Simultaneous or latent + active `>= 2/3` IMU loss | All Phases | Loss of reliable attitude/rate estimation -> Loss of Control In-Flight (LOC-I). | **Catastrophic** | `<= 10^-9/FH` (FDAL A) | Dissimilar MEMS lots, independent SPI/power (`beta_imu <= 10^-4`), continuous cross-check (`G5 [UNDEVELOPED]`). |
| **FHA-06** | Flight Control & RTA Compute | Total FCC loss or stuck/erroneous RTA switch | All Phases | Loss of control authority or unbounded Complex Function command -> LOC-I / CTR breach. | **Catastrophic** | `<= 10^-9/FH` (FDAL A) + No Single Failure | Software: 0-heap, static stack DAG, Cortex-M7 `<256K/<128K`, bounded DAQP (`Sn7..Sn10`). Hardware: `1oo3` COM-MON or `lambda_fcc <= 1.83e-5/FH` (`G5 [UNDEVELOPED]`). |
| **FHA-07** | High-Voltage ESS | Total 400 V / 22 kWh loss or thermal runaway | All Phases | Complete loss of thrust across all 8 rotors or in-flight fire. | **Catastrophic** | `<= 10^-9/FH` (FDAL A) + No Single Failure | Dual thermally/electrically isolated HV packs + independent avionics bus (`VTOL.2525`, `G5 [UNDEVELOPED]`). |
| **FHA-08** | Energy & Diversion | `Part-IAM` final reserve exhaustion / no reachable SLZ | Cruise, Approach | Inability to execute CMP go-around or reach alternate FATO -> forced off-site landing. | **Hazardous** / **Catastrophic** | `<= 10^-7` / `<= 10^-9/FH` (FDAL B/A) | 10 Hz `UAM.OP.VCA.191/.192/.195` energy monitor + 1-RC Thevenin sag cone + `MINIMUM FUEL` at 20% (`Sn3`, `Sn5`). |
| **FHA-09** | Airspace & Obstacle Containment | Breach of `LTFJ` / `LTBX` CTR walls or obstacle strike | Cruise, Divert | Unauthorized controlled-airspace entry (`SHGM SHT-İHA` Red zone) or urban collision. | **Hazardous** / **Catastrophic** | `<= 10^-7` / `<= 10^-9/FH` (FDAL B/A) | 100 Hz HJI-CBVF-QP + `150 m` CTR buffer + Dual-Res `ROG-Map` ESDF (`age > 250 ms` margin inflation) (`Sn5`, `Sn6`). |
| **FHA-10** | Advisory AI (`DR-6` / `DR-7`) | Hallucinated TR/EN command, bad SLZ over water, or crash | Contingency, Approach | Degraded advisory UI only; zero impact on deterministic RTA flight control. | **Minor / No Safety Effect** | `<= 10^-3/FH` (DAL E) | Hard symbol/include isolation (`Sn1`), read-only telemetry, pilot approval gate, zero actuation authority. |

---

## 2. Requirement Traceability Matrix (`REQ-SAF-01` .. `REQ-SAF-12`)

| Req ID | Parent FHA | Requirement Statement (Locked Spec Contract) | Owning Module | Verification Gate / GSN Leaf |
| --- | --- | --- | --- | --- |
| **REQ-SAF-01** | FHA-01, FHA-02 | Isolate `1M` and `2M-Cross` motor/ESC losses in `< 85 ms` and maintain 4-DOF CSFL via `f64` QR/Cholesky INDI+WLS. | `libvertirta_flight_core` (`mc_indi_wls_control`) | `Sn4`, `Sn5` (`S2`, `S3`) |
| **REQ-SAF-02** | FHA-03 | Latch `LATCHED_CEL` irreversibly within `<= 2.5 ms` on `SAME_ARM_DUAL_LOSS`; land with `|v_z| <= 2.5 / 1.5 / 0.8 m/s`. | `libvertirta_flight_core` (`rta_arbiter`, `V_2D`) | `Sn2`, `Sn5` (`S4`) |
| **REQ-SAF-03** | FHA-04, FHA-05 | Isolate any single IMU fault (freeze/drift/step/noise) in `<= 20 ms` via 400 Hz 3-IMU MVS + `chi2_24` voter. | `libvertirta_flight_core` (`tmr_voter`) | `Sn5` (`S5`) |
| **REQ-SAF-04** | FHA-06 | Enforce zero dynamic heap allocations (`operator new` link TU + `nm -u` incl. aligned `new`), static stack `<= 4096 B/frame`, `<= 16 KB` DAG. | `libvertirta_flight_core` | `Sn7` (`tier1_embedded_structural`) |
| **REQ-SAF-05** | FHA-06 | Compile `libvertirta_flight_core` cleanly for Cortex-M7 within `< 256 KB` Flash and `< 128 KB` RAM (`-Werror -Wvla`). | `libvertirta_flight_core` | `Sn7` (`check_embedded_footprint.py`) |
| **REQ-SAF-06** | FHA-06 | Achieve `0-ULP` SHA-256 bit-identical `.vrta` replay on Same-ISA and `< 1e-12` continuous / `0` mode divergence across `x86_64 <-> ARM64`. | `libvertirta_flight_core` (`vrec`) | `Sn8` (`tier2b_cross_isa_verify`) |
| **REQ-SAF-07** | FHA-06 | Enforce `ASTM F3269-21` hysteresis/dwell (`T_dwell >= 1.0 s`, `Delta_hyst >= 0.05`, `dh/dt > 0`) on Priority 3 CBVF recovery. | `libvertirta_rta_companion` (`rta_arbiter`) | `Sn2` (`rta_fsm_unit_tests`) |
| **REQ-SAF-08** | FHA-08 | Maintain `>= 2` reachable alternates past PoC (`R_SLZ >= 5.0 m`, `theta <= 8 deg`, `sigma_z <= 0.15 m`) with `15%` final reserve / `20%` alert. | `libvertirta_rta_companion` (`slz_reach`) | `Sn3`, `Sn5` |
| **REQ-SAF-09** | FHA-09 | Enforce `LTFJ` (`798.58 m HAE`, NW join `J2`) and `LTBX` (`646.35 m HAE`) polygon+arc walls with `0` breaches inside BRT (`ODD < 29.1900 E`). | `libvertirta_rta_companion` (`cbvf_qp`) | `Sn3`, `Sn4`, `Sn5` |
| **REQ-SAF-10** | FHA-09 | Maintain Dual-Res `ROG-Map`/FIESTA ESDF `< 2 ms` CPU latency and inflate obstacle margins when map `age > 250 ms`. | `libvertirta_rta_companion` (`rog_esdf`) | `Sn6` (`tier3_50_lidar`) |
| **REQ-SAF-11** | FHA-10 | Enforce zero `#include` or linker symbol coupling from `core/flight_core/` and `core/rta_companion/` into `DR-6` (`llama`/`z3`) or `DR-7` (`onnxruntime`). | `core/` build graph | `Sn1` (`check_arch_isolation.py`) |
| **REQ-SAF-12** | FHA-01..09 | Never mix CSFL (`S2+S3+S5`, `n=650`) and CEL (`S4`, `n=150`) denominators; log every failed run with `.vrta` trace in `failure_cliff_ledger.json`. | `verification/monte_carlo/` | `Sn4`, `Sn5`, `Sn6` |

---

## 3. Illustrative Hardware FTA Cut-Set Analysis (Corrected 2oo3 Triplex Math)

Illustrative baseline failure rates (literature values, exposure `T = 1.0 h` unless noted): single propulsion channel `lambda_ch = lambda_motor (9.24e-5) + lambda_esc (2.70e-4) = 3.624e-4 / FH`; tactical MEMS IMU `lambda_imu = 1.00e-5 / FH` (100k h MTBF); simplex FCC lane `lambda_fcc = 1.00e-4 / FH`; single HV battery pack `lambda_batt = 1.00e-6 / FH`.

| Minimal Cut-Set / Hardware Path | Exact Cut-Set Formula (`T = 1.0 h`) | Rate (`1/FH`) | Cat. Enhanced (`<= 10^-9/FH`) Engineering Verdict |
| --- | --- | --- | --- |
| Single Rotor Channel (`1 of 8`) | `8 * lambda_ch` | `2.90e-3` | Mitigated to Major by 4-DOF `LATCHED_CSFL`. |
| Simplex FCC Hardware | `lambda_fcc` | `1.00e-4` | **FAILS** No-Single-Failure (`100,000x` over target). |
| Same-Arm Dual Coaxial + Boom CCF (`beta_arm = 0.05`) | `4 * beta_arm * lambda_ch` | `7.25e-5` | **FAILS** (`72,480x` over); physical boom segregation mandatory (`G5`). |
| Cross-Arm Dual Rotor (`24` pairs, independent) | `24 * lambda_ch^2 * T` | `3.15e-6` | Mitigated by 4-DOF `LATCHED_CSFL` degraded envelope. |
| Single HV Battery Pack | `lambda_batt` | `1.00e-6` | **FAILS** No-Single-Failure (`1,000x` over). |
| Same-Arm Dual Coaxial (Independent, `T = 1.0 h`) | `4 * lambda_ch^2 * T` | `5.25e-7` | **FAILS** (`525x` over); requires qualified `lambda_ch <= 1.58e-5/FH` (`G5`); `RCF-CEL` is last-resort mitigation only. |
| **Triplex 2oo3 Voting FCC (Corrected — Any `>= 2` of 3 Lanes Fail!)** | **`3 * lambda_fcc^2 * T`** | **`3.00e-8`** | **FAILS (`30x` over `10^-9/FH`!)** — Plain 2oo3 majority voting fails when ANY 2 of 3 lanes fail (`C(3,2) = 3`). Meeting `<= 10^-9/FH` requires EITHER **1oo3 self-checking COM-MON lanes** (with proven monitor coverage `c -> 1`) OR lane reliability **`lambda_fcc <= 1.83e-5/FH`** (`3 * (1.83e-5)^2 = 1.00e-9/FH`), plus CCF `beta_fcc <= 10^-5` (`G5 [UNDEVELOPED]`). |
| Dual Independent FCC (Fail-Passive, `T = 1.0 h`) | `lambda_fcc^2 * T` | `1.00e-8` | **FAILS** (`10x` over; no voting without COM-MON self-check). |
| 3-IMU TMR (`>= 2 of 3` Independent, `T = 1.0 h`) | `3 * lambda_imu^2 * T` | `3.00e-10` | **PASSES** `<= 10^-9/FH` *only if* dissimilar lots/buses keep `beta_imu <= 10^-4`. |
| Dual Isolated HV Battery Packs (`T = 1.0 h`) | `lambda_batt^2 * T` | `1.00e-12` | **PASSES** `<= 10^-9/FH` *only with* thermal/electrical propagation isolation (`G5`). |
