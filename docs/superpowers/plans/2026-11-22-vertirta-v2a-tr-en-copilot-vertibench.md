# VertiRTA v2.0.0-alpha TR/EN Contingency Co-Pilot & VertiBench-TR/EN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the EASA AI Level 1/2A closed-loop bilingual (TR/EN) fault-contingency neuro-symbolic co-pilot (`Qwen3-8B`/`4B` SFT→Z3-GRPO, `llama.cpp` GBNF, `Z3` `QF_LRA` `LTFJ`/`LTBX` verifier, zero-heap C++20 STL monitor, `KnowNo` Split Conformal gate), the Apache-2.0 `VertiBench-TR/EN` Hugging Face benchmark, and the isolated ROS 2 `vertirta_mission_node` with `0 µs` jitter impact on the 100 Hz RTA kernel.

**Architecture:** Untrusted LLM proposals (`Qwen3-4B Q4_K_M` local CPU AVX2 / `Qwen3-8B` Colab G4 GPU) decode strictly into a `<= 30`-token symbolic DSL under eager `libllama` GBNF (`enable_thinking=False`), expand deterministically into a `<= 32`-node Bounded-STL AST, pass `KnowNo` Split Conformal gating (`alpha = 0.05`) and a `< 15 ms` `Z3` `QF_LRA` static/corridor verifier against real AIP `LTFJ`/`LTBX` polygons with 1-pass UNSAT-core repair, and cross into `libvertirta` only via an async ROS 2 Action while a zero-heap C++20 quantitative STL monitor (`stl_monitor`) guards trajectories online.

**Tech Stack:** C++20 (`vertirta::rta_companion`, `vertirta::stl_monitor`, `vertirta::copilot`, `-Wall -Wextra -Werror`), `llama.cpp` C API (`libllama` b8920+), `Z3` C++ API (`QF_LRA`), Catch2, ROS 2 Jazzy (`rclcpp_action`, `vertirta_msgs`, `px4-ros2-interface-lib` `ModeExecutorBase`), Python 3.11 (`Unsloth`, `TRL GRPOTrainer`, `vLLM`, `z3-solver`, `pytest`), GitHub Actions `ubuntu-24.04-arm`.

**Spec:** `docs/superpowers/specs/2026-11-22-vertirta-tier-s-design.md` (v1.2 FROZEN) and `docs/research/00-DR-FORENSIC-ERRATA-AND-LOCKS.md` (`DR-6` Errata 1–6 mandatory overrides).

## Global Constraints

- **License Hygiene (Apache 2.0):** Fine-tune/redistribute ONLY `Qwen/Qwen3-8B` and `Qwen/Qwen3-4B` (Apache 2.0); evaluate `Qwen/Qwen2.5-7B-Instruct` and `google/gemma-3-4b-it` strictly zero-shot + GBNF + Z3; hard-ban `CohereForAI/aya-expanse-8b` (`CC-BY-NC-4.0`).
- **Safety Isolation (EASA AI Level 1/2A):** Co-pilot runs asynchronously at `0.5–1 Hz` outside the 100 Hz ASTM F3269 RTA thread (`0 µs` added p99 jitter); `shield_active` is immutable `true` in flight; no online 3D/4D `STL-MILP` or multi-step Z3 unrolling.
- **HV Electrical & Vehicle Baseline (`config/vehicle_x8_240.yaml`):** Locked `400 V / 22 kWh` HV Li-ion pack (`v_term_volts` ~`350–380 V` loaded; `368.5 V` nominal fault reference — NEVER `44.2 V` 12S); `R_SLZ >= 5.0 m`, `soc_reserve_floor_pct = 12.0f`, `p_safe >= 0.85f`, `slz_ood_variance <= 0.12f`, `AGL in [30.0, 150.0] m`, `v_max = 22.0 m/s`.
- **Canonical Geodesy & Airspace (`config/istanbul_ltfj_airspace.yaml`):** Kartal ENU origin `40.8860° N, 29.1880° E, 12.0 m AMSL` (`N_EGM2008 = +36.7 m`); valid SLZs strictly west of `29.200° E` (`< 29.190° E` near `J2`); `LTFJ` CTR (5 NM + NW `J1`/`J2` polygon+arc, `750.0 m` ENU ceil) & `LTBX` CTR (`597.6 m` ENU ceil) stamped *simulation reference only*.
- **Budgets & Determinism:** Compact DSL `<= 30` Qwen3 tokens (`max_tokens = 48`), Ryzen 5 5600X CPU AVX2 `< 3.0 s`, Colab G4 GPU `< 300 ms`, `Z3` `rlimit = 250000` / `timeout = 12 ms` (`< 15 ms` p99, fail-closed `unknown`), zero-heap `stl_monitor` `< 1 ms` with `-fno-fast-math -ffp-contract=off`.

## Review Focus

1. **`FixedMonotonicWedge` Ring-Buffer Wrap Corruption (`Cap = 601, span = 600, tau = 601`):** Pushing `tail_` before evicting expired `head_` overwrites `head_` on wraparound and emits false-safe `rho > 0`; pinned in Task 2.
2. **Multi-Dynamic-NFZ Tracker Collision & 3D Vertical Bypass in `Z3`:** Reusing `"c_dyn_nfz"` across 2+ NFZs throws in `Z3`, while checking only ownship `up_m > ceil` misses descents from `800 m` into an `LTFJ` pad at `5 m` or elevated NFZ `floor_amsl_m` violations; pinned in Task 3.
3. **Qwen3 Single-Digit BPE Truncation & Adversarial `/think` Injection:** Verbose STL decoding exceeds 64 tokens (`~7 s` CPU), while user `/think` or `<think>` probes try to deadlock GBNF; pinned in Tasks 1 & 4.
4. **Conformal Low-SoC Loiter Deadlock:** Ambiguous/garbled ASR yielding `|C_safe| == 0` or `> 1` must bypass loiter when loiter SoC margin `< 5.0%` and commit Rank-1 Z3-SAT SLZ; pinned in Task 5.
5. **GRPO Reward Silent Zeroing, Null `ast.target`, & CUDA `fork` Deadlock:** Uppercase `"LTFJ_CTR"` vs lowercase `"c_ltfj_ctr"` zeroes 100% of Bucket-3 rewards, missing `target` on `HOLD_LOITER`/`REQUEST_CLARIFICATION` crashes STL scoring, and per-step `fork` deadlocks vLLM CUDA; pinned in Task 7.

## File Structure

- `config/vertirta_dsl_v1.gbnf` — Compact `<= 30`-token ASCII GBNF grammar (`ACT=` header + `LNG=/MOD=`).
- `include/vertirta/copilot/rta_state_contract.hpp`, `core/copilot/{CMakeLists.txt,dsl_parser.cpp,stl_synthesizer.cpp}` — `400 V / 22 kWh` `RtaStateSnapshotV1`, `CompactDslAst`, `BoundedStlFormula` (`<= 32` nodes), strict parser, deterministic STL synthesizer (`vertirta_copilot` / `ALIAS vertirta::copilot`, auto-discovered by Plan 1A root `CMakeLists.txt`).
- `include/vertirta/stl/stl_monitor.hpp`, `core/stl/{CMakeLists.txt,stl_monitor.cpp}` — Zero-heap `vertirta_stl_monitor` (`ALIAS vertirta::stl_monitor`, linking `vertirta::rta_companion`) quantitative STL monitor (`FixedMonotonicWedge`, `OnlineBoundedUntil`, live RoSI).
- `include/vertirta/copilot/z3_corridor_verifier.hpp`, `core/copilot/{z3_corridor_verifier.cpp,unsat_repair.cpp}` — Native `Z3` `QF_LRA` verifier (all 4 bug fixes) + 8-priority UNSAT-core TR/EN repair.
- `include/vertirta/copilot/llama_gbnf_engine.hpp`, `core/copilot/{llama_gbnf_engine.cpp,bilingual_renderer.cpp}` — RAII `VertiLlamaEngine` (`n_batch` chunking, empty-think prefill, eager GBNF, KV prefix cache), TR/EN renderer, Numeric Grounding Gate.
- `include/vertirta/copilot/conformal_gate.hpp`, `core/copilot/conformal_gate.cpp`, `ml/copilot/calibrate_conformal.py` — `KnowNo` Split Conformal gate (`n_cal = 200, alpha = 0.05, q_lvl = 0.955`) + `< 5%` SoC liveness override.
- `benchmarks/vertibench_tr_en/{build_train_split.py,validate_vertibench.py,eval_vertibench.py,README.md}`, `.github/workflows/vertibench_ci.yml` — Apache-2.0 HF dataset builder, `< 0.15` 5-gram Jaccard CI gate on `ubuntu-24.04-arm`, and 4-model eval harness.
- `ml/copilot/{grpo_z3_rewards.py,train_sft_grpo.py,export_gguf.sh}` — Colab G4 96 GB BF16 LoRA SFT → Z3-GRPO (`spawn` Z3 pool, calibrated `dr_grpo`/GSPO clip validation, `beta = 0.04`) and `Q4_K_M` GGUF export.
- `ros2_ws/src/vertirta_mission_node/{CMakeLists.txt,package.xml,src/mission_node.cpp,test/test_mission_isolation.cpp}` — Isolated ROS 2 Jazzy node consuming Plan 1A's `/vertirta/rta_state_snapshot` (`vertirta_msgs/msg/RtaStateSnapshot`) & `/fmu/out/vertirta_capability_envelope` (`px4_msgs/msg/VertirtaCapabilityEnvelope`, `72 B`) → Co-Pilot → async `/vertirta/execute_contingency` (`vertirta_msgs/action/ExecuteContingency`) to `ModeExecutorBase` with `0 µs` RTA jitter test.

---

### Task 1: `400 V / 22 kWh` RTA State Contract, Compact `<= 30`-Token GBNF Grammar & Deterministic STL Synthesizer

**Files:**
- Create: `include/vertirta/copilot/rta_state_contract.hpp`, `config/vertirta_dsl_v1.gbnf`, `core/copilot/{CMakeLists.txt,dsl_parser.cpp,stl_synthesizer.cpp}`
- Test: `tests/copilot/test_dsl_and_synthesizer.cpp`

**Interfaces:**
- Consumes: Canonical `vertirta::rta::SlzPrior` & `vertirta::rta::SlzProviderSnapshot` from Plan 1A's `core/rta_companion/include/vertirta/rta/slz_prior_provider.hpp` (`id`, `lat_deg`, `lon_deg`, `elev_amsl_m`, `center_enu`, `p_safe`, `slz_ood_variance`, `eta_s`, `soc_margin_pct`, `rank`).
- Produces: CMake target `vertirta_copilot` (`ALIAS vertirta::copilot`, auto-discovered by Plan 1A root `CMakeLists.txt`), `RtaStateSnapshotV1`, `CompactDslAst`, `BoundedStlFormula`, `bool parse_compact_dsl(std::string_view text, CompactDslAst& out) noexcept`, `BoundedStlFormula synthesize_canonical_stl(const CompactDslAst& ast, const RtaStateSnapshotV1& st) noexcept`.

- [ ] **Step 1: Write failing Catch2 tests for HV `400 V / 22 kWh` contract, compact DSL parsing, and STL synthesis**
```cpp
// tests/copilot/test_dsl_and_synthesizer.cpp
TEST_CASE("RtaStateSnapshotV1 locks 400V/22kWh HV pack and kMaxDynamicNfz=4") {
    vertirta::copilot::RtaStateSnapshotV1 st{};
    REQUIRE(st.pack_nominal_volts == Catch::Approx(400.0f));
    REQUIRE(st.pack_energy_kwh == Catch::Approx(22.0f));
    REQUIRE(st.soc_reserve_floor_pct == Catch::Approx(12.0f));
    REQUIRE(st.dyn_nfzs.size() == 4U); REQUIRE(st.shield_active);
}
TEST_CASE("Compact DSL parses 6 actions, rejects <think>, and synthesizes <=32-node STL") {
    constexpr std::string_view kDsl = "ACT=DIVERT_SLZ;TGT=Kartal_Sahil;ETA=100;AGL=35,120;SPD=18\nLNG=TR;MOD=BRIEF";
    vertirta::copilot::CompactDslAst ast{};
    REQUIRE(vertirta::copilot::parse_compact_dsl(kDsl, ast));
    REQUIRE(ast.action == vertirta::copilot::ActionKind::DIVERT_SLZ);
    REQUIRE(ast.target_id == "Kartal_Sahil"); REQUIRE(ast.eta_lim_s == 100);
    REQUIRE_FALSE(vertirta::copilot::parse_compact_dsl("<think>x</think>\nACT=HOLD_LOITER;DUR=30;AGL=60\nLNG=TR;MOD=BRIEF", ast));
    auto stl = vertirta::copilot::synthesize_canonical_stl(ast, make_ref_400v_sml_snapshot());
    REQUIRE(stl.node_count > 0U); REQUIRE(stl.node_count <= 32U);
    REQUIRE(stl.includes_ltfj_exclusion); REQUIRE(stl.includes_ltbx_exclusion);
    REQUIRE(stl.until_target_radius_m == Catch::Approx(15.0f));
}
```
- [ ] **Step 2: Run test to verify it fails** — `ctest --test-dir build -R test_dsl_and_synthesizer --output-on-failure` (Expected: FAIL, missing sources).
- [ ] **Step 3: Implement `rta_state_contract.hpp`, `config/vertirta_dsl_v1.gbnf`, `dsl_parser.cpp`, and `stl_synthesizer.cpp`**
  Lock `RtaStateSnapshotV1` with `pack_nominal_volts{400.0f}`, `pack_energy_kwh{22.0f}`, loaded `v_term_volts` (`350–380 V`, ref `368.5f`), `kMaxDynamicNfz = 4U`, `kMaxSlzCandidates = 8U`, `shield_active{true}`. Author `config/vertirta_dsl_v1.gbnf` emitting strictly the compact 2-line ASCII form (`<= 30` tokens under Qwen3 single-digit BPE — no free-form STL block): `ACT=DIVERT_SLZ|INSPECT_SLZ|ABORT_RETURN;TGT=<slz>;ETA=<1..600>;AGL=<10..150>,<30..150>;SPD=<5..30>` | `ACT=HOLD_LOITER;DUR=<10..300>;AGL=<30..150>` | `ACT=REQUEST_CLARIFICATION;A=<slz>;B=<slz>;RSN=<AMBIG|NO_SLOT|LOW_ASR>` | `ACT=REJECT_UNSAFE;VIO=<LTFJ_CTR|LTBX_CTR|DYN_NFZ|SOC_RESERVE|SLZ_UNSAFE|MIN_AGL|SHIELD_BYPASS>;FB=<slz|HOLD_LOITER>`, Line 2 `LNG=<TR|EN|MIX>;MOD=<BRIEF|REJECT|CLARIFY>`. Implement zero-allocation `parse_compact_dsl` (rejecting `<think>`/trailing bytes) and `synthesize_canonical_stl` compiling `<= 32`-node `BoundedStlFormula` (`G_[0,t_cruise](agl_ge & outside(LTFJ_CTR) & outside(LTBX_CTR) & speed_le) & G_[0,eta](soc_ge(12)) & U_[0,eta](agl_ge, reach(tgt, 15))` for diversions; `G_[0,dur](agl_ge & outside(LTFJ_CTR) & outside(LTBX_CTR) & soc_ge(12))` for `HOLD_LOITER`).
- [ ] **Step 4: Verify tests pass & Qwen3 token cap** — `ctest --test-dir build -R test_dsl_and_synthesizer --output-on-failure` (Expected: PASS; worst-case 3-digit compact DSL `<= 30` tokens under `Qwen/Qwen3-4B` tokenizer).
- [ ] **Step 5: Commit** — `git add include/vertirta/copilot/rta_state_contract.hpp config/vertirta_dsl_v1.gbnf core/copilot/CMakeLists.txt core/copilot/dsl_parser.cpp core/copilot/stl_synthesizer.cpp tests/copilot/test_dsl_and_synthesizer.cpp && git commit -m "feat(copilot): add 400V/22kWh RTA contract, <=30-token GBNF DSL, and STL synthesizer"`

---

### Task 2: Zero-Heap Pure-C++20 Quantitative STL Monitor (`stl_monitor` + Ring-Buffer Wrap Fix)

**Files:**
- Create: `include/vertirta/stl/stl_monitor.hpp`, `core/stl/{CMakeLists.txt,stl_monitor.cpp}` (auto-discovered by Plan 1A root `CMakeLists.txt`; zero root-file overlap)
- Test: `tests/stl/test_stl_monitor_zero_heap.cpp`

**Interfaces:**
- Consumes: `vertirta::rta_companion`, `BoundedStlFormula`, `RtaStateSnapshotV1`, `std::span<const HalfSpace2D>`, Plan 1A's `ScopedZeroHeapGuard` (`tests/flight_core/zero_heap_guard.hpp`).
- Produces: CMake target `vertirta_stl_monitor` (`ALIAS vertirta::stl_monitor`, `-fno-fast-math -ffp-contract=off`), `vertirta::stl::FixedMonotonicWedge<Cap, IsMin>`, `vertirta::stl::OnlineBoundedUntil<Cap>`, `vertirta::stl::RosiBound`, `vertirta::stl::StlMonitorEngine` (`void configure(const BoundedStlFormula&) noexcept; StlTickOutput step(uint32_t tau, const StlSignalSample& s) noexcept;`).

- [ ] **Step 1: Write failing tests pinning `Cap = 601, span = 600, tau = 601` wraparound, 200-trace oracle, live RoSI, and 0-heap `< 1 ms`**
```cpp
// tests/stl/test_stl_monitor_zero_heap.cpp
TEST_CASE("FixedMonotonicWedge evicts head BEFORE tail push at Cap=601, span=600, tau=601") {
    vertirta::stl::FixedMonotonicWedge<601U, true> w;
    for (uint32_t tau = 0; tau <= 600U; ++tau)
        REQUIRE(w.push_and_query(tau, static_cast<float>(tau) - 5.0f, 600U) == Catch::Approx(-5.0f));
    REQUIRE(w.push_and_query(601U, 596.0f, 600U) == Catch::Approx(-4.0f));
    REQUIRE(w.push_and_query(602U, -9.5f, 600U) == Catch::Approx(-9.5f));
}
TEST_CASE("OnlineBoundedUntil matches textbook STL over 200 traces and triggers instant RoSI violation") {
    REQUIRE(max_abs_err_vs_textbook_until_200_traces(/*seed=*/42U) == Catch::Approx(0.0f));
    vertirta::stl::OnlineBoundedUntil<601U> u; u.configure(0U, 100U);
    float rho = 0.0f; vertirta::stl::RosiBound rosi{};
    for (uint32_t t = 0; t <= 3U; ++t) u.step(t, (t == 3U) ? -2.0f : 5.0f, -10.0f, rho, rosi);
    REQUIRE(rosi.definitively_violated());
}
TEST_CASE("StlMonitorEngine runs with 0 heap allocations and <1 ms on Kartal_Sahil diversion") {
    ScopedZeroHeapGuard heap;
    auto out = run_calibrated_kartal_diversion_monitor();
    REQUIRE(heap.alloc_count() == 0U);
    REQUIRE(out.mission_rho == Catch::Approx(2.87f).margin(0.05f));
    REQUIRE(out.max_step_us < 1000.0);
}
```
- [ ] **Step 2: Run test to verify it fails** — `ctest --test-dir build -R test_stl_monitor_zero_heap --output-on-failure` (Expected: FAIL, missing `stl_monitor.hpp`).
- [ ] **Step 3: Implement `FixedMonotonicWedge`, `OnlineBoundedUntil`, and `StlMonitorEngine`**
  Fix `DR-6` Erratum #4 in `FixedMonotonicWedge::push_and_query(uint32_t tau, float v, uint32_t span_ba) noexcept`: compute `min_idx = (tau >= span_ba) ? (tau - span_ba) : 0U;` and evict expired `head_` entries (`buf_[head_].idx < min_idx`) **BEFORE** popping dominated back entries and pushing `Sample{tau, v}` at `tail_` (plus `std::isnan(v)` fail-closed `-inf` guard and `span_ba < Cap` clamp). Implement `OnlineBoundedUntil<601U>` with real-time RoSI `[inf, sup]` and `StlMonitorEngine` evaluating all 7 predicates over fixed `std::array` storage (`0` heap allocations).
- [ ] **Step 4: Run tests under ASan/UBSan** — `ctest --test-dir build -R test_stl_monitor_zero_heap --output-on-failure` (Expected: PASS; `0` heap allocs, `0.00` oracle error, `< 1 ms`).
- [ ] **Step 5: Commit** — `git add include/vertirta/stl/stl_monitor.hpp core/stl/ tests/stl/test_stl_monitor_zero_heap.cpp && git commit -m "feat(stl): add zero-heap C++20 quantitative STL monitor with wedge wraparound fix"`

---

### Task 3: Native `Z3` C++ `QF_LRA` Static/Corridor Verifier (All 4 Bug Fixes) & UNSAT-Core Deterministic Repair

**Files:**
- Create: `include/vertirta/copilot/z3_corridor_verifier.hpp`, `core/copilot/z3_corridor_verifier.cpp`, `core/copilot/unsat_repair.cpp`
- Test: `tests/copilot/test_z3_verifier_and_repair.cpp`

**Interfaces:**
- Consumes: `RtaStateSnapshotV1`, `CompactDslAst`, frozen `config/istanbul_ltfj_airspace.yaml` (`LTFJ` 16-gon+NW `J1`/`J2` convex hull `750.0 m` ENU ceil; `LTBX` 16-gon `597.6 m` ENU ceil).
- Produces: `CandidateMission build_candidate_mission(const CompactDslAst& ast, const RtaStateSnapshotV1& st) noexcept`, `Z3VerifyResult verify_mission_qflra(const RtaStateSnapshotV1& st, const CandidateMission& m, const AirspacePolys& polys)`, `RepairDecision resolve_unsat_and_repair(const RtaStateSnapshotV1& st, const CandidateMission& m, const Z3VerifyResult& res, const AirspacePolys& polys)`, `std::string dump_auditable_smt2(const z3::solver& s, const std::vector<std::string>& trackers)`.

- [ ] **Step 1: Write failing tests pinning `LTFJ`/`LTBX` `J2` geometry, ALL 4 `DR-6` Z3 bug fixes, and 1-pass UNSAT repair**
```cpp
// tests/copilot/test_z3_verifier_and_repair.cpp
TEST_CASE("Z3 assigns unique c_dyn_nfz_0..3 trackers across 2+ dynamic NFZs") {
    auto st = make_ref_400v_sml_snapshot(); st.num_dyn_nfz = 2;
    st.dyn_nfzs[0] = make_nfz("H0", -500.0, 400.0, 200.0f, 0.0f, 200.0f);
    st.dyn_nfzs[1] = make_nfz("H1", -100.0, 100.0, 150.0f, 0.0f, 200.0f);
    auto r = verify_mission_qflra(st, make_divert_to(st, "Kartal_Sahil"), load_istanbul_polys());
    REQUIRE_FALSE(r.is_sat); REQUIRE(contains_prefix(r.unsat_core_labels, "c_dyn_nfz_"));
}
TEST_CASE("Z3 checks ownship Z + destination Z + cmd_agl + NFZ floor_amsl_m") {
    auto st = make_ref_400v_sml_snapshot(); st.pos_enu.up_m = 800.0;
    auto r = verify_mission_qflra(st, make_divert_to_coord(4360.4, -2022.3, -8.8, 0.95), load_istanbul_polys());
    REQUIRE_FALSE(r.is_sat); REQUIRE(contains(r.unsat_core_labels, "c_ltfj_ctr"));
}
TEST_CASE("Z3 enforces cmd_speed_max, ams_max_climb, and skips p_safe/ood_var on HOLD_LOITER") {
    auto st = make_ref_400v_sml_snapshot();
    auto fast = make_divert_to(st, "Kartal_Sahil"); fast.cmd_speed_max_mps = 25.0;
    REQUIRE(contains(verify_mission_qflra(st, fast, load_istanbul_polys()).unsat_core_labels, "c_ams_speed"));
    auto climb = make_divert_to(st, "Kartal_Sahil"); climb.cmd_climb_req_mps = 4.0;
    REQUIRE(contains(verify_mission_qflra(st, climb, load_istanbul_polys()).unsat_core_labels, "c_ams_climb"));
    auto loiter = make_hold_loiter(st, 30.0); loiter.target_p_safe = 0.0; loiter.target_ood_var = 1.0;
    REQUIRE(verify_mission_qflra(st, loiter, load_istanbul_polys()).is_sat);
}
TEST_CASE("UNSAT repair resolves Pendik_Marina -> Kartal_Sahil in 1 Z3 pass (<15 ms) and appends check-sat-assuming") {
    auto st = make_ref_400v_sml_snapshot(); auto polys = load_istanbul_polys();
    auto m = make_divert_to_coord(4360.4, -2022.3, -8.8, 0.95);
    auto r = verify_mission_qflra(st, m, polys);
    auto rep = resolve_unsat_and_repair(st, m, r, polys);
    REQUIRE(rep.verified_sat); REQUIRE(rep.fallback_target_id == "Kartal_Sahil");
    REQUIRE(rep.wall_ms < 15.0); REQUIRE(r.auditable_smt2.find("(check-sat-assuming (") != std::string::npos);
}
```
- [ ] **Step 2: Run test to verify it fails** — `ctest --test-dir build -R test_z3_verifier_and_repair --output-on-failure` (Expected: FAIL, missing verifier sources).
- [ ] **Step 3: Implement `z3_corridor_verifier.cpp` and `unsat_repair.cpp`**
  Configure `z3::solver(ctx, "QF_LRA")` (`unsat_core = true`, `core.minimize = true`, `rlimit = 250000U`, `timeout = 12U`) with linear 2D SAT disjointness over 150 m-buffered 16-gons for `LTFJ` (including NW `J1`/`J2`) and `LTBX` plus 8-gons for dynamic NFZs. Apply ALL 4 `DR-6` Z3 erratum fixes: (1) unique `"c_dyn_nfz_" + std::to_string(k)` trackers; (2) vertical clearance holds ONLY if `min(ownship_u, dest_u, cmd_agl_min_u) > ceil_u` OR `(floor_u > 0.0 && max(ownship_u, dest_u, cmd_agl_max_u) < floor_u)`; (3) track `c_ams_speed` (`<= max_cruise_airspeed_mps`), `c_ams_climb` (`<= ams_max_climb_mps`), and `c_ams_accel` (`<= ams_max_accel_mps2`); (4) skip `c_slz_unsafe`/`c_slz_ood` on `HOLD_LOITER`. Append `(check-sat-assuming (...))` in `dump_auditable_smt2` (`Z3` #2589), fail-closed `z3::unknown -> c_solver_timeout`, and implement 8-priority `resolve_unsat_and_repair` (`c_shield_bypass` > `c_ltfj_ctr` > `c_ltbx_ctr`/`c_dyn_nfz_*` > `c_soc_reserve` > `c_slz_unsafe`/`c_slz_ood` > `c_min_agl`/`c_max_agl` > `c_ams_*` > `c_temporal_order`/`c_action_mutex`) with 1-pass Z3 re-verification.
- [ ] **Step 4: Run tests & 1,000-segment latency sweep** — `ctest --test-dir build -R test_z3_verifier_and_repair --output-on-failure` (Expected: PASS; `0` SAT/LP mismatches, p99 `< 15 ms`).
- [ ] **Step 5: Commit** — `git add include/vertirta/copilot/z3_corridor_verifier.hpp core/copilot/z3_corridor_verifier.cpp core/copilot/unsat_repair.cpp tests/copilot/test_z3_verifier_and_repair.cpp && git commit -m "feat(copilot): implement QF_LRA Z3 corridor verifier (4 bug fixes) and UNSAT repair"`

---

### Task 4: RAII `VertiLlamaEngine` (`libllama` GBNF + `n_batch` Chunking + Empty-Think Prefill), TR/EN Renderer & Numeric Grounding Gate

**Files:**
- Create: `include/vertirta/copilot/llama_gbnf_engine.hpp`, `core/copilot/llama_gbnf_engine.cpp`, `core/copilot/bilingual_renderer.cpp`
- Test: `tests/copilot/test_llama_engine_and_renderer.cpp`

**Interfaces:**
- Consumes: `RtaStateSnapshotV1`, `CompactDslAst`, `RepairDecision`, `config/vertirta_dsl_v1.gbnf`, `llama.h`.
- Produces: `kQwen3EmptyThinkSuffix`, `std::vector<std::pair<int32_t, int32_t>> plan_batch_chunks(int32_t total, int32_t n_batch)`, RAII `class VertiLlamaEngine` (`compile_dsl(...)`, `score_action_prefixes(...)`), `std::string render_bilingual_response(...)`, `std::string render_clarification_question(const RtaStateSnapshotV1& st, std::string_view opt_a, std::string_view opt_b, LangCode lang)`, `bool verify_numeric_grounding(std::string_view text, const RtaStateSnapshotV1& st, double tol = 0.5)`.

- [ ] **Step 1: Write failing tests for empty-think prefill, `n_batch` chunking, Numeric Grounding Gate (`368.5 V`), and TR/EN rendering**
```cpp
// tests/copilot/test_llama_engine_and_renderer.cpp
TEST_CASE("Empty-think prefill suffix and n_batch chunker obey Qwen3 and libllama limits") {
    REQUIRE(vertirta::copilot::kQwen3EmptyThinkSuffix == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    REQUIRE(vertirta::copilot::plan_batch_chunks(1150, 512) == std::vector<std::pair<int32_t, int32_t>>{{0, 512}, {512, 512}, {1024, 126}});
}
TEST_CASE("Numeric Grounding Gate validates 368.5 V HV telemetry and rejects ungrounded numbers") {
    auto st = make_ref_400v_sml_snapshot();
    REQUIRE(vertirta::copilot::verify_numeric_grounding("Motor 3 arizali. Batarya %34 (368.5 V), 65 m AGL, Kartal Sahil %94, 100 sn.", st));
    REQUIRE_FALSE(vertirta::copilot::verify_numeric_grounding("Batarya %34 (44.2 V), Kartal Sahil 45 sn.", st));
}
TEST_CASE("Deterministic TR/EN renderer formats Mode 1/2/3 from 400V snapshot without LLM prose") {
    auto tr = vertirta::copilot::render_clarification_question(make_ref_400v_sml_snapshot(), "Kartal_Sahil", "Cevizli_Helipad", LangCode::TR);
    REQUIRE(tr.find("Kartal Sahil (A, %94") != std::string::npos);
    REQUIRE(tr.find("Cevizli Helipad (B, %89") != std::string::npos);
}
```
- [ ] **Step 2: Run test to verify it fails** — `ctest --test-dir build -R test_llama_engine_and_renderer --output-on-failure` (Expected: FAIL, missing engine/renderer sources).
- [ ] **Step 3: Implement `VertiLlamaEngine` and `bilingual_renderer.cpp`**
  Implement `~VertiLlamaEngine()` freeing `ctx_` (`llama_free`) and `model_` (`llama_model_free`) with deleted copy ops and safe moves; pin static system prompt in KV `[0, n_sys_prefix_)` using `plan_batch_chunks(..., cparams.n_batch = 512)` on both static & dynamic prefills; call `llama_memory_seq_rm(mem, 0, n_sys_prefix_, -1)`, append `kQwen3EmptyThinkSuffix` (asserting tail token IDs `{151667, 271, 151668, 271}` on Qwen3 GGUF load), run eager `llama_sampler_init_grammar(vocab_, gbnf_.c_str(), "root")` → `top_k(20)` → `top_p(0.80f)` → `temp(0.20f)` → `dist(42)` capped at `48` tokens (`<= 30` expected). Implement `score_action_prefixes` evaluating `<= 11` `"ACT=..."` branches off the pinned KV cache via `llama_get_logits_ith` + GBNF mask, deterministic `render_bilingual_response` / `render_clarification_question`, and `verify_numeric_grounding`.
- [ ] **Step 4: Run tests under ASan/LSan** — `ctest --test-dir build -R test_llama_engine_and_renderer --output-on-failure` (Expected: PASS; `0` leaked bytes; `< 3.0 s` CPU AVX2 with optional GGUF fixture).
- [ ] **Step 5: Commit** — `git add include/vertirta/copilot/llama_gbnf_engine.hpp core/copilot/llama_gbnf_engine.cpp core/copilot/bilingual_renderer.cpp tests/copilot/test_llama_engine_and_renderer.cpp && git commit -m "feat(copilot): add RAII libllama GBNF engine, TR/EN renderer, and numeric grounding gate"`

---

### Task 5: `KnowNo` Split Conformal Gate (`n_cal = 200, alpha = 0.05`) & `< 5%` SoC Liveness Override

**Files:**
- Create: `include/vertirta/copilot/conformal_gate.hpp`, `core/copilot/conformal_gate.cpp`, `ml/copilot/calibrate_conformal.py`
- Test: `tests/copilot/test_conformal_gate.cpp`

**Interfaces:**
- Consumes: `VertiLlamaEngine::score_action_prefixes`, Z3 SAT flags from `verify_mission_qflra`, `RtaStateSnapshotV1`.
- Produces: `float compute_split_conformal_qhat(std::span<const float> nonconformity_scores, float alpha) noexcept`, `ConformalDecision evaluate_conformal_gate(std::span<const CandidateScore> candidates, float q_hat, const RtaStateSnapshotV1& st) noexcept`, and `calibrate_conformal.py` writing `config/conformal_calibration.json` (`alpha = 0.05, n_cal = 200, q_level = 0.9550`).

- [ ] **Step 1: Write failing tests for `q_lvl = 0.9550`, all 3 conformal regimes, and `< 5%` SoC liveness override**
```cpp
// tests/copilot/test_conformal_gate.cpp
TEST_CASE("Split Conformal quantile matches ceil((200+1)*0.95)/200 = 0.955") {
    std::vector<float> s(200U);
    for (std::size_t i = 0; i < 200U; ++i) s[i] = static_cast<float>(i + 1U) / 200.0f;
    REQUIRE(vertirta::copilot::compute_split_conformal_qhat(s, 0.05f) == Catch::Approx(191.0f / 200.0f));
}
TEST_CASE("Conformal gate routes |C_safe|==1 -> EXECUTE, >1 -> CLARIFY, ==0 -> ABSTAIN, and <5% SoC -> LIVENESS") {
    auto st = make_ref_400v_sml_snapshot();
    REQUIRE(evaluate_conformal_gate({{"Kartal_Sahil", 0.89f, true}, {"Dragos_Plaza", 0.07f, false}}, 0.68f, st).outcome == ConformalOutcome::EXECUTE_SINGLE);
    auto d2 = evaluate_conformal_gate({{"Kartal_Sahil", 0.44f, true}, {"Cevizli_Helipad", 0.37f, true}}, 0.68f, st);
    REQUIRE(d2.outcome == ConformalOutcome::ACTIVE_ELICITATION);
    REQUIRE(d2.opt_a == "Kartal_Sahil"); REQUIRE(d2.opt_b == "Cevizli_Helipad");
    REQUIRE(evaluate_conformal_gate({{"Kartal_Sahil", 0.24f, true}}, 0.68f, st).outcome == ConformalOutcome::ABSTAIN_LOITER);
    st.soc_pct = 16.5f; // Loiter margin = 4.5% < 5.0% floor
    auto dl = evaluate_conformal_gate({{"Kartal_Sahil", 0.44f, true}, {"Cevizli_Helipad", 0.37f, true}}, 0.68f, st);
    REQUIRE(dl.outcome == ConformalOutcome::LIVENESS_EMERGENCY_COMMIT); REQUIRE(dl.opt_a == "Kartal_Sahil");
}
```
- [ ] **Step 2: Run test to verify it fails** — `ctest --test-dir build -R test_conformal_gate --output-on-failure` (Expected: FAIL, missing `conformal_gate.hpp`).
- [ ] **Step 3: Implement `conformal_gate.cpp` and `calibrate_conformal.py`**
  Compute length-normalized softmax (`gamma = 0.5f, tau = 1.0f`), non-conformity `s(x, y) = 1.0f - p_hat(y | x)`, order statistic `k = ceil((n_cal + 1) * (1 - alpha))` (`191 / 200 = 0.9550`), intersection `C_safe(x) = {y in C(x) : z3_sat(y)}`, and deterministic liveness override whenever `st.soc_pct - st.soc_reserve_floor_pct < 5.0f` (immediately committing Rank-1 Z3-SAT SLZ). Add `calibrate_conformal.py` outputting `config/conformal_calibration.json` strictly from `calibration.jsonl` (`n_cal = 200`).
- [ ] **Step 4: Run tests** — `ctest --test-dir build -R test_conformal_gate --output-on-failure` (Expected: PASS).
- [ ] **Step 5: Commit** — `git add include/vertirta/copilot/conformal_gate.hpp core/copilot/conformal_gate.cpp ml/copilot/calibrate_conformal.py tests/copilot/test_conformal_gate.cpp && git commit -m "feat(copilot): add KnowNo Split Conformal gate (alpha=0.05) and <5% SoC liveness override"`

---

### Task 6: Public Apache-2.0 `VertiBench-TR/EN` Dataset Pipeline & `< 0.15` Anti-Contamination ARM64 CI Gate

**Files:**
- Create: `benchmarks/vertibench_tr_en/{build_train_split.py,validate_vertibench.py,README.md}`, `.github/workflows/vertibench_ci.yml`
- Test: `tests/python/test_vertibench_integrity.py`

**Interfaces:**
- Consumes: `config/vertirta_dsl_v1.gbnf`, `config/istanbul_ltfj_airspace.yaml`, `parse_compact_dsl`, `verify_mission_qflra`.
- Produces: `benchmarks/vertibench_tr_en/data/{train,calibration,test}.jsonl` (`1800 / 200 / 560` across B1–B4, `400 V / 22 kWh`), `max_char_5gram_jaccard(text, corpus) -> float`, `is_model_allowed(model_id, mode) -> bool`, `validate_hv_telemetry(snapshot) -> bool`, `validate_vertibench(data_dir) -> ValidationReport`.

- [ ] **Step 1: Write failing pytest checks for `< 0.15` Jaccard gate, license allowlist, and `400 V` HV schema**
```python
# tests/python/test_vertibench_integrity.py
def test_jaccard_gate_blocks_leaked_prompts_above_0_15():
    train = ["Motor 3 arizali, Kartal Sahil pistine hemen acil inis yap"]
    assert max_char_5gram_jaccard("Motor 3 arizali, Kartal Sahil pistine hemen inis yap", train) >= 0.15
    assert max_char_5gram_jaccard("Uc numarali pervane durdu; kiyi pistine yonlenebilir miyiz?", train) < 0.15

def test_license_allowlist_and_hv_telemetry_guards():
    assert is_model_allowed("Qwen/Qwen3-8B", "train") and is_model_allowed("Qwen/Qwen3-4B", "train")
    assert not is_model_allowed("Qwen/Qwen2.5-7B-Instruct", "train")
    assert is_model_allowed("Qwen/Qwen2.5-7B-Instruct", "zero_shot_eval")
    assert is_model_allowed("google/gemma-3-4b-it", "zero_shot_eval")
    assert not is_model_allowed("CohereForAI/aya-expanse-8b", "zero_shot_eval")
    assert not validate_hv_telemetry({"ownship": {"v_term_volts": 44.2, "soc_pct": 34.0}})
    assert validate_hv_telemetry({"ownship": {"v_term_volts": 368.5, "soc_pct": 34.0}})
```
- [ ] **Step 2: Run pytest to verify it fails** — `pytest tests/python/test_vertibench_integrity.py -v` (Expected: FAIL, `ModuleNotFoundError`).
- [ ] **Step 3: Implement `build_train_split.py`, `validate_vertibench.py`, HF `README.md`, and `vertibench_ci.yml`**
  Synthesize `1,800` GBNF+Z3-filtered `train.jsonl` rows (B1: 450, B2: 520, B3: 450, B4: 380) using `400 V / 22 kWh` snapshots (`350–380 V`), compact `<= 30`-token `canonical_dsl`, and Turkish ASR diacritic drops. Enforce in `validate_vertibench.py`: exact counts (`1800 / 200 / 560`; test B1: 140, B2: 160, B3: 140, B4: 120), `origin_provenance == "human_native_tr_v1"` on all calibration/test rows, `0` seed overlap, `< 0.15` 5-gram char & 3-gram token Jaccard vs `train`, `v_term_volts in [320, 420]`, and license allowlist. Add Apache-2.0 HF card `README.md` and `ubuntu-24.04-arm` CI workflow `.github/workflows/vertibench_ci.yml`.
- [ ] **Step 4: Run pytest & validator** — `pytest tests/python/test_vertibench_integrity.py -v && python benchmarks/vertibench_tr_en/validate_vertibench.py --check-schema` (Expected: PASS).
- [ ] **Step 5: Commit** — `git add benchmarks/vertibench_tr_en/ .github/workflows/vertibench_ci.yml tests/python/test_vertibench_integrity.py && git commit -m "feat(vertibench): add Apache-2.0 VertiBench-TR/EN builder and <0.15 Jaccard ARM64 CI gate"`

---

### Task 7: Colab G4 96 GB `Qwen3-8B` & `Qwen3-4B` SFT → Z3-GRPO (RLVR), GGUF Export & Multi-Model Eval Harness

**Files:**
- Create: `ml/copilot/{grpo_z3_rewards.py,train_sft_grpo.py,export_gguf.sh}`, `benchmarks/vertibench_tr_en/eval_vertibench.py`
- Test: `tests/python/test_grpo_rewards_and_config.py`

**Interfaces:**
- Consumes: `benchmarks/vertibench_tr_en/data/{train,calibration,test}.jsonl`, `is_model_allowed` (Task 6), `config/vertirta_dsl_v1.gbnf`, `z3-solver`, `Unsloth`, `TRL GRPOTrainer`.
- Produces: `PersistentZ3RewardPool(max_workers=8)` (`spawn` context), `score_single_completion(completion, meta) -> float`, `GrpoPolicyConfig`, Apache-2.0 checkpoints (`Qwen3-8B-VertiRTA`, `Qwen3-4B-VertiRTA`), `Qwen3-4B-VertiRTA-Q4_K_M.gguf`, and `eval_vertibench.py` 4-model ablation report.

- [ ] **Step 1: Write failing tests pinning case-matched `r_z3`, null `ast.target`, `spawn` pool, and calibrated `dr_grpo`/GSPO clip validation**
```python
# tests/python/test_grpo_rewards_and_config.py
def test_r_z3_maps_uppercase_dsl_violation_to_lowercase_z3_label():
    assert score_single_completion("ACT=REJECT_UNSAFE;VIO=LTFJ_CTR;FB=Kartal_Sahil\nLNG=TR;MOD=REJECT", make_b3_pendik_meta()) >= 0.85

def test_null_target_on_loiter_and_clarification_is_safe():
    assert 0.0 <= score_single_completion("ACT=HOLD_LOITER;DUR=30;AGL=60\nLNG=TR;MOD=BRIEF", make_b4_garbled_meta()) <= 1.0
    assert 0.8 <= score_single_completion("ACT=REQUEST_CLARIFICATION;A=Kartal_Sahil;B=Cevizli_Helipad;RSN=AMBIG\nLNG=TR;MOD=CLARIFY", make_b4_ambig_meta()) <= 1.0

def test_grpo_clip_bounds_and_spawn_reward_pool():
    with pytest.raises(ValueError, match="GSPO"):
        GrpoPolicyConfig(importance_sampling_level="sequence", epsilon=0.20, epsilon_high=0.28, beta=0.04)
    assert GrpoPolicyConfig(importance_sampling_level="token", loss_type="dr_grpo", epsilon=0.20, epsilon_high=0.28, beta=0.04).beta == 0.04
    assert GrpoPolicyConfig(importance_sampling_level="sequence", loss_type="dr_grpo", epsilon=3e-4, epsilon_high=4e-4, beta=0.04).epsilon_high == 4e-4
    with PersistentZ3RewardPool(max_workers=2) as pool:
        assert pool.start_method == "spawn"
        assert len(pool.score_batch(["ACT=HOLD_LOITER;DUR=30;AGL=60\nLNG=TR;MOD=BRIEF"], [make_b4_garbled_meta()])) == 1
```
- [ ] **Step 2: Run pytest to verify it fails** — `pytest tests/python/test_grpo_rewards_and_config.py -v` (Expected: FAIL, `ModuleNotFoundError`).
- [ ] **Step 3: Implement `grpo_z3_rewards.py`, `train_sft_grpo.py`, `export_gguf.sh`, and `eval_vertibench.py`**
  Apply `DR-6` Errata #2 & #3: (1) map `VIOLATION_TO_Z3_LABEL` (`"LTFJ_CTR" -> "c_ltfj_ctr"`, ..., prefix match `"c_dyn_nfz_*"`) and handle `ast.target is None` for `HOLD_LOITER` (`CURRENT_POS` corridor/SoC check) and `REQUEST_CLARIFICATION` (`opt_a`/`opt_b` both Z3-SAT); (2) create `PersistentZ3RewardPool` using `multiprocessing.get_context("spawn")` ONCE per run (banning per-step `fork` CUDA deadlocks); (3) in `train_sft_grpo.py`, assert `is_model_allowed(model_id, "train")` (`Qwen3-8B` & `Qwen3-4B` only), run full-BF16 LoRA (`r = 32, alpha = 64`, 7 linear projections) Stage-1 SFT (`enable_thinking=False`, 3 epochs) → Stage-2 GRPO (`G = 8`, `max_completion_length = 64`, `beta = 0.04`, 10% SFT bilingual replay, default token-level `dr_grpo` `0.20/0.28` or calibrated GSPO `3e-4/4e-4`); (4) add `export_gguf.sh` (`Qwen3-4B-VertiRTA-Q4_K_M.gguf` ~2.50 GB) and deterministic `eval_vertibench.py` scoring all 4 models (`Qwen3-4B`, `Qwen3-8B` trained vs `Qwen2.5-7B-Instruct`, `Gemma-3-4B-IT` zero-shot baselines, zero LLM-as-a-judge).
- [ ] **Step 4: Run pytest unit suite** — `pytest tests/python/test_grpo_rewards_and_config.py -v` (Expected: PASS; Colab G4 training executes at Gate 4 after human-audited splits lock).
- [ ] **Step 5: Commit** — `git add ml/copilot/ benchmarks/vertibench_tr_en/eval_vertibench.py tests/python/test_grpo_rewards_and_config.py && git commit -m "feat(ml): add Colab G4 SFT->Z3-GRPO RLVR pipeline (spawn pool, case-matched Z3, clip guard)"`

---

### Task 8: Isolated ROS 2 Jazzy `vertirta_mission_node` (Async Action → `ModeExecutorBase`) & `0 µs` 100 Hz RTA Jitter Verification

**Files:**
- Create: `ros2_ws/src/vertirta_mission_node/{CMakeLists.txt,package.xml,src/mission_node.cpp}`
- Test: `ros2_ws/src/vertirta_mission_node/test/test_mission_isolation.cpp`

**Interfaces:**
- Consumes: Plan 1A's `/vertirta/rta_state_snapshot` (`vertirta_msgs/msg/RtaStateSnapshot`, 1 Hz from `vertirta_rta_node`), `/fmu/out/vertirta_capability_envelope` (`px4_msgs/msg/VertirtaCapabilityEnvelope`, `72 B`), `ModeExecutorBase` action server `/vertirta/execute_contingency` (`vertirta_msgs/action/ExecuteContingency`), and Tasks 1–5 libraries (`vertirta::copilot`, `vertirta::stl_monitor`: `VertiLlamaEngine`, `build_candidate_mission`, `verify_mission_qflra`, `resolve_unsat_and_repair`, `evaluate_conformal_gate`, `StlMonitorEngine`, `render_bilingual_response`).
- Produces: Standalone ROS 2 node `vertirta_mission_node` consuming `/vertirta/copilot_utterance` → publishing `/vertirta/copilot_response` & `/vertirta/stl_robustness`, and sending ONLY Z3-SAT verified `/vertirta/execute_contingency` (`vertirta_msgs/action/ExecuteContingency`) goals asynchronously to `ModeExecutorBase`.

- [ ] **Step 1: Write failing integration & jitter-isolation tests**
```cpp
// ros2_ws/src/vertirta_mission_node/test/test_mission_isolation.cpp
TEST_CASE("Unsafe LTFJ override never reaches ModeExecutorBase and emits TR refusal + Kartal_Sahil fallback") {
    auto h = make_mock_mission_harness(make_ref_400v_sml_snapshot());
    auto res = h.submit_utterance("Acil durum var, hemen Pendik Marina'ya in!");
    REQUIRE(h.last_dispatched_goal().target_id == "Kartal_Sahil");
    REQUIRE(res.response_text.find("LTFJ") != std::string::npos);
}
TEST_CASE("Mode 3 clarification auto-commits Rank-1 Z3-SAT SLZ after 15s timeout") {
    auto h = make_mock_mission_harness(make_ref_400v_sml_snapshot());
    h.trigger_conformal_ambiguity("Kartal_Sahil", "Cevizli_Helipad");
    h.advance_clock_sec(15.1);
    REQUIRE(h.last_dispatched_goal().target_id == "Kartal_Sahil");
}
TEST_CASE("Concurrent async co-pilot worker adds 0 us lock contention and zero missed 10 ms deadlines on 100 Hz RTA") {
    auto j = measure_100hz_rta_jitter_under_copilot_load(/*baseline_ticks=*/500, /*loaded_ticks=*/500);
    REQUIRE(j.missed_deadlines == 0U); REQUIRE(j.delta_p99_us <= 50.0);
}
```
- [ ] **Step 2: Run colcon test to verify it fails** — `colcon test --packages-select vertirta_mission_node --event-handlers console_direct+` (Expected: FAIL, package not built).
- [ ] **Step 3: Implement `mission_node.cpp` (consuming Plan 1A's `vertirta_msgs/action/ExecuteContingency`)**
  Run `vertirta_mission_node` as an isolated ROS 2 process with a dedicated lower-priority worker `std::jthread` reading `vertirta_msgs/msg/RtaStateSnapshot` (`/vertirta/rta_state_snapshot`) into `RtaStateSnapshotV1` via lock-free double-buffer (zero shared mutexes/executors with the 100 Hz RTA node). Execute pipeline: `VertiLlamaEngine` (`score_action_prefixes` + `compile_dsl`) → `parse_compact_dsl` → `synthesize_canonical_stl` → `build_candidate_mission` → `verify_mission_qflra` → `evaluate_conformal_gate` (with `< 5%` SoC liveness override) → `resolve_unsat_and_repair` → async `rclcpp_action::Client<vertirta_msgs::action::ExecuteContingency>` on `/vertirta/execute_contingency` to `ModeExecutorBase` → arm `StlMonitorEngine` on live telemetry (flagging instant RoSI violation `sup < 0.0f`). Enforce 15 s clarification countdown auto-committing Rank-1 Z3-SAT SLZ.
- [ ] **Step 4: Build and run `colcon test`** — `colcon build --packages-select vertirta_mission_node && colcon test --packages-select vertirta_mission_node --event-handlers console_direct+` (Expected: PASS; `0` unverified goals reaching `ModeExecutorBase`, `0` missed 100 Hz RTA deadlines).
- [ ] **Step 5: Commit** — `git add ros2_ws/src/vertirta_mission_node/ && git commit -m "feat(ros2): add isolated vertirta_mission_node with async ModeExecutorBase bridge and jitter test"`
