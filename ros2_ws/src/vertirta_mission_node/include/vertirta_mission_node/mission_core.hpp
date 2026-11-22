// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#include "vertirta/copilot/conformal_gate.hpp"
#include "vertirta/copilot/llama_gbnf_engine.hpp"
#include "vertirta/copilot/rta_state_contract.hpp"
#include "vertirta/copilot/z3_corridor_verifier.hpp"
#include "vertirta/stl/stl_monitor.hpp"

namespace vertirta::mission {

// Zero-UB, 100% TSan-clean lock-free snapshot buffer isolating the 100 Hz RTA thread
// from the asynchronous 0.5-1 Hz EASA AI Level 1/2A co-pilot worker (0 us lock contention).
template <typename T>
class alignas(64) LockFreeSnapshotBuffer {
  static_assert(std::is_trivially_copyable_v<T>, "LockFreeSnapshotBuffer requires trivially copyable POD");
  static_assert(std::atomic<uint64_t>::is_always_lock_free, "64-bit atomics must be lock-free");

 public:
  static constexpr std::size_t kWords = (sizeof(T) + sizeof(uint64_t) - 1U) / sizeof(uint64_t);

  LockFreeSnapshotBuffer() noexcept {
    for (auto& w : words_) {
      w.store(0ULL, std::memory_order_relaxed);
    }
  }

  LockFreeSnapshotBuffer(const LockFreeSnapshotBuffer&) = delete;
  LockFreeSnapshotBuffer& operator=(const LockFreeSnapshotBuffer&) = delete;

  // Wait-free writer called by 100 Hz RTA telemetry ingress
  void write(const T& value) noexcept {
    std::array<uint64_t, kWords> buf{};
    std::memcpy(static_cast<void*>(buf.data()), static_cast<const void*>(&value), sizeof(T));

    const uint64_t s = seq_.load(std::memory_order_relaxed);
    seq_.store(s + 1ULL, std::memory_order_release);
    for (std::size_t i = 0U; i < kWords; ++i) {
      words_[i].store(buf[i], std::memory_order_release);
    }
    seq_.store(s + 2ULL, std::memory_order_release);
  }

  // Lock-free reader called by isolated co-pilot worker thread
  [[nodiscard]] T read() const noexcept {
    std::array<uint64_t, kWords> buf{};
    while (true) {
      const uint64_t s1 = seq_.load(std::memory_order_acquire);
      if ((s1 & 1ULL) != 0ULL) {
        continue;
      }
      for (std::size_t i = 0U; i < kWords; ++i) {
        buf[i] = words_[i].load(std::memory_order_acquire);
      }
      const uint64_t s2 = seq_.load(std::memory_order_acquire);
      if (s1 == s2) {
        T out{};
        std::memcpy(static_cast<void*>(&out), static_cast<const void*>(buf.data()), sizeof(T));
        return out;
      }
    }
  }

 private:
  alignas(64) std::atomic<uint64_t> seq_{0ULL};
  alignas(64) std::array<std::atomic<uint64_t>, kWords> words_{};
};

// Mirrors ros2_ws/src/vertirta_msgs/action/ExecuteContingency.action Goal
struct DispatchedContingencyGoal {
  static constexpr uint8_t kActionDivertSlz = 0U;
  static constexpr uint8_t kActionInspectSlz = 1U;
  static constexpr uint8_t kActionHoldLoiter = 2U;
  static constexpr uint8_t kActionAbortReturn = 3U;
  static constexpr uint8_t kActionCelLandNow = 4U;

  uint8_t action_kind{kActionHoldLoiter};
  std::string target_id{};
  copilot::EnuPoint target_enu{};
  float eta_lim_s{0.0f};
  float min_agl_m{35.0f};
  float max_agl_m{120.0f};
  float max_speed_mps{18.0f};
  float loiter_duration_s{0.0f};
  bool z3_verified{false};
  std::string z3_audit_hash{};
};

struct MissionPipelineResult {
  std::string dsl_raw{};
  copilot::CompactDslAst ast{};
  copilot::Z3VerifyResult verify{};
  copilot::RepairDecision repair{};
  copilot::ConformalDecision conformal{};
  std::string response_text{};
  bool numeric_grounded{false};
  bool goal_dispatched{false};
  bool clarification_pending{false};
  DispatchedContingencyGoal dispatched_goal{};
};

struct MissionCoreConfig {
  std::string gguf_path{};
  std::string gbnf_grammar{};
  std::string static_sys_prompt{"VertiRTA-X8-240 400V/22kWh TR/EN Contingency Co-Pilot"};
  bool use_cuda{false};
  float conformal_q_hat{0.70f};
  double clarification_timeout_s{15.0};
};

// Pure C++20 EASA AI Level 1/2A Contingency Co-Pilot Core (zero ROS 2 dependency).
// Wires VertiLlamaEngine -> parse_compact_dsl -> synthesize_canonical_stl ->
// verify_mission_qflra -> evaluate_conformal_gate -> resolve_unsat_and_repair ->
// StlMonitorEngine -> async ExecuteContingency goal dispatch.
class MissionCore {
 public:
  using GoalDispatchCallback = std::function<void(const DispatchedContingencyGoal&)>;
  using ResponseCallback = std::function<void(const MissionPipelineResult&)>;

  explicit MissionCore(
      const copilot::RtaStateSnapshotV1& initial_snapshot = copilot::make_reference_400v_sml_snapshot(),
      MissionCoreConfig cfg = {})
      : cfg_(std::move(cfg)),
        llama_(cfg_.gguf_path.empty()
                   ? copilot::VertiLlamaEngine::create_mock(cfg_.gbnf_grammar, cfg_.static_sys_prompt)
                   : copilot::VertiLlamaEngine(
                         cfg_.gguf_path, cfg_.gbnf_grammar, cfg_.static_sys_prompt, cfg_.use_cuda)),
        polys_(copilot::load_istanbul_airspace_polys()) {
    snapshot_buf_.write(initial_snapshot);
    const auto ltfj_hs = copilot::to_stl_halfspaces(polys_.ltfj);
    const auto ltbx_hs = copilot::to_stl_halfspaces(polys_.ltbx);
    stl_monitor_.set_zone_halfspaces(copilot::ZoneId::LTFJ_CTR, ltfj_hs);
    stl_monitor_.set_zone_halfspaces(copilot::ZoneId::LTBX_CTR, ltbx_hs);
  }

  ~MissionCore() { stop_async_worker(); }

  MissionCore(const MissionCore&) = delete;
  MissionCore& operator=(const MissionCore&) = delete;

  void set_goal_dispatch_callback(GoalDispatchCallback cb) {
    std::lock_guard<std::mutex> lk(copilot_mutex_);
    dispatch_cb_ = std::move(cb);
  }

  void set_response_callback(ResponseCallback cb) {
    std::lock_guard<std::mutex> lk(copilot_mutex_);
    response_cb_ = std::move(cb);
  }

  // Wait-free telemetry update called from 100 Hz RTA / DDS snapshot ingress (NEVER blocks)
  void update_snapshot(const copilot::RtaStateSnapshotV1& snap) noexcept {
    copilot::RtaStateSnapshotV1 safe_snap = snap;
    safe_snap.shield_active = true;  // Immutable true in flight (EASA AI Level 1/2A lock)
    snapshot_buf_.write(safe_snap);
  }

  [[nodiscard]] copilot::RtaStateSnapshotV1 current_snapshot() const noexcept {
    return snapshot_buf_.read();
  }

  // Synchronous full neuro-symbolic pipeline invocation (runs on isolated worker thread or test harness)
  [[nodiscard]] MissionPipelineResult submit_utterance(std::string_view utterance) {
    const std::string dsl = llama_.compile_dsl(utterance);
    return process_dsl(dsl, utterance);
  }

  // Directly tests Z3 verification & repair against an untrusted/adversarial raw DSL proposal
  [[nodiscard]] MissionPipelineResult submit_dsl_override(std::string_view raw_dsl) {
    return process_dsl(std::string(raw_dsl), raw_dsl);
  }

  // Directly verifies & repairs an arbitrary CandidateMission via Z3 QF_LRA (defense-in-depth
  // verification even if an upstream candidate bypasses the GBNF SLZ-ID whitelist)
  [[nodiscard]] MissionPipelineResult submit_candidate_mission(
      const copilot::CandidateMission& cand,
      copilot::LangCode lang = copilot::LangCode::TR) {
    std::lock_guard<std::mutex> lk(copilot_mutex_);
    const auto st = snapshot_buf_.read();

    MissionPipelineResult out{};
    out.ast.action = cand.action;
    out.ast.set_target_id(cand.target_id);
    out.ast.eta_lim_s = std::max<int32_t>(10, static_cast<int32_t>(std::lround(cand.target_eta_s)));
    out.ast.lang = lang;
    out.ast.valid = true;

    out.verify = copilot::verify_mission_qflra(st, cand, polys_, /*emit_smt2=*/true);
    out.repair = copilot::resolve_unsat_and_repair(st, cand, out.verify, polys_);
    const auto final_z3 = out.repair.repaired
                              ? copilot::verify_mission_qflra(st, out.repair.repaired_mission, polys_, true)
                              : out.verify;
    arm_stl_and_dispatch_locked(st, out.repair.repaired_mission, final_z3, lang, out);
    out.response_text = copilot::render_bilingual_response(
        st, out.ast, out.repair.repaired ? &out.repair : nullptr, lang);
    out.numeric_grounded = copilot::verify_numeric_grounding(out.response_text, st);
    return out;
  }

  // Triggers Mode 3 KnowNo conformal ambiguity between two Z3-SAT candidates (15.0 s countdown)
  [[nodiscard]] MissionPipelineResult trigger_conformal_ambiguity(
      std::string_view opt_a,
      std::string_view opt_b,
      copilot::LangCode lang = copilot::LangCode::TR) {
    std::lock_guard<std::mutex> lk(copilot_mutex_);
    const auto st = snapshot_buf_.read();

    const auto sat_a = verify_named_slz(st, opt_a);
    const auto sat_b = verify_named_slz(st, opt_b);

    std::array<copilot::CandidateScore, 2> scores{{
        copilot::CandidateScore{std::string(opt_a), 0.45f, sat_a.is_sat},
        copilot::CandidateScore{std::string(opt_b), 0.45f, sat_b.is_sat},
    }};
    const auto conf = copilot::evaluate_conformal_gate(scores, cfg_.conformal_q_hat, st);

    MissionPipelineResult out{};
    out.conformal = conf;
    out.ast.action = copilot::ActionKind::REQUEST_CLARIFICATION;
    out.ast.set_opt_a(opt_a);
    out.ast.set_opt_b(opt_b);
    out.ast.lang = lang;
    out.ast.mode = copilot::ModeCode::CLARIFY;
    out.ast.valid = true;

    if (conf.outcome == copilot::ConformalOutcome::LIVENESS_EMERGENCY_COMMIT) {
      // < 5.0% SoC margin: bypass loiter deadlock and commit Rank-1 Z3-SAT SLZ immediately
      out = commit_rank1_sat_slz_locked(st, lang, out.conformal);
      return out;
    }

    if (conf.outcome == copilot::ConformalOutcome::ACTIVE_ELICITATION) {
      clarification_pending_ = true;
      clarification_deadline_s_ = clock_s_ + cfg_.clarification_timeout_s;
      clarification_opt_a_ = conf.opt_a;
      clarification_opt_b_ = conf.opt_b;
      clarification_lang_ = lang;
      out.clarification_pending = true;
      out.response_text = copilot::render_clarification_question(st, conf.opt_a, conf.opt_b, lang);
      out.numeric_grounded = copilot::verify_numeric_grounding(out.response_text, st);
      return out;
    }

    // Single unambiguous Z3-SAT candidate or abstain fallback
    if (conf.outcome == copilot::ConformalOutcome::EXECUTE_SINGLE && !conf.opt_a.empty()) {
      out = commit_named_slz_locked(st, conf.opt_a, lang, conf);
      return out;
    }
    return commit_hold_loiter_locked(st, lang, conf);
  }

  // Advances mission clock; auto-commits Rank-1 Z3-SAT SLZ when 15.0 s clarification expires
  void advance_clock_sec(double dt_s) {
    if (!std::isfinite(dt_s) || dt_s <= 0.0) {
      return;
    }
    std::lock_guard<std::mutex> lk(copilot_mutex_);
    clock_s_ += dt_s;
    if (clarification_pending_ && clock_s_ >= clarification_deadline_s_) {
      clarification_pending_ = false;
      const auto st = snapshot_buf_.read();
      (void)commit_rank1_sat_slz_locked(st, clarification_lang_, copilot::ConformalDecision{});
    }
  }

  // Step online quantitative STL monitor on live telemetry (zero heap)
  [[nodiscard]] stl::StlTickOutput step_stl_monitor(uint32_t tau, const stl::StlSignalSample& s) noexcept {
    std::lock_guard<std::mutex> lk(stl_mutex_);
    return stl_monitor_.step(tau, s);
  }

  // Isolated lower-priority async worker thread (std::jthread) for ROS 2 / concurrent stress tests
  void start_async_worker() {
    if (worker_.joinable()) {
      return;
    }
    worker_ = std::jthread([this](std::stop_token stoken) {
      while (!stoken.stop_requested()) {
        std::string utt;
        {
          std::unique_lock<std::mutex> lk(queue_mutex_);
          queue_cv_.wait(lk, [&]() { return stoken.stop_requested() || !utterance_queue_.empty(); });
          if (stoken.stop_requested()) {
            break;
          }
          utt = std::move(utterance_queue_.front());
          utterance_queue_.pop_front();
        }
        auto res = submit_utterance(utt);
        processed_async_count_.fetch_add(1U, std::memory_order_release);
        ResponseCallback cb;
        {
          std::lock_guard<std::mutex> lk(copilot_mutex_);
          cb = response_cb_;
        }
        if (cb) {
          cb(res);
        }
      }
    });
  }

  void enqueue_utterance(std::string utterance) {
    {
      std::lock_guard<std::mutex> lk(queue_mutex_);
      utterance_queue_.push_back(std::move(utterance));
    }
    queue_cv_.notify_one();
  }

  void stop_async_worker() {
    if (worker_.joinable()) {
      worker_.request_stop();
      queue_cv_.notify_all();
      worker_.join();
    }
  }

  [[nodiscard]] uint64_t processed_async_count() const noexcept {
    return processed_async_count_.load(std::memory_order_acquire);
  }

  [[nodiscard]] bool has_dispatched_goal() const {
    std::lock_guard<std::mutex> lk(copilot_mutex_);
    return has_dispatched_goal_;
  }

  [[nodiscard]] DispatchedContingencyGoal last_dispatched_goal() const {
    std::lock_guard<std::mutex> lk(copilot_mutex_);
    return last_dispatched_goal_;
  }

  [[nodiscard]] bool clarification_pending() const {
    std::lock_guard<std::mutex> lk(copilot_mutex_);
    return clarification_pending_;
  }

  [[nodiscard]] std::size_t dispatched_goal_count() const {
    std::lock_guard<std::mutex> lk(copilot_mutex_);
    return dispatched_goal_count_;
  }

 private:
  [[nodiscard]] static std::string fnv1a_hex(std::string_view data) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : data) {
      h ^= static_cast<uint64_t>(c);
      h *= 1099511628211ULL;
    }
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << h;
    return oss.str();
  }

  [[nodiscard]] static std::string violation_to_z3_label(copilot::ViolationCode code) {
    switch (code) {
      case copilot::ViolationCode::LTFJ_CTR:
        return "c_ltfj_ctr";
      case copilot::ViolationCode::LTBX_CTR:
        return "c_ltbx_ctr";
      case copilot::ViolationCode::DYN_NFZ:
        return "c_dyn_nfz_0";
      case copilot::ViolationCode::SOC_RESERVE:
        return "c_soc_reserve";
      case copilot::ViolationCode::SLZ_UNSAFE:
        return "c_slz_unsafe";
      case copilot::ViolationCode::MIN_AGL:
        return "c_min_agl";
      case copilot::ViolationCode::SHIELD_BYPASS:
        return "c_shield_bypass";
      default:
        return "c_unknown";
    }
  }

  [[nodiscard]] static uint8_t to_action_kind_byte(copilot::ActionKind k) noexcept {
    switch (k) {
      case copilot::ActionKind::DIVERT_SLZ:
        return DispatchedContingencyGoal::kActionDivertSlz;
      case copilot::ActionKind::INSPECT_SLZ:
        return DispatchedContingencyGoal::kActionInspectSlz;
      case copilot::ActionKind::HOLD_LOITER:
        return DispatchedContingencyGoal::kActionHoldLoiter;
      case copilot::ActionKind::ABORT_RETURN:
        return DispatchedContingencyGoal::kActionAbortReturn;
      default:
        return DispatchedContingencyGoal::kActionHoldLoiter;
    }
  }

  [[nodiscard]] copilot::Z3VerifyResult verify_named_slz(
      const copilot::RtaStateSnapshotV1& st,
      std::string_view slz_id) const {
    copilot::CompactDslAst ast{};
    ast.action = copilot::ActionKind::DIVERT_SLZ;
    ast.set_target_id(slz_id);
    if (const auto* slz = st.find_slz(slz_id)) {
      ast.eta_lim_s = std::max<int32_t>(10, static_cast<int32_t>(std::lround(slz->eta_s)));
    } else {
      ast.eta_lim_s = 100;
    }
    ast.valid = true;
    const auto cand = copilot::build_candidate_mission(ast, st);
    return copilot::verify_mission_qflra(st, cand, polys_, /*emit_smt2=*/false);
  }

  void arm_stl_and_dispatch_locked(
      const copilot::RtaStateSnapshotV1& st,
      const copilot::CandidateMission& verified_mission,
      const copilot::Z3VerifyResult& final_z3,
      copilot::LangCode lang,
      MissionPipelineResult& out) {
    // Hard EASA AI Level 1/2A invariant: NEVER dispatch unless Z3 returned SAT
    if (!final_z3.is_sat) {
      out.goal_dispatched = false;
      return;
    }

    copilot::CompactDslAst final_ast{};
    final_ast.action = verified_mission.action;
    final_ast.set_target_id(verified_mission.target_id);
    final_ast.eta_lim_s = std::max<int32_t>(10, static_cast<int32_t>(std::lround(verified_mission.target_eta_s)));
    final_ast.duration_s = final_ast.eta_lim_s;
    final_ast.agl_min_m = static_cast<int32_t>(std::lround(verified_mission.cmd_agl_min_m));
    final_ast.agl_max_m = static_cast<int32_t>(std::lround(verified_mission.cmd_agl_max_m));
    final_ast.speed_max_mps = static_cast<int32_t>(std::lround(verified_mission.cmd_speed_max_mps));
    final_ast.lang = lang;
    final_ast.valid = true;

    const auto formula = copilot::synthesize_canonical_stl(final_ast, st);
    {
      std::lock_guard<std::mutex> slk(stl_mutex_);
      stl_monitor_.configure(formula);
    }

    DispatchedContingencyGoal goal{};
    goal.action_kind = to_action_kind_byte(verified_mission.action);
    goal.target_id = verified_mission.target_id;
    goal.target_enu = {verified_mission.target_e_m, verified_mission.target_n_m, verified_mission.target_u_m};
    goal.eta_lim_s = static_cast<float>(verified_mission.target_eta_s);
    goal.min_agl_m = static_cast<float>(verified_mission.cmd_agl_min_m);
    goal.max_agl_m = static_cast<float>(verified_mission.cmd_agl_max_m);
    goal.max_speed_mps = static_cast<float>(verified_mission.cmd_speed_max_mps);
    goal.loiter_duration_s =
        (verified_mission.action == copilot::ActionKind::HOLD_LOITER) ? goal.eta_lim_s : 0.0f;
    goal.z3_verified = true;
    goal.z3_audit_hash = fnv1a_hex(final_z3.auditable_smt2);

    last_dispatched_goal_ = goal;
    has_dispatched_goal_ = true;
    ++dispatched_goal_count_;
    out.dispatched_goal = goal;
    out.goal_dispatched = true;

    if (dispatch_cb_) {
      dispatch_cb_(goal);
    }
  }

  [[nodiscard]] MissionPipelineResult commit_named_slz_locked(
      const copilot::RtaStateSnapshotV1& st,
      std::string_view slz_id,
      copilot::LangCode lang,
      const copilot::ConformalDecision& conf) {
    MissionPipelineResult out{};
    out.conformal = conf;
    out.ast.action = copilot::ActionKind::DIVERT_SLZ;
    out.ast.set_target_id(slz_id);
    if (const auto* slz = st.find_slz(slz_id)) {
      out.ast.eta_lim_s = std::max<int32_t>(10, static_cast<int32_t>(std::lround(slz->eta_s)));
    } else {
      out.ast.eta_lim_s = 100;
    }
    out.ast.lang = lang;
    out.ast.valid = true;

    const auto cand = copilot::build_candidate_mission(out.ast, st);
    out.verify = copilot::verify_mission_qflra(st, cand, polys_, /*emit_smt2=*/true);
    out.repair = copilot::resolve_unsat_and_repair(st, cand, out.verify, polys_);
    const auto final_z3 = out.repair.repaired
                              ? copilot::verify_mission_qflra(st, out.repair.repaired_mission, polys_, true)
                              : out.verify;
    arm_stl_and_dispatch_locked(st, out.repair.repaired_mission, final_z3, lang, out);
    out.response_text = copilot::render_bilingual_response(
        st, out.ast, out.repair.repaired ? &out.repair : nullptr, lang);
    out.numeric_grounded = copilot::verify_numeric_grounding(out.response_text, st);
    return out;
  }

  [[nodiscard]] MissionPipelineResult commit_rank1_sat_slz_locked(
      const copilot::RtaStateSnapshotV1& st,
      copilot::LangCode lang,
      const copilot::ConformalDecision& conf) {
    const std::size_t n = std::min<std::size_t>(st.num_slz, copilot::kMaxSlzCandidates);
    for (std::size_t i = 0U; i < n; ++i) {
      const std::string_view id = st.ranked_slzs[i].id_view();
      if (!id.empty() && verify_named_slz(st, id).is_sat) {
        return commit_named_slz_locked(st, id, lang, conf);
      }
    }
    return commit_hold_loiter_locked(st, lang, conf);
  }

  [[nodiscard]] MissionPipelineResult commit_hold_loiter_locked(
      const copilot::RtaStateSnapshotV1& st,
      copilot::LangCode lang,
      const copilot::ConformalDecision& conf) {
    MissionPipelineResult out{};
    out.conformal = conf;
    out.ast.action = copilot::ActionKind::HOLD_LOITER;
    out.ast.duration_s = 30;
    out.ast.agl_min_m = 60;
    out.ast.agl_max_m = 120;
    out.ast.lang = lang;
    out.ast.valid = true;

    const auto cand = copilot::build_candidate_mission(out.ast, st);
    out.verify = copilot::verify_mission_qflra(st, cand, polys_, /*emit_smt2=*/true);
    out.repair = copilot::resolve_unsat_and_repair(st, cand, out.verify, polys_);
    const auto final_z3 = out.repair.repaired
                              ? copilot::verify_mission_qflra(st, out.repair.repaired_mission, polys_, true)
                              : out.verify;
    arm_stl_and_dispatch_locked(st, out.repair.repaired_mission, final_z3, lang, out);
    out.response_text = copilot::render_bilingual_response(
        st, out.ast, out.repair.repaired ? &out.repair : nullptr, lang);
    out.numeric_grounded = copilot::verify_numeric_grounding(out.response_text, st);
    return out;
  }

  [[nodiscard]] MissionPipelineResult process_dsl(std::string dsl, std::string_view /*utterance*/) {
    std::lock_guard<std::mutex> lk(copilot_mutex_);
    const auto st = snapshot_buf_.read();

    MissionPipelineResult out{};
    out.dsl_raw = std::move(dsl);

    if (!copilot::parse_compact_dsl(out.dsl_raw, out.ast)) {
      return commit_hold_loiter_locked(st, copilot::LangCode::TR, copilot::ConformalDecision{});
    }

    if (out.ast.action == copilot::ActionKind::REQUEST_CLARIFICATION) {
      // Delegate to KnowNo conformal clarification path (15 s countdown)
      const std::string a(out.ast.opt_a());
      const std::string b(out.ast.opt_b());
      const copilot::LangCode lang = out.ast.lang;
      clarification_pending_ = true;
      clarification_deadline_s_ = clock_s_ + cfg_.clarification_timeout_s;
      clarification_opt_a_ = a;
      clarification_opt_b_ = b;
      clarification_lang_ = lang;
      out.clarification_pending = true;
      out.response_text = copilot::render_clarification_question(st, a, b, lang);
      out.numeric_grounded = copilot::verify_numeric_grounding(out.response_text, st);
      return out;
    }

    if (out.ast.action == copilot::ActionKind::REJECT_UNSAFE) {
      // Synthesize formal UNSAT core matching the rejected violation, then run Z3 repair
      copilot::CandidateMission unsafe_cand = copilot::build_candidate_mission(out.ast, st);
      copilot::Z3VerifyResult synth_unsat{};
      synth_unsat.is_sat = false;
      synth_unsat.unsat_core_labels.push_back(violation_to_z3_label(out.ast.violation_code));
      out.verify = synth_unsat;
      out.repair = copilot::resolve_unsat_and_repair(st, unsafe_cand, synth_unsat, polys_);

      const auto final_z3 = copilot::verify_mission_qflra(
          st, out.repair.repaired_mission, polys_, /*emit_smt2=*/true);
      arm_stl_and_dispatch_locked(st, out.repair.repaired_mission, final_z3, out.ast.lang, out);
      out.response_text = copilot::render_bilingual_response(st, out.ast, &out.repair, out.ast.lang);
      out.numeric_grounded = copilot::verify_numeric_grounding(out.response_text, st);
      return out;
    }

    // Standard DIVERT_SLZ / INSPECT_SLZ / ABORT_RETURN / HOLD_LOITER -> strict Z3 QF_LRA gate
    const copilot::CandidateMission cand = copilot::build_candidate_mission(out.ast, st);
    out.verify = copilot::verify_mission_qflra(st, cand, polys_, /*emit_smt2=*/true);
    out.repair = copilot::resolve_unsat_and_repair(st, cand, out.verify, polys_);
    const auto final_z3 = out.repair.repaired
                              ? copilot::verify_mission_qflra(st, out.repair.repaired_mission, polys_, true)
                              : out.verify;
    arm_stl_and_dispatch_locked(st, out.repair.repaired_mission, final_z3, out.ast.lang, out);
    out.response_text = copilot::render_bilingual_response(
        st, out.ast, out.repair.repaired ? &out.repair : nullptr, out.ast.lang);
    out.numeric_grounded = copilot::verify_numeric_grounding(out.response_text, st);
    return out;
  }

  MissionCoreConfig cfg_{};
  copilot::VertiLlamaEngine llama_;
  copilot::AirspacePolys polys_{};
  LockFreeSnapshotBuffer<copilot::RtaStateSnapshotV1> snapshot_buf_{};

  mutable std::mutex stl_mutex_{};
  stl::StlMonitorEngine stl_monitor_{};

  mutable std::mutex copilot_mutex_{};
  double clock_s_{0.0};
  bool clarification_pending_{false};
  double clarification_deadline_s_{0.0};
  std::string clarification_opt_a_{};
  std::string clarification_opt_b_{};
  copilot::LangCode clarification_lang_{copilot::LangCode::TR};
  bool has_dispatched_goal_{false};
  std::size_t dispatched_goal_count_{0U};
  DispatchedContingencyGoal last_dispatched_goal_{};
  GoalDispatchCallback dispatch_cb_{};
  ResponseCallback response_cb_{};

  std::mutex queue_mutex_{};
  std::condition_variable queue_cv_{};
  std::deque<std::string> utterance_queue_{};
  std::atomic<uint64_t> processed_async_count_{0ULL};
  std::jthread worker_{};
};

}  // namespace vertirta::mission
