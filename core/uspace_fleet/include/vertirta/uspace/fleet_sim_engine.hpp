// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/core/indi_wls.hpp"
#include "vertirta/core/sim_plant_6dof.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"
#include "vertirta/uspace/cbf_orca_3d.hpp"
#include "vertirta/uspace/corridor_network.hpp"
#include "vertirta/uspace/part_is_security.hpp"

namespace vertirta::uspace {

inline constexpr std::size_t kMaxFleetAgents = 20U;

struct FleetTickSummary {
  uint32_t tick_index{0U};
  uint32_t active_agents{0U};
  uint32_t cel_active_agents{0U};
  uint32_t airspace_breaches{0U};
  uint16_t max_rta_hazard_rows{0U};
  double min_inter_agent_sep_m{1e6};
  double sim_time_s{0.0};
};

// Hybrid 5-20 eVTOL Structure-of-Arrays (SoA) deterministic simulation engine:
//   - Cache-aligned SoA kinematics/telemetry buffers (alignas(64))
//   - 4x 400 Hz CoaxialPlant6Dof + IndiWlsController substeps + 1x 100 Hz CBF-ORCA + CbvfSafetyFilter
//   - Optional fixed thread pool (1-4 threads) with TSan-clean barrier synchronization
//   - Zero edits to frozen core/flight_core or core/rta_companion
class FleetSimEngine {
 public:
  explicit FleetSimEngine(uint32_t n_agents = 20U, uint32_t threads = 4U);
  ~FleetSimEngine();

  FleetSimEngine(const FleetSimEngine&) = delete;
  FleetSimEngine& operator=(const FleetSimEngine&) = delete;
  FleetSimEngine(FleetSimEngine&&) = delete;
  FleetSimEngine& operator=(FleetSimEngine&&) = delete;

  void reset_corridor_schedule(uint64_t seed) noexcept;
  void inject_same_arm_cel(uint32_t agent_idx, uint32_t arm_idx) noexcept;
  void mark_peer_suspect(uint32_t agent_idx) noexcept;

  [[nodiscard]] FleetTickSummary step_100hz() noexcept;

  [[nodiscard]] uint32_t agent_count() const noexcept { return n_agents_; }
  [[nodiscard]] bool agent_cel_latched(uint32_t agent_idx) const noexcept;
  [[nodiscard]] core::Vec3Ned agent_pos_ned(uint32_t agent_idx) const noexcept;
  [[nodiscard]] core::Vec3Ned agent_vel_ned(uint32_t agent_idx) const noexcept;
  [[nodiscard]] uint8_t agent_fault_mask(uint32_t agent_idx) const noexcept;
  [[nodiscard]] const CorridorNetwork& corridor() const noexcept { return corridor_; }

 private:
  struct WorkerResult {
    uint32_t cel_count{0U};
    uint32_t breaches{0U};
    uint16_t max_hazards{0U};
  };

  void step_agent_range(uint32_t begin_idx, uint32_t end_idx, WorkerResult& out) noexcept;
  void worker_loop(uint32_t worker_id) noexcept;

  uint32_t n_agents_{20U};
  uint32_t num_threads_{1U};
  uint32_t tick_index_{0U};
  CorridorNetwork corridor_{};

  // Cache-aligned SoA double-buffered peer kinematics (read snapshot -> write next state, zero races)
  alignas(64) std::array<PeerKinematics, kMaxFleetAgents> peer_snapshot_{};
  alignas(64) std::array<core::Vec3Ned, kMaxFleetAgents> goal_ned_{};
  alignas(64) std::array<uint8_t, kMaxFleetAgents> fault_mask_{};
  alignas(64) std::array<uint8_t, kMaxFleetAgents> cel_latched_{};
  alignas(64) std::array<uint8_t, kMaxFleetAgents> suspect_peer_{};
  alignas(64) std::array<uint8_t, kMaxFleetAgents> is_external_hero_{};
  alignas(64) std::array<uint8_t, kMaxFleetAgents> dest_vp_idx_{};

  // Per-agent frozen flight_core & rta_companion instances (disjoint per agent index)
  std::array<core::CoaxialPlant6Dof, kMaxFleetAgents> plants_{};
  std::array<core::IndiWlsController, kMaxFleetAgents> controllers_;
  std::array<std::unique_ptr<rta::CbvfSafetyFilter>, kMaxFleetAgents> rta_filters_{};
  std::array<CyberResilienceFsm, kMaxFleetAgents> cyber_fsms_{};

  // Fixed worker pool (only spawned when num_threads_ > 1)
  std::mutex pool_mtx_{};
  std::condition_variable cv_start_{};
  std::condition_variable cv_done_{};
  uint64_t generation_{0ULL};
  uint32_t active_workers_{0U};
  bool stop_pool_{false};
  std::array<WorkerResult, 4U> worker_results_{};
  std::vector<std::thread> workers_{};
};

}  // namespace vertirta::uspace
