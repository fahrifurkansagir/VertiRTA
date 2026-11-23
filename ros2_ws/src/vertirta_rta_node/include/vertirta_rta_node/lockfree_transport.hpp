#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace vertirta::ros {

// Cache-line-isolated, zero-heap, lock-free Single-Producer Single-Consumer ring buffer.
// Capacity must be a power of 2 >= 2; holds up to (Capacity - 1) elements.
template <typename T, std::size_t Capacity>
class alignas(64) SpscRingBuffer {
  static_assert(std::is_trivially_copyable_v<T>, "SpscRingBuffer requires trivially copyable POD");
  static_assert(Capacity >= 2U && ((Capacity & (Capacity - 1U)) == 0U),
                "SpscRingBuffer Capacity must be a power of 2 >= 2");

 public:
  static constexpr std::size_t kMask = Capacity - 1U;

  SpscRingBuffer() noexcept = default;
  SpscRingBuffer(const SpscRingBuffer&) = delete;
  SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;

  [[nodiscard]] bool try_push(const T& item) noexcept {
    const std::size_t tail = tail_.load(std::memory_order_relaxed);
    const std::size_t next = (tail + 1U) & kMask;
    if (next == head_.load(std::memory_order_acquire)) {
      return false;
    }
    slots_[tail] = item;
    tail_.store(next, std::memory_order_release);
    return true;
  }

  [[nodiscard]] bool try_pop(T& out) noexcept {
    const std::size_t head = head_.load(std::memory_order_relaxed);
    if (head == tail_.load(std::memory_order_acquire)) {
      return false;
    }
    out = slots_[head];
    head_.store((head + 1U) & kMask, std::memory_order_release);
    return true;
  }

  // Drains all queued items and outputs ONLY the most recent frame (Thread 2 RT semantics).
  [[nodiscard]] bool pop_latest(T& out) noexcept {
    bool got = false;
    T tmp{};
    while (try_pop(tmp)) {
      out = tmp;
      got = true;
    }
    return got;
  }

  [[nodiscard]] bool empty() const noexcept {
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::size_t size_approx() const noexcept {
    const std::size_t tail = tail_.load(std::memory_order_acquire);
    const std::size_t head = head_.load(std::memory_order_acquire);
    return (tail - head) & kMask;
  }

 private:
  alignas(64) std::atomic<std::size_t> head_{0U};
  alignas(64) std::atomic<std::size_t> tail_{0U};
  alignas(64) std::array<T, Capacity> slots_{};
};

// Zero-UB, 100% TSan-clean C++20 Seqlock Snapshot (Single Writer Thread 3 -> Reader Thread 2).
// Stores payload as lock-free std::atomic<uint64_t> words with acquire/release sequence framing,
// guaranteeing zero C++ memory-model data races and zero torn reads on x86_64 and ARM64.
template <typename T>
class alignas(64) SeqlockSnapshot {
  static_assert(std::is_trivially_copyable_v<T>, "SeqlockSnapshot requires trivially copyable POD");
  static_assert(std::atomic<uint64_t>::is_always_lock_free, "64-bit atomics must be lock-free");

 public:
  static constexpr std::size_t kWords = (sizeof(T) + sizeof(uint64_t) - 1U) / sizeof(uint64_t);

  SeqlockSnapshot() noexcept {
    for (auto& w : words_) {
      w.store(0ULL, std::memory_order_relaxed);
    }
  }

  SeqlockSnapshot(const SeqlockSnapshot&) = delete;
  SeqlockSnapshot& operator=(const SeqlockSnapshot&) = delete;

  void publish(const T& value) noexcept {
    std::array<uint64_t, kWords> buf{};
    std::memcpy(static_cast<void*>(buf.data()), static_cast<const void*>(&value), sizeof(T));

    const uint64_t s = seq_.load(std::memory_order_relaxed);
    seq_.store(s + 1ULL, std::memory_order_release);
    for (std::size_t i = 0U; i < kWords; ++i) {
      words_[i].store(buf[i], std::memory_order_release);
    }
    seq_.store(s + 2ULL, std::memory_order_release);
  }

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

  [[nodiscard]] uint64_t sequence() const noexcept {
    return seq_.load(std::memory_order_acquire);
  }

 private:
  alignas(64) std::atomic<uint64_t> seq_{0ULL};
  alignas(64) std::array<std::atomic<uint64_t>, kWords> words_{};
};

// Thread 1 (DDS) -> Thread 2 (100 Hz RT) telemetry ingress POD
struct TelemetryIngressFrame {
  uint64_t seq{0ULL};
  uint64_t stamp_ns{0ULL};
  double pos_ned[3]{0.0, 0.0, 0.0};
  float vel_ned[3]{0.0f, 0.0f, 0.0f};
  float q_frd_ned[4]{1.0f, 0.0f, 0.0f, 0.0f};
  float v_term_v{403.2f};
  float i_pack_a{0.0f};
  float soc{0.80f};
  float t_max_n{5886.0f};
  float a_xy_max{6.87f};
  float tilt_max_rad{0.45f};
  float vz_sink_max{3.0f};
  float chi2_residual{0.0f};
  uint8_t flight_regime{0U};
  uint8_t fault_bitmask{0U};
  uint8_t tmr_state{0U};
  uint8_t imu_health_mask{0x07U};
  uint64_t checksum{0ULL};
};

// Thread 2 (100 Hz RT) -> Thread 1 (DDS) command egress POD
struct RtaCommandEgressFrame {
  uint64_t seq{0ULL};
  uint64_t stamp_ns{0ULL};
  std::array<float, 3> accel_ned{0.0f, 0.0f, 0.0f};
  float thrust_des_n{0.0f};
  float yaw_sp{0.0f};
  float yawspeed_sp{0.0f};
  float vz_limit_mps{2.5f};
  float cbvf_min_margin_m{0.0f};
  float ltfj_margin_m{0.0f};
  float ltbx_margin_m{0.0f};
  uint8_t control_mode{0U};
  bool cel_latched{false};
  bool perception_stale{false};
  bool divert_requested{false};
  bool valid{false};
};

// Fixed-capacity trivially copyable hazard slot transported lock-free via Seqlock (Thread 3 -> Thread 2)
struct HazardPod {
  double pos_ned[3]{0.0, 0.0, 0.0};
  double vel_ned[3]{0.0, 0.0, 0.0};
  float radius_h_m{0.0f};
  float radius_z_m{0.0f};
  float gamma{1.2f};
  uint8_t kind{0U};  // 0=ESDF, 1=WIRE, 2=INTRUDER, 3=CTR, 4=TERRAIN
  uint8_t pad[3]{0U, 0U, 0U};
};

// Thread 3 (20 Hz ESDF / SLZ / Wire / Intruder) -> Thread 2 (100 Hz RT) Seqlock snapshot POD
struct PerceptionHazardSnapshot {
  static constexpr std::size_t kMaxHazards = 15U;

  uint64_t version{0ULL};
  uint64_t stamp_ns{0ULL};
  uint8_t hazard_count{0U};
  uint8_t slz_count{0U};
  bool stale{false};
  uint8_t active_decision_state{0U};
  float terrain_elev_m{0.0f};  // Local DEM elevation relative to Kartal datum [m]
  std::array<int32_t, 4> min_clearance_mm{0, 0, 0, 0};
  float best_slz_enu[3]{0.0f, 0.0f, 0.0f};
  float best_slz_p_safe{0.0f};
  float gimbal_pitch_rad{0.0f};
  float gimbal_yaw_rad{0.0f};
  std::array<HazardPod, kMaxHazards> hazards{};
  uint64_t checksum{0ULL};
};

// Computes DEM-referenced AGL height [m] relative to Kartal datum (never assumes flat sea-level ground)
[[nodiscard]] inline float compute_terrain_agl_m(
    double pos_down_ned_m,
    float terrain_elev_rel_datum_m) noexcept {
  const double alt_rel_datum_m = -pos_down_ned_m;
  const double agl = alt_rel_datum_m - static_cast<double>(terrain_elev_rel_datum_m);
  return (agl > 0.0) ? static_cast<float>(agl) : 0.0f;
}

// Analytical Maltepe-Dragos-Kartal coastal terrain elevation relative to Kartal datum (12 m AMSL).
// Positive inland (Dragos hill / Kartal hospital ridge up to +33 m), slightly negative toward Maltepe coast (-7 m).
[[nodiscard]] inline float estimate_kartal_corridor_terrain_rel_m(
    double east_m,
    double north_m) noexcept {
  // Coastal slope rising northeastward from Marmara shoreline (~0 m near coast, up to +33 m inland)
  const double inland_m = 0.62 * east_m + 0.78 * north_m;
  if (inland_m <= 0.0) {
    // Seaward / Maltepe reclaimed shore (~5 m AMSL => -7 m relative to 12 m Kartal datum)
    const double coast_drop = (inland_m < -1200.0) ? -7.0 : (7.0 * (inland_m / 1200.0));
    return static_cast<float>(coast_drop);
  }
  const double rise = (inland_m / 1800.0) * 33.0;
  return static_cast<float>((rise > 33.0) ? 33.0 : rise);
}

// Thread 2 (100 Hz RT) -> Thread 4 (Async .vrta Disk Writer) log record slot POD
struct VrtaLogRecordSlot {
  uint64_t seq{0ULL};
  uint64_t t_sim_us{0ULL};
  double pos_enu[3]{0.0, 0.0, 0.0};
  float vel_enu[3]{0.0f, 0.0f, 0.0f};
  float a_safe_enu[3]{0.0f, 0.0f, 0.0f};
  float v_term_v{403.2f};
  float soc_pct{80.0f};
  float cbvf_min_v{0.0f};
  float ltfj_margin_m{0.0f};
  float ltbx_margin_m{0.0f};
  float chi2_residual{0.0f};
  float agl_m{0.0f};
  uint8_t active_motor_mask{0xFFU};
  uint8_t flight_regime{0U};
  uint8_t tmr_state{0U};
  uint8_t cel_latched{0U};
};

// Unified 4-thread lock-free transport bundle for vertirta_rta_node
struct FourThreadRtaTransport {
  SpscRingBuffer<TelemetryIngressFrame, 256U> dds_to_rt_ring{};
  SpscRingBuffer<RtaCommandEgressFrame, 256U> rt_to_dds_ring{};
  SeqlockSnapshot<PerceptionHazardSnapshot> map_to_rt_seqlock{};
  SpscRingBuffer<VrtaLogRecordSlot, 512U> rt_to_disk_ring{};
};

// Best-effort real-time SCHED_FIFO promotion for Thread 2 (gracefully falls back without CAP_SYS_NICE)
inline bool try_configure_sched_fifo(int priority = 80) noexcept {
#if defined(__linux__)
  sched_param sp{};
  sp.sched_priority = priority;
  return pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0;
#else
  (void)priority;
  return false;
#endif
}

}  // namespace vertirta::ros
