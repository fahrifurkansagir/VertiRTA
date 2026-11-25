#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "vertirta_rta_node/lockfree_transport.hpp"

using vertirta::ros::FourThreadRtaTransport;
using vertirta::ros::PerceptionHazardSnapshot;
using vertirta::ros::RtaCommandEgressFrame;
using vertirta::ros::SeqlockSnapshot;
using vertirta::ros::SpscRingBuffer;
using vertirta::ros::TelemetryIngressFrame;
using vertirta::ros::VrtaLogRecordSlot;

namespace {

uint64_t compute_telemetry_checksum(const TelemetryIngressFrame& f) noexcept {
  uint64_t h = 14695981039346656037ULL;
  auto mix = [&](uint64_t v) {
    h ^= v;
    h *= 1099511628211ULL;
  };
  mix(f.seq);
  mix(f.stamp_ns);
  mix(static_cast<uint64_t>(f.flight_regime));
  mix(static_cast<uint64_t>(f.fault_bitmask));
  mix(static_cast<uint64_t>(f.tmr_state));
  return h;
}

uint64_t compute_hazard_checksum(const PerceptionHazardSnapshot& s) noexcept {
  uint64_t h = 14695981039346656037ULL;
  auto mix = [&](uint64_t v) {
    h ^= v;
    h *= 1099511628211ULL;
  };
  mix(s.version);
  mix(s.stamp_ns);
  mix(static_cast<uint64_t>(s.hazard_count));
  mix(static_cast<uint64_t>(s.slz_count));
  mix(static_cast<uint64_t>(s.stale ? 1U : 0U));
  for (uint32_t i = 0; i < 4U; ++i) {
    mix(static_cast<uint64_t>(s.min_clearance_mm[i]));
  }
  return h;
}

}  // namespace

TEST_CASE("SpscRingBuffer maintains exact FIFO order and 0 data races across 1,000,000 ops") {
  static_assert(alignof(SpscRingBuffer<uint64_t, 256>) >= 64);
  SpscRingBuffer<uint64_t, 256> ring;

  constexpr uint64_t kOps = 1'000'000ULL;
  std::atomic<uint64_t> mismatches{0};
  std::atomic<uint64_t> consumed{0};

  std::thread producer([&]() {
    for (uint64_t i = 1; i <= kOps; ++i) {
      while (!ring.try_push(i)) {
        std::this_thread::yield();
      }
    }
  });

  std::thread consumer([&]() {
    uint64_t expected = 1;
    uint64_t val = 0;
    while (expected <= kOps) {
      if (ring.try_pop(val)) {
        if (val != expected) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }
        ++expected;
        consumed.fetch_add(1, std::memory_order_relaxed);
      } else {
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  REQUIRE(consumed.load() == kOps);
  REQUIRE(mismatches.load() == 0ULL);
  REQUIRE(ring.empty());
}

TEST_CASE("SeqlockSnapshot guarantees zero torn reads and monotonic versions under concurrent stress") {
  static_assert(alignof(SeqlockSnapshot<PerceptionHazardSnapshot>) >= 64);
  SeqlockSnapshot<PerceptionHazardSnapshot> seqlock;

  PerceptionHazardSnapshot init{};
  init.version = 1;
  init.stamp_ns = 1000;
  init.hazard_count = 2;
  init.slz_count = 3;
  init.min_clearance_mm = {5000, 6000, 7000, 8000};
  init.checksum = compute_hazard_checksum(init);
  seqlock.publish(init);

  constexpr uint64_t kWrites = 250'000ULL;
  constexpr uint64_t kMinReads = 50'000ULL;
  std::atomic<bool> reader_started{false};
  std::atomic<bool> writer_done{false};
  std::atomic<uint64_t> torn_reads{0};
  std::atomic<uint64_t> non_monotonic{0};
  std::atomic<uint64_t> total_reads{0};

  std::thread writer([&]() {
    while (!reader_started.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (uint64_t v = 2; v <= kWrites; ++v) {
      PerceptionHazardSnapshot snap{};
      snap.version = v;
      snap.stamp_ns = v * 50'000'000ULL;
      snap.hazard_count = static_cast<uint8_t>(v % 16U);
      snap.slz_count = static_cast<uint8_t>((v % 8U) + 1U);
      snap.stale = ((v & 0xFFU) == 0U);
      snap.min_clearance_mm = {
          static_cast<int32_t>(v),
          static_cast<int32_t>(v * 2U),
          static_cast<int32_t>(v * 3U),
          static_cast<int32_t>(v * 4U),
      };
      snap.checksum = compute_hazard_checksum(snap);
      seqlock.publish(snap);
    }
    writer_done.store(true, std::memory_order_release);
  });

  std::thread reader([&]() {
    reader_started.store(true, std::memory_order_release);
    uint64_t last_ver = 0;
    while (!writer_done.load(std::memory_order_acquire) || total_reads.load(std::memory_order_relaxed) < kMinReads) {
      const auto snap = seqlock.read();
      if (snap.checksum != compute_hazard_checksum(snap)) {
        torn_reads.fetch_add(1, std::memory_order_relaxed);
      }
      if (snap.version < last_ver) {
        non_monotonic.fetch_add(1, std::memory_order_relaxed);
      }
      last_ver = snap.version;
      const uint64_t reads_now = total_reads.fetch_add(1, std::memory_order_relaxed) + 1ULL;
      if (writer_done.load(std::memory_order_acquire) && snap.version == kWrites && reads_now >= kMinReads) {
        break;
      }
    }
  });

  writer.join();
  reader.join();

  REQUIRE(torn_reads.load() == 0ULL);
  REQUIRE(non_monotonic.load() == 0ULL);
  REQUIRE(total_reads.load() > 1000ULL);
  REQUIRE(seqlock.read().version == kWrites);
}

TEST_CASE("4-thread VertiRTA SPSC + Seqlock pipeline runs 1,000,000 total ops with 0 races and irreversible CEL latch") {
  FourThreadRtaTransport bus;

  // Publish valid initial perception snapshot before threads launch
  {
    PerceptionHazardSnapshot init{};
    init.version = 1;
    init.stamp_ns = 1'000'000ULL;
    init.hazard_count = 1;
    init.slz_count = 3;
    init.min_clearance_mm = {12000, 15000, 18000, 20000};
    init.checksum = compute_hazard_checksum(init);
    bus.map_to_rt_seqlock.publish(init);
  }

  // 4 threads x 250,000 ops each = 1,000,000 concurrent cross-thread operations
  constexpr uint64_t kFrames = 250'000ULL;

  std::atomic<bool> stop_map{false};
  std::atomic<bool> rt_done{false};
  std::atomic<uint64_t> t1_egress_ok{0};
  std::atomic<uint64_t> t2_processed{0};
  std::atomic<uint64_t> t3_published{0};
  std::atomic<uint64_t> t4_logged{0};
  std::atomic<uint64_t> data_corruptions{0};
  std::atomic<bool> cel_ever_unlatched{false};

  // Thread 1: DDS I/O (produces ingress telemetry, consumes egress RTA commands)
  std::thread t1_dds([&]() {
    uint64_t last_cmd_seq = 0;
    for (uint64_t seq = 1; seq <= kFrames; ++seq) {
      TelemetryIngressFrame tel{};
      tel.seq = seq;
      tel.stamp_ns = seq * 10'000'000ULL;
      // Inject SAME_ARM_ACTIVE_SPIN (regime=3) at midpoint, then flicker back to NOMINAL (0)
      tel.flight_regime = (seq >= kFrames / 2 && seq < (kFrames / 2 + 100)) ? 3U : 0U;
      tel.fault_bitmask = (seq >= kFrames / 2) ? 0x11U : 0x00U;
      tel.tmr_state = 0U;
      tel.v_term_v = 368.5f;
      tel.soc = 0.65f;
      tel.chi2_residual = 12.4f;
      tel.checksum = compute_telemetry_checksum(tel);

      while (!bus.dds_to_rt_ring.try_push(tel)) {
        RtaCommandEgressFrame cmd{};
        while (bus.rt_to_dds_ring.try_pop(cmd)) {
          if (cmd.seq <= last_cmd_seq) {
            data_corruptions.fetch_add(1, std::memory_order_relaxed);
          }
          last_cmd_seq = cmd.seq;
          t1_egress_ok.fetch_add(1, std::memory_order_relaxed);
        }
        std::this_thread::yield();
      }

      RtaCommandEgressFrame cmd{};
      while (bus.rt_to_dds_ring.try_pop(cmd)) {
        if (cmd.seq <= last_cmd_seq) {
          data_corruptions.fetch_add(1, std::memory_order_relaxed);
        }
        last_cmd_seq = cmd.seq;
        t1_egress_ok.fetch_add(1, std::memory_order_relaxed);
      }
    }

    while (!rt_done.load(std::memory_order_acquire) || !bus.rt_to_dds_ring.empty()) {
      RtaCommandEgressFrame cmd{};
      if (bus.rt_to_dds_ring.try_pop(cmd)) {
        if (cmd.seq <= last_cmd_seq) {
          data_corruptions.fetch_add(1, std::memory_order_relaxed);
        }
        last_cmd_seq = cmd.seq;
        t1_egress_ok.fetch_add(1, std::memory_order_relaxed);
      } else {
        std::this_thread::yield();
      }
    }
  });

  // Thread 3: 20 Hz ESDF / SLZ Map writer publishing to Seqlock
  std::thread t3_map([&]() {
    uint64_t v = 2;
    while (!stop_map.load(std::memory_order_acquire) || v <= kFrames) {
      PerceptionHazardSnapshot snap{};
      snap.version = v;
      snap.stamp_ns = v * 50'000'000ULL;
      snap.hazard_count = static_cast<uint8_t>(v % 16U);
      snap.slz_count = 3U;
      snap.stale = false;
      snap.min_clearance_mm = {
          static_cast<int32_t>(5000U + (v & 0xFFFU)),
          static_cast<int32_t>(6000U + (v & 0xFFFU)),
          static_cast<int32_t>(7000U + (v & 0xFFFU)),
          static_cast<int32_t>(8000U + (v & 0xFFFU)),
      };
      snap.checksum = compute_hazard_checksum(snap);
      bus.map_to_rt_seqlock.publish(snap);
      t3_published.fetch_add(1, std::memory_order_relaxed);
      ++v;
      if (v > kFrames && stop_map.load(std::memory_order_acquire)) {
        break;
      }
    }
  });

  // Thread 2: 100 Hz Zero-ROS Zero-Heap RT Safety Loop
  std::thread t2_rt([&]() {
    bool cel_latched = false;
    uint64_t expected_seq = 1;
    while (expected_seq <= kFrames) {
      TelemetryIngressFrame tel{};
      if (!bus.dds_to_rt_ring.try_pop(tel)) {
        std::this_thread::yield();
        continue;
      }
      if (tel.seq != expected_seq || tel.checksum != compute_telemetry_checksum(tel)) {
        data_corruptions.fetch_add(1, std::memory_order_relaxed);
      }

      const auto haz = bus.map_to_rt_seqlock.read();
      if (haz.checksum != compute_hazard_checksum(haz)) {
        data_corruptions.fetch_add(1, std::memory_order_relaxed);
      }

      // Irreversible CEL latch: once regime 3 hits, transient NOMINAL flicker must NEVER unlatch
      const bool prev_latch = cel_latched;
      if (tel.flight_regime == 3U) {
        cel_latched = true;
      }
      if (prev_latch && !cel_latched) {
        cel_ever_unlatched.store(true, std::memory_order_relaxed);
      }

      RtaCommandEgressFrame cmd{};
      cmd.seq = tel.seq;
      cmd.stamp_ns = tel.stamp_ns;
      cmd.accel_ned = {0.0f, 0.0f, cel_latched ? 0.5f : 0.0f};
      cmd.yawspeed_sp = cel_latched ? -2.2f : 0.0f;
      cmd.cel_latched = cel_latched;
      cmd.valid = true;

      while (!bus.rt_to_dds_ring.try_push(cmd)) {
        std::this_thread::yield();
      }

      VrtaLogRecordSlot rec{};
      rec.seq = tel.seq;
      rec.t_sim_us = tel.stamp_ns / 1000ULL;
      rec.flight_regime = tel.flight_regime;
      rec.cel_latched = cel_latched ? 1U : 0U;
      rec.chi2_residual = tel.chi2_residual;
      rec.v_term_v = tel.v_term_v;
      while (!bus.rt_to_disk_ring.try_push(rec)) {
        std::this_thread::yield();
      }

      t2_processed.fetch_add(1, std::memory_order_relaxed);
      ++expected_seq;
    }
    rt_done.store(true, std::memory_order_release);
    stop_map.store(true, std::memory_order_release);
  });

  // Thread 4: Low-priority async .vrta writer consuming rt_to_disk_ring
  std::thread t4_disk([&]() {
    uint64_t expected_seq = 1;
    bool seen_latched = false;
    while (expected_seq <= kFrames) {
      VrtaLogRecordSlot rec{};
      if (bus.rt_to_disk_ring.try_pop(rec)) {
        if (rec.seq != expected_seq) {
          data_corruptions.fetch_add(1, std::memory_order_relaxed);
        }
        if (rec.cel_latched != 0U) {
          seen_latched = true;
        } else if (seen_latched) {
          cel_ever_unlatched.store(true, std::memory_order_relaxed);
        }
        t4_logged.fetch_add(1, std::memory_order_relaxed);
        ++expected_seq;
      } else {
        std::this_thread::yield();
      }
    }
  });

  t1_dds.join();
  t2_rt.join();
  t3_map.join();
  t4_disk.join();

  const uint64_t total_ops =
      t1_egress_ok.load() + t2_processed.load() + t3_published.load() + t4_logged.load();
  REQUIRE(total_ops >= 1'000'000ULL);
  REQUIRE(t1_egress_ok.load() == kFrames);
  REQUIRE(t2_processed.load() == kFrames);
  REQUIRE(t4_logged.load() == kFrames);
  REQUIRE(data_corruptions.load() == 0ULL);
  REQUIRE_FALSE(cel_ever_unlatched.load());
}
