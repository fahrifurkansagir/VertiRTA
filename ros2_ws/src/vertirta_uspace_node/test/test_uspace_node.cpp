// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>
#include <string>

#include "vertirta_uspace_node/uspace_core.hpp"

using namespace vertirta::uspace_node;

TEST_CASE("ASTM F3548-21 bijective priority map, irreversible Contingent state, and InterUSS #1549 NOTAM tiling") {
  REQUIRE(to_astm_priority(RtaPriorityClass::PRIORITY_0_EMERGENCY_CEL) == 100);
  REQUIRE(to_astm_priority(RtaPriorityClass::PRIORITY_1_REVENUE) == 10);
  REQUIRE(to_astm_priority(RtaPriorityClass::PRIORITY_2_REPOSITION) == 5);
  REQUIRE(from_astm_priority(100) == RtaPriorityClass::PRIORITY_0_EMERGENCY_CEL);
  REQUIRE(from_astm_priority(10) == RtaPriorityClass::PRIORITY_1_REVENUE);
  REQUIRE(from_astm_priority(5) == RtaPriorityClass::PRIORITY_2_REPOSITION);

  UspaceCore core{};
  // Reject OIV creation east of 29.1895 E (LTFJ CTR breach)
  const auto rejected = core.create_oiv("UA-BAD", RtaPriorityClass::PRIORITY_1_REVENUE, 40.8846, 29.2057, 70.0);
  REQUIRE(rejected.state == OivState::NONE);

  const auto acc = core.create_oiv("UA-01", RtaPriorityClass::PRIORITY_1_REVENUE, /*lat=*/40.90, /*lon=*/29.15, /*alt_msl_m=*/70.0);
  REQUIRE(acc.state == OivState::ACCEPTED);
  REQUIRE(acc.alt_w84_m > 106.0);  // Explicit EGM2008 +36.7 m applied for ASTM W84

  REQUIRE(core.transition_oiv("UA-01", OivState::ACTIVATED, /*omit_ovn=*/false));
  // Trigger Priority-0 CEL Contingent: omits OVN key for zero-block DSS write, irreversible!
  REQUIRE(core.trigger_contingent_preemption("UA-01", "VPT-CVZ"));
  REQUIRE(core.oiv_state("UA-01") == OivState::CONTINGENT);
  REQUIRE(core.oiv_astm_priority("UA-01") == 100);
  REQUIRE(core.oiv_omitted_ovn("UA-01"));
  REQUIRE_FALSE(core.transition_oiv("UA-01", OivState::ACTIVATED, false));  // Irreversible
  REQUIRE(core.transition_oiv("UA-01", OivState::ENDED, true));

  // S2 Level-13 coarse DAR false-positive filter in local Kartal ENU 4D
  const Volume4DEnu own{-100.0, 100.0, -100.0, 100.0, 65.0, 75.0, 0.0, 60.0};
  const Volume4DEnu dar_non_overlapping{-100.0, 100.0, -100.0, 100.0, 100.0, 110.0, 0.0, 60.0};
  const Volume4DEnu dar_overlapping{-50.0, 50.0, -50.0, 50.0, 70.0, 80.0, 30.0, 90.0};
  REQUIRE_FALSE(volumes_intersect_4d_enu(own, dar_non_overlapping));
  REQUIRE(volumes_intersect_4d_enu(own, dar_overlapping));

  // InterUSS #1549 guard: wide Istanbul NOTAM (> 2,500 km^2) auto-tiled into sub-100 km^2 tiles
  const auto tiles = tile_notam_polygon_km2(/*area_km2=*/3200.0, /*max_tile_km2=*/100.0);
  REQUIRE(tiles.size() >= 32U);
  for (double a : tiles) {
    REQUIRE(a <= 100.0);
    REQUIRE(a > 0.0);
  }
}

TEST_CASE("ASTM F3411-22a + RFC 9575 Extended DRIP Wrapper packs exactly 228 bytes (9 x 25 B + 3 B hdr)") {
  const auto pack = encode_f3411_drip_message_pack("VRTA-X8-01", 40.9038, 29.1545, 70.0, /*cel_emg=*/true);
  REQUIRE(pack.size() == 228U);  // 3 B MsgPack hdr + 9 x 25 B messages (4 core + 5 DRIP Auth pages)
  REQUIRE(pack[0] == 0xF2U);     // ProtoVersion = 2 (F3411-22a), MsgType = 0xF (MessagePack)
  REQUIRE(pack[1] == 25U);       // 25 B per message
  REQUIRE(pack[2] == 9U);        // Exactly 9 messages

  // Verify Self-ID emergency string "EMG-CEL-SPIN-VPT-CVZ" in Msg 2 [53..77]
  const std::string self_id(reinterpret_cast<const char*>(pack.data() + 55U), 20U);
  REQUIRE(self_id == "EMG-CEL-SPIN-VPT-CVZ");

  // Verify RFC 9575 Extended DRIP Wrapper IANA SAM ID = 0x02 in Auth page 0 [103..127]
  REQUIRE(pack[103] == 0x22U);
  REQUIRE(pack[104] == 0x50U);  // AuthType=5, Page=0
  REQUIRE(pack[105] == 4U);     // LastPageIndex = 4 (5 pages total)
  REQUIRE(pack[106] == 109U);   // 109 B wrapper payload
  REQUIRE(pack[111] == 0x02U);  // IANA SAM 0x02 (DRIP Wrapper)
}
