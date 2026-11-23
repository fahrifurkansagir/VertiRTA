// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

#include "vertirta/uspace/part_is_security.hpp"

using namespace vertirta::uspace;

TEST_CASE("40 B VertiRtaSecureHeader + ReplayWindow128 closes OMG DDS Security 1.2 replay gap") {
  static_assert(sizeof(VertiRtaSecureHeader) == 40U, "VertiRtaSecureHeader must be exactly 40 bytes");
  REQUIRE(check_libsodium_min_version(1U, 0U, 21U));
  REQUIRE(check_libsodium_min_version(1U, 0U, 22U));
  REQUIRE_FALSE(check_libsodium_min_version(1U, 0U, 20U));  // Dec 2025 Ed25519 subgroup CVE ban
  REQUIRE_FALSE(check_libsodium_min_version(1U, 0U, 18U));  // Ubuntu 24.04 apt CVE ban

  ReplayWindow128 win{};
  const uint64_t t_now_ns = 1'760'000'000'000'000'000ULL;

  // Accept fresh monotonic nonces
  REQUIRE(win.check_and_accept(100ULL, t_now_ns, t_now_ns) == ReplayStatus::ACCEPT);
  REQUIRE(win.check_and_accept(105ULL, t_now_ns + 50'000'000ULL, t_now_ns + 50'000'000ULL) == ReplayStatus::ACCEPT);
  // Accept slightly out-of-order within 128-packet window
  REQUIRE(win.check_and_accept(102ULL, t_now_ns + 60'000'000ULL, t_now_ns + 55'000'000ULL) == ReplayStatus::ACCEPT);
  // Reject exact replay (same nonce)
  REQUIRE(win.check_and_accept(102ULL, t_now_ns + 70'000'000ULL, t_now_ns + 65'000'000ULL) == ReplayStatus::REJECT_REPLAY);
  // Reject stale TAI clock skew (> +/-250 ms)
  REQUIRE(win.check_and_accept(106ULL, t_now_ns + 500'000'000ULL, t_now_ns) == ReplayStatus::REJECT_TAI_SKEW);
  // Reject out-of-window ancient nonce
  REQUIRE(win.check_and_accept(250ULL, t_now_ns, t_now_ns) == ReplayStatus::ACCEPT);
  REQUIRE(win.check_and_accept(110ULL, t_now_ns, t_now_ns) == ReplayStatus::REJECT_REPLAY);
}

TEST_CASE("Mode B crypto and CyberResilienceFsm isolate spoofed peers and quarantine rogue USSP within RTO <= 10 s") {
  PartIsCryptoContext ctx(/*key_epoch=*/1U);
  uint8_t payload[64]{0x42U, 0x11U};
  SecuredPacket pkt{};
  REQUIRE(ctx.seal_mode_b_v2v(payload, sizeof(payload), /*nonce=*/1ULL, /*tai_ns=*/1'000'000'000ULL, pkt));
  REQUIRE(ctx.verify_mode_b(pkt, /*now_tai_ns=*/1'000'000'000ULL));

  // Replay of identical authentic packet rejected by ReplayWindow128
  REQUIRE_FALSE(ctx.verify_mode_b(pkt, /*now_tai_ns=*/1'000'000'000ULL));

  // Signed Priority-0 / RID packet verifies on fresh nonce, fails on 1-bit tamper
  SecuredPacket p0_pkt{};
  REQUIRE(ctx.seal_mode_b_priority0_signed(payload, sizeof(payload), /*nonce=*/2ULL, /*tai_ns=*/1'010'000'000ULL, p0_pkt));
  REQUIRE(p0_pkt.has_ed25519_sig);
  REQUIRE(ctx.verify_mode_b(p0_pkt, /*now_tai_ns=*/1'010'000'000ULL));

  SecuredPacket tampered{};
  REQUIRE(ctx.seal_mode_b_v2v(payload, sizeof(payload), /*nonce=*/3ULL, /*tai_ns=*/1'020'000'000ULL, tampered));
  tampered.payload[0] ^= 0x01U;
  REQUIRE_FALSE(ctx.verify_mode_b(tampered, /*now_tai_ns=*/1'020'000'000ULL));

  // Compromised-key peer failing 3-sigma Mahalanobis gate -> demoted to non-cooperative (beta_ij=1.0, P0 stripped)
  CyberResilienceFsm fsm{};
  const auto peer_act = fsm.on_peer_kinematic_anomaly(/*peer_id=*/7U, /*mahalanobis_sigma=*/4.8);
  REQUIRE(fsm.state() == CyberState::SEC_SUSPECT_PEER);
  REQUIRE(peer_act.demote_to_non_cooperative);
  REQUIRE(peer_act.strip_priority0_rights);

  // Spoofed USSP commanding diversion east of 29.1895 E into LTFJ CTR -> instant quarantine + RTB <= 10 s
  fsm.reset();
  const auto ussp_act = fsm.on_ussp_command_check(
      /*sig_valid=*/true, /*cmd_lon_deg=*/29.2100, /*cmd_agl_m=*/80.0, /*t_s=*/10.0);
  REQUIRE(fsm.state() == CyberState::SEC_USSP_QUARANTINE);
  REQUIRE(ussp_act.drop_ussp_directives);
  REQUIRE(ussp_act.emit_is_or_230_draft);

  const auto rtb_act = fsm.step_time(/*t_s=*/20.0, /*ussp_auth_restored=*/false);  // 10.0 s elapsed == RTO
  REQUIRE(fsm.state() == CyberState::SEC_AUTONOMOUS_RTB);
  REQUIRE(rtb_act.divert_to_nearest_fato_emg);
}
