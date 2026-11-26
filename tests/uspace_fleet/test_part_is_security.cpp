// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>

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

TEST_CASE("Part-IS Mode-B rejects signature downgrade attacks and CyberResilienceFsm fails closed on NaN/Inf") {
  const double qnan = std::numeric_limits<double>::quiet_NaN();
  const double pinf = std::numeric_limits<double>::infinity();
  const double ninf = -std::numeric_limits<double>::infinity();

  // 1. Mode-B Priority-0 signature downgrade MUST be rejected by header.flags_and_mode & 0x02U check!
  PartIsCryptoContext ctx(/*key_epoch=*/1U);
  uint8_t payload[64]{0x42U, 0x11U, 0x99U};
  SecuredPacket p0_pkt{};
  REQUIRE(ctx.seal_mode_b_priority0_signed(
      payload, sizeof(payload), /*nonce=*/10ULL, /*tai_ns=*/2'000'000'000ULL, p0_pkt));
  REQUIRE((p0_pkt.header.flags_and_mode & 0x02U) != 0U);
  REQUIRE(p0_pkt.has_ed25519_sig);

  // Attack A: strip unauthenticated bool has_ed25519_sig=false + corrupt signature bytes.
  SecuredPacket stripped_sig = p0_pkt;
  stripped_sig.ed25519_sig[0] ^= 0xFFU;
  stripped_sig.has_ed25519_sig = false;
  REQUIRE_FALSE(ctx.verify_mode_b(stripped_sig, /*now_tai_ns=*/2'000'000'000ULL));

  // Attack B: tamper flags_and_mode to clear bit 0x02U (AAD mismatch rejects).
  SecuredPacket cleared_flag = p0_pkt;
  cleared_flag.header.flags_and_mode &= ~0x02U;
  cleared_flag.has_ed25519_sig = false;
  REQUIRE_FALSE(ctx.verify_mode_b(cleared_flag, /*now_tai_ns=*/2'000'000'000ULL));

  // Attack C: tamper Ed25519 signature while leaving has_ed25519_sig=true.
  SecuredPacket bad_sig = p0_pkt;
  bad_sig.ed25519_sig[15] ^= 0x01U;
  REQUIRE_FALSE(ctx.verify_mode_b(bad_sig, /*now_tai_ns=*/2'000'000'000ULL));

  // Attack D: set has_ed25519_sig=true on an unsigned V2V packet (header bit 0x02U == 0).
  SecuredPacket v2v_pkt{};
  REQUIRE(ctx.seal_mode_b_v2v(
      payload, sizeof(payload), /*nonce=*/11ULL, /*tai_ns=*/2'010'000'000ULL, v2v_pkt));
  SecuredPacket forged_flag_bool = v2v_pkt;
  forged_flag_bool.has_ed25519_sig = true;
  REQUIRE_FALSE(ctx.verify_mode_b(forged_flag_bool, /*now_tai_ns=*/2'010'000'000ULL));

  // Original unmodified packets still verify cleanly (replay window was NOT polluted by rejected attacks).
  REQUIRE(ctx.verify_mode_b(p0_pkt, /*now_tai_ns=*/2'000'000'000ULL));
  REQUIRE(ctx.verify_mode_b(v2v_pkt, /*now_tai_ns=*/2'010'000'000ULL));

  // 2. Peer Mahalanobis anomaly gate MUST fail closed on NaN / +Inf / -Inf / negative sigma.
  for (const double bad_sigma : {qnan, pinf, ninf, -1.0}) {
    CyberResilienceFsm fsm{};
    const auto act = fsm.on_peer_kinematic_anomaly(/*peer_id=*/42U, bad_sigma);
    REQUIRE(fsm.state() == CyberState::SEC_SUSPECT_PEER);
    REQUIRE(act.demote_to_non_cooperative);
    REQUIRE(act.strip_priority0_rights);
  }

  // 3. USSP command check MUST quarantine fail-closed on NaN / Inf lon, AGL, or timestamp.
  struct BadUsspCase {
    double lon;
    double agl;
    double t;
  };
  const BadUsspCase bad_ussp_cases[] = {
      {qnan, 60.0, 10.0},
      {pinf, 60.0, 10.0},
      {ninf, 60.0, 10.0},
      {29.1500, qnan, 10.0},
      {29.1500, pinf, 10.0},
      {29.1500, ninf, 10.0},
      {29.1500, 60.0, qnan},
  };
  for (const auto& tc : bad_ussp_cases) {
    CyberResilienceFsm fsm{};
    const auto act = fsm.on_ussp_command_check(/*sig_valid=*/true, tc.lon, tc.agl, tc.t);
    REQUIRE(fsm.state() == CyberState::SEC_USSP_QUARANTINE);
    REQUIRE(act.drop_ussp_directives);
    REQUIRE(act.emit_is_or_230_draft);
  }

  // 4. Non-finite t_s in step_time while quarantined MUST NOT clear quarantine; escalates fail-closed to RTB.
  CyberResilienceFsm fsm{};
  (void)fsm.on_ussp_command_check(/*sig_valid=*/false, 29.1500, 60.0, 10.0);
  REQUIRE(fsm.state() == CyberState::SEC_USSP_QUARANTINE);
  const auto rtb_nan = fsm.step_time(qnan, /*ussp_auth_restored=*/true);
  REQUIRE(fsm.state() == CyberState::SEC_AUTONOMOUS_RTB);
  REQUIRE(rtb_nan.divert_to_nearest_fato_emg);
}
