// SPDX-License-Identifier: Apache-2.0
//! Cross-platform live kernel UDP loopback 2,500-packet adversarial proof
//! (exact workload parity with `tools/mavlink_rta_bridge/src/udp_bridge.cpp`).

use std::net::{SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::codec::{
    decode_frame, encode_message, AttitudeQuaternion, CommandLong, Heartbeat, LocalPositionNed,
    MavMessage, SetPositionTargetLocalNed, SysStatus, MAX_FRAME_BYTES,
    MAV_CMD_COMPONENT_ARM_DISARM, MAV_CMD_DO_FLIGHT_TERMINATION, MAV_FRAME_LOCAL_NED,
    MAV_MODE_FLAG_SAFETY_ARMED, MAV_RESULT_DENIED, TYPE_MASK_FORCE, TYPE_MASK_IGNORE_AFX,
    TYPE_MASK_IGNORE_AFY, TYPE_MASK_IGNORE_AFZ, TYPE_MASK_IGNORE_YAW, TYPE_MASK_IGNORE_Z,
    TYPE_MASK_VELOCITY_ONLY,
};
use crate::shield::{OffboardRtaShield, ShieldConfig, ShieldStats};

#[derive(Debug, Clone)]
pub struct LatencyStatsUs {
    pub sample_count: usize,
    pub min_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
    pub mean_us: f64,
}

#[derive(Debug, Clone)]
pub struct UdpLoopbackStressReport {
    pub passed: bool,
    pub total_offboard_packets_sent: u64,
    pub nominal_sent: u64,
    pub plunge_overspeed_sent: u64,
    pub ltfj_nfz_sent: u64,
    pub hospital_nfz_sent: u64,
    pub ltbx_nfz_sent: u64,
    pub airborne_disarm_sent: u64,
    pub flight_term_sent: u64,
    pub corrupted_crc_sent: u64,
    pub deadman_timeout_sent: u64,
    pub fcu_valid_packets_received: u64,
    pub fcu_unsafe_packets_leaked: u64,
    pub fcu_corrupted_packets_leaked: u64,
    pub valid_packets_dropped: u64,
    pub planner_acks_denied_received: u64,
    pub planner_statustext_alerts_received: u64,
    pub shield_stats: ShieldStats,
    pub latency_us: LatencyStatsUs,
    pub elapsed_ms: f64,
}

fn compute_latency_stats(mut samples: Vec<f64>) -> LatencyStatsUs {
    if samples.is_empty() {
        return LatencyStatsUs {
            sample_count: 0,
            min_us: 0.0,
            p50_us: 0.0,
            p95_us: 0.0,
            p99_us: 0.0,
            max_us: 0.0,
            mean_us: 0.0,
        };
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let n = samples.len();
    let pct = |q: f64| -> f64 {
        let idx = q * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = idx - lo as f64;
        samples[lo] * (1.0 - frac) + samples[hi] * frac
    };
    let sum: f64 = samples.iter().sum();
    LatencyStatsUs {
        sample_count: n,
        min_us: samples[0],
        p50_us: pct(0.50),
        p95_us: pct(0.95),
        p99_us: pct(0.99),
        max_us: samples[n - 1],
        mean_us: sum / n as f64,
    }
}

fn bind_loopback() -> std::io::Result<(UdpSocket, SocketAddr)> {
    let sock = UdpSocket::bind("127.0.0.1:0")?;
    let addr = sock.local_addr()?;
    Ok((sock, addr))
}

pub fn run_udp_loopback_stress_proof(total_packets: u32) -> UdpLoopbackStressReport {
    let t_start = Instant::now();
    let n_packets = total_packets.max(2000);

    let (fcu_sock, fcu_addr) = bind_loopback().expect("bind fcu_sock");
    let (bridge_fcu_sock, bridge_fcu_addr) = bind_loopback().expect("bind bridge_fcu_sock");
    let (bridge_off_sock, bridge_off_addr) = bind_loopback().expect("bind bridge_off_sock");
    let (planner_sock, planner_addr) = bind_loopback().expect("bind planner_sock");

    fcu_sock
        .set_read_timeout(Some(Duration::from_millis(2)))
        .expect("fcu timeout");
    bridge_fcu_sock
        .set_nonblocking(true)
        .expect("bridge_fcu nonblocking");
    bridge_off_sock
        .set_nonblocking(true)
        .expect("bridge_off nonblocking");
    planner_sock
        .set_nonblocking(true)
        .expect("planner nonblocking");

    let cfg = ShieldConfig::default();
    let stop_bridge = Arc::new(AtomicBool::new(false));
    let stop_fcu = Arc::new(AtomicBool::new(false));
    let pause_hb = Arc::new(AtomicBool::new(false));
    let zone = Arc::new(AtomicI32::new(0));

    let fcu_rx_valid = Arc::new(AtomicU64::new(0));
    let fcu_unsafe_leaked = Arc::new(AtomicU64::new(0));
    let fcu_corrupt_leaked = Arc::new(AtomicU64::new(0));

    // Thread 1: MockFCU (emits 50 Hz telemetry + audits every received command)
    let fcu_handle = {
        let stop_fcu = Arc::clone(&stop_fcu);
        let pause_hb = Arc::clone(&pause_hb);
        let zone = Arc::clone(&zone);
        let fcu_rx_valid = Arc::clone(&fcu_rx_valid);
        let fcu_unsafe_leaked = Arc::clone(&fcu_unsafe_leaked);
        let fcu_corrupt_leaked = Arc::clone(&fcu_corrupt_leaked);
        let fcu_sock_tx = fcu_sock.try_clone().expect("clone fcu_sock");

        thread::spawn(move || {
            let mut rx_buf = [0u8; MAX_FRAME_BYTES];
            let mut fcu_seq = 0u8;
            let mut last_telem = Instant::now() - Duration::from_millis(100);
            let t0 = Instant::now();

            while !stop_fcu.load(Ordering::Relaxed) {
                if !pause_hb.load(Ordering::Relaxed)
                    && last_telem.elapsed() >= Duration::from_millis(20)
                {
                    last_telem = Instant::now();
                    let hb = encode_message(
                        &MavMessage::Heartbeat(Heartbeat {
                            mav_type: 2,
                            autopilot: 12,
                            base_mode: MAV_MODE_FLAG_SAFETY_ARMED,
                            system_status: 4,
                            mavlink_version: 3,
                            ..Default::default()
                        }),
                        fcu_seq,
                        1,
                        1,
                    );
                    fcu_seq = fcu_seq.wrapping_add(1);
                    let _ = fcu_sock_tx.send_to(&hb, bridge_fcu_addr);

                    let (x, y, z) = match zone.load(Ordering::Relaxed) {
                        1 => (500.0, 90.0, -60.0),
                        2 => (3200.0, -1165.0, -60.0),
                        3 => (6020.0, 2365.0, -60.0),
                        _ => (500.0, -400.0, -60.0),
                    };
                    let lp = encode_message(
                        &MavMessage::LocalPositionNed(LocalPositionNed {
                            time_boot_ms: t0.elapsed().as_millis() as u32,
                            x,
                            y,
                            z,
                            vx: 4.0,
                            vy: 0.0,
                            vz: 0.0,
                        }),
                        fcu_seq,
                        1,
                        1,
                    );
                    fcu_seq = fcu_seq.wrapping_add(1);
                    let _ = fcu_sock_tx.send_to(&lp, bridge_fcu_addr);
                }

                if let Ok((n, _)) = fcu_sock.recv_from(&mut rx_buf) {
                    match decode_frame(&rx_buf[..n]) {
                        Err(_) => {
                            fcu_corrupt_leaked.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(dec) => {
                            fcu_rx_valid.fetch_add(1, Ordering::Relaxed);
                            match dec.message {
                                MavMessage::CommandLong(cmd) => {
                                    if cmd.command == MAV_CMD_DO_FLIGHT_TERMINATION
                                        || (cmd.command == MAV_CMD_COMPONENT_ARM_DISARM
                                            && (!cmd.param1.is_finite()
                                                || (cmd.param1 - 1.0).abs() >= 0.5))
                                    {
                                        fcu_unsafe_leaked.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                                MavMessage::SetPositionTargetLocalNed(sp) => {
                                    let v_xy = f64::from(sp.vx).hypot(f64::from(sp.vy));
                                    let bad_speed = v_xy > cfg.max_horiz_speed_mps + 0.05
                                        || f64::from(sp.vz) > cfg.max_descent_rate_mps + 0.05
                                        || f64::from(sp.vz) < -cfg.max_ascent_rate_mps - 0.05;
                                    let bad_alt = (sp.type_mask & TYPE_MASK_IGNORE_Z) == 0
                                        && f64::from(sp.z) > -cfg.min_agl_m + 0.05;
                                    let bad_frame = sp.coordinate_frame != MAV_FRAME_LOCAL_NED
                                        || (sp.type_mask & TYPE_MASK_FORCE) != 0;
                                    if !sp.vx.is_finite()
                                        || !sp.vy.is_finite()
                                        || !sp.vz.is_finite()
                                        || bad_speed
                                        || bad_alt
                                        || bad_frame
                                    {
                                        fcu_unsafe_leaked.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        })
    };

    // Thread 2: Rust OffboardRtaShield Bridge
    let bridge_handle = {
        let stop_bridge = Arc::clone(&stop_bridge);
        thread::spawn(move || {
            let mut shield = OffboardRtaShield::new(cfg);
            let mut latencies_us = Vec::with_capacity(n_packets as usize + 128);
            let mut rx_buf = [0u8; MAX_FRAME_BYTES];
            let t0 = Instant::now();

            while !stop_bridge.load(Ordering::Relaxed) {
                let mut did_work = false;
                while let Ok((n, _)) = bridge_fcu_sock.recv_from(&mut rx_buf) {
                    did_work = true;
                    let now_us = t0.elapsed().as_micros() as u64;
                    let out = shield.process_fcu_packet(&rx_buf[..n], now_us);
                    for pkt in &out.to_offboard {
                        let _ = bridge_off_sock.send_to(pkt, planner_addr);
                    }
                }
                while let Ok((n, _)) = bridge_off_sock.recv_from(&mut rx_buf) {
                    did_work = true;
                    let ts0 = Instant::now();
                    let now_us = t0.elapsed().as_micros() as u64;
                    let out = shield.process_offboard_packet(&rx_buf[..n], now_us);
                    for pkt in &out.to_fcu {
                        let _ = bridge_fcu_sock.send_to(pkt, fcu_addr);
                    }
                    for pkt in &out.to_offboard {
                        let _ = bridge_off_sock.send_to(pkt, planner_addr);
                    }
                    latencies_us.push(ts0.elapsed().as_nanos() as f64 * 1e-3);
                }
                if !did_work {
                    thread::sleep(Duration::from_micros(50));
                }
            }
            (shield.stats(), latencies_us)
        })
    };

    let mut planner_acks_denied = 0u64;
    let mut planner_statustext = 0u64;
    let mut rx_drain = [0u8; MAX_FRAME_BYTES];
    let mut drain_planner = |acks: &mut u64, st: &mut u64| {
        while let Ok((n, _)) = planner_sock.recv_from(&mut rx_drain) {
            if let Ok(dec) = decode_frame(&rx_drain[..n]) {
                match dec.message {
                    MavMessage::CommandAck(a) if a.result == MAV_RESULT_DENIED => *acks += 1,
                    MavMessage::StatusText(_) => *st += 1,
                    _ => {}
                }
            }
        }
    };

    // Wait for initial FCU HEARTBEAT + LOCAL_POSITION_NED + exercise SYS_STATUS & ATTITUDE_QUATERNION
    thread::sleep(Duration::from_millis(60));
    let _ = planner_sock.send_to(
        &encode_message(
            &MavMessage::SysStatus(SysStatus {
                voltage_battery: 40_320,
                battery_remaining: 82,
                ..Default::default()
            }),
            200,
            1,
            1,
        ),
        bridge_fcu_addr,
    );
    let _ = planner_sock.send_to(
        &encode_message(
            &MavMessage::AttitudeQuaternion(AttitudeQuaternion {
                time_boot_ms: 1000,
                q1: 1.0,
                ..Default::default()
            }),
            201,
            1,
            1,
        ),
        bridge_fcu_addr,
    );
    thread::sleep(Duration::from_millis(15));
    drain_planner(&mut planner_acks_denied, &mut planner_statustext);

    let mut p_seq = 0u8;
    let mut nominal_sent = 0u64;
    let mut plunge_overspeed_sent = 0u64;
    let mut ltfj_nfz_sent = 0u64;
    let mut hospital_nfz_sent = 0u64;
    let mut ltbx_nfz_sent = 0u64;
    let mut airborne_disarm_sent = 0u64;
    let mut flight_term_sent = 0u64;
    let mut corrupted_crc_sent = 0u64;
    let mut deadman_timeout_sent = 0u64;

    let deadman_pkts = 40u32;
    let active_pkts = n_packets - deadman_pkts;
    let phase_a = (active_pkts * 65) / 100;
    let phase_b = (active_pkts * 15) / 100;
    let phase_c = (active_pkts * 10) / 100;
    let phase_d = active_pkts - phase_a - phase_b - phase_c;

    let mut send_paced = |pkt: &[u8], idx: u32, acks: &mut u64, st: &mut u64| {
        let _ = planner_sock.send_to(pkt, bridge_off_addr);
        if idx.is_multiple_of(16) {
            drain_planner(acks, st);
            thread::sleep(Duration::from_micros(150));
        }
    };

    // Phase A: Nominal, Plunge/Overspeed, Airborne Disarm, Flight Termination, Bad CRC
    for i in 0..phase_a {
        let slot = i % 20;
        let pkt = if slot < 10 {
            nominal_sent += 1;
            encode_message(
                &MavMessage::SetPositionTargetLocalNed(SetPositionTargetLocalNed {
                    time_boot_ms: i,
                    vx: 5.0,
                    vy: -1.0,
                    vz: 0.0,
                    type_mask: TYPE_MASK_VELOCITY_ONLY,
                    ..Default::default()
                }),
                p_seq,
                255,
                190,
            )
        } else if slot < 14 {
            plunge_overspeed_sent += 1;
            encode_message(
                &MavMessage::SetPositionTargetLocalNed(SetPositionTargetLocalNed {
                    time_boot_ms: i,
                    x: 500.0,
                    y: -400.0,
                    z: 40.0,
                    vx: 55.0,
                    vy: 0.0,
                    vz: 25.0,
                    type_mask: TYPE_MASK_IGNORE_AFX
                        | TYPE_MASK_IGNORE_AFY
                        | TYPE_MASK_IGNORE_AFZ
                        | TYPE_MASK_IGNORE_YAW,
                    ..Default::default()
                }),
                p_seq,
                255,
                190,
            )
        } else if slot < 16 {
            airborne_disarm_sent += 1;
            encode_message(
                &MavMessage::CommandLong(CommandLong {
                    command: MAV_CMD_COMPONENT_ARM_DISARM,
                    param1: 0.0,
                    ..Default::default()
                }),
                p_seq,
                255,
                190,
            )
        } else if slot == 16 {
            flight_term_sent += 1;
            encode_message(
                &MavMessage::CommandLong(CommandLong {
                    command: MAV_CMD_DO_FLIGHT_TERMINATION,
                    param1: 1.0,
                    ..Default::default()
                }),
                p_seq,
                255,
                190,
            )
        } else {
            corrupted_crc_sent += 1;
            let mut bad = encode_message(
                &MavMessage::SetPositionTargetLocalNed(SetPositionTargetLocalNed {
                    time_boot_ms: i,
                    vx: 40.0,
                    type_mask: TYPE_MASK_VELOCITY_ONLY,
                    ..Default::default()
                }),
                p_seq,
                255,
                190,
            );
            let last = bad.len() - 1;
            bad[last] ^= 0x5A;
            bad
        };
        p_seq = p_seq.wrapping_add(1);
        send_paced(&pkt, i, &mut planner_acks_denied, &mut planner_statustext);
    }

    // Phase B: LTFJ CTR breach
    zone.store(1, Ordering::Relaxed);
    thread::sleep(Duration::from_millis(50));
    for i in 0..phase_b {
        ltfj_nfz_sent += 1;
        let pkt = encode_message(
            &MavMessage::SetPositionTargetLocalNed(SetPositionTargetLocalNed {
                time_boot_ms: phase_a + i,
                x: 500.0,
                y: 450.0,
                z: -60.0,
                vx: 0.0,
                vy: 20.0,
                vz: 0.0,
                type_mask: TYPE_MASK_IGNORE_AFX
                    | TYPE_MASK_IGNORE_AFY
                    | TYPE_MASK_IGNORE_AFZ
                    | TYPE_MASK_IGNORE_YAW,
                ..Default::default()
            }),
            p_seq,
            255,
            190,
        );
        p_seq = p_seq.wrapping_add(1);
        send_paced(&pkt, i, &mut planner_acks_denied, &mut planner_statustext);
    }

    // Phase C: Kartal Hospital helipad NFZ breach
    zone.store(2, Ordering::Relaxed);
    thread::sleep(Duration::from_millis(50));
    for i in 0..phase_c {
        hospital_nfz_sent += 1;
        let pkt = encode_message(
            &MavMessage::SetPositionTargetLocalNed(SetPositionTargetLocalNed {
                time_boot_ms: phase_a + phase_b + i,
                x: 3375.0,
                y: -1165.0,
                z: -60.0,
                vx: 18.0,
                vy: 0.0,
                vz: 0.0,
                type_mask: TYPE_MASK_IGNORE_AFX
                    | TYPE_MASK_IGNORE_AFY
                    | TYPE_MASK_IGNORE_AFZ
                    | TYPE_MASK_IGNORE_YAW,
                ..Default::default()
            }),
            p_seq,
            255,
            190,
        );
        p_seq = p_seq.wrapping_add(1);
        send_paced(&pkt, i, &mut planner_acks_denied, &mut planner_statustext);
    }

    // Phase D: LTBX CTR breach
    zone.store(3, Ordering::Relaxed);
    thread::sleep(Duration::from_millis(50));
    for i in 0..phase_d {
        ltbx_nfz_sent += 1;
        let pkt = encode_message(
            &MavMessage::SetPositionTargetLocalNed(SetPositionTargetLocalNed {
                time_boot_ms: phase_a + phase_b + phase_c + i,
                x: 11500.0,
                y: 2365.0,
                z: -60.0,
                vx: 20.0,
                vy: 0.0,
                vz: 0.0,
                type_mask: TYPE_MASK_IGNORE_AFX
                    | TYPE_MASK_IGNORE_AFY
                    | TYPE_MASK_IGNORE_AFZ
                    | TYPE_MASK_IGNORE_YAW,
                ..Default::default()
            }),
            p_seq,
            255,
            190,
        );
        p_seq = p_seq.wrapping_add(1);
        send_paced(&pkt, i, &mut planner_acks_denied, &mut planner_statustext);
    }

    // Phase E: Deadman 500 ms FCU heartbeat timeout
    pause_hb.store(true, Ordering::Relaxed);
    thread::sleep(Duration::from_millis(540));
    for i in 0..deadman_pkts {
        deadman_timeout_sent += 1;
        let pkt = encode_message(
            &MavMessage::SetPositionTargetLocalNed(SetPositionTargetLocalNed {
                time_boot_ms: active_pkts + i,
                vx: 8.0,
                vy: 0.0,
                vz: 0.0,
                type_mask: TYPE_MASK_VELOCITY_ONLY,
                ..Default::default()
            }),
            p_seq,
            255,
            190,
        );
        p_seq = p_seq.wrapping_add(1);
        send_paced(&pkt, i, &mut planner_acks_denied, &mut planner_statustext);
    }

    // Flush remaining datagrams
    for _ in 0..20 {
        thread::sleep(Duration::from_millis(5));
        drain_planner(&mut planner_acks_denied, &mut planner_statustext);
    }

    stop_bridge.store(true, Ordering::Relaxed);
    stop_fcu.store(true, Ordering::Relaxed);
    let (shield_stats, latencies_us) = bridge_handle.join().expect("join bridge");
    fcu_handle.join().expect("join fcu");

    let total_sent = nominal_sent
        + plunge_overspeed_sent
        + ltfj_nfz_sent
        + hospital_nfz_sent
        + ltbx_nfz_sent
        + airborne_disarm_sent
        + flight_term_sent
        + corrupted_crc_sent
        + deadman_timeout_sent;
    let expected_fcu = nominal_sent
        + plunge_overspeed_sent
        + ltfj_nfz_sent
        + hospital_nfz_sent
        + ltbx_nfz_sent
        + deadman_timeout_sent;
    let fcu_rx = fcu_rx_valid.load(Ordering::Relaxed);
    let unsafe_leaked = fcu_unsafe_leaked.load(Ordering::Relaxed);
    let corrupt_leaked = fcu_corrupt_leaked.load(Ordering::Relaxed);
    let dropped = expected_fcu.saturating_sub(fcu_rx);
    let expected_denied = airborne_disarm_sent + flight_term_sent;

    let passed = unsafe_leaked == 0
        && corrupt_leaked == 0
        && dropped == 0
        && shield_stats.commands_denied == expected_denied
        && shield_stats.crc_rejected >= corrupted_crc_sent
        && shield_stats.deadman_overrides == deadman_timeout_sent
        && planner_acks_denied == expected_denied;

    UdpLoopbackStressReport {
        passed,
        total_offboard_packets_sent: total_sent,
        nominal_sent,
        plunge_overspeed_sent,
        ltfj_nfz_sent,
        hospital_nfz_sent,
        ltbx_nfz_sent,
        airborne_disarm_sent,
        flight_term_sent,
        corrupted_crc_sent,
        deadman_timeout_sent,
        fcu_valid_packets_received: fcu_rx,
        fcu_unsafe_packets_leaked: unsafe_leaked,
        fcu_corrupted_packets_leaked: corrupt_leaked,
        valid_packets_dropped: dropped,
        planner_acks_denied_received: planner_acks_denied,
        planner_statustext_alerts_received: planner_statustext,
        shield_stats,
        latency_us: compute_latency_stats(latencies_us),
        elapsed_ms: t_start.elapsed().as_secs_f64() * 1000.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_udp_2500_packet_adversarial_loopback_zero_leaks_zero_drops() {
        let r = run_udp_loopback_stress_proof(2500);
        assert!(r.passed, "UDP proof failed: {r:#?}");
        assert_eq!(r.fcu_unsafe_packets_leaked, 0);
        assert_eq!(r.fcu_corrupted_packets_leaked, 0);
        assert_eq!(r.valid_packets_dropped, 0);
        assert_eq!(r.total_offboard_packets_sent, 2500);
    }
}
