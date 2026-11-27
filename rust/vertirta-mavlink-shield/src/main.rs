// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]

use vertirta_mavlink_shield::udp_proof::run_udp_loopback_stress_proof;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut packets: u32 = 2500;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--packets" && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse::<u32>() {
                packets = v;
            }
            i += 2;
        } else {
            i += 1;
        }
    }

    let r = run_udp_loopback_stress_proof(packets);
    println!("{{");
    println!("  \"schema_version\": \"vertirta.rust_mavlink_v2_rta_pilot/v1.0.0\",");
    println!("  \"license\": \"Apache-2.0 (Zero-GPL, Zero-Unsafe, Zero-Deps Rust)\",");
    println!("  \"passed\": {},", r.passed);
    println!("  \"total_offboard_packets_sent\": {},", r.total_offboard_packets_sent);
    println!("  \"fcu_valid_packets_received\": {},", r.fcu_valid_packets_received);
    println!("  \"fcu_unsafe_packets_leaked\": {},", r.fcu_unsafe_packets_leaked);
    println!("  \"fcu_corrupted_packets_leaked\": {},", r.fcu_corrupted_packets_leaked);
    println!("  \"valid_packets_dropped\": {},", r.valid_packets_dropped);
    println!("  \"setpoints_passed\": {},", r.shield_stats.setpoints_passed);
    println!("  \"setpoints_clamped\": {},", r.shield_stats.setpoints_clamped);
    println!("  \"setpoints_clamped_nfz\": {},", r.shield_stats.setpoints_clamped_nfz);
    println!("  \"setpoints_clamped_plunge_speed\": {},", r.shield_stats.setpoints_clamped_plunge_speed);
    println!("  \"deadman_overrides\": {},", r.shield_stats.deadman_overrides);
    println!("  \"commands_denied\": {},", r.shield_stats.commands_denied);
    println!("  \"crc_rejected\": {},", r.shield_stats.crc_rejected);
    println!("  \"planner_acks_denied_received\": {},", r.planner_acks_denied_received);
    println!("  \"latency_us\": {{");
    println!("    \"sample_count\": {},", r.latency_us.sample_count);
    println!("    \"min_us\": {:.3},", r.latency_us.min_us);
    println!("    \"p50_us\": {:.3},", r.latency_us.p50_us);
    println!("    \"p95_us\": {:.3},", r.latency_us.p95_us);
    println!("    \"p99_us\": {:.3},", r.latency_us.p99_us);
    println!("    \"max_us\": {:.3},", r.latency_us.max_us);
    println!("    \"mean_us\": {:.3}", r.latency_us.mean_us);
    println!("  }},");
    println!("  \"elapsed_ms\": {:.3}", r.elapsed_ms);
    println!("}}");

    if !r.passed {
        std::process::exit(1);
    }
}
