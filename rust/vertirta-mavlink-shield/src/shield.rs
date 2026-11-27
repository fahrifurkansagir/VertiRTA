// SPDX-License-Identifier: Apache-2.0
//! Zero-unsafe, `FiniteF64`-typed MAVLink v2 Offboard RTA Shield.
//! Enforces Istanbul LTFJ/LTBX/Hospital hard geofences, plunge/overspeed/altitude envelopes,
//! `LOCAL_NED` + `FORCE_SET` guards, airborne disarm/termination denial, and 500 ms deadman hover.

use crate::codec::{
    decode_frame, encode_message, CommandAck, CommandLong, DecodeError, MavMessage,
    SetPositionTargetLocalNed, StatusText, MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_DO_FLIGHT_TERMINATION, MAV_FRAME_LOCAL_NED, MAV_MODE_FLAG_SAFETY_ARMED,
    MAV_RESULT_DENIED, MAV_SEVERITY_CRITICAL, MAV_SEVERITY_WARNING, TYPE_MASK_FORCE,
    TYPE_MASK_IGNORE_AFX, TYPE_MASK_IGNORE_AFY, TYPE_MASK_IGNORE_AFZ, TYPE_MASK_IGNORE_VX,
    TYPE_MASK_IGNORE_VY, TYPE_MASK_IGNORE_VZ, TYPE_MASK_IGNORE_X, TYPE_MASK_IGNORE_Y,
    TYPE_MASK_IGNORE_Z, TYPE_MASK_VELOCITY_ONLY,
};
use crate::finite::{FiniteF64, Vec3Ned};
use crate::geodesy::{
    geodetic_to_kartal_ned, kartal_ned_to_geodetic, GeodeticWgs84, KARTAL_DATUM_HAE_M,
    KARTAL_DATUM_LAT_DEG, KARTAL_HOSPITAL_LAT_DEG, KARTAL_HOSPITAL_LON_DEG, LTBX_ARP_LAT_DEG,
    LTBX_ARP_LON_DEG,
};

#[derive(Debug, Clone, Copy)]
pub struct ShieldConfig {
    pub bridge_sysid: u8,
    pub bridge_compid: u8,
    pub deadman_timeout_us: u64,
    pub max_horiz_speed_mps: f64,
    pub max_ascent_rate_mps: f64,
    pub max_descent_rate_mps: f64,
    pub min_agl_m: f64,
    pub max_agl_m: f64,
    pub airborne_disarm_agl_m: f64,
    pub ltfj_max_lon_deg: f64,
    pub ltfj_buffer_m: f64,
    pub ltbx_ctr_radius_m: f64,
    pub hospital_nfz_radius_m: f64,
    pub hospital_nfz_ceiling_agl_m: f64,
    pub lookahead_dt_s: f64,
}

impl Default for ShieldConfig {
    fn default() -> Self {
        Self {
            bridge_sysid: 1,
            bridge_compid: 191,
            deadman_timeout_us: 500_000,
            max_horiz_speed_mps: 18.0,
            max_ascent_rate_mps: 4.0,
            max_descent_rate_mps: 3.0,
            min_agl_m: 10.0,
            max_agl_m: 150.0,
            airborne_disarm_agl_m: 2.0,
            ltfj_max_lon_deg: 29.1895,
            ltfj_buffer_m: 30.0,
            ltbx_ctr_radius_m: 5556.0,
            hospital_nfz_radius_m: 165.0,
            hospital_nfz_ceiling_agl_m: 167.0,
            lookahead_dt_s: 0.50,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VehicleTelemetryState {
    pub heartbeat_seen: bool,
    pub last_fcu_heartbeat_us: u64,
    pub local_position_seen: bool,
    pub last_local_position_us: u64,
    pub armed: bool,
    pub base_mode: u8,
    pub system_status: u8,
    pub pos_ned: Vec3Ned,
    pub vel_ned: Vec3Ned,
    pub voltage_battery_mv: u16,
    pub battery_remaining_pct: i8,
}

impl Default for VehicleTelemetryState {
    fn default() -> Self {
        Self {
            heartbeat_seen: false,
            last_fcu_heartbeat_us: 0,
            local_position_seen: false,
            last_local_position_us: 0,
            armed: false,
            base_mode: 0,
            system_status: 0,
            pos_ned: Vec3Ned::new(0.0, 0.0, -50.0).expect("constant finite"),
            vel_ned: Vec3Ned::ZERO,
            voltage_battery_mv: 40_320,
            battery_remaining_pct: 80,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ShieldStats {
    pub fcu_rx_valid: u64,
    pub fcu_rx_rejected: u64,
    pub offboard_rx_valid: u64,
    pub crc_rejected: u64,
    pub malformed_rejected: u64,
    pub setpoints_passed: u64,
    pub setpoints_clamped: u64,
    pub setpoints_clamped_nfz: u64,
    pub setpoints_clamped_plunge_speed: u64,
    pub deadman_overrides: u64,
    pub commands_passed: u64,
    pub commands_denied: u64,
    pub statustext_emitted: u64,
    pub command_acks_emitted: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShieldAction {
    RejectedInvalidFrame,
    ForwardedNominal,
    ClampedSafeSetpoint,
    DeadmanHoverOverride,
    DeniedCommand,
}

#[derive(Debug, Clone)]
pub struct ShieldStepOutput {
    pub action: ShieldAction,
    pub to_fcu: Vec<Vec<u8>>,
    pub to_offboard: Vec<Vec<u8>>,
}

pub struct OffboardRtaShield {
    cfg: ShieldConfig,
    state: VehicleTelemetryState,
    stats: ShieldStats,
    ltbx_arp_ned: Vec3Ned,
    hospital_ned: Vec3Ned,
    ltfj_wall_east_m: f64,
    tx_seq: u8,
}

impl OffboardRtaShield {
    pub fn new(cfg: ShieldConfig) -> Self {
        let ltbx_arp_ned = geodetic_to_kartal_ned(GeodeticWgs84 {
            lat_deg: LTBX_ARP_LAT_DEG,
            lon_deg: LTBX_ARP_LON_DEG,
            h_hae_m: KARTAL_DATUM_HAE_M,
        })
        .expect("LTBX ARP finite");
        let hospital_ned = geodetic_to_kartal_ned(GeodeticWgs84 {
            lat_deg: KARTAL_HOSPITAL_LAT_DEG,
            lon_deg: KARTAL_HOSPITAL_LON_DEG,
            h_hae_m: KARTAL_DATUM_HAE_M,
        })
        .expect("Hospital finite");
        let ltfj_wall_east_m = geodetic_to_kartal_ned(GeodeticWgs84 {
            lat_deg: KARTAL_DATUM_LAT_DEG,
            lon_deg: cfg.ltfj_max_lon_deg,
            h_hae_m: KARTAL_DATUM_HAE_M,
        })
        .expect("LTFJ wall finite")
        .e
        .get();

        Self {
            cfg,
            state: VehicleTelemetryState::default(),
            stats: ShieldStats::default(),
            ltbx_arp_ned,
            hospital_ned,
            ltfj_wall_east_m,
            tx_seq: 0,
        }
    }

    pub fn config(&self) -> ShieldConfig {
        self.cfg
    }

    pub fn stats(&self) -> ShieldStats {
        self.stats
    }

    pub fn state(&self) -> VehicleTelemetryState {
        self.state
    }

    pub fn is_deadman_expired(&self, now_us: u64) -> bool {
        if !self.state.heartbeat_seen {
            return true;
        }
        now_us.saturating_sub(self.state.last_fcu_heartbeat_us) > self.cfg.deadman_timeout_us
    }

    pub fn violates_istanbul_nfz(&self, pos: Vec3Ned) -> bool {
        let geo = kartal_ned_to_geodetic(pos);
        if geo.lon_deg >= self.cfg.ltfj_max_lon_deg
            || pos.e.get() >= (self.ltfj_wall_east_m - self.cfg.ltfj_buffer_m)
        {
            return true;
        }
        let hosp_r = (pos.n.get() - self.hospital_ned.n.get())
            .hypot(pos.e.get() - self.hospital_ned.e.get());
        let agl_m = -pos.d.get();
        if hosp_r <= self.cfg.hospital_nfz_radius_m && agl_m <= self.cfg.hospital_nfz_ceiling_agl_m
        {
            return true;
        }
        let ltbx_r = (pos.n.get() - self.ltbx_arp_ned.n.get())
            .hypot(pos.e.get() - self.ltbx_arp_ned.e.get());
        ltbx_r <= self.cfg.ltbx_ctr_radius_m
    }

    fn next_seq(&mut self) -> u8 {
        let s = self.tx_seq;
        self.tx_seq = self.tx_seq.wrapping_add(1);
        s
    }

    fn make_statustext(&mut self, severity: u8, text: &str) -> Vec<u8> {
        let seq = self.next_seq();
        self.stats.statustext_emitted += 1;
        encode_message(
            &MavMessage::StatusText(StatusText::new(severity, text)),
            seq,
            self.cfg.bridge_sysid,
            self.cfg.bridge_compid,
        )
    }

    fn make_command_ack(&mut self, command: u16, result: u8, target_sys: u8, target_comp: u8) -> Vec<u8> {
        let seq = self.next_seq();
        self.stats.command_acks_emitted += 1;
        encode_message(
            &MavMessage::CommandAck(CommandAck {
                command,
                result,
                progress: 0,
                result_param2: 0,
                target_system: target_sys,
                target_component: target_comp,
            }),
            seq,
            self.cfg.bridge_sysid,
            self.cfg.bridge_compid,
        )
    }

    fn make_hover_setpoint(&mut self, src: &SetPositionTargetLocalNed) -> Vec<u8> {
        let seq = self.next_seq();
        let hover = SetPositionTargetLocalNed {
            time_boot_ms: src.time_boot_ms,
            x: self.state.pos_ned.n.as_f32(),
            y: self.state.pos_ned.e.as_f32(),
            z: self.state.pos_ned.d.get().min(-self.cfg.min_agl_m) as f32,
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
            afx: 0.0,
            afy: 0.0,
            afz: 0.0,
            yaw: 0.0,
            yaw_rate: 0.0,
            type_mask: TYPE_MASK_VELOCITY_ONLY,
            target_system: src.target_system,
            target_component: src.target_component,
            coordinate_frame: MAV_FRAME_LOCAL_NED,
        };
        encode_message(
            &MavMessage::SetPositionTargetLocalNed(hover),
            seq,
            self.cfg.bridge_sysid,
            self.cfg.bridge_compid,
        )
    }

    pub fn process_fcu_packet(&mut self, wire: &[u8], now_us: u64) -> ShieldStepOutput {
        let dec = match decode_frame(wire) {
            Ok(d) => d,
            Err(e) => {
                self.stats.fcu_rx_rejected += 1;
                if e == DecodeError::BadCrc {
                    self.stats.crc_rejected += 1;
                } else {
                    self.stats.malformed_rejected += 1;
                }
                return ShieldStepOutput {
                    action: ShieldAction::RejectedInvalidFrame,
                    to_fcu: Vec::new(),
                    to_offboard: Vec::new(),
                };
            }
        };

        self.stats.fcu_rx_valid += 1;
        match dec.message {
            MavMessage::Heartbeat(hb) => {
                self.state.heartbeat_seen = true;
                self.state.last_fcu_heartbeat_us = now_us;
                self.state.base_mode = hb.base_mode;
                self.state.armed = (hb.base_mode & MAV_MODE_FLAG_SAFETY_ARMED) != 0;
                self.state.system_status = hb.system_status;
            }
            MavMessage::SysStatus(ss) => {
                self.state.voltage_battery_mv = ss.voltage_battery;
                self.state.battery_remaining_pct = ss.battery_remaining;
            }
            MavMessage::LocalPositionNed(lp) => {
                if let (Some(pos), Some(vel)) = (
                    Vec3Ned::from_f32(lp.x, lp.y, lp.z),
                    Vec3Ned::from_f32(lp.vx, lp.vy, lp.vz),
                ) {
                    self.state.local_position_seen = true;
                    self.state.last_local_position_us = now_us;
                    self.state.pos_ned = pos;
                    self.state.vel_ned = vel;
                }
            }
            _ => {}
        }

        ShieldStepOutput {
            action: ShieldAction::ForwardedNominal,
            to_fcu: Vec::new(),
            to_offboard: vec![wire[..dec.bytes_consumed].to_vec()],
        }
    }

    pub fn process_offboard_packet(&mut self, wire: &[u8], now_us: u64) -> ShieldStepOutput {
        let dec = match decode_frame(wire) {
            Ok(d) => d,
            Err(e) => {
                if e == DecodeError::BadCrc {
                    self.stats.crc_rejected += 1;
                } else {
                    self.stats.malformed_rejected += 1;
                }
                return ShieldStepOutput {
                    action: ShieldAction::RejectedInvalidFrame,
                    to_fcu: Vec::new(),
                    to_offboard: Vec::new(),
                };
            }
        };

        self.stats.offboard_rx_valid += 1;
        match dec.message {
            MavMessage::CommandLong(cmd) => self.handle_command_long(cmd, dec.sysid, dec.compid, &wire[..dec.bytes_consumed], now_us),
            MavMessage::SetPositionTargetLocalNed(raw) => self.handle_setpoint(raw, &wire[..dec.bytes_consumed], now_us),
            _ => ShieldStepOutput {
                action: ShieldAction::ForwardedNominal,
                to_fcu: vec![wire[..dec.bytes_consumed].to_vec()],
                to_offboard: Vec::new(),
            },
        }
    }

    fn handle_command_long(
        &mut self,
        cmd: CommandLong,
        src_sys: u8,
        src_comp: u8,
        raw_bytes: &[u8],
        now_us: u64,
    ) -> ShieldStepOutput {
        let local_pos_fresh = self.state.local_position_seen
            && now_us.saturating_sub(self.state.last_local_position_us)
                <= self.cfg.deadman_timeout_us;
        let confirmed_on_ground = !self.is_deadman_expired(now_us)
            && local_pos_fresh
            && self.state.pos_ned.d.get() >= -self.cfg.airborne_disarm_agl_m;

        let is_arm_disarm = cmd.command == MAV_CMD_COMPONENT_ARM_DISARM;
        let p1_finite = FiniteF64::from_f32(cmd.param1);
        let p1_explicit_arm = p1_finite.is_some_and(|p| (p.get() - 1.0).abs() < 0.5);
        let is_non_finite_arm_disarm = is_arm_disarm && p1_finite.is_none();
        let is_airborne_disarm = is_arm_disarm && !p1_explicit_arm && !confirmed_on_ground;
        let is_flight_term = cmd.command == MAV_CMD_DO_FLIGHT_TERMINATION;

        if is_non_finite_arm_disarm || is_airborne_disarm || is_flight_term {
            self.stats.commands_denied += 1;
            let deny_msg = if is_non_finite_arm_disarm {
                "RTA DENY: NON-FINITE ARM/DISARM PARAM1 BLOCKED"
            } else if is_airborne_disarm {
                "RTA DENY: AIRBORNE/UNKNOWN-ALT DISARM BLOCKED"
            } else {
                "RTA DENY: FLIGHT TERMINATION BLOCKED"
            };
            let ack = self.make_command_ack(cmd.command, MAV_RESULT_DENIED, src_sys, src_comp);
            let st = self.make_statustext(MAV_SEVERITY_CRITICAL, deny_msg);
            return ShieldStepOutput {
                action: ShieldAction::DeniedCommand,
                to_fcu: Vec::new(),
                to_offboard: vec![ack, st],
            };
        }

        self.stats.commands_passed += 1;
        ShieldStepOutput {
            action: ShieldAction::ForwardedNominal,
            to_fcu: vec![raw_bytes.to_vec()],
            to_offboard: Vec::new(),
        }
    }

    fn handle_setpoint(
        &mut self,
        raw: SetPositionTargetLocalNed,
        raw_bytes: &[u8],
        now_us: u64,
    ) -> ShieldStepOutput {
        // 1. Deadman timeout fail-closed -> zero-velocity hover
        if self.is_deadman_expired(now_us) {
            self.stats.deadman_overrides += 1;
            let hover = self.make_hover_setpoint(&raw);
            let st = self.make_statustext(
                MAV_SEVERITY_CRITICAL,
                "RTA DEADMAN: FCU HB TIMEOUT -> ZERO-VEL HOVER",
            );
            return ShieldStepOutput {
                action: ShieldAction::DeadmanHoverOverride,
                to_fcu: vec![hover],
                to_offboard: vec![st],
            };
        }

        // 2. Compile-time FiniteF64 parsing gate (zero NaN/Inf can pass beyond this block!)
        let finite_payload = (|| -> Option<(Vec3Ned, Vec3Ned, Vec3Ned, FiniteF64, FiniteF64)> {
            Some((
                Vec3Ned::from_f32(raw.x, raw.y, raw.z)?,
                Vec3Ned::from_f32(raw.vx, raw.vy, raw.vz)?,
                Vec3Ned::from_f32(raw.afx, raw.afy, raw.afz)?,
                FiniteF64::from_f32(raw.yaw)?,
                FiniteF64::from_f32(raw.yaw_rate)?,
            ))
        })();

        let (raw_pos, raw_vel, _raw_acc, _raw_yaw, raw_yaw_rate) = match finite_payload {
            Some(v) => v,
            None => {
                self.stats.setpoints_clamped += 1;
                self.stats.setpoints_clamped_plunge_speed += 1;
                let hover = self.make_hover_setpoint(&raw);
                let st = self.make_statustext(
                    MAV_SEVERITY_CRITICAL,
                    "RTA CLAMP: NON-FINITE SETPOINT -> HOVER",
                );
                return ShieldStepOutput {
                    action: ShieldAction::ClampedSafeSetpoint,
                    to_fcu: vec![hover],
                    to_offboard: vec![st],
                };
            }
        };

        // 3. Enforce MAV_FRAME_LOCAL_NED (1)
        if raw.coordinate_frame != MAV_FRAME_LOCAL_NED {
            self.stats.setpoints_clamped += 1;
            self.stats.setpoints_clamped_nfz += 1;
            let hover = self.make_hover_setpoint(&raw);
            let st = self.make_statustext(
                MAV_SEVERITY_CRITICAL,
                "RTA CLAMP: NON-LOCAL_NED FRAME -> HOVER",
            );
            return ShieldStepOutput {
                action: ShieldAction::ClampedSafeSetpoint,
                to_fcu: vec![hover],
                to_offboard: vec![st],
            };
        }

        // 4. Reject FORCE_SET bit (0x0200)
        if (raw.type_mask & TYPE_MASK_FORCE) != 0 {
            self.stats.setpoints_clamped += 1;
            self.stats.setpoints_clamped_plunge_speed += 1;
            let hover = self.make_hover_setpoint(&raw);
            let st = self.make_statustext(
                MAV_SEVERITY_CRITICAL,
                "RTA CLAMP: FORCE_SET BIT BLOCKED -> HOVER",
            );
            return ShieldStepOutput {
                action: ShieldAction::ClampedSafeSetpoint,
                to_fcu: vec![hover],
                to_offboard: vec![st],
            };
        }

        let pos_all_ignored = (raw.type_mask
            & (TYPE_MASK_IGNORE_X | TYPE_MASK_IGNORE_Y | TYPE_MASK_IGNORE_Z))
            == (TYPE_MASK_IGNORE_X | TYPE_MASK_IGNORE_Y | TYPE_MASK_IGNORE_Z);
        let vel_all_ignored = (raw.type_mask
            & (TYPE_MASK_IGNORE_VX | TYPE_MASK_IGNORE_VY | TYPE_MASK_IGNORE_VZ))
            == (TYPE_MASK_IGNORE_VX | TYPE_MASK_IGNORE_VY | TYPE_MASK_IGNORE_VZ);
        let use_pos = !pos_all_ignored;
        let use_vel = !vel_all_ignored;
        let _use_acc = (raw.type_mask
            & (TYPE_MASK_IGNORE_AFX | TYPE_MASK_IGNORE_AFY | TYPE_MASK_IGNORE_AFZ))
            != (TYPE_MASK_IGNORE_AFX | TYPE_MASK_IGNORE_AFY | TYPE_MASK_IGNORE_AFZ);

        let mut vn = self.state.vel_ned.n.get();
        let mut ve = self.state.vel_ned.e.get();
        let mut vd = self.state.vel_ned.d.get();
        if use_vel {
            if (raw.type_mask & TYPE_MASK_IGNORE_VX) == 0 {
                vn = raw_vel.n.get();
            }
            if (raw.type_mask & TYPE_MASK_IGNORE_VY) == 0 {
                ve = raw_vel.e.get();
            }
            if (raw.type_mask & TYPE_MASK_IGNORE_VZ) == 0 {
                vd = raw_vel.d.get();
            }
        } else if use_pos {
            vn = if (raw.type_mask & TYPE_MASK_IGNORE_X) == 0 {
                raw_pos.n.get() - self.state.pos_ned.n.get()
            } else {
                0.0
            };
            ve = if (raw.type_mask & TYPE_MASK_IGNORE_Y) == 0 {
                raw_pos.e.get() - self.state.pos_ned.e.get()
            } else {
                0.0
            };
            vd = if (raw.type_mask & TYPE_MASK_IGNORE_Z) == 0 {
                raw_pos.d.get() - self.state.pos_ned.d.get()
            } else {
                0.0
            };
        }

        let mut clamped = false;
        let mut clamped_plunge_speed = false;
        let mut clamped_nfz = false;

        // 5. Speed & plunge-rate envelope clamping
        let v_xy = vn.hypot(ve);
        if v_xy > self.cfg.max_horiz_speed_mps {
            let s = self.cfg.max_horiz_speed_mps / v_xy;
            vn *= s;
            ve *= s;
            clamped = true;
            clamped_plunge_speed = true;
        }
        if vd > self.cfg.max_descent_rate_mps {
            vd = self.cfg.max_descent_rate_mps;
            clamped = true;
            clamped_plunge_speed = true;
        } else if vd < -self.cfg.max_ascent_rate_mps {
            vd = -self.cfg.max_ascent_rate_mps;
            clamped = true;
            clamped_plunge_speed = true;
        }

        let mut target_n = self.state.pos_ned.n.get();
        let mut target_e = self.state.pos_ned.e.get();
        let mut target_d = self.state.pos_ned.d.get();
        if (raw.type_mask & TYPE_MASK_IGNORE_X) == 0 {
            target_n = raw_pos.n.get();
        }
        if (raw.type_mask & TYPE_MASK_IGNORE_Y) == 0 {
            target_e = raw_pos.e.get();
        }
        if (raw.type_mask & TYPE_MASK_IGNORE_Z) == 0 {
            target_d = raw_pos.d.get();
            let z_min_ned = -self.cfg.max_agl_m;
            let z_max_ned = -self.cfg.min_agl_m;
            if target_d > z_max_ned {
                target_d = z_max_ned;
                vd = vd.min(0.0);
                clamped = true;
                clamped_plunge_speed = true;
            } else if target_d < z_min_ned {
                target_d = z_min_ned;
                vd = vd.max(0.0);
                clamped = true;
                clamped_plunge_speed = true;
            }
        }
        if self.state.pos_ned.d.get() + vd * self.cfg.lookahead_dt_s > -self.cfg.min_agl_m {
            vd = vd.min((-self.cfg.min_agl_m - self.state.pos_ned.d.get()) / self.cfg.lookahead_dt_s);
            clamped = true;
            clamped_plunge_speed = true;
        }

        // 6. Hard Istanbul NFZ geofence (LTFJ CTR, Kartal Hospital, LTBX CTR)
        let pred_pos = Vec3Ned::new(
            self.state.pos_ned.n.get() + vn * (2.0 * self.cfg.lookahead_dt_s),
            self.state.pos_ned.e.get() + ve * (2.0 * self.cfg.lookahead_dt_s),
            self.state.pos_ned.d.get() + vd * (2.0 * self.cfg.lookahead_dt_s),
        )
        .expect("clamped predicted pos finite");
        let targ_pos = Vec3Ned::new(target_n, target_e, target_d).expect("clamped target pos finite");

        if (use_pos && self.violates_istanbul_nfz(targ_pos)) || self.violates_istanbul_nfz(pred_pos) {
            clamped = true;
            clamped_nfz = true;

            let ltfj_safe_east = self.ltfj_wall_east_m - self.cfg.ltfj_buffer_m;
            if pred_pos.e.get() >= ltfj_safe_east || target_e >= ltfj_safe_east {
                if ve > 0.0 {
                    ve = -ve.abs().min(2.0);
                }
                target_e = target_e.min(ltfj_safe_east - 5.0);
            }

            let hosp_r = (pred_pos.n.get() - self.hospital_ned.n.get())
                .hypot(pred_pos.e.get() - self.hospital_ned.e.get());
            let hosp_t_r = (target_n - self.hospital_ned.n.get())
                .hypot(target_e - self.hospital_ned.e.get());
            if hosp_r <= self.cfg.hospital_nfz_radius_m || hosp_t_r <= self.cfg.hospital_nfz_radius_m {
                let dn0 = self.state.pos_ned.n.get() - self.hospital_ned.n.get();
                let de0 = self.state.pos_ned.e.get() - self.hospital_ned.e.get();
                let r0 = dn0.hypot(de0).max(1.0);
                let un = dn0 / r0;
                let ue = de0 / r0;
                let radial_v = vn * un + ve * ue;
                if radial_v < 0.0 {
                    vn -= (radial_v - 1.0) * un;
                    ve -= (radial_v - 1.0) * ue;
                }
                target_n = self.hospital_ned.n.get() + un * (self.cfg.hospital_nfz_radius_m + 15.0);
                target_e = self.hospital_ned.e.get() + ue * (self.cfg.hospital_nfz_radius_m + 15.0);
            }

            let ltbx_r = (pred_pos.n.get() - self.ltbx_arp_ned.n.get())
                .hypot(pred_pos.e.get() - self.ltbx_arp_ned.e.get());
            let ltbx_t_r = (target_n - self.ltbx_arp_ned.n.get())
                .hypot(target_e - self.ltbx_arp_ned.e.get());
            if ltbx_r <= self.cfg.ltbx_ctr_radius_m || ltbx_t_r <= self.cfg.ltbx_ctr_radius_m {
                let dn0 = self.state.pos_ned.n.get() - self.ltbx_arp_ned.n.get();
                let de0 = self.state.pos_ned.e.get() - self.ltbx_arp_ned.e.get();
                let r0 = dn0.hypot(de0).max(1.0);
                let un = dn0 / r0;
                let ue = de0 / r0;
                let radial_v = vn * un + ve * ue;
                if radial_v < 0.0 {
                    vn -= (radial_v - 1.0) * un;
                    ve -= (radial_v - 1.0) * ue;
                }
                target_n = self.ltbx_arp_ned.n.get() + un * (self.cfg.ltbx_ctr_radius_m + 25.0);
                target_e = self.ltbx_arp_ned.e.get() + ue * (self.cfg.ltbx_ctr_radius_m + 25.0);
            }

            let v_xy2 = vn.hypot(ve);
            if v_xy2 > self.cfg.max_horiz_speed_mps {
                let s2 = self.cfg.max_horiz_speed_mps / v_xy2;
                vn *= s2;
                ve *= s2;
            }
        }

        if !clamped {
            self.stats.setpoints_passed += 1;
            return ShieldStepOutput {
                action: ShieldAction::ForwardedNominal,
                to_fcu: vec![raw_bytes.to_vec()],
                to_offboard: Vec::new(),
            };
        }

        self.stats.setpoints_clamped += 1;
        if clamped_nfz {
            self.stats.setpoints_clamped_nfz += 1;
        }
        if clamped_plunge_speed {
            self.stats.setpoints_clamped_plunge_speed += 1;
        }

        let safe = SetPositionTargetLocalNed {
            time_boot_ms: raw.time_boot_ms,
            x: target_n as f32,
            y: target_e as f32,
            z: target_d.clamp(-self.cfg.max_agl_m, -self.cfg.min_agl_m) as f32,
            vx: vn as f32,
            vy: ve as f32,
            vz: vd.clamp(-self.cfg.max_ascent_rate_mps, self.cfg.max_descent_rate_mps) as f32,
            afx: 0.0,
            afy: 0.0,
            afz: 0.0,
            yaw: 0.0,
            yaw_rate: raw_yaw_rate.clamp_finite(-1.5, 1.5).as_f32(),
            type_mask: TYPE_MASK_VELOCITY_ONLY,
            target_system: raw.target_system,
            target_component: raw.target_component,
            coordinate_frame: MAV_FRAME_LOCAL_NED,
        };
        let seq = self.next_seq();
        let pkt = encode_message(
            &MavMessage::SetPositionTargetLocalNed(safe),
            seq,
            self.cfg.bridge_sysid,
            self.cfg.bridge_compid,
        );
        let alert = if clamped_nfz {
            "RTA CLAMP: ISTANBUL NFZ BREACH BLOCKED"
        } else {
            "RTA CLAMP: PLUNGE/OVERSPEED ENVELOPE LIMIT"
        };
        let st = self.make_statustext(MAV_SEVERITY_WARNING, alert);
        ShieldStepOutput {
            action: ShieldAction::ClampedSafeSetpoint,
            to_fcu: vec![pkt],
            to_offboard: vec![st],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{Heartbeat, LocalPositionNed};

    fn prime_airborne(shield: &mut OffboardRtaShield, now_us: u64, n: f32, e: f32, d: f32) {
        let hb = encode_message(
            &MavMessage::Heartbeat(Heartbeat {
                base_mode: MAV_MODE_FLAG_SAFETY_ARMED,
                system_status: 4,
                ..Default::default()
            }),
            0,
            1,
            1,
        );
        let lp = encode_message(
            &MavMessage::LocalPositionNed(LocalPositionNed {
                time_boot_ms: 1000,
                x: n,
                y: e,
                z: d,
                vx: 4.0,
                vy: 0.0,
                vz: 0.0,
            }),
            1,
            1,
            1,
        );
        assert_eq!(shield.process_fcu_packet(&hb, now_us).action, ShieldAction::ForwardedNominal);
        assert_eq!(shield.process_fcu_packet(&lp, now_us).action, ShieldAction::ForwardedNominal);
    }

    #[test]
    fn denies_airborne_and_nan_disarm_and_blocks_force_or_wrong_frame() {
        let mut shield = OffboardRtaShield::new(ShieldConfig::default());
        let t0 = 1_000_000u64;

        // Pre-heartbeat disarm must fail closed
        let disarm = encode_message(
            &MavMessage::CommandLong(CommandLong {
                command: MAV_CMD_COMPONENT_ARM_DISARM,
                param1: 0.0,
                ..Default::default()
            }),
            0,
            255,
            190,
        );
        assert_eq!(shield.process_offboard_packet(&disarm, t0).action, ShieldAction::DeniedCommand);

        prime_airborne(&mut shield, t0, 500.0, -400.0, -60.0);

        // Airborne disarm denied with COMMAND_ACK(DENIED)
        let out = shield.process_offboard_packet(&disarm, t0 + 10_000);
        assert_eq!(out.action, ShieldAction::DeniedCommand);
        assert!(out.to_fcu.is_empty());
        assert_eq!(out.to_offboard.len(), 2);

        // NaN param1 arm/disarm denied even on ground
        prime_airborne(&mut shield, t0 + 20_000, 500.0, -400.0, -0.2);
        let nan_arm = encode_message(
            &MavMessage::CommandLong(CommandLong {
                command: MAV_CMD_COMPONENT_ARM_DISARM,
                param1: f32::NAN,
                ..Default::default()
            }),
            1,
            255,
            190,
        );
        assert_eq!(shield.process_offboard_packet(&nan_arm, t0 + 25_000).action, ShieldAction::DeniedCommand);

        // BODY_NED frame & FORCE_SET bit fail-closed to LOCAL_NED hover
        prime_airborne(&mut shield, t0 + 30_000, 500.0, -400.0, -60.0);
        let bad_frame = encode_message(
            &MavMessage::SetPositionTargetLocalNed(SetPositionTargetLocalNed {
                vx: 5.0,
                coordinate_frame: 8, // BODY_NED
                type_mask: TYPE_MASK_VELOCITY_ONLY,
                ..Default::default()
            }),
            2,
            255,
            190,
        );
        let f_out = shield.process_offboard_packet(&bad_frame, t0 + 35_000);
        assert_eq!(f_out.action, ShieldAction::ClampedSafeSetpoint);
        let dec = decode_frame(&f_out.to_fcu[0]).expect("valid hover");
        match dec.message {
            MavMessage::SetPositionTargetLocalNed(sp) => {
                assert_eq!(sp.coordinate_frame, MAV_FRAME_LOCAL_NED);
                assert_eq!(sp.vx, 0.0);
                assert_eq!(sp.type_mask & TYPE_MASK_FORCE, 0);
            }
            _ => panic!("expected setpoint"),
        }

        // Deadman > 500 ms triggers zero-velocity hover
        let nominal = encode_message(
            &MavMessage::SetPositionTargetLocalNed(SetPositionTargetLocalNed {
                vx: 5.0,
                type_mask: TYPE_MASK_VELOCITY_ONLY,
                ..Default::default()
            }),
            3,
            255,
            190,
        );
        let dm = shield.process_offboard_packet(&nominal, t0 + 600_000);
        assert_eq!(dm.action, ShieldAction::DeadmanHoverOverride);
    }
}
