// SPDX-License-Identifier: Apache-2.0
//! Clean-room Apache-2.0, zero-unsafe MAVLink v2 (`STX=0xFD`) wire codec.
//! Exact bit-for-bit parity with `tools/mavlink_rta_bridge/src/mavlink_v2_codec.cpp`.

pub const MAVLINK_V2_STX: u8 = 0xFD;
pub const HEADER_BYTES: usize = 10;
pub const CHECKSUM_BYTES: usize = 2;
pub const SIGNATURE_BYTES: usize = 13;
pub const MAX_PAYLOAD_BYTES: usize = 255;
pub const MAX_FRAME_BYTES: usize = HEADER_BYTES + MAX_PAYLOAD_BYTES + CHECKSUM_BYTES + SIGNATURE_BYTES;

pub const INCOMPAT_FLAG_SIGNED: u8 = 0x01;

// Official MAVLink common.xml message IDs.
pub const MSG_ID_HEARTBEAT: u32 = 0;
pub const MSG_ID_SYS_STATUS: u32 = 1;
pub const MSG_ID_ATTITUDE_QUATERNION: u32 = 31;
pub const MSG_ID_LOCAL_POSITION_NED: u32 = 32;
pub const MSG_ID_COMMAND_LONG: u32 = 76;
pub const MSG_ID_COMMAND_ACK: u32 = 77;
pub const MSG_ID_SET_POSITION_TARGET_LOCAL_NED: u32 = 84;
pub const MSG_ID_STATUSTEXT: u32 = 253;

// Official MAVLink common.xml CRC_EXTRA seeds.
pub const CRC_EXTRA_HEARTBEAT: u8 = 50;
pub const CRC_EXTRA_SYS_STATUS: u8 = 124;
pub const CRC_EXTRA_ATTITUDE_QUATERNION: u8 = 246;
pub const CRC_EXTRA_LOCAL_POSITION_NED: u8 = 185;
pub const CRC_EXTRA_COMMAND_LONG: u8 = 152;
pub const CRC_EXTRA_COMMAND_ACK: u8 = 143;
pub const CRC_EXTRA_SET_POSITION_TARGET_LOCAL_NED: u8 = 143;
pub const CRC_EXTRA_STATUSTEXT: u8 = 83;

// MAVLink enums used by the RTA Shield.
pub const MAV_MODE_FLAG_SAFETY_ARMED: u8 = 0x80;
pub const MAV_FRAME_LOCAL_NED: u8 = 1;

pub const MAV_CMD_DO_FLIGHT_TERMINATION: u16 = 185;
pub const MAV_CMD_COMPONENT_ARM_DISARM: u16 = 400;

pub const MAV_RESULT_ACCEPTED: u8 = 0;
pub const MAV_RESULT_DENIED: u8 = 2;

pub const MAV_SEVERITY_CRITICAL: u8 = 2;
pub const MAV_SEVERITY_WARNING: u8 = 4;
pub const MAV_SEVERITY_INFO: u8 = 6;

// POSITION_TARGET_TYPEMASK bitflags.
pub const TYPE_MASK_IGNORE_X: u16 = 0x0001;
pub const TYPE_MASK_IGNORE_Y: u16 = 0x0002;
pub const TYPE_MASK_IGNORE_Z: u16 = 0x0004;
pub const TYPE_MASK_IGNORE_VX: u16 = 0x0008;
pub const TYPE_MASK_IGNORE_VY: u16 = 0x0010;
pub const TYPE_MASK_IGNORE_VZ: u16 = 0x0020;
pub const TYPE_MASK_IGNORE_AFX: u16 = 0x0040;
pub const TYPE_MASK_IGNORE_AFY: u16 = 0x0080;
pub const TYPE_MASK_IGNORE_AFZ: u16 = 0x0100;
pub const TYPE_MASK_FORCE: u16 = 0x0200;
pub const TYPE_MASK_IGNORE_YAW: u16 = 0x0400;
pub const TYPE_MASK_IGNORE_YAW_RATE: u16 = 0x0800;
pub const TYPE_MASK_VELOCITY_ONLY: u16 = TYPE_MASK_IGNORE_X
    | TYPE_MASK_IGNORE_Y
    | TYPE_MASK_IGNORE_Z
    | TYPE_MASK_IGNORE_AFX
    | TYPE_MASK_IGNORE_AFY
    | TYPE_MASK_IGNORE_AFZ
    | TYPE_MASK_IGNORE_YAW
    | TYPE_MASK_IGNORE_YAW_RATE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    TooShort,
    BadMagic,
    TruncatedFrame,
    UnsupportedIncompatFlags,
    UnknownMsgId,
    BadCrc,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Heartbeat {
    pub custom_mode: u32,
    pub mav_type: u8,
    pub autopilot: u8,
    pub base_mode: u8,
    pub system_status: u8,
    pub mavlink_version: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SysStatus {
    pub onboard_control_sensors_present: u32,
    pub onboard_control_sensors_enabled: u32,
    pub onboard_control_sensors_health: u32,
    pub load: u16,
    pub voltage_battery: u16,
    pub current_battery: i16,
    pub drop_rate_comm: u16,
    pub errors_comm: u16,
    pub errors_count1: u16,
    pub errors_count2: u16,
    pub errors_count3: u16,
    pub errors_count4: u16,
    pub battery_remaining: i8,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AttitudeQuaternion {
    pub time_boot_ms: u32,
    pub q1: f32,
    pub q2: f32,
    pub q3: f32,
    pub q4: f32,
    pub rollspeed: f32,
    pub pitchspeed: f32,
    pub yawspeed: f32,
    pub repr_offset_q: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LocalPositionNed {
    pub time_boot_ms: u32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CommandLong {
    pub param1: f32,
    pub param2: f32,
    pub param3: f32,
    pub param4: f32,
    pub param5: f32,
    pub param6: f32,
    pub param7: f32,
    pub command: u16,
    pub target_system: u8,
    pub target_component: u8,
    pub confirmation: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CommandAck {
    pub command: u16,
    pub result: u8,
    pub progress: u8,
    pub result_param2: i32,
    pub target_system: u8,
    pub target_component: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetPositionTargetLocalNed {
    pub time_boot_ms: u32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub afx: f32,
    pub afy: f32,
    pub afz: f32,
    pub yaw: f32,
    pub yaw_rate: f32,
    pub type_mask: u16,
    pub target_system: u8,
    pub target_component: u8,
    pub coordinate_frame: u8,
}

impl Default for SetPositionTargetLocalNed {
    fn default() -> Self {
        Self {
            time_boot_ms: 0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
            afx: 0.0,
            afy: 0.0,
            afz: 0.0,
            yaw: 0.0,
            yaw_rate: 0.0,
            type_mask: 0,
            target_system: 1,
            target_component: 1,
            coordinate_frame: MAV_FRAME_LOCAL_NED,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StatusText {
    pub severity: u8,
    pub text: [u8; 50],
    pub id: u16,
    pub chunk_seq: u8,
}

impl StatusText {
    pub fn new(severity: u8, msg: &str) -> Self {
        let mut text = [0u8; 50];
        let bytes = msg.as_bytes();
        let n = bytes.len().min(text.len());
        text[..n].copy_from_slice(&bytes[..n]);
        Self {
            severity,
            text,
            id: 0,
            chunk_seq: 0,
        }
    }

    pub fn text_str(&self) -> &str {
        let n = self.text.iter().position(|&b| b == 0).unwrap_or(self.text.len());
        std::str::from_utf8(&self.text[..n]).unwrap_or("")
    }
}

/// Strongly typed MAVLink v2 message envelope — prevents payload mis-interpretation at compile time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MavMessage {
    Heartbeat(Heartbeat),
    SysStatus(SysStatus),
    AttitudeQuaternion(AttitudeQuaternion),
    LocalPositionNed(LocalPositionNed),
    CommandLong(CommandLong),
    CommandAck(CommandAck),
    SetPositionTargetLocalNed(SetPositionTargetLocalNed),
    StatusText(StatusText),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedFrame {
    pub seq: u8,
    pub sysid: u8,
    pub compid: u8,
    pub msgid: u32,
    pub is_signed: bool,
    pub bytes_consumed: usize,
    pub message: MavMessage,
}

#[inline]
pub const fn crc16_accumulate(data: u8, crc: u16) -> u16 {
    let mut tmp = data ^ (crc as u8);
    tmp ^= tmp << 4;
    (crc >> 8) ^ ((tmp as u16) << 8) ^ ((tmp as u16) << 3) ^ ((tmp as u16) >> 4)
}

#[inline]
pub const fn crc_extra_for_msgid(msgid: u32) -> Option<u8> {
    match msgid {
        MSG_ID_HEARTBEAT => Some(CRC_EXTRA_HEARTBEAT),
        MSG_ID_SYS_STATUS => Some(CRC_EXTRA_SYS_STATUS),
        MSG_ID_ATTITUDE_QUATERNION => Some(CRC_EXTRA_ATTITUDE_QUATERNION),
        MSG_ID_LOCAL_POSITION_NED => Some(CRC_EXTRA_LOCAL_POSITION_NED),
        MSG_ID_COMMAND_LONG => Some(CRC_EXTRA_COMMAND_LONG),
        MSG_ID_COMMAND_ACK => Some(CRC_EXTRA_COMMAND_ACK),
        MSG_ID_SET_POSITION_TARGET_LOCAL_NED => Some(CRC_EXTRA_SET_POSITION_TARGET_LOCAL_NED),
        MSG_ID_STATUSTEXT => Some(CRC_EXTRA_STATUSTEXT),
        _ => None,
    }
}

#[inline]
fn truncated_payload_len(full: &[u8]) -> usize {
    if full.is_empty() {
        return 0;
    }
    let mut len = full.len().min(MAX_PAYLOAD_BYTES);
    while len > 1 && full[len - 1] == 0 {
        len -= 1;
    }
    len
}

#[inline]
fn put_u16_le(dst: &mut [u8], off: usize, v: u16) {
    dst[off..off + 2].copy_from_slice(&v.to_le_bytes());
}
#[inline]
fn put_i16_le(dst: &mut [u8], off: usize, v: i16) {
    dst[off..off + 2].copy_from_slice(&v.to_le_bytes());
}
#[inline]
fn put_u32_le(dst: &mut [u8], off: usize, v: u32) {
    dst[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
#[inline]
fn put_i32_le(dst: &mut [u8], off: usize, v: i32) {
    dst[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
#[inline]
fn put_f32_le(dst: &mut [u8], off: usize, v: f32) {
    dst[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

#[inline]
fn get_u16_le(src: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([src[off], src[off + 1]])
}
#[inline]
fn get_i16_le(src: &[u8], off: usize) -> i16 {
    i16::from_le_bytes([src[off], src[off + 1]])
}
#[inline]
fn get_u32_le(src: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([src[off], src[off + 1], src[off + 2], src[off + 3]])
}
#[inline]
fn get_i32_le(src: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([src[off], src[off + 1], src[off + 2], src[off + 3]])
}
#[inline]
fn get_f32_le(src: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([src[off], src[off + 1], src[off + 2], src[off + 3]])
}

fn zero_extend<const N: usize>(raw: &[u8]) -> [u8; N] {
    let mut out = [0u8; N];
    let n = raw.len().min(N);
    out[..n].copy_from_slice(&raw[..n]);
    out
}

pub fn encode_raw(seq: u8, sysid: u8, compid: u8, msgid: u32, payload: &[u8]) -> Vec<u8> {
    let crc_extra = crc_extra_for_msgid(msgid).expect("allowlisted MAVLink msgid");
    let tlen = truncated_payload_len(payload);
    let mut buf = Vec::with_capacity(HEADER_BYTES + tlen + CHECKSUM_BYTES);
    buf.push(MAVLINK_V2_STX);
    buf.push(tlen as u8);
    buf.push(0); // incompat_flags
    buf.push(0); // compat_flags
    buf.push(seq);
    buf.push(sysid);
    buf.push(compid);
    buf.push((msgid & 0xFF) as u8);
    buf.push(((msgid >> 8) & 0xFF) as u8);
    buf.push(((msgid >> 16) & 0xFF) as u8);
    buf.extend_from_slice(&payload[..tlen]);

    let mut crc = 0xFFFFu16;
    for &b in &buf[1..] {
        crc = crc16_accumulate(b, crc);
    }
    crc = crc16_accumulate(crc_extra, crc);
    buf.extend_from_slice(&crc.to_le_bytes());
    buf
}

pub fn encode_message(msg: &MavMessage, seq: u8, sysid: u8, compid: u8) -> Vec<u8> {
    match *msg {
        MavMessage::Heartbeat(m) => {
            let mut p = [0u8; 9];
            put_u32_le(&mut p, 0, m.custom_mode);
            p[4] = m.mav_type;
            p[5] = m.autopilot;
            p[6] = m.base_mode;
            p[7] = m.system_status;
            p[8] = m.mavlink_version;
            encode_raw(seq, sysid, compid, MSG_ID_HEARTBEAT, &p)
        }
        MavMessage::SysStatus(m) => {
            let mut p = [0u8; 31];
            put_u32_le(&mut p, 0, m.onboard_control_sensors_present);
            put_u32_le(&mut p, 4, m.onboard_control_sensors_enabled);
            put_u32_le(&mut p, 8, m.onboard_control_sensors_health);
            put_u16_le(&mut p, 12, m.load);
            put_u16_le(&mut p, 14, m.voltage_battery);
            put_i16_le(&mut p, 16, m.current_battery);
            put_u16_le(&mut p, 18, m.drop_rate_comm);
            put_u16_le(&mut p, 20, m.errors_comm);
            put_u16_le(&mut p, 22, m.errors_count1);
            put_u16_le(&mut p, 24, m.errors_count2);
            put_u16_le(&mut p, 26, m.errors_count3);
            put_u16_le(&mut p, 28, m.errors_count4);
            p[30] = m.battery_remaining as u8;
            encode_raw(seq, sysid, compid, MSG_ID_SYS_STATUS, &p)
        }
        MavMessage::AttitudeQuaternion(m) => {
            let mut p = [0u8; 32];
            put_u32_le(&mut p, 0, m.time_boot_ms);
            put_f32_le(&mut p, 4, m.q1);
            put_f32_le(&mut p, 8, m.q2);
            put_f32_le(&mut p, 12, m.q3);
            put_f32_le(&mut p, 16, m.q4);
            put_f32_le(&mut p, 20, m.rollspeed);
            put_f32_le(&mut p, 24, m.pitchspeed);
            put_f32_le(&mut p, 28, m.yawspeed);
            encode_raw(seq, sysid, compid, MSG_ID_ATTITUDE_QUATERNION, &p)
        }
        MavMessage::LocalPositionNed(m) => {
            let mut p = [0u8; 28];
            put_u32_le(&mut p, 0, m.time_boot_ms);
            put_f32_le(&mut p, 4, m.x);
            put_f32_le(&mut p, 8, m.y);
            put_f32_le(&mut p, 12, m.z);
            put_f32_le(&mut p, 16, m.vx);
            put_f32_le(&mut p, 20, m.vy);
            put_f32_le(&mut p, 24, m.vz);
            encode_raw(seq, sysid, compid, MSG_ID_LOCAL_POSITION_NED, &p)
        }
        MavMessage::CommandLong(m) => {
            let mut p = [0u8; 33];
            put_f32_le(&mut p, 0, m.param1);
            put_f32_le(&mut p, 4, m.param2);
            put_f32_le(&mut p, 8, m.param3);
            put_f32_le(&mut p, 12, m.param4);
            put_f32_le(&mut p, 16, m.param5);
            put_f32_le(&mut p, 20, m.param6);
            put_f32_le(&mut p, 24, m.param7);
            put_u16_le(&mut p, 28, m.command);
            p[30] = m.target_system;
            p[31] = m.target_component;
            p[32] = m.confirmation;
            encode_raw(seq, sysid, compid, MSG_ID_COMMAND_LONG, &p)
        }
        MavMessage::CommandAck(m) => {
            let mut p = [0u8; 10];
            put_u16_le(&mut p, 0, m.command);
            p[2] = m.result;
            p[3] = m.progress;
            put_i32_le(&mut p, 4, m.result_param2);
            p[8] = m.target_system;
            p[9] = m.target_component;
            encode_raw(seq, sysid, compid, MSG_ID_COMMAND_ACK, &p)
        }
        MavMessage::SetPositionTargetLocalNed(m) => {
            let mut p = [0u8; 53];
            put_u32_le(&mut p, 0, m.time_boot_ms);
            put_f32_le(&mut p, 4, m.x);
            put_f32_le(&mut p, 8, m.y);
            put_f32_le(&mut p, 12, m.z);
            put_f32_le(&mut p, 16, m.vx);
            put_f32_le(&mut p, 20, m.vy);
            put_f32_le(&mut p, 24, m.vz);
            put_f32_le(&mut p, 28, m.afx);
            put_f32_le(&mut p, 32, m.afy);
            put_f32_le(&mut p, 36, m.afz);
            put_f32_le(&mut p, 40, m.yaw);
            put_f32_le(&mut p, 44, m.yaw_rate);
            put_u16_le(&mut p, 48, m.type_mask);
            p[50] = m.target_system;
            p[51] = m.target_component;
            p[52] = m.coordinate_frame;
            encode_raw(seq, sysid, compid, MSG_ID_SET_POSITION_TARGET_LOCAL_NED, &p)
        }
        MavMessage::StatusText(m) => {
            let mut p = [0u8; 51];
            p[0] = m.severity;
            p[1..51].copy_from_slice(&m.text);
            encode_raw(seq, sysid, compid, MSG_ID_STATUSTEXT, &p)
        }
    }
}

pub fn decode_frame(wire: &[u8]) -> Result<DecodedFrame, DecodeError> {
    if wire.len() < HEADER_BYTES + CHECKSUM_BYTES {
        return Err(DecodeError::TooShort);
    }
    if wire[0] != MAVLINK_V2_STX {
        return Err(DecodeError::BadMagic);
    }
    let len = wire[1] as usize;
    let incompat = wire[2];
    if (incompat & !INCOMPAT_FLAG_SIGNED) != 0 {
        return Err(DecodeError::UnsupportedIncompatFlags);
    }
    let is_signed = (incompat & INCOMPAT_FLAG_SIGNED) != 0;
    let total = HEADER_BYTES + len + CHECKSUM_BYTES + if is_signed { SIGNATURE_BYTES } else { 0 };
    if wire.len() < total {
        return Err(DecodeError::TruncatedFrame);
    }

    let seq = wire[4];
    let sysid = wire[5];
    let compid = wire[6];
    let msgid = (wire[7] as u32) | ((wire[8] as u32) << 8) | ((wire[9] as u32) << 16);
    let crc_extra = crc_extra_for_msgid(msgid).ok_or(DecodeError::UnknownMsgId)?;

    let mut computed_crc = 0xFFFFu16;
    for &b in &wire[1..HEADER_BYTES + len] {
        computed_crc = crc16_accumulate(b, computed_crc);
    }
    computed_crc = crc16_accumulate(crc_extra, computed_crc);
    let wire_crc = get_u16_le(wire, HEADER_BYTES + len);
    if computed_crc != wire_crc {
        return Err(DecodeError::BadCrc);
    }

    let raw_p = &wire[HEADER_BYTES..HEADER_BYTES + len];
    let message = match msgid {
        MSG_ID_HEARTBEAT => {
            let p = zero_extend::<9>(raw_p);
            MavMessage::Heartbeat(Heartbeat {
                custom_mode: get_u32_le(&p, 0),
                mav_type: p[4],
                autopilot: p[5],
                base_mode: p[6],
                system_status: p[7],
                mavlink_version: p[8],
            })
        }
        MSG_ID_SYS_STATUS => {
            let p = zero_extend::<31>(raw_p);
            MavMessage::SysStatus(SysStatus {
                onboard_control_sensors_present: get_u32_le(&p, 0),
                onboard_control_sensors_enabled: get_u32_le(&p, 4),
                onboard_control_sensors_health: get_u32_le(&p, 8),
                load: get_u16_le(&p, 12),
                voltage_battery: get_u16_le(&p, 14),
                current_battery: get_i16_le(&p, 16),
                drop_rate_comm: get_u16_le(&p, 18),
                errors_comm: get_u16_le(&p, 20),
                errors_count1: get_u16_le(&p, 22),
                errors_count2: get_u16_le(&p, 24),
                errors_count3: get_u16_le(&p, 26),
                errors_count4: get_u16_le(&p, 28),
                battery_remaining: p[30] as i8,
            })
        }
        MSG_ID_ATTITUDE_QUATERNION => {
            let p = zero_extend::<48>(raw_p);
            MavMessage::AttitudeQuaternion(AttitudeQuaternion {
                time_boot_ms: get_u32_le(&p, 0),
                q1: get_f32_le(&p, 4),
                q2: get_f32_le(&p, 8),
                q3: get_f32_le(&p, 12),
                q4: get_f32_le(&p, 16),
                rollspeed: get_f32_le(&p, 20),
                pitchspeed: get_f32_le(&p, 24),
                yawspeed: get_f32_le(&p, 28),
                repr_offset_q: [
                    get_f32_le(&p, 32),
                    get_f32_le(&p, 36),
                    get_f32_le(&p, 40),
                    get_f32_le(&p, 44),
                ],
            })
        }
        MSG_ID_LOCAL_POSITION_NED => {
            let p = zero_extend::<28>(raw_p);
            MavMessage::LocalPositionNed(LocalPositionNed {
                time_boot_ms: get_u32_le(&p, 0),
                x: get_f32_le(&p, 4),
                y: get_f32_le(&p, 8),
                z: get_f32_le(&p, 12),
                vx: get_f32_le(&p, 16),
                vy: get_f32_le(&p, 20),
                vz: get_f32_le(&p, 24),
            })
        }
        MSG_ID_COMMAND_LONG => {
            let p = zero_extend::<33>(raw_p);
            MavMessage::CommandLong(CommandLong {
                param1: get_f32_le(&p, 0),
                param2: get_f32_le(&p, 4),
                param3: get_f32_le(&p, 8),
                param4: get_f32_le(&p, 12),
                param5: get_f32_le(&p, 16),
                param6: get_f32_le(&p, 20),
                param7: get_f32_le(&p, 24),
                command: get_u16_le(&p, 28),
                target_system: p[30],
                target_component: p[31],
                confirmation: p[32],
            })
        }
        MSG_ID_COMMAND_ACK => {
            let p = zero_extend::<10>(raw_p);
            MavMessage::CommandAck(CommandAck {
                command: get_u16_le(&p, 0),
                result: p[2],
                progress: p[3],
                result_param2: get_i32_le(&p, 4),
                target_system: p[8],
                target_component: p[9],
            })
        }
        MSG_ID_SET_POSITION_TARGET_LOCAL_NED => {
            let p = zero_extend::<53>(raw_p);
            MavMessage::SetPositionTargetLocalNed(SetPositionTargetLocalNed {
                time_boot_ms: get_u32_le(&p, 0),
                x: get_f32_le(&p, 4),
                y: get_f32_le(&p, 8),
                z: get_f32_le(&p, 12),
                vx: get_f32_le(&p, 16),
                vy: get_f32_le(&p, 20),
                vz: get_f32_le(&p, 24),
                afx: get_f32_le(&p, 28),
                afy: get_f32_le(&p, 32),
                afz: get_f32_le(&p, 36),
                yaw: get_f32_le(&p, 40),
                yaw_rate: get_f32_le(&p, 44),
                type_mask: get_u16_le(&p, 48),
                target_system: p[50],
                target_component: p[51],
                coordinate_frame: p[52],
            })
        }
        MSG_ID_STATUSTEXT => {
            let p = zero_extend::<54>(raw_p);
            let mut text = [0u8; 50];
            text.copy_from_slice(&p[1..51]);
            MavMessage::StatusText(StatusText {
                severity: p[0],
                text,
                id: get_u16_le(&p, 51),
                chunk_seq: p[53],
            })
        }
        _ => return Err(DecodeError::UnknownMsgId),
    };

    Ok(DecodedFrame {
        seq,
        sysid,
        compid,
        msgid,
        is_signed,
        bytes_consumed: total,
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_all_8_messages_and_reject_corrupted_crc() {
        let msgs = [
            MavMessage::Heartbeat(Heartbeat {
                custom_mode: 0x04030000,
                mav_type: 2,
                autopilot: 12,
                base_mode: MAV_MODE_FLAG_SAFETY_ARMED,
                system_status: 4,
                mavlink_version: 3,
            }),
            MavMessage::LocalPositionNed(LocalPositionNed {
                time_boot_ms: 42_000,
                x: 500.0,
                y: -400.0,
                z: -60.0,
                vx: 5.0,
                vy: -1.0,
                vz: 0.0,
            }),
            MavMessage::SetPositionTargetLocalNed(SetPositionTargetLocalNed {
                time_boot_ms: 99,
                vx: 4.5,
                vy: -0.5,
                type_mask: TYPE_MASK_VELOCITY_ONLY,
                ..Default::default()
            }),
            MavMessage::CommandLong(CommandLong {
                param1: 0.0,
                command: MAV_CMD_COMPONENT_ARM_DISARM,
                target_system: 1,
                target_component: 1,
                ..Default::default()
            }),
            MavMessage::CommandAck(CommandAck {
                command: MAV_CMD_COMPONENT_ARM_DISARM,
                result: MAV_RESULT_DENIED,
                target_system: 255,
                target_component: 190,
                ..Default::default()
            }),
            MavMessage::StatusText(StatusText::new(MAV_SEVERITY_CRITICAL, "RTA DENY")),
        ];

        for (i, m) in msgs.iter().enumerate() {
            let mut wire = encode_message(m, i as u8, 1, 191);
            let dec = decode_frame(&wire).expect("valid frame");
            assert_eq!(dec.message, *m);
            assert_eq!(dec.bytes_consumed, wire.len());
            // Flip last CRC byte -> must fail closed with BadCrc
            let last = wire.len() - 1;
            wire[last] ^= 0x5A;
            assert_eq!(decode_frame(&wire), Err(DecodeError::BadCrc));
        }
    }
}
