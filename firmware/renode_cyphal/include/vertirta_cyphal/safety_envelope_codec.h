/* SPDX-License-Identifier: Apache-2.0 */
/* Nunavut-aligned C11 Serializer/Deserializer for vertirta.flight.SafetyEnvelope.1.0
 * and uavcan.node.Heartbeat.1.0 (DR-08 Sec 2.2, zero heap, endian-safe). */
#ifndef VERTIRTA_CYPHAL_SAFETY_ENVELOPE_CODEC_H
#define VERTIRTA_CYPHAL_SAFETY_ENVELOPE_CODEC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* OpenCyphal v1.0 fixed Subject-IDs and CAN-FD MTU contracts. */
#define VERTIRTA_SAFETY_ENVELOPE_SUBJECT_ID 1024u
#define VERTIRTA_HEARTBEAT_SUBJECT_ID 7509u
#define VERTIRTA_CANFD_MTU_BYTES 64u
#define VERTIRTA_CYPHAL_TAIL_BYTE_SIZE 1u
#define VERTIRTA_CANFD_SINGLE_FRAME_MAX_PAYLOAD_BYTES 63u

#define VERTIRTA_SAFETY_ENVELOPE_EXTENT_BYTES 48u
#define VERTIRTA_SAFETY_ENVELOPE_SERIALIZATION_BUFFER_SIZE_BYTES 48u
#define VERTIRTA_HEARTBEAT_EXTENT_BYTES 7u

/* ISO 11898-1:2015 CAN-FD wire-time (1 Mbps nominal / 5 Mbps data, 29-bit extended ID)
 * and <20 ms dual-FCC failover budget constants (DR-08 Sec 2.3). */
#define VERTIRTA_CANFD_WIRE_TIME_48B_TENTHS_US 1606u   /* 160.6 us */
#define VERTIRTA_CANFD_WIRE_TIME_64B_TENTHS_US 1926u   /* 192.6 us */
#define VERTIRTA_FAILOVER_PERIOD_US 2500u              /* 400 Hz = 2.50 ms */
#define VERTIRTA_FAILOVER_WATCHDOG_4FRAME_US 10500u    /* 10.50 ms (4 missed + 0.50 ms jitter) */
#define VERTIRTA_FAILOVER_WATCHDOG_3FRAME_US 8000u     /*  8.00 ms (3 missed + 0.50 ms jitter) */
#define VERTIRTA_FAILOVER_BUS_HANDOVER_US 400u         /*  0.40 ms blocking + Exceptional handover */
#define VERTIRTA_FAILOVER_SW_ALLOWANCE_US 350u         /*  0.35 ms FDCAN ISR + o1heap + FSM */
#define VERTIRTA_FAILOVER_FIRST_FRAME_US 2500u         /*  2.50 ms first active actuator frame */
#define VERTIRTA_FAILOVER_MODELED_TOTAL_US 13750u      /* 13.75 ms (< 20.00 ms, 6.25 ms margin) */
#define VERTIRTA_FAILOVER_HARD_CEILING_US 20000u       /* 20.00 ms hard upper bound */

/* Codec return codes (aligned with Nunavut C11 negative error convention). */
#define VERTIRTA_CODEC_OK 0
#define VERTIRTA_CODEC_ERR_INVALID_ARG (-2)
#define VERTIRTA_CODEC_ERR_BUF_TOO_SMALL (-3)
#define VERTIRTA_CODEC_ERR_CRC_MISMATCH (-4)
#define VERTIRTA_CODEC_ERR_NON_FINITE (-5)

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(VERTIRTA_SAFETY_ENVELOPE_EXTENT_BYTES == 48u,
               "SafetyEnvelope.1.0 must serialize to exactly 48 bytes (384 bits)");
_Static_assert((VERTIRTA_SAFETY_ENVELOPE_EXTENT_BYTES + VERTIRTA_CYPHAL_TAIL_BYTE_SIZE) <=
                   VERTIRTA_CANFD_MTU_BYTES,
               "SafetyEnvelope.1.0 + Cyphal tail byte must fit in one 64-byte CAN-FD frame");
_Static_assert(VERTIRTA_FAILOVER_MODELED_TOTAL_US < VERTIRTA_FAILOVER_HARD_CEILING_US,
               "Modeled failover budget must remain strictly below 20 ms");
#endif

typedef struct {
  float wrench_min_frd[4]; /* [Fz, Mx, My, Mz] */
  float wrench_max_frd[4]; /* [Fz, Mx, My, Mz] */
  uint32_t seq;
  uint32_t t_us;
  uint8_t fault_mask;
  uint8_t flight_regime;
  uint8_t tmr_state;
  uint8_t cel_latched;
  uint32_t crc32c;
} vertirta_flight_SafetyEnvelope_1_0;

typedef struct {
  uint32_t uptime;
  uint8_t health;
  uint8_t mode;
  uint8_t vendor_specific_status_code;
} vertirta_uavcan_node_Heartbeat_1_0;

/* RFC 3720 / iSCSI Castagnoli CRC-32C (polynomial 0x82F63B78). */
uint32_t vertirta_crc32c_c11(const uint8_t* data, size_t len);

/* Serializes SafetyEnvelope.1.0 into exact 48 little-endian bytes and populates CRC-32C. */
int32_t vertirta_flight_SafetyEnvelope_1_0_serialize_(
    const vertirta_flight_SafetyEnvelope_1_0* obj,
    uint8_t* buffer,
    size_t* inout_buffer_size_bytes);

/* Deserializes exact 48 bytes, verifying CRC-32C and IEEE-754 finiteness of all wrench axes. */
int32_t vertirta_flight_SafetyEnvelope_1_0_deserialize_(
    vertirta_flight_SafetyEnvelope_1_0* out_obj,
    const uint8_t* buffer,
    size_t* inout_buffer_size_bytes);

/* Serializes / deserializes standard OpenCyphal uavcan.node.Heartbeat.1.0 (7 B, Subject-ID 7509). */
int32_t vertirta_uavcan_node_Heartbeat_1_0_serialize_(
    const vertirta_uavcan_node_Heartbeat_1_0* obj,
    uint8_t* buffer,
    size_t* inout_buffer_size_bytes);

int32_t vertirta_uavcan_node_Heartbeat_1_0_deserialize_(
    vertirta_uavcan_node_Heartbeat_1_0* out_obj,
    const uint8_t* buffer,
    size_t* inout_buffer_size_bytes);

#ifdef __cplusplus
}
#endif

#endif /* VERTIRTA_CYPHAL_SAFETY_ENVELOPE_CODEC_H */
