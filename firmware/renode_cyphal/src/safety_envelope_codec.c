/* SPDX-License-Identifier: Apache-2.0 */
#include "vertirta_cyphal/safety_envelope_codec.h"

#include <string.h>

static bool float_is_finite_u32(uint32_t bits) {
  return (bits & 0x7F800000u) != 0x7F800000u;
}

static void store_u32_le(uint8_t* dst, uint32_t val) {
  dst[0] = (uint8_t)(val & 0xFFu);
  dst[1] = (uint8_t)((val >> 8u) & 0xFFu);
  dst[2] = (uint8_t)((val >> 16u) & 0xFFu);
  dst[3] = (uint8_t)((val >> 24u) & 0xFFu);
}

static uint32_t load_u32_le(const uint8_t* src) {
  return ((uint32_t)src[0]) | (((uint32_t)src[1]) << 8u) | (((uint32_t)src[2]) << 16u) |
         (((uint32_t)src[3]) << 24u);
}

uint32_t vertirta_crc32c_c11(const uint8_t* data, size_t len) {
  uint32_t crc = 0xFFFFFFFFu;
  if (data == NULL) {
    return 0u;
  }
  for (size_t i = 0u; i < len; ++i) {
    crc ^= (uint32_t)data[i];
    for (uint32_t bit = 0u; bit < 8u; ++bit) {
      const uint32_t mask = (uint32_t)-(int32_t)(crc & 1u);
      crc = (crc >> 1u) ^ (0x82F63B78u & mask);
    }
  }
  return ~crc;
}

int32_t vertirta_flight_SafetyEnvelope_1_0_serialize_(
    const vertirta_flight_SafetyEnvelope_1_0* obj,
    uint8_t* buffer,
    size_t* inout_buffer_size_bytes) {
  if ((obj == NULL) || (buffer == NULL) || (inout_buffer_size_bytes == NULL)) {
    return VERTIRTA_CODEC_ERR_INVALID_ARG;
  }
  if (*inout_buffer_size_bytes < VERTIRTA_SAFETY_ENVELOPE_EXTENT_BYTES) {
    return VERTIRTA_CODEC_ERR_BUF_TOO_SMALL;
  }

  size_t off = 0u;
  for (size_t i = 0u; i < 4u; ++i) {
    uint32_t bits = 0u;
    memcpy(&bits, &obj->wrench_min_frd[i], sizeof(bits));
    if (!float_is_finite_u32(bits)) {
      return VERTIRTA_CODEC_ERR_NON_FINITE;
    }
    store_u32_le(&buffer[off], bits);
    off += 4u;
  }
  for (size_t i = 0u; i < 4u; ++i) {
    uint32_t bits = 0u;
    memcpy(&bits, &obj->wrench_max_frd[i], sizeof(bits));
    if (!float_is_finite_u32(bits)) {
      return VERTIRTA_CODEC_ERR_NON_FINITE;
    }
    store_u32_le(&buffer[off], bits);
    off += 4u;
  }

  store_u32_le(&buffer[off], obj->seq);
  off += 4u;
  store_u32_le(&buffer[off], obj->t_us);
  off += 4u;

  buffer[off++] = obj->fault_mask;
  buffer[off++] = obj->flight_regime;
  buffer[off++] = obj->tmr_state;
  buffer[off++] = obj->cel_latched;

  /* off == 44u here; append Castagnoli CRC-32C covering bytes [0..43]. */
  const uint32_t crc = vertirta_crc32c_c11(buffer, off);
  store_u32_le(&buffer[off], crc);
  off += 4u;

  *inout_buffer_size_bytes = off;
  return VERTIRTA_CODEC_OK;
}

int32_t vertirta_flight_SafetyEnvelope_1_0_deserialize_(
    vertirta_flight_SafetyEnvelope_1_0* out_obj,
    const uint8_t* buffer,
    size_t* inout_buffer_size_bytes) {
  if ((out_obj == NULL) || (buffer == NULL) || (inout_buffer_size_bytes == NULL)) {
    return VERTIRTA_CODEC_ERR_INVALID_ARG;
  }
  if (*inout_buffer_size_bytes < VERTIRTA_SAFETY_ENVELOPE_EXTENT_BYTES) {
    return VERTIRTA_CODEC_ERR_BUF_TOO_SMALL;
  }

  const uint32_t expected_crc = load_u32_le(&buffer[44u]);
  const uint32_t actual_crc = vertirta_crc32c_c11(buffer, 44u);
  if (expected_crc != actual_crc) {
    return VERTIRTA_CODEC_ERR_CRC_MISMATCH;
  }

  vertirta_flight_SafetyEnvelope_1_0 tmp;
  size_t off = 0u;
  for (size_t i = 0u; i < 4u; ++i) {
    const uint32_t bits = load_u32_le(&buffer[off]);
    if (!float_is_finite_u32(bits)) {
      return VERTIRTA_CODEC_ERR_NON_FINITE;
    }
    memcpy(&tmp.wrench_min_frd[i], &bits, sizeof(bits));
    off += 4u;
  }
  for (size_t i = 0u; i < 4u; ++i) {
    const uint32_t bits = load_u32_le(&buffer[off]);
    if (!float_is_finite_u32(bits)) {
      return VERTIRTA_CODEC_ERR_NON_FINITE;
    }
    memcpy(&tmp.wrench_max_frd[i], &bits, sizeof(bits));
    off += 4u;
  }

  tmp.seq = load_u32_le(&buffer[off]);
  off += 4u;
  tmp.t_us = load_u32_le(&buffer[off]);
  off += 4u;
  tmp.fault_mask = buffer[off++];
  tmp.flight_regime = buffer[off++];
  tmp.tmr_state = buffer[off++];
  tmp.cel_latched = buffer[off++];
  tmp.crc32c = expected_crc;

  *out_obj = tmp;
  *inout_buffer_size_bytes = VERTIRTA_SAFETY_ENVELOPE_EXTENT_BYTES;
  return VERTIRTA_CODEC_OK;
}

int32_t vertirta_uavcan_node_Heartbeat_1_0_serialize_(
    const vertirta_uavcan_node_Heartbeat_1_0* obj,
    uint8_t* buffer,
    size_t* inout_buffer_size_bytes) {
  if ((obj == NULL) || (buffer == NULL) || (inout_buffer_size_bytes == NULL)) {
    return VERTIRTA_CODEC_ERR_INVALID_ARG;
  }
  if (*inout_buffer_size_bytes < VERTIRTA_HEARTBEAT_EXTENT_BYTES) {
    return VERTIRTA_CODEC_ERR_BUF_TOO_SMALL;
  }
  store_u32_le(&buffer[0u], obj->uptime);
  buffer[4u] = (uint8_t)(obj->health & 0x03u);
  buffer[5u] = (uint8_t)(obj->mode & 0x07u);
  buffer[6u] = obj->vendor_specific_status_code;
  *inout_buffer_size_bytes = VERTIRTA_HEARTBEAT_EXTENT_BYTES;
  return VERTIRTA_CODEC_OK;
}

int32_t vertirta_uavcan_node_Heartbeat_1_0_deserialize_(
    vertirta_uavcan_node_Heartbeat_1_0* out_obj,
    const uint8_t* buffer,
    size_t* inout_buffer_size_bytes) {
  if ((out_obj == NULL) || (buffer == NULL) || (inout_buffer_size_bytes == NULL)) {
    return VERTIRTA_CODEC_ERR_INVALID_ARG;
  }
  if (*inout_buffer_size_bytes < VERTIRTA_HEARTBEAT_EXTENT_BYTES) {
    return VERTIRTA_CODEC_ERR_BUF_TOO_SMALL;
  }
  out_obj->uptime = load_u32_le(&buffer[0u]);
  out_obj->health = (uint8_t)(buffer[4u] & 0x03u);
  out_obj->mode = (uint8_t)(buffer[5u] & 0x07u);
  out_obj->vendor_specific_status_code = buffer[6u];
  *inout_buffer_size_bytes = VERTIRTA_HEARTBEAT_EXTENT_BYTES;
  return VERTIRTA_CODEC_OK;
}
