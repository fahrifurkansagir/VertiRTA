#include "vertirta/rta/mmap_cbvf_lut.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vertirta::rta {

namespace {

constexpr uint32_t kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

inline uint32_t rotr32(uint32_t x, uint32_t n) noexcept {
  return (x >> n) | (x << (32u - n));
}

void sha256_transform(uint32_t state[8], const uint8_t block[64]) noexcept {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<uint32_t>(block[4 * i]) << 24u) |
           (static_cast<uint32_t>(block[4 * i + 1]) << 16u) |
           (static_cast<uint32_t>(block[4 * i + 2]) << 8u) |
           static_cast<uint32_t>(block[4 * i + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = rotr32(w[i - 15], 7u) ^ rotr32(w[i - 15], 18u) ^ (w[i - 15] >> 3u);
    const uint32_t s1 = rotr32(w[i - 2], 17u) ^ rotr32(w[i - 2], 19u) ^ (w[i - 2] >> 10u);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }

  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t S1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t temp1 = h + S1 + ch + kSha256K[i] + w[i];
    const uint32_t S0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temp2 = S0 + maj;

    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

// Exact 152 B binary header matching reachability/solve_cbvf_luts.py (<8sII4I4f4f2f4f32s4fI3f)
struct RawVrtaLutHeader {
  char magic[8];
  uint32_t version;
  uint32_t num_regimes;
  uint32_t dims[4];
  float lo[4];
  float hi[4];
  float scale;
  float offset;
  float eps_grid[4];
  uint8_t payload_sha256[32];
  float v_cap_f[4];
  uint32_t flags;
  float horizon_s;
  float d_ar_max;
  float d_az_max;
};
static_assert(sizeof(RawVrtaLutHeader) == 152, "RawVrtaLutHeader size mismatch");

}  // namespace

void compute_sha256(const void* data, std::size_t len, uint8_t out_32[32]) noexcept {
  uint32_t state[8] = {
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
  };
  const auto* bytes = static_cast<const uint8_t*>(data);
  std::size_t pos = 0;
  while (pos + 64u <= len) {
    sha256_transform(state, bytes + pos);
    pos += 64u;
  }

  uint8_t tail[128]{};
  const std::size_t rem = len - pos;
  if (rem > 0u) {
    std::memcpy(tail, bytes + pos, rem);
  }
  tail[rem] = 0x80u;
  const std::size_t pad_len = (rem < 56u) ? 64u : 128u;
  const uint64_t bit_len = static_cast<uint64_t>(len) * 8ULL;
  for (int i = 0; i < 8; ++i) {
    tail[pad_len - 1u - static_cast<std::size_t>(i)] =
        static_cast<uint8_t>((bit_len >> (8u * static_cast<uint32_t>(i))) & 0xFFu);
  }
  sha256_transform(state, tail);
  if (pad_len == 128u) {
    sha256_transform(state, tail + 64u);
  }

  for (int i = 0; i < 8; ++i) {
    out_32[4 * i] = static_cast<uint8_t>((state[i] >> 24u) & 0xFFu);
    out_32[4 * i + 1] = static_cast<uint8_t>((state[i] >> 16u) & 0xFFu);
    out_32[4 * i + 2] = static_cast<uint8_t>((state[i] >> 8u) & 0xFFu);
    out_32[4 * i + 3] = static_cast<uint8_t>(state[i] & 0xFFu);
  }
}

MmapCbvf4DLut::~MmapCbvf4DLut() noexcept {
  unload();
}

MmapCbvf4DLut::MmapCbvf4DLut(MmapCbvf4DLut&& other) noexcept
    : mapped_(other.mapped_),
      mapped_size_(other.mapped_size_),
      fd_(other.fd_),
      loaded_(other.loaded_),
      data_(other.data_),
      dims_(other.dims_),
      lo_(other.lo_),
      hi_(other.hi_),
      inv_dx_(other.inv_dx_),
      eps_grid_(other.eps_grid_),
      v_cap_f_(other.v_cap_f_),
      payload_sha256_(other.payload_sha256_),
      scale_(other.scale_),
      offset_(other.offset_),
      horizon_s_(other.horizon_s_),
      d_ar_max_(other.d_ar_max_),
      d_az_max_(other.d_az_max_) {
  other.mapped_ = nullptr;
  other.mapped_size_ = 0;
  other.fd_ = -1;
  other.loaded_ = false;
  other.data_ = nullptr;
}

MmapCbvf4DLut& MmapCbvf4DLut::operator=(MmapCbvf4DLut&& other) noexcept {
  if (this != &other) {
    unload();
    mapped_ = other.mapped_;
    mapped_size_ = other.mapped_size_;
    fd_ = other.fd_;
    loaded_ = other.loaded_;
    data_ = other.data_;
    dims_ = other.dims_;
    lo_ = other.lo_;
    hi_ = other.hi_;
    inv_dx_ = other.inv_dx_;
    eps_grid_ = other.eps_grid_;
    v_cap_f_ = other.v_cap_f_;
    payload_sha256_ = other.payload_sha256_;
    scale_ = other.scale_;
    offset_ = other.offset_;
    horizon_s_ = other.horizon_s_;
    d_ar_max_ = other.d_ar_max_;
    d_az_max_ = other.d_az_max_;

    other.mapped_ = nullptr;
    other.mapped_size_ = 0;
    other.fd_ = -1;
    other.loaded_ = false;
    other.data_ = nullptr;
  }
  return *this;
}

void MmapCbvf4DLut::unload() noexcept {
  if (mapped_ != nullptr && mapped_ != MAP_FAILED) {
    (void)::munlock(mapped_, mapped_size_);
    (void)::munmap(mapped_, mapped_size_);
  }
  if (fd_ >= 0) {
    (void)::close(fd_);
  }
  mapped_ = nullptr;
  mapped_size_ = 0;
  fd_ = -1;
  data_ = nullptr;
  loaded_ = false;
}

bool MmapCbvf4DLut::load(std::string_view path, bool verify_sha256) noexcept {
  unload();
  const std::string path_str(path);
  const int fd = ::open(path_str.c_str(), O_RDONLY);
  if (fd < 0) {
    return false;
  }

  struct stat st {};
  if (::fstat(fd, &st) != 0 || static_cast<std::size_t>(st.st_size) != kTotalBytes) {
    (void)::close(fd);
    return false;
  }

  void* addr = ::mmap(nullptr, kTotalBytes, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) {
    (void)::close(fd);
    return false;
  }
  (void)::madvise(addr, kTotalBytes, MADV_WILLNEED);
  (void)::mlock(addr, kTotalBytes);  // Best-effort pin (unprivileged containers may enforce low RLIMIT_MEMLOCK)

  // Pre-fault all 4 KiB pages into resident RAM so RT 100 Hz eval() sees zero minor page faults.
  const auto* raw_bytes = static_cast<const volatile uint8_t*>(addr);
  volatile uint8_t prefault_acc = 0;
  for (std::size_t off = 0; off < kTotalBytes; off += 4096u) {
    prefault_acc ^= raw_bytes[off];
  }
  (void)prefault_acc;

  RawVrtaLutHeader hdr{};
  std::memcpy(&hdr, addr, sizeof(hdr));
  if (std::memcmp(hdr.magic, "VRTALUT1", 8) != 0 || hdr.version != 1u ||
      hdr.num_regimes != kNumRegimes || hdr.dims[0] != kN0 || hdr.dims[1] != kN1 ||
      hdr.dims[2] != kN2 || hdr.dims[3] != kN3) {
    (void)::munlock(addr, kTotalBytes);
    (void)::munmap(addr, kTotalBytes);
    (void)::close(fd);
    return false;
  }

  if (!std::isfinite(hdr.scale) || hdr.scale <= 0.0f || !std::isfinite(hdr.offset)) {
    (void)::munlock(addr, kTotalBytes);
    (void)::munmap(addr, kTotalBytes);
    (void)::close(fd);
    return false;
  }

  for (int d = 0; d < 4; ++d) {
    if (!std::isfinite(hdr.lo[d]) || !std::isfinite(hdr.hi[d]) || hdr.hi[d] <= hdr.lo[d] ||
        !std::isfinite(hdr.eps_grid[d]) || !std::isfinite(hdr.v_cap_f[d])) {
      (void)::munlock(addr, kTotalBytes);
      (void)::munmap(addr, kTotalBytes);
      (void)::close(fd);
      return false;
    }
  }

  const auto* payload_ptr = static_cast<const uint8_t*>(addr) + kHeaderBytes;
  if (verify_sha256) {
    uint8_t digest[32]{};
    compute_sha256(payload_ptr, kPayloadBytes, digest);
    if (std::memcmp(digest, hdr.payload_sha256, 32) != 0) {
      (void)::munlock(addr, kTotalBytes);
      (void)::munmap(addr, kTotalBytes);
      (void)::close(fd);
      return false;
    }
  }

  mapped_ = addr;
  mapped_size_ = kTotalBytes;
  fd_ = fd;
  data_ = reinterpret_cast<const int16_t*>(payload_ptr);
  for (int d = 0; d < 4; ++d) {
    dims_[static_cast<std::size_t>(d)] = hdr.dims[d];
    lo_[static_cast<std::size_t>(d)] = static_cast<double>(hdr.lo[d]);
    hi_[static_cast<std::size_t>(d)] = static_cast<double>(hdr.hi[d]);
    inv_dx_[static_cast<std::size_t>(d)] =
        static_cast<double>(hdr.dims[d] - 1u) /
        (hi_[static_cast<std::size_t>(d)] - lo_[static_cast<std::size_t>(d)]);
    eps_grid_[static_cast<std::size_t>(d)] = static_cast<double>(hdr.eps_grid[d]);
    v_cap_f_[static_cast<std::size_t>(d)] = static_cast<double>(hdr.v_cap_f[d]);
  }
  std::memcpy(payload_sha256_.data(), hdr.payload_sha256, 32);
  scale_ = static_cast<double>(hdr.scale);
  offset_ = static_cast<double>(hdr.offset);
  horizon_s_ = std::isfinite(hdr.horizon_s) ? static_cast<double>(hdr.horizon_s) : 6.0;
  d_ar_max_ = std::isfinite(hdr.d_ar_max) ? static_cast<double>(hdr.d_ar_max) : 1.75;
  d_az_max_ = std::isfinite(hdr.d_az_max) ? static_cast<double>(hdr.d_az_max) : 0.75;
  loaded_ = true;
  return true;
}

CbvfQueryResult MmapCbvf4DLut::eval(
    uint8_t regime_idx,
    double r,
    double vr,
    double z,
    double vz) const noexcept {
  // DR-2 Erratum #4: strict guard before any pointer arithmetic or indexing.
  if (!loaded_ || data_ == nullptr || regime_idx >= kNumRegimes ||
      !std::isfinite(r) || !std::isfinite(vr) || !std::isfinite(z) || !std::isfinite(vz)) {
    return CbvfQueryResult{};
  }

  constexpr std::size_t kS2 = kN3;
  constexpr std::size_t kS1 = static_cast<std::size_t>(kN2) * kN3;
  constexpr std::size_t kS0 = static_cast<std::size_t>(kN1) * kN2 * kN3;

  const double q[4] = {r, vr, z, vz};
  uint32_t idx[4]{};
  double frac[4]{};
  for (int d = 0; d < 4; ++d) {
    const auto du = static_cast<std::size_t>(d);
    const double u = (q[d] - lo_[du]) * inv_dx_[du];
    const double uc = std::clamp(u, 0.0, static_cast<double>(dims_[du] - 1u) - 1e-9);
    idx[d] = static_cast<uint32_t>(uc);
    frac[d] = uc - static_cast<double>(idx[d]);
  }

  const int16_t* base =
      data_ + static_cast<std::size_t>(regime_idx) * kElementsPerRegime +
      static_cast<std::size_t>(idx[0]) * kS0 +
      static_cast<std::size_t>(idx[1]) * kS1 +
      static_cast<std::size_t>(idx[2]) * kS2 +
      static_cast<std::size_t>(idx[3]);

  double c[16];
  for (uint32_t b0 = 0; b0 < 2u; ++b0) {
    for (uint32_t b1 = 0; b1 < 2u; ++b1) {
      for (uint32_t b2 = 0; b2 < 2u; ++b2) {
        const int16_t* p = base + b0 * kS0 + b1 * kS1 + b2 * kS2;
        const uint32_t k = (b0 << 3u) | (b1 << 2u) | (b2 << 1u);
        c[k] = static_cast<double>(p[0]) * scale_ + offset_;
        c[k | 1u] = static_cast<double>(p[1]) * scale_ + offset_;
      }
    }
  }

  const double f0 = frac[0], f1 = frac[1], f2 = frac[2], f3 = frac[3];
  const double o0 = 1.0 - f0, o1 = 1.0 - f1, o2 = 1.0 - f2, o3 = 1.0 - f3;

  double a[8], da3[8];
  for (int k = 0; k < 8; ++k) {
    const double v0 = c[2 * k];
    const double v1 = c[2 * k + 1];
    a[k] = v0 * o3 + v1 * f3;
    da3[k] = v1 - v0;
  }

  double b[4], db2[4], db3[4];
  for (int k = 0; k < 4; ++k) {
    b[k] = a[2 * k] * o2 + a[2 * k + 1] * f2;
    db2[k] = a[2 * k + 1] - a[2 * k];
    db3[k] = da3[2 * k] * o2 + da3[2 * k + 1] * f2;
  }

  double g[2], dg1[2], dg2[2], dg3[2];
  for (int k = 0; k < 2; ++k) {
    g[k] = b[2 * k] * o1 + b[2 * k + 1] * f1;
    dg1[k] = b[2 * k + 1] - b[2 * k];
    dg2[k] = db2[2 * k] * o1 + db2[2 * k + 1] * f1;
    dg3[k] = db3[2 * k] * o1 + db3[2 * k + 1] * f1;
  }

  CbvfQueryResult res{};
  res.value_m = (g[0] * o0 + g[1] * f0) - eps_grid_[regime_idx];
  res.grad[0] = (g[1] - g[0]) * inv_dx_[0];
  res.grad[1] = (dg1[0] * o0 + dg1[1] * f0) * inv_dx_[1];
  res.grad[2] = (dg2[0] * o0 + dg2[1] * f0) * inv_dx_[2];
  res.grad[3] = (dg3[0] * o0 + dg3[1] * f0) * inv_dx_[3];
  res.valid = true;
  return res;
}

}  // namespace vertirta::rta
