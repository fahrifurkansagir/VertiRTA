// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#ifdef VERTIRTA_HAS_ORT
#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>
#endif

namespace vertirta::perception {

inline constexpr uint32_t kSlzInputWidth = 512u;
inline constexpr uint32_t kSlzInputHeight = 512u;
inline constexpr uint32_t kSlzInputChannels = 3u;
inline constexpr uint32_t kSlzNumClasses = 5u;
inline constexpr uint32_t kSlzNormalChannels = 3u;
inline constexpr uint32_t kSlzBdryWidth = 64u;
inline constexpr uint32_t kSlzBdryHeight = 64u;

inline constexpr std::size_t kSlzRgbElements =
    static_cast<std::size_t>(kSlzInputChannels) * kSlzInputHeight * kSlzInputWidth;
inline constexpr std::size_t kSlzLogitsElements =
    static_cast<std::size_t>(kSlzNumClasses) * kSlzInputHeight * kSlzInputWidth;
inline constexpr std::size_t kSlzNormalsElements =
    static_cast<std::size_t>(kSlzNormalChannels) * kSlzInputHeight * kSlzInputWidth;
inline constexpr std::size_t kSlzPixels =
    static_cast<std::size_t>(kSlzInputHeight) * kSlzInputWidth;
inline constexpr std::size_t kSlzBdryElements =
    static_cast<std::size_t>(kSlzBdryHeight) * kSlzBdryWidth;

struct SlzOrtConfig {
  std::string onnx_path{};
  std::string trt_cache_dir{};
  std::string ood_calibration_path{};
  bool use_tensorrt{false};
  bool enable_fp16{false};
  bool enable_cuda_graph{true};
  bool allow_mock_fallback{true};
  std::string arena_extend_strategy{"kSameAsRequested"};
  std::string cudnn_conv_algo_search{"AUTO"};
  float t_ood{1.0f};
  float t_prob{1.25f};
  float kappa_b{1.5f};
  float tau_ood{-8.0f};
};

struct CudaProviderTuning {
  std::string arena_extend_strategy{"kSameAsRequested"};
  std::string cudnn_conv_algo_search{"HEURISTIC"};
  bool cudnn_conv_use_max_workspace{false};
  bool enable_cuda_graph{true};
};

// Resolves platform-specific ORT CUDA ExecutionProvider tuning for TU116 (4 GB GTX 1650 SUPER):
//   "AUTO" -> Windows WDDM: EXHAUSTIVE + max_workspace=1; Linux/WSL2: HEURISTIC + max_workspace=0.
[[nodiscard]] CudaProviderTuning ResolveCudaProviderTuning(
    const SlzOrtConfig& cfg,
#ifdef _WIN32
    bool is_windows_host = true
#else
    bool is_windows_host = false
#endif
) noexcept;

// Parses ood_calibration.json text (T_ood, T_prob, kappa_b, tau_ood) into cfg with finite bounds checks.
// Zero external JSON dependencies; leaves cfg unchanged and returns false on any malformed/out-of-range value.
[[nodiscard]] bool ParseOodCalibrationJson(
    std::string_view json_text,
    SlzOrtConfig& cfg) noexcept;

// Loads ood_calibration.json from disk and updates cfg.{t_ood, t_prob, kappa_b, tau_ood}.
[[nodiscard]] bool LoadOodCalibrationJson(
    const std::string& path,
    SlzOrtConfig& cfg);

// Computes per-pixel Helmholtz Free Energy: E(u) = -T * log(sum_c exp(f_c(u) / T))
// from NCHW seg_logits [5, 512, 512]. Zero heap allocations.
[[nodiscard]] bool ComputeFreeEnergy(
    std::span<const float> seg_logits_5x512x512,
    float t_ood,
    std::span<float> e_raw_512) noexcept;

// Bilinearly upsamples PIDNet-S 64x64 boundary probability map to 512x512 and computes
// boundary-compensated OOD energy: E_tilde(u) = E_raw(u) - kappa_b * b_up(u).
// Zero heap allocations.
[[nodiscard]] bool ApplyUpsampledBoundaryCompensation(
    std::span<const float> e_raw_512,
    std::span<const float> bdry_64,
    float kappa_b,
    std::span<float> e_out_512) noexcept;

// Non-copyable, non-movable RAII ONNX Runtime / TensorRT V2 inference engine.
// Guarantees session & I/O binding teardown strictly BEFORE CUDA stream destruction.
// Falls back cleanly to a deterministic zero-heap mock runner when compiled without ORT
// or when cfg.onnx_path is empty with allow_mock_fallback == true.
class SlzOrtRunner {
 public:
  explicit SlzOrtRunner(const SlzOrtConfig& cfg);
  ~SlzOrtRunner() noexcept;

  SlzOrtRunner(const SlzOrtRunner&) = delete;
  SlzOrtRunner& operator=(const SlzOrtRunner&) = delete;
  SlzOrtRunner(SlzOrtRunner&&) = delete;
  SlzOrtRunner& operator=(SlzOrtRunner&&) = delete;

  // Executes single forward pass into caller-owned spans (zero heap in hot loop).
  [[nodiscard]] bool Run(
      std::span<const float> rgb_nchw_512,
      std::span<float> seg_logits_5x512x512,
      std::span<float> surf_normals_3x512x512,
      std::span<float> bdry_prob_64x64) noexcept;

  // Configures constant class logits returned by deterministic mock mode (testing only).
  void SetMockOutputs(
      const float class_logits[kSlzNumClasses],
      float bdry_prob_val = 0.0f) noexcept;

  [[nodiscard]] bool is_mock_backend() const noexcept { return using_mock_; }
  [[nodiscard]] bool is_ready() const noexcept { return ready_; }
  [[nodiscard]] const SlzOrtConfig& config() const noexcept { return cfg_; }

 private:
  SlzOrtConfig cfg_{};
  bool using_mock_{true};
  bool ready_{false};
  float mock_logits_[kSlzNumClasses]{-1.0f, 12.0f, -3.0f, -4.0f, -4.0f};
  float mock_bdry_{0.0f};
  bool use_custom_mock_{false};

#ifdef VERTIRTA_HAS_ORT
  void ReleaseCudaResources() noexcept;
  void AllocateStaticIoBindings();

  // Member declaration order enforces reverse destruction: io_binding_ -> session_ -> env_,
  // while ~SlzOrtRunner explicitly destroys session_ BEFORE cudaStreamDestroy(stream_).
  cudaStream_t stream_{nullptr};
  float* d_input_{nullptr};
  float* d_logits_{nullptr};
  float* d_normals_{nullptr};
  float* d_bdry_{nullptr};
  std::unique_ptr<Ort::Env> env_{};
  std::unique_ptr<Ort::Session> session_{};
  std::unique_ptr<Ort::IoBinding> io_binding_{};
#endif
};

}  // namespace vertirta::perception
