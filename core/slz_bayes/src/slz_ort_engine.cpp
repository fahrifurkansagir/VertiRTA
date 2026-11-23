// SPDX-License-Identifier: Apache-2.0
#include "vertirta/perception/slz_ort_engine.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace vertirta::perception {

namespace {

[[nodiscard]] bool ExtractJsonFloatField(
    std::string_view json_text,
    std::string_view key,
    float& out_val) noexcept {
  const std::string needle = "\"" + std::string(key) + "\"";
  std::size_t pos = json_text.find(needle);
  if (pos == std::string_view::npos) {
    return false;
  }
  pos += needle.size();
  while (pos < json_text.size() && std::isspace(static_cast<unsigned char>(json_text[pos]))) {
    ++pos;
  }
  if (pos >= json_text.size() || json_text[pos] != ':') {
    return false;
  }
  ++pos;
  while (pos < json_text.size() && std::isspace(static_cast<unsigned char>(json_text[pos]))) {
    ++pos;
  }
  if (pos >= json_text.size()) {
    return false;
  }
  char buf[64]{};
  std::size_t n = 0u;
  while (pos < json_text.size() && n + 1u < sizeof(buf)) {
    const char ch = json_text[pos];
    if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-' || ch == '+' ||
        ch == '.' || ch == 'e' || ch == 'E') {
      buf[n++] = ch;
      ++pos;
    } else {
      break;
    }
  }
  if (n == 0u) {
    return false;
  }
  buf[n] = '\0';
  char* end_ptr = nullptr;
  const double parsed = std::strtod(buf, &end_ptr);
  if (end_ptr == buf || *end_ptr != '\0' || !std::isfinite(parsed)) {
    return false;
  }
  out_val = static_cast<float>(parsed);
  return std::isfinite(out_val);
}

}  // namespace

CudaProviderTuning ResolveCudaProviderTuning(
    const SlzOrtConfig& cfg,
    bool is_windows_host) noexcept {
  CudaProviderTuning out{};
  out.arena_extend_strategy =
      cfg.arena_extend_strategy.empty() ? "kSameAsRequested" : cfg.arena_extend_strategy;
  out.enable_cuda_graph = cfg.enable_cuda_graph;

  std::string algo = cfg.cudnn_conv_algo_search;
  for (char& ch : algo) {
    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  }
  if (algo.empty() || algo == "AUTO") {
    if (is_windows_host) {
      out.cudnn_conv_algo_search = "EXHAUSTIVE";
      out.cudnn_conv_use_max_workspace = true;
    } else {
      out.cudnn_conv_algo_search = "HEURISTIC";
      out.cudnn_conv_use_max_workspace = false;
    }
  } else {
    out.cudnn_conv_algo_search = std::move(algo);
    out.cudnn_conv_use_max_workspace = (out.cudnn_conv_algo_search == "EXHAUSTIVE");
  }
  return out;
}

bool ParseOodCalibrationJson(std::string_view json_text, SlzOrtConfig& cfg) noexcept {
  float t_ood = 0.0f;
  float t_prob = 0.0f;
  float kappa_b = 0.0f;
  float tau_ood = 0.0f;
  if (!ExtractJsonFloatField(json_text, "T_ood", t_ood) ||
      !ExtractJsonFloatField(json_text, "T_prob", t_prob) ||
      !ExtractJsonFloatField(json_text, "kappa_b", kappa_b) ||
      !ExtractJsonFloatField(json_text, "tau_ood", tau_ood)) {
    return false;
  }
  if (!(t_ood > 1e-4f && t_ood <= 10.0f) ||
      !(t_prob >= 0.5f && t_prob <= 3.0f) ||
      !(kappa_b >= 0.0f && kappa_b <= 10.0f) ||
      !std::isfinite(tau_ood)) {
    return false;
  }
  cfg.t_ood = t_ood;
  cfg.t_prob = t_prob;
  cfg.kappa_b = kappa_b;
  cfg.tau_ood = tau_ood;
  return true;
}

bool LoadOodCalibrationJson(const std::string& path, SlzOrtConfig& cfg) {
  if (path.empty()) {
    return false;
  }
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return false;
  }
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return ParseOodCalibrationJson(oss.str(), cfg);
}

bool ComputeFreeEnergy(
    std::span<const float> seg_logits_5x512x512,
    float t_ood,
    std::span<float> e_raw_512) noexcept {
  if (seg_logits_5x512x512.size() != kSlzLogitsElements ||
      e_raw_512.size() != kSlzPixels ||
      !std::isfinite(t_ood) || t_ood <= 1e-6f) {
    return false;
  }

  const float inv_t = 1.0f / t_ood;
  for (std::size_t idx = 0; idx < kSlzPixels; ++idx) {
    float max_scaled = -std::numeric_limits<float>::infinity();
    bool finite_ok = true;
    for (uint32_t c = 0; c < kSlzNumClasses; ++c) {
      const float v = seg_logits_5x512x512[static_cast<std::size_t>(c) * kSlzPixels + idx];
      if (!std::isfinite(v)) {
        finite_ok = false;
        break;
      }
      const float scaled = v * inv_t;
      if (scaled > max_scaled) {
        max_scaled = scaled;
      }
    }
    if (!finite_ok) {
      e_raw_512[idx] = 0.0f;  // fail-closed high free energy (OOD)
      continue;
    }
    float sum_exp = 0.0f;
    for (uint32_t c = 0; c < kSlzNumClasses; ++c) {
      const float v = seg_logits_5x512x512[static_cast<std::size_t>(c) * kSlzPixels + idx];
      sum_exp += std::exp(v * inv_t - max_scaled);
    }
    e_raw_512[idx] = -t_ood * (max_scaled + std::log(std::max(sum_exp, 1e-30f)));
  }
  return true;
}

bool ApplyUpsampledBoundaryCompensation(
    std::span<const float> e_raw_512,
    std::span<const float> bdry_64,
    float kappa_b,
    std::span<float> e_out_512) noexcept {
  if (e_raw_512.size() != kSlzPixels ||
      bdry_64.size() != kSlzBdryElements ||
      e_out_512.size() != kSlzPixels ||
      !std::isfinite(kappa_b)) {
    return false;
  }

  constexpr float kScaleY = static_cast<float>(kSlzBdryHeight) / static_cast<float>(kSlzInputHeight);
  constexpr float kScaleX = static_cast<float>(kSlzBdryWidth) / static_cast<float>(kSlzInputWidth);
  constexpr float kMaxY = static_cast<float>(kSlzBdryHeight - 1u);
  constexpr float kMaxX = static_cast<float>(kSlzBdryWidth - 1u);

  for (uint32_t y = 0; y < kSlzInputHeight; ++y) {
    const float sy = std::clamp((static_cast<float>(y) + 0.5f) * kScaleY - 0.5f, 0.0f, kMaxY);
    const uint32_t y0 = static_cast<uint32_t>(sy);
    const uint32_t y1 = std::min(y0 + 1u, kSlzBdryHeight - 1u);
    const float wy = sy - static_cast<float>(y0);
    const float wy0 = 1.0f - wy;

    const std::size_t row_out = static_cast<std::size_t>(y) * kSlzInputWidth;
    const std::size_t row_b0 = static_cast<std::size_t>(y0) * kSlzBdryWidth;
    const std::size_t row_b1 = static_cast<std::size_t>(y1) * kSlzBdryWidth;

    for (uint32_t x = 0; x < kSlzInputWidth; ++x) {
      const float sx = std::clamp((static_cast<float>(x) + 0.5f) * kScaleX - 0.5f, 0.0f, kMaxX);
      const uint32_t x0 = static_cast<uint32_t>(sx);
      const uint32_t x1 = std::min(x0 + 1u, kSlzBdryWidth - 1u);
      const float wx = sx - static_cast<float>(x0);
      const float wx0 = 1.0f - wx;

      const float b00 = bdry_64[row_b0 + x0];
      const float b01 = bdry_64[row_b0 + x1];
      const float b10 = bdry_64[row_b1 + x0];
      const float b11 = bdry_64[row_b1 + x1];

      float b_up = wy0 * (wx0 * b00 + wx * b01) + wy * (wx0 * b10 + wx * b11);
      if (!std::isfinite(b_up)) {
        b_up = 0.0f;
      } else {
        b_up = std::clamp(b_up, 0.0f, 1.0f);
      }

      const float e_in = e_raw_512[row_out + x];
      e_out_512[row_out + x] = std::isfinite(e_in) ? (e_in - kappa_b * b_up) : 0.0f;
    }
  }
  return true;
}

SlzOrtRunner::SlzOrtRunner(const SlzOrtConfig& cfg) : cfg_(cfg) {
  if (!cfg_.ood_calibration_path.empty()) {
    if (!LoadOodCalibrationJson(cfg_.ood_calibration_path, cfg_) && !cfg_.allow_mock_fallback) {
      throw std::runtime_error(
          "SlzOrtRunner: failed to load OOD calibration JSON: " + cfg_.ood_calibration_path);
    }
  }
#ifdef VERTIRTA_HAS_ORT
  if (!cfg_.onnx_path.empty()) {
    try {
      env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "VertiRTA_SLZ");
      const auto& api = Ort::GetApi();
      Ort::SessionOptions opts;
      opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      opts.SetIntraOpNumThreads(1);
      opts.SetInterOpNumThreads(1);

      if (cudaStreamCreate(&stream_) != cudaSuccess) {
        throw std::runtime_error("SlzOrtRunner: cudaStreamCreate failed");
      }

      if (cfg_.use_tensorrt) {
        OrtTensorRTProviderOptionsV2* trt_opts = nullptr;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt_opts));
        std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)>
            trt_guard(trt_opts, api.ReleaseTensorRTProviderOptions);

        std::vector<const char*> keys = {
            "device_id",
            "trt_max_workspace_size",
            "trt_fp16_enable",
            "trt_builder_optimization_level",
            "trt_cuda_graph_enable",
        };
        std::vector<const char*> vals = {
            "0",
            "1073741824",
            cfg_.enable_fp16 ? "1" : "0",
            "3",
            cfg_.enable_cuda_graph ? "1" : "0",
        };
        if (!cfg_.trt_cache_dir.empty()) {
          keys.push_back("trt_engine_cache_enable");
          vals.push_back("1");
          keys.push_back("trt_engine_cache_path");
          vals.push_back(cfg_.trt_cache_dir.c_str());
          keys.push_back("trt_timing_cache_enable");
          vals.push_back("1");
          keys.push_back("trt_timing_cache_path");
          vals.push_back(cfg_.trt_cache_dir.c_str());
        }
        Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
            trt_opts, keys.data(), vals.data(), keys.size()));
        Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(
            trt_opts, "user_compute_stream", stream_));
        opts.AppendExecutionProvider_TensorRT_V2(*trt_opts);
      }

      const CudaProviderTuning tuning = ResolveCudaProviderTuning(cfg_);
      OrtCUDAProviderOptionsV2* cuda_opts = nullptr;
      Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_opts));
      std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)>
          cuda_guard(cuda_opts, api.ReleaseCUDAProviderOptions);
      const char* ckeys[] = {
          "device_id",
          "arena_extend_strategy",
          "cudnn_conv_algo_search",
          "cudnn_conv_use_max_workspace",
          "enable_cuda_graph",
      };
      const char* cvals[] = {
          "0",
          tuning.arena_extend_strategy.c_str(),
          tuning.cudnn_conv_algo_search.c_str(),
          tuning.cudnn_conv_use_max_workspace ? "1" : "0",
          tuning.enable_cuda_graph ? "1" : "0",
      };
      Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda_opts, ckeys, cvals, 5));
      Ort::ThrowOnError(api.UpdateCUDAProviderOptionsWithValue(
          cuda_opts, "user_compute_stream", stream_));
      opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);

      session_ = std::make_unique<Ort::Session>(*env_, cfg_.onnx_path.c_str(), opts);
      AllocateStaticIoBindings();
      using_mock_ = false;
      ready_ = true;
      return;
    } catch (...) {
      // Strictly destroy IO binding & session BEFORE CUDA buffers/stream on ctor failure.
      io_binding_.reset();
      session_.reset();
      env_.reset();
      ReleaseCudaResources();
      if (!cfg_.allow_mock_fallback) {
        throw;
      }
    }
  }
#endif

  if (cfg_.allow_mock_fallback) {
    using_mock_ = true;
    ready_ = true;
  } else {
    using_mock_ = false;
    ready_ = false;
  }
}

SlzOrtRunner::~SlzOrtRunner() noexcept {
#ifdef VERTIRTA_HAS_ORT
  // Critical RAII teardown order (DR-7 Erratum #4):
  // Destroy IoBinding and Ort::Session strictly BEFORE releasing device buffers & cudaStream_t.
  io_binding_.reset();
  session_.reset();
  env_.reset();
  ReleaseCudaResources();
#endif
}

#ifdef VERTIRTA_HAS_ORT
void SlzOrtRunner::ReleaseCudaResources() noexcept {
  if (stream_ != nullptr) {
    (void)cudaStreamSynchronize(stream_);
  }
  if (d_bdry_ != nullptr) {
    (void)cudaFree(d_bdry_);
    d_bdry_ = nullptr;
  }
  if (d_normals_ != nullptr) {
    (void)cudaFree(d_normals_);
    d_normals_ = nullptr;
  }
  if (d_logits_ != nullptr) {
    (void)cudaFree(d_logits_);
    d_logits_ = nullptr;
  }
  if (d_input_ != nullptr) {
    (void)cudaFree(d_input_);
    d_input_ = nullptr;
  }
  if (stream_ != nullptr) {
    (void)cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void SlzOrtRunner::AllocateStaticIoBindings() {
  if (cudaMalloc(reinterpret_cast<void**>(&d_input_), kSlzRgbElements * sizeof(float)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_logits_), kSlzLogitsElements * sizeof(float)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_normals_), kSlzNormalsElements * sizeof(float)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_bdry_), kSlzBdryElements * sizeof(float)) != cudaSuccess) {
    throw std::runtime_error("SlzOrtRunner: cudaMalloc static IO buffers failed");
  }

  Ort::MemoryInfo mem_cuda("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
  const int64_t in_shape[4] = {1, kSlzInputChannels, kSlzInputHeight, kSlzInputWidth};
  const int64_t log_shape[4] = {1, kSlzNumClasses, kSlzInputHeight, kSlzInputWidth};
  const int64_t norm_shape[4] = {1, kSlzNormalChannels, kSlzInputHeight, kSlzInputWidth};
  const int64_t bdry_shape[4] = {1, 1, kSlzBdryHeight, kSlzBdryWidth};

  io_binding_ = std::make_unique<Ort::IoBinding>(*session_);
  io_binding_->BindInput(
      "rgb_nchw",
      Ort::Value::CreateTensor<float>(mem_cuda, d_input_, kSlzRgbElements, in_shape, 4));
  io_binding_->BindOutput(
      "seg_logits",
      Ort::Value::CreateTensor<float>(mem_cuda, d_logits_, kSlzLogitsElements, log_shape, 4));
  io_binding_->BindOutput(
      "surf_normals",
      Ort::Value::CreateTensor<float>(mem_cuda, d_normals_, kSlzNormalsElements, norm_shape, 4));
  io_binding_->BindOutput(
      "bdry_prob",
      Ort::Value::CreateTensor<float>(mem_cuda, d_bdry_, kSlzBdryElements, bdry_shape, 4));
}
#endif

void SlzOrtRunner::SetMockOutputs(
    const float class_logits[kSlzNumClasses],
    float bdry_prob_val) noexcept {
  if (class_logits == nullptr) {
    return;
  }
  for (uint32_t c = 0; c < kSlzNumClasses; ++c) {
    mock_logits_[c] = class_logits[c];
  }
  mock_bdry_ = std::clamp(bdry_prob_val, 0.0f, 1.0f);
  use_custom_mock_ = true;
}

bool SlzOrtRunner::Run(
    std::span<const float> rgb_nchw_512,
    std::span<float> seg_logits_5x512x512,
    std::span<float> surf_normals_3x512x512,
    std::span<float> bdry_prob_64x64) noexcept {
  if (!ready_ ||
      rgb_nchw_512.size() != kSlzRgbElements ||
      seg_logits_5x512x512.size() != kSlzLogitsElements ||
      surf_normals_3x512x512.size() != kSlzNormalsElements ||
      bdry_prob_64x64.size() != kSlzBdryElements) {
    return false;
  }

#ifdef VERTIRTA_HAS_ORT
  if (!using_mock_ && session_ != nullptr && io_binding_ != nullptr) {
    try {
      if (cudaMemcpyAsync(d_input_, rgb_nchw_512.data(), kSlzRgbElements * sizeof(float),
                          cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
        return false;
      }
      Ort::RunOptions run_opts;
      session_->Run(run_opts, *io_binding_);
      if (cudaMemcpyAsync(seg_logits_5x512x512.data(), d_logits_,
                          kSlzLogitsElements * sizeof(float), cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
          cudaMemcpyAsync(surf_normals_3x512x512.data(), d_normals_,
                          kSlzNormalsElements * sizeof(float), cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
          cudaMemcpyAsync(bdry_prob_64x64.data(), d_bdry_,
                          kSlzBdryElements * sizeof(float), cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
        return false;
      }
      return cudaStreamSynchronize(stream_) == cudaSuccess;
    } catch (...) {
      return false;
    }
  }
#endif

  // Deterministic zero-heap mock backend.
  const float* r_plane = rgb_nchw_512.data();
  const float* g_plane = rgb_nchw_512.data() + kSlzPixels;
  const float* b_plane = rgb_nchw_512.data() + 2u * kSlzPixels;

  for (std::size_t idx = 0; idx < kSlzPixels; ++idx) {
    if (use_custom_mock_) {
      for (uint32_t c = 0; c < kSlzNumClasses; ++c) {
        seg_logits_5x512x512[static_cast<std::size_t>(c) * kSlzPixels + idx] = mock_logits_[c];
      }
    } else {
      const float r = std::clamp(r_plane[idx], 0.0f, 1.0f);
      const float g = std::clamp(g_plane[idx], 0.0f, 1.0f);
      const float b = std::clamp(b_plane[idx], 0.0f, 1.0f);
      const float luma = 0.299f * r + 0.587f * g + 0.114f * b;
      const bool blue_water = (b > r + 0.25f) && (b > g + 0.15f);
      const bool red_dyn = (r > g + 0.35f) && (r > b + 0.35f);
      seg_logits_5x512x512[0u * kSlzPixels + idx] = (g > r + 0.10f) ? 9.0f : -2.0f;
      seg_logits_5x512x512[1u * kSlzPixels + idx] = (!blue_water && !red_dyn && luma > 0.25f) ? 11.0f : -3.0f;
      seg_logits_5x512x512[2u * kSlzPixels + idx] = (luma <= 0.25f) ? 8.0f : -4.0f;
      seg_logits_5x512x512[3u * kSlzPixels + idx] = red_dyn ? 10.0f : -5.0f;
      seg_logits_5x512x512[4u * kSlzPixels + idx] = blue_water ? 10.0f : -5.0f;
    }
    surf_normals_3x512x512[0u * kSlzPixels + idx] = 0.0f;
    surf_normals_3x512x512[1u * kSlzPixels + idx] = 0.0f;
    surf_normals_3x512x512[2u * kSlzPixels + idx] = 1.0f;
  }

  const float bval = use_custom_mock_ ? mock_bdry_ : 0.05f;
  std::fill(bdry_prob_64x64.begin(), bdry_prob_64x64.end(), bval);
  return true;
}

}  // namespace vertirta::perception
