#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vertirta/copilot/rta_state_contract.hpp"
#include "vertirta/copilot/z3_corridor_verifier.hpp"

#ifdef VERTIRTA_HAS_LLAMA
struct llama_model;
struct llama_context;
struct llama_vocab;
#endif

namespace vertirta::copilot {

// Official Qwen3 non-thinking closing suffix (DR-6 Section B.2; prevents Issue #20345 GBNF trap)
inline constexpr std::string_view kQwen3EmptyThinkSuffix =
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

inline constexpr std::array<int32_t, 4> kQwen3EmptyThinkTailTokens = {151667, 271, 151668, 271};
inline constexpr int32_t kDefaultLlamaBatchSize = 512;
inline constexpr int32_t kMaxCompactDslDecodeTokens = 48;

// Partition a prefill token span into contiguous <= n_batch chunks (DR-6 Critical Erratum #6).
[[nodiscard]] std::vector<std::pair<int32_t, int32_t>> plan_batch_chunks(int32_t total_tokens,
                                                                         int32_t n_batch);

class VertiLlamaEngine {
 public:
  // Production GGUF constructor (uses native libllama when VERTIRTA_HAS_LLAMA is defined).
  VertiLlamaEngine(const std::string& gguf_path,
                   const std::string& gbnf_str,
                   const std::string& static_sys_prompt,
                   bool use_cuda,
                   int32_t n_batch = kDefaultLlamaBatchSize);

  // Deterministic offline/mock constructor for host CI & unit testing without GGUF weights.
  [[nodiscard]] static VertiLlamaEngine create_mock(std::string gbnf_str,
                                                    std::string static_sys_prompt,
                                                    int32_t n_batch = kDefaultLlamaBatchSize);

  ~VertiLlamaEngine() noexcept;

  VertiLlamaEngine(const VertiLlamaEngine&) = delete;
  VertiLlamaEngine& operator=(const VertiLlamaEngine&) = delete;

  VertiLlamaEngine(VertiLlamaEngine&& other) noexcept;
  VertiLlamaEngine& operator=(VertiLlamaEngine&& other) noexcept;

  [[nodiscard]] std::string compile_dsl(std::string_view telemetry_and_user);

  // Score up to 11 discrete "ACT=..." branch prefixes off the pinned KV cache for KnowNo Conformal.
  [[nodiscard]] std::vector<float> score_action_prefixes(
      std::string_view telemetry_and_user,
      std::span<const std::string_view> prefixes);

  [[nodiscard]] bool is_mock_mode() const noexcept { return mock_mode_; }
  [[nodiscard]] int32_t batch_size() const noexcept { return n_batch_; }
  [[nodiscard]] bool has_pinned_prefix() const noexcept { return prefix_pinned_; }

 private:
  VertiLlamaEngine() = default;
  void release() noexcept;

  std::string gbnf_;
  std::string static_sys_prompt_;
  int32_t n_batch_{kDefaultLlamaBatchSize};
  bool mock_mode_{true};
  bool prefix_pinned_{false};

#ifdef VERTIRTA_HAS_LLAMA
  llama_model* model_{nullptr};
  const llama_vocab* vocab_{nullptr};
  llama_context* ctx_{nullptr};
  int32_t n_sys_prefix_{0};
#endif
};

// Deterministic C++20 bilingual (TR/EN) renderer reading directly from verified snapshot & repair.
[[nodiscard]] std::string render_bilingual_response(const RtaStateSnapshotV1& st,
                                                    const CompactDslAst& ast,
                                                    const RepairDecision* repair,
                                                    LangCode lang);

[[nodiscard]] std::string render_clarification_question(const RtaStateSnapshotV1& st,
                                                        std::string_view opt_a,
                                                        std::string_view opt_b,
                                                        LangCode lang);

// Numeric Grounding Gate: asserts every numeric literal in text matches a whitelisted field
// of RtaStateSnapshotV1 within +/-tol (eliminates hallucinated voltages/ETAs/altitudes).
[[nodiscard]] bool verify_numeric_grounding(std::string_view text,
                                            const RtaStateSnapshotV1& st,
                                            double tol = 0.5) noexcept;

}  // namespace vertirta::copilot
