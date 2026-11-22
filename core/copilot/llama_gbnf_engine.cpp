#include "vertirta/copilot/llama_gbnf_engine.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#ifdef VERTIRTA_HAS_LLAMA
#include "llama.h"
#endif

namespace vertirta::copilot {

std::vector<std::pair<int32_t, int32_t>> plan_batch_chunks(int32_t total_tokens, int32_t n_batch) {
  std::vector<std::pair<int32_t, int32_t>> chunks;
  if (total_tokens <= 0 || n_batch <= 0) {
    return chunks;
  }
  chunks.reserve(static_cast<std::size_t>((total_tokens + n_batch - 1) / n_batch));
  for (int32_t offset = 0; offset < total_tokens; offset += n_batch) {
    const int32_t len = std::min(n_batch, total_tokens - offset);
    chunks.emplace_back(offset, len);
  }
  return chunks;
}

#ifdef VERTIRTA_HAS_LLAMA
namespace {

std::vector<llama_token> tokenize_string(const llama_vocab* vocab,
                                         std::string_view s,
                                         bool add_bos,
                                         bool parse_special) {
  const int32_t raw_n = -llama_tokenize(vocab, s.data(), static_cast<int32_t>(s.size()), nullptr,
                                        0, add_bos, parse_special);
  if (raw_n <= 0) {
    return {};
  }
  std::vector<llama_token> toks(static_cast<std::size_t>(raw_n));
  const int32_t got = llama_tokenize(vocab, s.data(), static_cast<int32_t>(s.size()), toks.data(),
                                     raw_n, add_bos, parse_special);
  if (got < 0) {
    toks.clear();
  } else {
    toks.resize(static_cast<std::size_t>(got));
  }
  return toks;
}

void decode_chunked(llama_context* ctx,
                    const std::vector<llama_token>& toks,
                    llama_pos start_pos,
                    int32_t n_batch,
                    bool logits_on_last) {
  const auto chunks = plan_batch_chunks(static_cast<int32_t>(toks.size()), n_batch);
  for (std::size_t c_idx = 0U; c_idx < chunks.size(); ++c_idx) {
    const int32_t offset = chunks[c_idx].first;
    const int32_t len = chunks[c_idx].second;
    const bool is_last_chunk = (c_idx + 1U == chunks.size());

    llama_batch batch = llama_batch_init(len, 0, 1);
    for (int32_t i = 0; i < len; ++i) {
      batch.token[i] = toks[static_cast<std::size_t>(offset + i)];
      batch.pos[i] = start_pos + offset + i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = (logits_on_last && is_last_chunk && i == len - 1) ? 1 : 0;
    }
    batch.n_tokens = len;
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
      throw std::runtime_error("VertiLlamaEngine: llama_decode failed in chunked prefill");
    }
  }
}

}  // namespace
#endif

VertiLlamaEngine::VertiLlamaEngine(const std::string& gguf_path,
                                   const std::string& gbnf_str,
                                   const std::string& static_sys_prompt,
                                   bool use_cuda,
                                   int32_t n_batch)
    : gbnf_(gbnf_str),
      static_sys_prompt_(static_sys_prompt),
      n_batch_(std::max(64, n_batch)),
      mock_mode_(false),
      prefix_pinned_(false) {
#ifdef VERTIRTA_HAS_LLAMA
  llama_backend_init();
  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = use_cuda ? 99 : 0;
  model_ = llama_model_load_from_file(gguf_path.c_str(), mparams);
  if (!model_) {
    throw std::runtime_error("VertiLlamaEngine: failed to load GGUF model: " + gguf_path);
  }
  vocab_ = llama_model_get_vocab(model_);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 2048;
  cparams.n_batch = static_cast<uint32_t>(n_batch_);
  cparams.n_threads = 6;
  cparams.n_threads_batch = 6;
  cparams.type_k = GGML_TYPE_Q8_0;
  cparams.type_v = GGML_TYPE_Q8_0;
  ctx_ = llama_init_from_model(model_, cparams);
  if (!ctx_) {
    release();
    throw std::runtime_error("VertiLlamaEngine: llama_init_from_model failed");
  }

  const auto sys_toks = tokenize_string(vocab_, static_sys_prompt_, true, true);
  if (!sys_toks.empty()) {
    decode_chunked(ctx_, sys_toks, 0, n_batch_, /*logits_on_last=*/false);
    n_sys_prefix_ = static_cast<int32_t>(sys_toks.size());
  }
  prefix_pinned_ = true;
#else
  (void)gguf_path;
  (void)use_cuda;
  mock_mode_ = true;
  prefix_pinned_ = true;
#endif
}

VertiLlamaEngine VertiLlamaEngine::create_mock(std::string gbnf_str,
                                               std::string static_sys_prompt,
                                               int32_t n_batch) {
  VertiLlamaEngine eng;
  eng.gbnf_ = std::move(gbnf_str);
  eng.static_sys_prompt_ = std::move(static_sys_prompt);
  eng.n_batch_ = std::max(64, n_batch);
  eng.mock_mode_ = true;
  eng.prefix_pinned_ = true;
  return eng;
}

void VertiLlamaEngine::release() noexcept {
#ifdef VERTIRTA_HAS_LLAMA
  if (ctx_ != nullptr) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_ != nullptr) {
    llama_model_free(model_);
    model_ = nullptr;
  }
  vocab_ = nullptr;
  n_sys_prefix_ = 0;
#endif
  prefix_pinned_ = false;
}

VertiLlamaEngine::~VertiLlamaEngine() noexcept {
  release();
}

VertiLlamaEngine::VertiLlamaEngine(VertiLlamaEngine&& other) noexcept
    : gbnf_(std::move(other.gbnf_)),
      static_sys_prompt_(std::move(other.static_sys_prompt_)),
      n_batch_(other.n_batch_),
      mock_mode_(other.mock_mode_),
      prefix_pinned_(other.prefix_pinned_)
#ifdef VERTIRTA_HAS_LLAMA
      ,
      model_(other.model_),
      vocab_(other.vocab_),
      ctx_(other.ctx_),
      n_sys_prefix_(other.n_sys_prefix_)
#endif
{
  other.prefix_pinned_ = false;
#ifdef VERTIRTA_HAS_LLAMA
  other.model_ = nullptr;
  other.vocab_ = nullptr;
  other.ctx_ = nullptr;
  other.n_sys_prefix_ = 0;
#endif
}

VertiLlamaEngine& VertiLlamaEngine::operator=(VertiLlamaEngine&& other) noexcept {
  if (this != &other) {
    release();
    gbnf_ = std::move(other.gbnf_);
    static_sys_prompt_ = std::move(other.static_sys_prompt_);
    n_batch_ = other.n_batch_;
    mock_mode_ = other.mock_mode_;
    prefix_pinned_ = other.prefix_pinned_;
    other.prefix_pinned_ = false;
#ifdef VERTIRTA_HAS_LLAMA
    model_ = other.model_;
    vocab_ = other.vocab_;
    ctx_ = other.ctx_;
    n_sys_prefix_ = other.n_sys_prefix_;
    other.model_ = nullptr;
    other.vocab_ = nullptr;
    other.ctx_ = nullptr;
    other.n_sys_prefix_ = 0;
#endif
  }
  return *this;
}

std::string VertiLlamaEngine::compile_dsl(std::string_view telemetry_and_user) {
#ifdef VERTIRTA_HAS_LLAMA
  if (!mock_mode_ && ctx_ != nullptr && vocab_ != nullptr) {
    llama_memory_t mem = llama_get_memory(ctx_);
    if (!llama_memory_seq_rm(mem, 0, n_sys_prefix_, -1)) {
      throw std::runtime_error("VertiLlamaEngine: KV tail eviction refused");
    }

    std::string turn;
    turn.reserve(telemetry_and_user.size() + kQwen3EmptyThinkSuffix.size());
    turn.append(telemetry_and_user);
    turn.append(kQwen3EmptyThinkSuffix);

    const auto dyn_toks = tokenize_string(vocab_, turn, false, true);
    decode_chunked(ctx_, dyn_toks, n_sys_prefix_, n_batch_, /*logits_on_last=*/true);

    llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_grammar(vocab_, gbnf_.c_str(), "root"));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.80f, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(0.20f));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(42));

    std::string out;
    llama_pos cur_pos = n_sys_prefix_ + static_cast<llama_pos>(dyn_toks.size());
    for (int32_t step = 0; step < kMaxCompactDslDecodeTokens; ++step) {
      const llama_token tok = llama_sampler_sample(chain, ctx_, -1);
      if (llama_vocab_is_eog(vocab_, tok)) {
        break;
      }
      char buf[64];
      const int n = llama_token_to_piece(vocab_, tok, buf, static_cast<int32_t>(sizeof(buf)), 0,
                                         false);
      if (n > 0) {
        out.append(buf, static_cast<std::size_t>(n));
      }
      llama_batch next = llama_batch_init(1, 0, 1);
      next.token[0] = tok;
      next.pos[0] = cur_pos++;
      next.n_seq_id[0] = 1;
      next.seq_id[0][0] = 0;
      next.logits[0] = 1;
      next.n_tokens = 1;
      const int rc = llama_decode(ctx_, next);
      llama_batch_free(next);
      if (rc != 0) {
        break;
      }
    }
    llama_sampler_free(chain);
    return out;
  }
#endif

  // Deterministic offline mock router obeying vertirta_dsl_v1.gbnf (<= 30 tokens).
  const bool mentions_shield =
      (telemetry_and_user.find("kalkan") != std::string_view::npos ||
       telemetry_and_user.find("CBF") != std::string_view::npos ||
       telemetry_and_user.find("shield") != std::string_view::npos);
  if (mentions_shield) {
    return "ACT=REJECT_UNSAFE;VIO=SHIELD_BYPASS;FB=Kartal_Sahil\nLNG=TR;MOD=REJECT";
  }
  const bool mentions_pendik =
      (telemetry_and_user.find("Pendik") != std::string_view::npos ||
       telemetry_and_user.find("Sabiha") != std::string_view::npos ||
       telemetry_and_user.find("LTFJ") != std::string_view::npos);
  if (mentions_pendik) {
    return "ACT=REJECT_UNSAFE;VIO=LTFJ_CTR;FB=Kartal_Sahil\nLNG=TR;MOD=REJECT";
  }
  if (telemetry_and_user.find("Cevizli") != std::string_view::npos) {
    return "ACT=DIVERT_SLZ;TGT=Cevizli_Helipad;ETA=76;AGL=35,120;SPD=18\nLNG=TR;MOD=BRIEF";
  }
  if (telemetry_and_user.find("Kartal") != std::string_view::npos) {
    return "ACT=DIVERT_SLZ;TGT=Kartal_Sahil;ETA=100;AGL=35,120;SPD=18\nLNG=TR;MOD=BRIEF";
  }
  if (telemetry_and_user.find("en yakin") != std::string_view::npos ||
      telemetry_and_user.find("sahil") != std::string_view::npos) {
    return "ACT=REQUEST_CLARIFICATION;A=Kartal_Sahil;B=Cevizli_Helipad;RSN=AMBIG\nLNG=TR;MOD=CLARIFY";
  }
  return "ACT=HOLD_LOITER;DUR=30;AGL=60\nLNG=TR;MOD=BRIEF";
}

std::vector<float> VertiLlamaEngine::score_action_prefixes(
    std::string_view telemetry_and_user,
    std::span<const std::string_view> prefixes) {
  std::vector<float> scores(prefixes.size(), -5.0f);
  if (prefixes.empty()) {
    return scores;
  }

#ifdef VERTIRTA_HAS_LLAMA
  if (!mock_mode_ && ctx_ != nullptr && vocab_ != nullptr) {
    llama_memory_t mem = llama_get_memory(ctx_);
    if (!llama_memory_seq_rm(mem, 0, n_sys_prefix_, -1)) {
      throw std::runtime_error("VertiLlamaEngine: KV eviction failed during prefix scoring");
    }
    std::string turn;
    turn.reserve(telemetry_and_user.size() + kQwen3EmptyThinkSuffix.size());
    turn.append(telemetry_and_user);
    turn.append(kQwen3EmptyThinkSuffix);
    const auto dyn_toks = tokenize_string(vocab_, turn, false, true);
    decode_chunked(ctx_, dyn_toks, n_sys_prefix_, n_batch_, /*logits_on_last=*/true);
    const llama_pos base_pos = n_sys_prefix_ + static_cast<llama_pos>(dyn_toks.size());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab_);

    for (std::size_t p_idx = 0U; p_idx < prefixes.size(); ++p_idx) {
      (void)llama_memory_seq_rm(mem, 0, base_pos, -1);
      const auto p_toks = tokenize_string(vocab_, prefixes[p_idx], false, false);
      if (p_toks.empty()) {
        continue;
      }
      double sum_lp = 0.0;
      for (std::size_t j = 0U; j < p_toks.size(); ++j) {
        const float* logits = llama_get_logits_ith(ctx_, -1);
        float max_l = logits[0];
        for (int32_t v = 1; v < n_vocab; ++v) {
          if (logits[v] > max_l) {
            max_l = logits[v];
          }
        }
        double denom = 0.0;
        for (int32_t v = 0; v < n_vocab; ++v) {
          denom += std::exp(static_cast<double>(logits[v] - max_l));
        }
        const double lp = static_cast<double>(logits[p_toks[j]] - max_l) - std::log(denom);
        sum_lp += lp;

        if (j + 1U < p_toks.size()) {
          llama_batch step_b = llama_batch_init(1, 0, 1);
          step_b.token[0] = p_toks[j];
          step_b.pos[0] = base_pos + static_cast<llama_pos>(j);
          step_b.n_seq_id[0] = 1;
          step_b.seq_id[0][0] = 0;
          step_b.logits[0] = 1;
          step_b.n_tokens = 1;
          (void)llama_decode(ctx_, step_b);
          llama_batch_free(step_b);
        }
      }
      scores[p_idx] = static_cast<float>(sum_lp);
    }
    return scores;
  }
#endif

  // Deterministic offline prefix log-probability scoring for unit & CI testing.
  for (std::size_t i = 0U; i < prefixes.size(); ++i) {
    const std::string_view p = prefixes[i];
    if (p.find("Kartal_Sahil") != std::string_view::npos &&
        telemetry_and_user.find("Kartal") != std::string_view::npos) {
      scores[i] = -0.15f;
    } else if (p.find("Cevizli_Helipad") != std::string_view::npos &&
               telemetry_and_user.find("Cevizli") != std::string_view::npos) {
      scores[i] = -0.15f;
    } else if (p.find("HOLD_LOITER") != std::string_view::npos) {
      scores[i] = -2.20f;
    } else {
      scores[i] = -1.40f - 0.10f * static_cast<float>(i);
    }
  }
  return scores;
}

}  // namespace vertirta::copilot
