#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "transformer/transformer_internal.h"
#include "gguf_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace qwen3_tts {

namespace {

bool is_cuda_backend(ggml_backend_t backend) {
    ggml_backend_dev_t device = backend ? ggml_backend_get_device(backend) : nullptr;
    const char * name = device ? ggml_backend_dev_name(device) : nullptr;
    return name && std::strstr(name, "CUDA") != nullptr;
}

bool talker_replay_enabled() {
    const char * value = std::getenv("QWEN3_TTS_TALKER_REPLAY_GRAPHS");
    return !value || value[0] != '0';
}

int32_t talker_step_n_kv_pad(const tts_transformer_state & state, int32_t n_past) {
    return std::min<int32_t>(state.cache.n_ctx, GGML_PAD(n_past + 1, 256));
}

bool ensure_talker_replay_graph(TTSTransformer & self, tts_transformer_private & impl,
                                int32_t n_past, bool use_frame_codes,
                                bool read_hidden, bool has_hidden_out) {
    auto & state = impl.state;
    // Replay graphs pad KV to 256-token buckets for CUDA; on CPU exact unpadded KV is faster
    if (!is_cuda_backend(state.backend) || !talker_replay_enabled()) {
        return false;
    }
    if (!use_frame_codes || read_hidden || has_hidden_out) {
        return false;
    }
    if (state.talker_replay_failed || !state.backend) {
        return false;
    }

    const int32_t n_kv_pad = talker_step_n_kv_pad(state, n_past);
    if (state.talker_replay_ready && state.talker_replay_n_kv_pad == n_kv_pad) {
        return true;
    }

    if (state.talker_replay_sched) {
        ggml_backend_sched_reset(state.talker_replay_sched);
    } else {
        std::vector<ggml_backend_t> backends;
        backends.push_back(state.backend);
        if (state.backend_cpu) {
            backends.push_back(state.backend_cpu);
        }
        state.talker_replay_sched = ggml_backend_sched_new(
            backends.data(), nullptr, (int) backends.size(), QWEN3_TTS_MAX_NODES, false, true);
        if (!state.talker_replay_sched) {
            state.talker_replay_failed = true;
            fprintf(stderr, "  Talker replay graph scheduler failed to initialize; using dynamic graphs\n");
            return false;
        }
    }

    state.talker_replay_graph = transformer_internal::ops::build_step_graph(
        self, n_past, true, &state.talker_replay_compute_meta);
    if (!state.talker_replay_graph) {
        state.talker_replay_failed = true;
        return false;
    }

    if (!ggml_backend_sched_alloc_graph(state.talker_replay_sched, state.talker_replay_graph)) {
        ggml_backend_sched_reset(state.talker_replay_sched);
        state.talker_replay_ready = false;
        state.talker_replay_failed = true;
        fprintf(stderr, "  Talker replay graph allocation failed; using dynamic graphs\n");
        return false;
    }

    state.talker_replay_ready = true;
    state.talker_replay_n_kv_pad = n_kv_pad;
    fprintf(stderr, "  Talker replay graph: enabled for %d-token bucket (experimental)\n", n_kv_pad);
    return true;
}

void read_and_unpack_talker_logits(const tts_transformer_model & model,
                                   struct ggml_tensor * logits_tensor,
                                   std::vector<float> & dst,
                                   std::vector<float> & compact_scratch) {
    const int32_t full_vocab = model.config.codec_vocab_size;
    dst.resize((size_t) full_vocab);
    if (model.codec_head_compacted) {
        const size_t raw_n = (size_t) logits_tensor->ne[0];
        compact_scratch.resize(raw_n);
        ggml_backend_tensor_get(logits_tensor, compact_scratch.data(), 0, raw_n * sizeof(float));
        std::memcpy(dst.data(), compact_scratch.data(), 2048 * sizeof(float));
        for (int32_t i = 2048; i < full_vocab; ++i) {
            dst[(size_t) i] = -INFINITY;
        }
        dst[(size_t) model.config.codec_eos_id] = compact_scratch[2048];
    } else {
        ggml_backend_tensor_get(logits_tensor, dst.data(), 0, (size_t) full_vocab * sizeof(float));
    }
}

bool can_sum_frame_codes_on_host(const tts_transformer_model & model) {
    if (!model.codec_embd || !model.codec_embd->buffer ||
        !ggml_backend_buffer_is_host(model.codec_embd->buffer) || !model.codec_embd->data) {
        return false;
    }
    for (int cb = 1; cb < model.config.n_codebooks; ++cb) {
        struct ggml_tensor * t = model.code_pred_embd[(size_t) (cb - 1)];
        if (!t || !t->buffer || !ggml_backend_buffer_is_host(t->buffer) || !t->data) {
            return false;
        }
    }
    return true;
}

bool sum_frame_codes_on_host(const tts_transformer_model & model,
                             const int32_t * frame_codes,
                             const float * overlay_embd,
                             std::vector<float> & out_step_embd,
                             std::vector<float> & row_scratch) {
    const int32_t hidden_size = model.config.hidden_size;
    out_step_embd.resize((size_t) hidden_size);
    row_scratch.resize((size_t) hidden_size);

    const int32_t cb0 = frame_codes[0];
    if (cb0 < 0 || cb0 >= model.codec_embd->ne[1]) {
        return false;
    }
    const uint8_t * p0 = static_cast<const uint8_t *>(model.codec_embd->data) +
                         (size_t) cb0 * model.codec_embd->nb[1];
    if (!convert_tensor_bytes_to_f32(model.codec_embd->type, p0, out_step_embd.data(), hidden_size)) {
        return false;
    }

    for (int cb = 1; cb < model.config.n_codebooks; ++cb) {
        struct ggml_tensor * t = model.code_pred_embd[(size_t) (cb - 1)];
        const int32_t tok = frame_codes[cb];
        if (tok < 0 || tok >= t->ne[1]) {
            return false;
        }
        const uint8_t * pt = static_cast<const uint8_t *>(t->data) + (size_t) tok * t->nb[1];
        if (!convert_tensor_bytes_to_f32(t->type, pt, row_scratch.data(), hidden_size)) {
            return false;
        }
        for (int32_t h = 0; h < hidden_size; ++h) {
            out_step_embd[(size_t) h] += row_scratch[(size_t) h];
        }
    }

    for (int32_t h = 0; h < hidden_size; ++h) {
        out_step_embd[(size_t) h] += overlay_embd[h];
    }
    return true;
}

} // namespace

bool TTSTransformer::forward_prefill(const float * prefill_embd, int32_t n_tokens,
                                     int32_t n_past, std::vector<float> & output,
                                     std::vector<float> * logits_out) {
    // A direct prefill call may overwrite rows owned by a cached generate()
    // prefix. generate() records a new exact key after its own successful miss.
    impl_->cached_icl_prefill_valid = false;
    if (!impl_->model.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!prefill_embd) {
        error_msg_ = "prefill_embd is null";
        return false;
    }
    if (n_tokens <= 0) {
        error_msg_ = "n_tokens must be > 0";
        return false;
    }

    if (impl_->state.cache.n_ctx == 0) {
        const int32_t min_ctx = std::max<int32_t>(256, n_past + n_tokens + 16);
        if (!init_kv_cache(min_ctx)) {
            return false;
        }
    }

    if (n_past + n_tokens > impl_->state.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_cgraph * gf = transformer_internal::ops::build_prefill_forward_graph(*this, n_tokens, n_past);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_prefill_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!ggml_backend_sched_alloc_graph(impl_->state.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_prefill_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_tensor * inp_prefill = ggml_graph_get_tensor(gf, "inp_prefill_embd");
    if (inp_prefill) {
        ggml_backend_tensor_set(inp_prefill, prefill_embd, 0,
                                (size_t) n_tokens * impl_->model.config.hidden_size * sizeof(float));
    }

    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            positions[i] = n_past + i;
        }
        size_t write_size = positions.size() * sizeof(int32_t);
        if (write_size > ggml_nbytes(inp_pos)) {
            error_msg_ = "Talker position input is smaller than the requested prefill write";
            ggml_backend_sched_reset(impl_->state.sched);
            return false;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, write_size);
    }

    struct ggml_tensor * inp_mrope_pos = ggml_graph_get_tensor(gf, "inp_mrope_pos");
    if (inp_mrope_pos && impl_->model.config.use_mrope) {
        std::vector<int32_t> positions(n_tokens * 4);
        for (int i = 0; i < n_tokens; ++i) {
            int32_t p = n_past + i;
            positions[i + n_tokens * 0] = p;
            positions[i + n_tokens * 1] = p;
            positions[i + n_tokens * 2] = p;
            positions[i + n_tokens * 3] = 0;
        }
        size_t write_size = positions.size() * sizeof(int32_t);
        if (write_size > ggml_nbytes(inp_mrope_pos)) {
            error_msg_ = "Talker M-RoPE input is smaller than the requested prefill write";
            ggml_backend_sched_reset(impl_->state.sched);
            return false;
        }
        ggml_backend_tensor_set(inp_mrope_pos, positions.data(), 0, write_size);
    }

    struct ggml_tensor * inp_mask = ggml_graph_get_tensor(gf, "inp_mask");
    if (inp_mask) {
        const int32_t n_kv_pad = (int32_t) inp_mask->ne[0];
        const size_t mask_size = (size_t) n_tokens * (size_t) n_kv_pad;
        auto & mask = impl_->state.talker_mask;
        if (mask.size() != mask_size) {
            mask.resize(mask_size);
        }
        const ggml_fp16_t zero_fp16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf_fp16 = ggml_fp32_to_fp16(-INFINITY);
        for (size_t i = 0; i < mask.size(); ++i) {
            mask[i] = neg_inf_fp16;
        }
        for (int32_t q = 0; q < n_tokens; ++q) {
            const int32_t q_pos = n_past + q;
            const int32_t last_k = std::min<int32_t>(q_pos, n_kv_pad - 1);
            ggml_fp16_t * row = mask.data() + (size_t) q * (size_t) n_kv_pad;
            for (int32_t k = 0; k <= last_k; ++k) {
                row[k] = zero_fp16;
            }
        }
        ggml_backend_tensor_set(inp_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_prefill_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (ggml_backend_sched_graph_compute(impl_->state.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(impl_->state.sched);
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_prefill_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");
    if (!hidden) {
        error_msg_ = "Failed to find hidden_states tensor";
        ggml_backend_sched_reset(impl_->state.sched);
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    const size_t hidden_size = (size_t) impl_->model.config.hidden_size;
    output.resize((size_t) n_tokens * hidden_size);
    ggml_backend_tensor_get(hidden, output.data(), 0, output.size() * sizeof(float));

    last_hidden_.resize(hidden_size);
    std::memcpy(last_hidden_.data(),
                output.data() + (size_t) (n_tokens - 1) * hidden_size,
                hidden_size * sizeof(float));

    if (logits_out) {
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor";
            ggml_backend_sched_reset(impl_->state.sched);
            return false;
        }

        // build_prefill_forward_graph slices last_cur before codec_head, so logits has 1 row at offset 0
        read_and_unpack_talker_logits(impl_->model, logits, *logits_out, impl_->compact_logits_scratch);
    }

    impl_->state.cache.n_used = n_past + n_tokens;

    ggml_backend_sched_reset(impl_->state.sched);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_prefill_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    return true;
}

bool TTSTransformer::forward_text(const int32_t * text_tokens, int32_t n_tokens,
                                  const float * speaker_embd, int32_t n_past,
                                  std::vector<float> & output) {
    if (!text_tokens) {
        error_msg_ = "text_tokens is null";
        return false;
    }
    if (n_tokens <= 0) {
        error_msg_ = "n_tokens must be > 0";
        return false;
    }

    std::vector<float> projected;
    if (!transformer_internal::ops::project_text_tokens(*this, text_tokens, n_tokens, projected)) {
        return false;
    }

    if (speaker_embd) {
        const int32_t hidden_size = impl_->model.config.hidden_size;
        for (int32_t t = 0; t < n_tokens; ++t) {
            float * row = projected.data() + (size_t) t * hidden_size;
            for (int32_t h = 0; h < hidden_size; ++h) {
                row[h] += speaker_embd[h];
            }
        }
    }

    return forward_prefill(projected.data(), n_tokens, n_past, output, nullptr);
}

bool TTSTransformer::forward_step_internal(const float * step_embd,
                                           const int32_t * frame_codes,
                                           const float * overlay_embd,
                                           int32_t n_past,
                                           std::vector<float> & output,
                                           std::vector<float> * hidden_out,
                                           bool read_hidden) {
    if (!impl_->model.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    bool use_frame_codes = frame_codes != nullptr && overlay_embd != nullptr;
    if (!step_embd && !use_frame_codes) {
        error_msg_ = "step_embd is null";
        return false;
    }

    if (impl_->state.cache.n_ctx == 0) {
        const int32_t min_ctx = std::max<int32_t>(256, n_past + 1 + 16);
        if (!init_kv_cache(min_ctx)) {
            return false;
        }
    }

    if (n_past + 1 > impl_->state.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

    // On CPU, sum the 16 codebook rows + overlay directly from zero-copy host embeddings in L1 cache,
    // avoiding 64 tiny 1-row GGML graph nodes and 32 threadpool barriers per Talker step.
    if (use_frame_codes && !is_cuda_backend(impl_->state.backend) &&
        can_sum_frame_codes_on_host(impl_->model)) {
        if (sum_frame_codes_on_host(impl_->model, frame_codes, overlay_embd,
                                    impl_->step_embd_scratch, impl_->compact_logits_scratch)) {
            step_embd = impl_->step_embd_scratch.data();
            use_frame_codes = false;
        }
    }

    const bool use_talker_replay =
        ensure_talker_replay_graph(*this, *impl_, n_past, use_frame_codes,
                                   read_hidden, hidden_out != nullptr);
    ggml_backend_sched_t step_sched = use_talker_replay
        ? impl_->state.talker_replay_sched
        : impl_->state.sched;

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_cgraph * gf = use_talker_replay
        ? impl_->state.talker_replay_graph
        : transformer_internal::ops::build_step_graph(*this, n_past, use_frame_codes);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing && !use_talker_replay) {
        impl_->timing->t_talker_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!use_talker_replay) {
        if (!ggml_backend_sched_alloc_graph(step_sched, gf)) {
            error_msg_ = "Failed to allocate graph";
            return false;
        }
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing && !use_talker_replay) {
        impl_->timing->t_talker_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (use_frame_codes) {
        struct ggml_tensor * inp_frame_codes = ggml_graph_get_tensor(gf, "inp_frame_codes");
        if (inp_frame_codes) {
            ggml_backend_tensor_set(inp_frame_codes, frame_codes, 0,
                                    impl_->model.config.n_codebooks * sizeof(int32_t));
        }
        struct ggml_tensor * inp_overlay = ggml_graph_get_tensor(gf, "inp_overlay_embd");
        if (inp_overlay) {
            ggml_backend_tensor_set(inp_overlay, overlay_embd, 0,
                                    impl_->model.config.hidden_size * sizeof(float));
        }
    } else {
        struct ggml_tensor * inp_step = ggml_graph_get_tensor(gf, "inp_step_embd");
        if (!inp_step) {
            error_msg_ = "Failed to find inp_step_embd tensor";
            return false;
        }
        ggml_backend_tensor_set(inp_step, step_embd, 0,
                                impl_->model.config.hidden_size * sizeof(float));
    }

    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        int32_t pos = n_past;
        ggml_backend_tensor_set(inp_pos, &pos, 0, sizeof(int32_t));
    }

    struct ggml_tensor * inp_mrope_pos = ggml_graph_get_tensor(gf, "inp_mrope_pos");
    if (inp_mrope_pos && impl_->model.config.use_mrope) {
        int32_t positions[4] = {n_past, n_past, n_past, 0};
        ggml_backend_tensor_set(inp_mrope_pos, positions, 0, 4 * sizeof(int32_t));
    }

    struct ggml_tensor * inp_mask = ggml_graph_get_tensor(gf, "inp_mask");
    if (inp_mask) {
        const int32_t n_kv_pad = (int32_t) inp_mask->ne[0];
        auto & mask = impl_->state.talker_mask;
        if (mask.size() != (size_t) n_kv_pad) {
            mask.resize((size_t) n_kv_pad);
        }
        const ggml_fp16_t zero_fp16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf_fp16 = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < n_kv_pad; ++i) {
            mask[(size_t) i] = (i <= n_past) ? zero_fp16 : neg_inf_fp16;
        }
        ggml_backend_tensor_set(inp_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) {
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        impl_->timing->t_talker_data_ms += dt_ms;
        impl_->timing->t_talker_input_upload_ms += dt_ms;
    }
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (ggml_backend_sched_graph_compute(step_sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        if (use_talker_replay) {
            impl_->state.talker_replay_ready = false;
            impl_->state.talker_replay_failed = true;
        }
        ggml_backend_sched_reset(step_sched);
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_talker_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");
    if (!hidden) {
        error_msg_ = "Failed to find hidden_states tensor";
        if (!use_talker_replay) {
            ggml_backend_sched_reset(step_sched);
        }
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (read_hidden || hidden_out) {
        last_hidden_.resize(impl_->model.config.hidden_size);
    }
    if (hidden_out) {
        hidden_out->resize(impl_->model.config.hidden_size);
        ggml_backend_tensor_get(hidden, hidden_out->data(), 0,
                                impl_->model.config.hidden_size * sizeof(float));
        last_hidden_ = *hidden_out;
    } else if (read_hidden) {
        ggml_backend_tensor_get(hidden, last_hidden_.data(), 0,
                                impl_->model.config.hidden_size * sizeof(float));
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) {
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        impl_->timing->t_talker_data_ms += dt_ms;
        impl_->timing->t_talker_hidden_read_ms += dt_ms;
    }
    t0 = clk::now();
#endif

    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        error_msg_ = "Failed to find logits tensor";
        if (!use_talker_replay) {
            ggml_backend_sched_reset(step_sched);
        }
        return false;
    }

    read_and_unpack_talker_logits(impl_->model, logits, output, impl_->compact_logits_scratch);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) {
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        impl_->timing->t_talker_data_ms += dt_ms;
        impl_->timing->t_talker_logits_read_ms += dt_ms;
    }
    t0 = clk::now();
#endif

    impl_->state.cache.n_used = n_past + 1;

    if (!use_talker_replay) {
        ggml_backend_sched_reset(step_sched);
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing && !use_talker_replay) {
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        impl_->timing->t_talker_data_ms += dt_ms;
        impl_->timing->t_talker_sched_reset_ms += dt_ms;
    }
#endif

    return true;
}

bool TTSTransformer::forward_step(const float * step_embd, int32_t n_past,
                                  std::vector<float> & output,
                                  std::vector<float> * hidden_out) {
    impl_->cached_icl_prefill_valid = false;
    return forward_step_internal(step_embd, nullptr, nullptr, n_past, output, hidden_out, true);
}

bool TTSTransformer::forward_codec(int32_t codec_token, int32_t n_past,
                                   std::vector<float> & output) {
    std::vector<float> codec_row;
    if (!transformer_internal::ops::lookup_embedding_rows(*this, impl_->model.codec_embd, &codec_token, 1,
                               "inp_legacy_codec_token", "legacy_codec_row",
                               codec_row)) {
        return false;
    }

    return forward_step(codec_row.data(), n_past, output, nullptr);
}

} // namespace qwen3_tts
