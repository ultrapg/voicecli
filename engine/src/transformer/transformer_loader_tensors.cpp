#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "gguf_loader.h"
#include "ggml-alloc.h"
#include "transformer/transformer_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace qwen3_tts {

namespace {

constexpr int64_t kCompactedCodecHeadRows = 2056; // 2048 audio tokens + EOS + 7 pad rows (multiple of 8 for repack)

bool fused_weights_enabled() {
    const char * v = std::getenv("VOICECLI_DISABLE_FUSED_WEIGHTS");
    return !(v && v[0] != '\0' && v[0] != '0');
}

} // namespace

bool transformer_internal::ops::create_tensors(TTSTransformer & self, struct gguf_context * ctx) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const auto & cfg = impl->model.config;

    impl->model.codec_head_compacted = false;

    const bool enable_fused = fused_weights_enabled();
    const bool can_compact_codec_head =
        !transformer_internal::get_debug_trace_config().enabled &&
        cfg.codec_vocab_size > kCompactedCodecHeadRows &&
        cfg.code_pred_vocab_size == 2048 &&
        cfg.codec_eos_id >= 2048 &&
        cfg.codec_eos_id < cfg.codec_vocab_size;

    const size_t extra_fused_tensors =
        (size_t) cfg.n_layers * 2 + (size_t) cfg.code_pred_layers * 2 + 16;
    const size_t ctx_size = ((size_t) n_tensors + extra_fused_tensors) * ggml_tensor_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    impl->model.ctx = ggml_init(params);
    if (!impl->model.ctx) {
        error_msg = "Failed to create GGML context";
        return false;
    }

    impl->model.layers.resize(cfg.n_layers);
    impl->model.code_pred_layers.resize(cfg.code_pred_layers);
    impl->model.code_pred_embd.resize(cfg.n_codebooks - 1);
    impl->model.code_pred_head.resize(cfg.n_codebooks - 1);

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);

        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        int n_dims = 0;

        if (strstr(name, "spk_enc.") || strstr(name, "tok_")) {
            continue;
        }

        if (strstr(name, "talker.text_embd.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.text_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc1.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.text_embd_dim;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc1.bias")) {
            ne[0] = cfg.text_embd_dim;
            n_dims = 1;
        } else if (strstr(name, "talker.text_proj.fc2.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.hidden_size;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc2.bias")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "talker.codec_embd.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.codec_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.codec_head.weight")) {
            ne[0] = cfg.hidden_size;
            if (can_compact_codec_head) {
                ne[1] = kCompactedCodecHeadRows;
                impl->model.codec_head_compacted = true;
            } else {
                ne[1] = cfg.codec_vocab_size;
            }
            n_dims = 2;
        } else if (strstr(name, "talker.output_norm.weight")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "talker.blk.%d.", &layer_idx) == 1 &&
                layer_idx >= 0 && layer_idx < cfg.n_layers) {
                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_k_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_attention_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = cfg.n_attention_heads * cfg.head_dim;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = cfg.intermediate_size;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.small_to_mtp.weight") || strstr(name, "code_pred.mtp_proj.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.code_pred_hidden_size;
            n_dims = 2;
        } else if (strstr(name, "code_pred.small_to_mtp.bias") || strstr(name, "code_pred.mtp_proj.bias")) {
            ne[0] = cfg.code_pred_hidden_size;
            n_dims = 1;
        } else if (strstr(name, "code_pred.blk.")) {
            if (impl->skip_ggml_code_pred_layers) {
                continue;
            }
            int layer_idx = -1;
            if (sscanf(name, "code_pred.blk.%d.", &layer_idx) == 1 &&
                layer_idx >= 0 && layer_idx < cfg.code_pred_layers) {
                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight")) {
                    ne[0] = cfg.code_pred_head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_k_norm.weight")) {
                    ne[0] = cfg.code_pred_head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_attention_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_key_value_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_key_value_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = cfg.code_pred_n_attention_heads * cfg.code_pred_head_dim;
                    ne[1] = cfg.code_pred_hidden_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = cfg.code_pred_intermediate_size;
                    ne[1] = cfg.code_pred_hidden_size;
                    n_dims = 2;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.codec_embd.")) {
            int cb_idx = -1;
            if (sscanf(name, "code_pred.codec_embd.%d.weight", &cb_idx) == 1 &&
                cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                ne[0] = cfg.hidden_size;
                ne[1] = cfg.code_pred_vocab_size;
                n_dims = 2;
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.lm_head.")) {
            if (impl->skip_ggml_code_pred_layers) {
                continue;
            }
            int cb_idx = -1;
            if (sscanf(name, "code_pred.lm_head.%d.weight", &cb_idx) == 1 &&
                cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                ne[0] = cfg.code_pred_hidden_size;
                ne[1] = cfg.code_pred_vocab_size;
                n_dims = 2;
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.output_norm.weight")) {
            if (impl->skip_ggml_code_pred_layers) {
                continue;
            }
            ne[0] = cfg.code_pred_hidden_size;
            n_dims = 1;
        } else {
            continue;
        }

        // Pre-convert all 1D norm and bias vectors to F32 at load time so ggml_mul / ggml_add
        // hit direct F32 SIMD vector kernels instead of per-token BF16/F16 upcasting.
        if (n_dims == 1) {
            type = GGML_TYPE_F32;
        }

        struct ggml_tensor * tensor = ggml_new_tensor(impl->model.ctx, type, n_dims, ne);
        if (!tensor) {
            error_msg = "Failed to create tensor: " + std::string(name);
            return false;
        }
        ggml_set_name(tensor, name);
        impl->model.tensors[name] = tensor;

        if (strstr(name, "talker.text_embd.weight")) {
            impl->model.text_embd = tensor;
        } else if (strstr(name, "talker.text_proj.fc1.weight")) {
            impl->model.text_proj_fc1 = tensor;
        } else if (strstr(name, "talker.text_proj.fc1.bias")) {
            impl->model.text_proj_fc1_bias = tensor;
        } else if (strstr(name, "talker.text_proj.fc2.weight")) {
            impl->model.text_proj_fc2 = tensor;
        } else if (strstr(name, "talker.text_proj.fc2.bias")) {
            impl->model.text_proj_fc2_bias = tensor;
        } else if (strstr(name, "talker.codec_embd.weight")) {
            impl->model.codec_embd = tensor;
        } else if (strstr(name, "talker.codec_head.weight")) {
            impl->model.codec_head = tensor;
        } else if (strstr(name, "talker.output_norm.weight")) {
            impl->model.output_norm = tensor;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            sscanf(name, "talker.blk.%d.", &layer_idx);
            if (layer_idx >= 0 && layer_idx < cfg.n_layers) {
                auto & layer = impl->model.layers[layer_idx];
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        } else if (strstr(name, "code_pred.small_to_mtp.weight") || strstr(name, "code_pred.mtp_proj.weight")) {
            impl->model.code_pred_small_to_mtp_weight = tensor;
        } else if (strstr(name, "code_pred.small_to_mtp.bias") || strstr(name, "code_pred.mtp_proj.bias")) {
            impl->model.code_pred_small_to_mtp_bias = tensor;
        } else if (strstr(name, "code_pred.blk.")) {
            int layer_idx = -1;
            sscanf(name, "code_pred.blk.%d.", &layer_idx);
            if (layer_idx >= 0 && layer_idx < cfg.code_pred_layers) {
                auto & layer = impl->model.code_pred_layers[layer_idx];
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        } else if (strstr(name, "code_pred.codec_embd.")) {
            int cb_idx = -1;
            sscanf(name, "code_pred.codec_embd.%d.weight", &cb_idx);
            if (cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                impl->model.code_pred_embd[cb_idx] = tensor;
            }
        } else if (strstr(name, "code_pred.lm_head.")) {
            int cb_idx = -1;
            sscanf(name, "code_pred.lm_head.%d.weight", &cb_idx);
            if (cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                impl->model.code_pred_head[cb_idx] = tensor;
            }
        } else if (strstr(name, "code_pred.output_norm.weight")) {
            impl->model.code_pred_output_norm = tensor;
        }
    }

    if (enable_fused && !impl->skip_ggml_code_pred_layers) {
        for (int32_t il = 0; il < cfg.code_pred_layers; ++il) {
            auto & layer = impl->model.code_pred_layers[(size_t) il];
            if (layer.attn_q && layer.attn_k && layer.attn_v &&
                layer.attn_q->type == layer.attn_k->type &&
                layer.attn_q->type == layer.attn_v->type &&
                layer.attn_q->ne[0] == layer.attn_k->ne[0] &&
                layer.attn_q->ne[0] == layer.attn_v->ne[0]) {
                layer.attn_qkv = ggml_new_tensor_2d(
                    impl->model.ctx,
                    layer.attn_q->type,
                    layer.attn_q->ne[0],
                    layer.attn_q->ne[1] + layer.attn_k->ne[1] + layer.attn_v->ne[1]);
                if (!layer.attn_qkv) {
                    error_msg = "Failed to create packed Code Predictor QKV tensor for layer " +
                        std::to_string(il);
                    return false;
                }
                ggml_format_name(layer.attn_qkv, "code_pred.blk.%d.attn_qkv.packed", il);
            }

            if (layer.ffn_gate && layer.ffn_up &&
                layer.ffn_gate->type == layer.ffn_up->type &&
                layer.ffn_gate->ne[0] == layer.ffn_up->ne[0] &&
                layer.ffn_gate->ne[1] == layer.ffn_up->ne[1]) {
                layer.ffn_gate_up = ggml_new_tensor_2d(
                    impl->model.ctx,
                    layer.ffn_gate->type,
                    layer.ffn_gate->ne[0],
                    layer.ffn_gate->ne[1] + layer.ffn_up->ne[1]);
                if (!layer.ffn_gate_up) {
                    error_msg = "Failed to create packed Code Predictor FFN gate_up tensor for layer " +
                        std::to_string(il);
                    return false;
                }
                ggml_format_name(layer.ffn_gate_up, "code_pred.blk.%d.ffn_gate_up.packed", il);
            }
        }
    }

    if (enable_fused) {
        for (int32_t il = 0; il < cfg.n_layers; ++il) {
            auto & layer = impl->model.layers[(size_t) il];
            if (layer.attn_q && layer.attn_k && layer.attn_v &&
                layer.attn_q->type == layer.attn_k->type &&
                layer.attn_q->type == layer.attn_v->type &&
                layer.attn_q->ne[0] == layer.attn_k->ne[0] &&
                layer.attn_q->ne[0] == layer.attn_v->ne[0]) {
                layer.attn_qkv = ggml_new_tensor_2d(
                    impl->model.ctx,
                    layer.attn_q->type,
                    layer.attn_q->ne[0],
                    layer.attn_q->ne[1] + layer.attn_k->ne[1] + layer.attn_v->ne[1]);
                if (!layer.attn_qkv) {
                    error_msg = "Failed to create packed Talker QKV tensor for layer " +
                        std::to_string(il);
                    return false;
                }
                ggml_format_name(layer.attn_qkv, "talker.blk.%d.attn_qkv.packed", il);
            }

            if (layer.ffn_gate && layer.ffn_up &&
                layer.ffn_gate->type == layer.ffn_up->type &&
                layer.ffn_gate->ne[0] == layer.ffn_up->ne[0] &&
                layer.ffn_gate->ne[1] == layer.ffn_up->ne[1]) {
                layer.ffn_gate_up = ggml_new_tensor_2d(
                    impl->model.ctx,
                    layer.ffn_gate->type,
                    layer.ffn_gate->ne[0],
                    layer.ffn_gate->ne[1] + layer.ffn_up->ne[1]);
                if (!layer.ffn_gate_up) {
                    error_msg = "Failed to create packed Talker FFN gate_up tensor for layer " +
                        std::to_string(il);
                    return false;
                }
                ggml_format_name(layer.ffn_gate_up, "talker.blk.%d.ffn_gate_up.packed", il);
            }
        }
    }

    return true;
}

bool transformer_internal::ops::load_tensor_data(TTSTransformer & self, const std::string & path, struct gguf_context * ctx) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;
    const auto & cfg = impl->model.config;

    bool own_backend = false;
    ggml_backend_t backend = impl->state.backend;
    if (!backend) {
        backend = init_preferred_backend("TTSTransformer", &error_msg);
        if (!backend) {
            return false;
        }
        own_backend = true;
    }
    auto cleanup_backend = [&]() {
        if (own_backend && backend) {
            release_preferred_backend(backend);
            backend = nullptr;
        }
    };

    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    const bool is_cpu_backend = device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU;

    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    // Map GGUF file read-only on CPU so untouched weights and embedding tables stay zero-copy
    // and fused/repacked source ranges can be loaded without FILE* seeks/reads.
    std::shared_ptr<MappedFile> mapped;
    ggml_backend_buffer_t mmap_buf = nullptr;
    if (is_cpu_backend) {
        auto candidate = std::make_shared<MappedFile>();
        std::string mmap_err;
        if (candidate->open(path, mmap_err) && candidate->size() > data_offset) {
            ggml_backend_dev_t cpu_dev = device;
            mmap_buf = ggml_backend_dev_buffer_from_host_ptr(
                cpu_dev,
                candidate->mutable_data() + data_offset,
                candidate->size() - data_offset,
                candidate->size() - data_offset);
            if (mmap_buf) {
                ggml_backend_buffer_set_usage(mmap_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                mapped = std::move(candidate);
                impl->model.mapped_file = mapped;
                impl->model.mmap_buffer = mmap_buf;
            }
        }
    }

    // Index GGUF source info for every model tensor
    struct GgufTensorInfo {
        size_t offset = 0;
        size_t size = 0;
        enum ggml_type src_type = GGML_TYPE_F32;
    };
    std::unordered_map<struct ggml_tensor *, GgufTensorInfo> src_info;
    src_info.reserve((size_t) n_tensors);

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        auto it = impl->model.tensors.find(name);
        if (it == impl->model.tensors.end()) {
            continue;
        }
        GgufTensorInfo info;
        info.offset = gguf_get_tensor_offset(ctx, i);
        info.size = gguf_get_tensor_size(ctx, i);
        info.src_type = gguf_get_tensor_type(ctx, i);
        src_info[it->second] = info;
    }

    // Track embedding tables (must remain host-accessible for ggml_get_rows & direct host row lookups)
    // and superseded raw Q/K/V/gate/up weights (must remain zero-copy in mmap_buffer, never repacked).
    std::unordered_set<struct ggml_tensor *> embedding_tensors;
    if (impl->model.text_embd) embedding_tensors.insert(impl->model.text_embd);
    if (impl->model.codec_embd) embedding_tensors.insert(impl->model.codec_embd);
    for (struct ggml_tensor * t : impl->model.code_pred_embd) {
        if (t) embedding_tensors.insert(t);
    }

    std::unordered_set<struct ggml_tensor *> superseded_raw_tensors;
    auto mark_superseded = [&](const transformer_layer & layer) {
        if (layer.attn_qkv) {
            if (layer.attn_q) superseded_raw_tensors.insert(layer.attn_q);
            if (layer.attn_k) superseded_raw_tensors.insert(layer.attn_k);
            if (layer.attn_v) superseded_raw_tensors.insert(layer.attn_v);
        }
        if (layer.ffn_gate_up) {
            if (layer.ffn_gate) superseded_raw_tensors.insert(layer.ffn_gate);
            if (layer.ffn_up) superseded_raw_tensors.insert(layer.ffn_up);
        }
    };
    for (const auto & layer : impl->model.layers) {
        mark_superseded(layer);
    }
    for (const auto & layer : impl->model.code_pred_layers) {
        mark_superseded(layer);
    }

    // Stage 1: Identify 2D matmul weights eligible for GGML CPU_REPACK (e.g. Q4_K -> q4_K_8x8, Q8_0 -> q8_0_4x8)
    ggml_backend_buffer_type_t repack_buft = is_cpu_backend ? get_cpu_repack_buffer_type(backend) : nullptr;
    std::vector<struct ggml_tensor *> repack_tensors;
    std::unordered_set<struct ggml_tensor *> repack_set;

    if (repack_buft && device) {
        for (struct ggml_tensor * t = ggml_get_first_tensor(impl->model.ctx);
             t != nullptr;
             t = ggml_get_next_tensor(impl->model.ctx, t)) {
            if (t->view_src != nullptr) continue;
            if (ggml_n_dims(t) != 2) continue;
            if (embedding_tensors.count(t) || superseded_raw_tensors.count(t)) continue;
            if (can_repack_mul_mat_weight(device, repack_buft, t->type, t->ne[0], t->ne[1])) {
                repack_tensors.push_back(t);
                repack_set.insert(t);
            }
        }
    }

    if (!repack_tensors.empty()) {
        const size_t repack_align = ggml_backend_buft_get_alignment(repack_buft);
        size_t repack_total_bytes = 0;
        for (struct ggml_tensor * t : repack_tensors) {
            repack_total_bytes += GGML_PAD(ggml_backend_buft_get_alloc_size(repack_buft, t), repack_align);
        }
        impl->model.repack_buffer = ggml_backend_buft_alloc_buffer(repack_buft, repack_total_bytes);
        if (impl->model.repack_buffer) {
            ggml_backend_buffer_set_usage(impl->model.repack_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            struct ggml_tallocr repack_tallocr = ggml_tallocr_new(impl->model.repack_buffer);
            for (struct ggml_tensor * t : repack_tensors) {
                if (ggml_tallocr_alloc(&repack_tallocr, t) != GGML_STATUS_SUCCESS) {
                    error_msg = "Failed to allocate repacked tensor: " + std::string(ggml_get_name(t));
                    cleanup_backend();
                    return false;
                }
            }
            const char * dbg = std::getenv("VOICECLI_DEBUG");
            if (dbg && dbg[0] != '\0' && dbg[0] != '0') {
                fprintf(stderr, "  TTSTransformer CPU_REPACK: repacking %zu matmul weight tensors (%.1f MiB)\n",
                        repack_tensors.size(), (double) repack_total_bytes / (1024.0 * 1024.0));
            }
        } else {
            repack_tensors.clear();
            repack_set.clear();
        }
    }

    // Stage 2: Bind zero-copy mmap tensors (untouched weights, embedding tables, and superseded raw tensors)
    std::unordered_set<struct ggml_tensor *> zero_copy_set;
    if (mapped && mmap_buf) {
        const size_t alignment = ggml_backend_buffer_get_alignment(mmap_buf);
        for (const auto & kv : src_info) {
            struct ggml_tensor * t = kv.first;
            const GgufTensorInfo & info = kv.second;
            if (t->data != nullptr || t->view_src != nullptr) continue;
            if (repack_set.count(t)) continue;
            if (t == impl->model.codec_head && impl->model.codec_head_compacted) continue;
            if (t->type != info.src_type) continue;

            const size_t nbytes = ggml_nbytes(t);
            if (info.size != nbytes || data_offset + info.offset + nbytes > mapped->size()) {
                continue;
            }
            uint8_t * ptr = mapped->mutable_data() + data_offset + info.offset;
            if (alignment > 1 && (reinterpret_cast<uintptr_t>(ptr) % alignment) != 0) {
                continue;
            }

            t->buffer = mmap_buf;
            t->data = ptr;
            zero_copy_set.insert(t);
        }
    }

    // Stage 3: Allocate any remaining tensors (converted 1D F32 norms/biases, non-repacked fused QKV/FFN,
    // non-repacked compacted codec_head, or all tensors on non-CPU backends)
    bool has_unallocated = false;
    for (struct ggml_tensor * t = ggml_get_first_tensor(impl->model.ctx);
         t != nullptr;
         t = ggml_get_next_tensor(impl->model.ctx, t)) {
        if (t->data == nullptr && t->view_src == nullptr) {
            has_unallocated = true;
            break;
        }
    }
    if (has_unallocated) {
        impl->model.buffer = ggml_backend_alloc_ctx_tensors(impl->model.ctx, backend);
        if (!impl->model.buffer) {
            error_msg = "Failed to allocate tensor buffer";
            cleanup_backend();
            return false;
        }
        ggml_backend_buffer_set_usage(impl->model.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // Helper to read raw GGUF bytes for a tensor (zero-copy slice from mmap when available, else fread)
    FILE * f = nullptr;
    if (!mapped) {
        f = fopen(path.c_str(), "rb");
        if (!f) {
            error_msg = "Failed to open file for reading: " + path;
            cleanup_backend();
            return false;
        }
    }
    auto close_file = [&]() {
        if (f) {
            fclose(f);
            f = nullptr;
        }
    };

    std::vector<uint8_t> read_buf;
    auto get_src_bytes = [&](const GgufTensorInfo & info, const char * label) -> const uint8_t * {
        if (mapped) {
            if (data_offset + info.offset + info.size > mapped->size()) {
                error_msg = "Corrupt tensor offset/size in GGUF: " + std::string(label);
                return nullptr;
            }
            return mapped->data() + data_offset + info.offset;
        }
        read_buf.resize(info.size);
        const int64_t file_pos = (int64_t) data_offset + (int64_t) info.offset;
#ifdef _WIN32
        if (_fseeki64(f, file_pos, SEEK_SET) != 0) {
#else
        if (fseek(f, (long) file_pos, SEEK_SET) != 0) {
#endif
            error_msg = "Failed to seek to tensor data: " + std::string(label);
            return nullptr;
        }
        if (fread(read_buf.data(), 1, info.size, f) != info.size) {
            error_msg = "Failed to read tensor data: " + std::string(label);
            return nullptr;
        }
        return read_buf.data();
    };

    // Populate individual GGUF-backed tensors that are NOT zero-copy and NOT superseded raw tensors
    std::vector<float> f32_conv_buf;
    std::vector<uint8_t> compact_buf;

    for (const auto & kv : src_info) {
        struct ggml_tensor * tensor = kv.first;
        const GgufTensorInfo & info = kv.second;

        if (zero_copy_set.count(tensor) || superseded_raw_tensors.count(tensor)) {
            continue;
        }

        const uint8_t * src_ptr = get_src_bytes(info, ggml_get_name(tensor));
        if (!src_ptr) {
            close_file();
            cleanup_backend();
            return false;
        }

        if (tensor == impl->model.codec_head && impl->model.codec_head_compacted) {
            const size_t row_bytes = ggml_row_size(info.src_type, cfg.hidden_size);
            if (info.size < (size_t) cfg.codec_vocab_size * row_bytes) {
                error_msg = "Unexpected size for talker.codec_head.weight";
                close_file();
                cleanup_backend();
                return false;
            }
            compact_buf.resize((size_t) kCompactedCodecHeadRows * row_bytes);
            // Rows 0..2047: audio codebook 0 logits
            std::memcpy(compact_buf.data(), src_ptr, 2048 * row_bytes);
            // Row 2048: codec_eos_id logit
            std::memcpy(compact_buf.data() + 2048 * row_bytes,
                        src_ptr + (size_t) cfg.codec_eos_id * row_bytes,
                        row_bytes);
            // Rows 2049..2055: valid padding rows (copy of row 0) to keep ne[1] % 8 == 0 for CPU_REPACK
            for (int64_t r = 2049; r < kCompactedCodecHeadRows; ++r) {
                std::memcpy(compact_buf.data() + (size_t) r * row_bytes, src_ptr, row_bytes);
            }
            ggml_backend_tensor_set(tensor, compact_buf.data(), 0, compact_buf.size());
        } else if (tensor->type != info.src_type) {
            if (tensor->type != GGML_TYPE_F32) {
                error_msg = "Unsupported tensor type conversion for: " + std::string(ggml_get_name(tensor));
                close_file();
                cleanup_backend();
                return false;
            }
            const int64_t n_elems = ggml_nelements(tensor);
            f32_conv_buf.resize((size_t) n_elems);
            if (!convert_tensor_bytes_to_f32(info.src_type, src_ptr, f32_conv_buf.data(), n_elems)) {
                error_msg = "Failed to convert tensor to F32: " + std::string(ggml_get_name(tensor));
                close_file();
                cleanup_backend();
                return false;
            }
            ggml_backend_tensor_set(tensor, f32_conv_buf.data(), 0, (size_t) n_elems * sizeof(float));
        } else {
            const size_t nbytes = ggml_nbytes(tensor);
            if (info.size != nbytes) {
                error_msg = "Tensor byte size mismatch: " + std::string(ggml_get_name(tensor));
                close_file();
                cleanup_backend();
                return false;
            }
            ggml_backend_tensor_set(tensor, src_ptr, 0, nbytes);
        }

        if (mapped) {
            mapped->advise_dontneed(data_offset + info.offset, info.size);
        }
    }

    // Populate fused QKV ([Q | K | V]) and fused FFN ([gate | up]) in single full-tensor writes
    // (required by ggml_backend_cpu_repack_buffer_set_tensor which asserts offset == 0 && size == nbytes)
    std::vector<uint8_t> fused_buf;
    auto populate_layer_fused = [&](transformer_layer & layer) -> bool {
        if (layer.attn_qkv) {
            const auto & iq = src_info.at(layer.attn_q);
            const auto & ik = src_info.at(layer.attn_k);
            const auto & iv = src_info.at(layer.attn_v);
            const size_t q_bytes = ggml_nbytes(layer.attn_q);
            const size_t k_bytes = ggml_nbytes(layer.attn_k);
            const size_t v_bytes = ggml_nbytes(layer.attn_v);
            const size_t total_bytes = q_bytes + k_bytes + v_bytes;
            fused_buf.resize(total_bytes);

            const uint8_t * pq = get_src_bytes(iq, ggml_get_name(layer.attn_q));
            if (!pq) return false;
            std::memcpy(fused_buf.data(), pq, q_bytes);

            const uint8_t * pk = get_src_bytes(ik, ggml_get_name(layer.attn_k));
            if (!pk) return false;
            std::memcpy(fused_buf.data() + q_bytes, pk, k_bytes);

            const uint8_t * pv = get_src_bytes(iv, ggml_get_name(layer.attn_v));
            if (!pv) return false;
            std::memcpy(fused_buf.data() + q_bytes + k_bytes, pv, v_bytes);

            ggml_backend_tensor_set(layer.attn_qkv, fused_buf.data(), 0, total_bytes);
            if (mapped) {
                mapped->advise_dontneed(data_offset + iq.offset, iq.size);
                mapped->advise_dontneed(data_offset + ik.offset, ik.size);
                mapped->advise_dontneed(data_offset + iv.offset, iv.size);
            }
        }

        if (layer.ffn_gate_up) {
            const auto & ig = src_info.at(layer.ffn_gate);
            const auto & iu = src_info.at(layer.ffn_up);
            const size_t g_bytes = ggml_nbytes(layer.ffn_gate);
            const size_t u_bytes = ggml_nbytes(layer.ffn_up);
            const size_t total_bytes = g_bytes + u_bytes;
            fused_buf.resize(total_bytes);

            const uint8_t * pg = get_src_bytes(ig, ggml_get_name(layer.ffn_gate));
            if (!pg) return false;
            std::memcpy(fused_buf.data(), pg, g_bytes);

            const uint8_t * pu = get_src_bytes(iu, ggml_get_name(layer.ffn_up));
            if (!pu) return false;
            std::memcpy(fused_buf.data() + g_bytes, pu, u_bytes);

            ggml_backend_tensor_set(layer.ffn_gate_up, fused_buf.data(), 0, total_bytes);
            if (mapped) {
                mapped->advise_dontneed(data_offset + ig.offset, ig.size);
                mapped->advise_dontneed(data_offset + iu.offset, iu.size);
            }
        }
        return true;
    };

    for (auto & layer : impl->model.code_pred_layers) {
        if (!populate_layer_fused(layer)) {
            close_file();
            cleanup_backend();
            return false;
        }
    }
    for (auto & layer : impl->model.layers) {
        if (!populate_layer_fused(layer)) {
            close_file();
            cleanup_backend();
            return false;
        }
    }

    close_file();
    cleanup_backend();
    return true;
}

} // namespace qwen3_tts
