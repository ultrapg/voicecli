#pragma once

#include "audio_tokenizer_encoder.h"
#include "encoder/encoder_internal.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <map>
#include <vector>

namespace qwen3_tts {

struct res2net_block {
    struct ggml_tensor * tdnn1_w = nullptr;
    struct ggml_tensor * tdnn1_b = nullptr;

    struct ggml_tensor * res2net_w[7] = {nullptr};
    struct ggml_tensor * res2net_b[7] = {nullptr};

    struct ggml_tensor * tdnn2_w = nullptr;
    struct ggml_tensor * tdnn2_b = nullptr;

    struct ggml_tensor * se_conv1_w = nullptr;
    struct ggml_tensor * se_conv1_b = nullptr;
    struct ggml_tensor * se_conv2_w = nullptr;
    struct ggml_tensor * se_conv2_b = nullptr;
};

struct speaker_encoder_model {
    speaker_encoder_config config;

    struct ggml_tensor * conv0_w = nullptr;
    struct ggml_tensor * conv0_b = nullptr;

    res2net_block blocks[3];

    struct ggml_tensor * mfa_w = nullptr;
    struct ggml_tensor * mfa_b = nullptr;

    struct ggml_tensor * asp_conv_w = nullptr;
    struct ggml_tensor * asp_conv_b = nullptr;
    struct ggml_tensor * asp_tdnn_w = nullptr;
    struct ggml_tensor * asp_tdnn_b = nullptr;

    struct ggml_tensor * fc_w = nullptr;
    struct ggml_tensor * fc_b = nullptr;

    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct sparse_mel_band {
    int32_t k_start = 0;
    int32_t k_end = 0;
    int32_t weight_offset = 0;
};

struct speaker_encoder_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
    std::vector<float> mel_filterbank;
    std::vector<sparse_mel_band> mel_bands;
    std::vector<float> mel_sparse_weights;
    std::vector<float> stft_window;
    std::vector<int32_t> fft_bit_reverse;
    std::vector<float> fft_twiddle_r;
    std::vector<float> fft_twiddle_i;
    std::vector<float> scratch_padded;
    std::vector<float> scratch_frame;
    std::vector<float> scratch_fft_real;
    std::vector<float> scratch_fft_imag;
    std::vector<float> scratch_magnitude;
};

struct speaker_encoder_private {
    speaker_encoder_model model;
    speaker_encoder_state state;
};

} // namespace qwen3_tts
