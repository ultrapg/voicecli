#include "audio_tokenizer_encoder.h"
#include "encoder/encoder_state_internal.h"
#include "gguf_loader.h"

namespace qwen3_tts {

AudioTokenizerEncoder::AudioTokenizerEncoder()
    : impl_(std::make_unique<speaker_encoder_private>()) {
}

AudioTokenizerEncoder::~AudioTokenizerEncoder() {
    unload_model();
}

void AudioTokenizerEncoder::unload_model() {
    auto & state = impl_->state;

    if (state.sched) {
        ggml_backend_sched_free(state.sched);
        state.sched = nullptr;
    }
    if (state.backend) {
        release_preferred_backend(state.backend);
        state.backend = nullptr;
    }
    if (state.backend_cpu) {
        release_preferred_backend(state.backend_cpu);
        state.backend_cpu = nullptr;
    }

    state.compute_meta.clear();
    state.mel_filterbank.clear();
    state.mel_bands.clear();
    state.mel_sparse_weights.clear();
    state.stft_window.clear();
    state.fft_bit_reverse.clear();
    state.fft_twiddle_r.clear();
    state.fft_twiddle_i.clear();
    state.scratch_padded.clear();
    state.scratch_frame.clear();
    state.scratch_fft_real.clear();
    state.scratch_fft_imag.clear();
    state.scratch_magnitude.clear();
    free_speaker_encoder_model(impl_->model);
}

const speaker_encoder_config & AudioTokenizerEncoder::get_config() const {
    return impl_->model.config;
}

void free_speaker_encoder_model(speaker_encoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
}

} // namespace qwen3_tts
