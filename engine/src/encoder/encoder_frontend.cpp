#include "audio_tokenizer_encoder.h"
#include "encoder/encoder_state_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

static constexpr float k_pi = 3.14159265358979323846f;

void compute_mel_filterbank_slaney(float * filterbank,
                                   std::vector<qwen3_tts::sparse_mel_band> & bands,
                                   std::vector<float> & sparse_weights,
                                   int n_mels, int n_fft,
                                   int sample_rate, float f_min, float f_max) {
    auto hz_to_mel_slaney = [](float hz) -> float {
        const float f_sp = 200.0f / 3.0f;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = min_log_hz / f_sp;
        const float logstep = logf(6.4f) / 27.0f;

        if (hz < min_log_hz) {
            return hz / f_sp;
        }
        return min_log_mel + logf(hz / min_log_hz) / logstep;
    };

    auto mel_to_hz_slaney = [](float mel) -> float {
        const float f_sp = 200.0f / 3.0f;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = min_log_hz / f_sp;
        const float logstep = logf(6.4f) / 27.0f;

        if (mel < min_log_mel) {
            return f_sp * mel;
        }
        return min_log_hz * expf(logstep * (mel - min_log_mel));
    };

    const float mel_min = hz_to_mel_slaney(f_min);
    const float mel_max = hz_to_mel_slaney(f_max);
    const int n_fft_bins = n_fft / 2 + 1;

    std::vector<float> mel_points(n_mels + 2);
    std::vector<float> hz_points(n_mels + 2);
    std::vector<float> fft_freqs(n_fft_bins);

    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        hz_points[i] = mel_to_hz_slaney(mel_points[i]);
    }

    for (int i = 0; i < n_fft_bins; ++i) {
        fft_freqs[i] = (float) i * sample_rate / n_fft;
    }

    memset(filterbank, 0, (size_t) n_mels * (size_t) n_fft_bins * sizeof(float));
    bands.assign((size_t) n_mels, {});
    sparse_weights.clear();
    sparse_weights.reserve((size_t) n_mels * 16);

    for (int m = 0; m < n_mels; ++m) {
        const float f_left = hz_points[m];
        const float f_center = hz_points[m + 1];
        const float f_right = hz_points[m + 2];
        const float enorm = 2.0f / (f_right - f_left);

        int first_nz = -1;
        int last_nz = -1;

        for (int k = 0; k < n_fft_bins; ++k) {
            const float freq = fft_freqs[k];
            float w = 0.0f;
            if (freq >= f_left && freq <= f_center) {
                if (f_center > f_left) {
                    w = enorm * (freq - f_left) / (f_center - f_left);
                }
            } else if (freq > f_center && freq <= f_right) {
                if (f_right > f_center) {
                    w = enorm * (f_right - freq) / (f_right - f_center);
                }
            }
            filterbank[m * n_fft_bins + k] = w;
            if (w != 0.0f) {
                if (first_nz < 0) {
                    first_nz = k;
                }
                last_nz = k + 1;
            }
        }

        if (first_nz >= 0 && last_nz > first_nz) {
            bands[(size_t) m].k_start = first_nz;
            bands[(size_t) m].k_end = last_nz;
            bands[(size_t) m].weight_offset = (int32_t) sparse_weights.size();
            for (int k = first_nz; k < last_nz; ++k) {
                sparse_weights.push_back(filterbank[m * n_fft_bins + k]);
            }
        }
    }
}

void compute_dft(const float * input, float * real, float * imag, int n) {
    for (int k = 0; k < n; ++k) {
        real[k] = 0.0f;
        imag[k] = 0.0f;
        for (int t = 0; t < n; ++t) {
            const float angle = -2.0f * k_pi * k * t / n;
            real[k] += input[t] * cosf(angle);
            imag[k] += input[t] * sinf(angle);
        }
    }
}

bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

void compute_fft_tables(std::vector<int32_t> & bit_reverse,
                        std::vector<float> & twiddle_r,
                        std::vector<float> & twiddle_i,
                        int n) {
    bit_reverse.resize((size_t) n);
    int bits = 0;
    while ((1 << bits) < n) {
        bits++;
    }

    for (int i = 0; i < n; ++i) {
        int x = i;
        int y = 0;
        for (int b = 0; b < bits; ++b) {
            y = (y << 1) | (x & 1);
            x >>= 1;
        }
        bit_reverse[(size_t) i] = y;
    }

    // Precompute exact twiddles for all stages len = 2, 4, ..., n (total = n - 1 entries)
    twiddle_r.resize((size_t) n);
    twiddle_i.resize((size_t) n);
    size_t offset = 0;
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const double inv_len = -2.0 * 3.14159265358979323846264338327950288 / (double) len;
        for (int j = 0; j < half; ++j) {
            const double angle = inv_len * (double) j;
            twiddle_r[offset + (size_t) j] = (float) std::cos(angle);
            twiddle_i[offset + (size_t) j] = (float) std::sin(angle);
        }
        offset += (size_t) half;
    }
}

void compute_fft_radix2(const float * input,
                        const std::vector<int32_t> & bit_reverse,
                        const float * twiddle_r,
                        const float * twiddle_i,
                        float * real,
                        float * imag,
                        int n) {
    // Stage len=2 fused with bit-reversal load (imag == 0 on real input!)
    const int32_t * br = bit_reverse.data();
    for (int i = 0; i < n; i += 2) {
        const float u = input[br[i]];
        const float v = input[br[i + 1]];
        real[i]     = u + v;
        imag[i]     = 0.0f;
        real[i + 1] = u - v;
        imag[i + 1] = 0.0f;
    }

    size_t tw_offset = 1; // Skip len=2 stage (already computed above)
    for (int len = 4; len <= n; len <<= 1) {
        const int half = len >> 1;
        const float * wr_tab = twiddle_r + tw_offset;
        const float * wi_tab = twiddle_i + tw_offset;

        for (int i = 0; i < n; i += len) {
            // j = 0 fast path (wr = 1, wi = 0)
            {
                const int even = i;
                const int odd = i + half;
                const float ur = real[even];
                const float ui = imag[even];
                const float vr = real[odd];
                const float vi = imag[odd];
                real[even] = ur + vr;
                imag[even] = ui + vi;
                real[odd]  = ur - vr;
                imag[odd]  = ui - vi;
            }
            for (int j = 1; j < half; ++j) {
                const int even = i + j;
                const int odd = even + half;
                const float wr = wr_tab[j];
                const float wi = wi_tab[j];
                const float or_val = real[odd];
                const float oi_val = imag[odd];
                const float ur = real[even];
                const float ui = imag[even];
                const float vr = or_val * wr - oi_val * wi;
                const float vi = or_val * wi + oi_val * wr;

                real[even] = ur + vr;
                imag[even] = ui + vi;
                real[odd]  = ur - vr;
                imag[odd]  = ui - vi;
            }
        }
        tw_offset += (size_t) half;
    }
}

void compute_centered_window(float * window, int n_fft, int win_length) {
    memset(window, 0, (size_t) n_fft * sizeof(float));

    const int offset = (n_fft - win_length) / 2;
    for (int i = 0; i < win_length; ++i) {
        window[offset + i] = 0.5f * (1.0f - cosf(2.0f * k_pi * i / win_length));
    }
}

} // namespace

namespace qwen3_tts {

void encoder_internal::ops::init_frontend_cache(AudioTokenizerEncoder & self) {
    auto & model = self.impl_->model;
    auto & state = self.impl_->state;

    const int n_fft_bins = model.config.n_fft / 2 + 1;
    state.mel_filterbank.resize((size_t) model.config.n_mels * (size_t) n_fft_bins);
    compute_mel_filterbank_slaney(state.mel_filterbank.data(),
                                  state.mel_bands,
                                  state.mel_sparse_weights,
                                  model.config.n_mels, model.config.n_fft,
                                  model.config.sample_rate, model.config.f_min, model.config.f_max);

    state.stft_window.resize(model.config.n_fft);
    compute_centered_window(state.stft_window.data(), model.config.n_fft, model.config.win_length);

    if (is_power_of_two(model.config.n_fft)) {
        compute_fft_tables(state.fft_bit_reverse, state.fft_twiddle_r, state.fft_twiddle_i, model.config.n_fft);
    } else {
        state.fft_bit_reverse.clear();
        state.fft_twiddle_r.clear();
        state.fft_twiddle_i.clear();
    }
}

bool encoder_internal::ops::compute_mel_spectrogram(AudioTokenizerEncoder & self,
                                                    const float * samples,
                                                    int32_t n_samples,
                                                    std::vector<float> & mel,
                                                    int32_t & n_frames) {
    const auto & cfg = self.impl_->model.config;
    auto & state = self.impl_->state;

    if (!samples || n_samples <= 0) {
        self.error_msg_ = "Audio too short for mel spectrogram";
        return false;
    }

    const int padding = (cfg.n_fft - cfg.hop_length) / 2;
    const int padded_length = n_samples + 2 * padding;

    state.scratch_padded.resize((size_t) padded_length);
    float * padded = state.scratch_padded.data();

    // Left reflect pad
    for (int i = 0; i < padding; ++i) {
        int src_idx = std::max(0, std::min(n_samples - 1, padding - i));
        padded[i] = samples[src_idx];
    }
    // Contiguous center copy
    std::memcpy(padded + padding, samples, (size_t) n_samples * sizeof(float));
    // Right reflect pad
    for (int i = padding + n_samples; i < padded_length; ++i) {
        int src_idx = std::max(0, std::min(n_samples - 1, 2 * n_samples - (i - padding) - 2));
        padded[i] = samples[src_idx];
    }

    n_frames = (padded_length - cfg.n_fft) / cfg.hop_length + 1;
    if (n_frames <= 0) {
        self.error_msg_ = "Audio too short for mel spectrogram";
        return false;
    }

    const int n_fft_bins = cfg.n_fft / 2 + 1;
    if (state.mel_filterbank.empty() || state.stft_window.empty()) {
        self.error_msg_ = "Speaker encoder frontend cache is not initialized";
        return false;
    }

    mel.resize((size_t) cfg.n_mels * (size_t) n_frames);

    state.scratch_frame.resize((size_t) cfg.n_fft);
    state.scratch_fft_real.resize((size_t) cfg.n_fft);
    state.scratch_fft_imag.resize((size_t) cfg.n_fft);
    state.scratch_magnitude.resize((size_t) n_fft_bins);

    float * frame = state.scratch_frame.data();
    float * fft_real = state.scratch_fft_real.data();
    float * fft_imag = state.scratch_fft_imag.data();
    float * magnitude = state.scratch_magnitude.data();
    const float * win = state.stft_window.data();
    const bool use_radix2 = !state.fft_bit_reverse.empty();
    const bool use_sparse_mel = (int) state.mel_bands.size() == cfg.n_mels;

    for (int32_t f = 0; f < n_frames; ++f) {
        const float * src = padded + (size_t) f * (size_t) cfg.hop_length;
        for (int i = 0; i < cfg.n_fft; ++i) {
            frame[i] = src[i] * win[i];
        }

        if (use_radix2) {
            compute_fft_radix2(frame, state.fft_bit_reverse,
                               state.fft_twiddle_r.data(), state.fft_twiddle_i.data(),
                               fft_real, fft_imag, cfg.n_fft);
        } else {
            compute_dft(frame, fft_real, fft_imag, cfg.n_fft);
        }

        for (int k = 0; k < n_fft_bins; ++k) {
            magnitude[k] = sqrtf(fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k] + 1e-9f);
        }

        if (use_sparse_mel) {
            const float * sparse_w = state.mel_sparse_weights.data();
            for (int m = 0; m < cfg.n_mels; ++m) {
                const sparse_mel_band & band = state.mel_bands[(size_t) m];
                const float * w_ptr = sparse_w + band.weight_offset;
                const float * mag_ptr = magnitude + band.k_start;
                const int count = band.k_end - band.k_start;
                float sum = 0.0f;
                for (int k = 0; k < count; ++k) {
                    sum += w_ptr[k] * mag_ptr[k];
                }
                mel[(size_t) m * (size_t) n_frames + (size_t) f] = logf(std::max(sum, 1e-5f));
            }
        } else {
            for (int m = 0; m < cfg.n_mels; ++m) {
                float sum = 0.0f;
                const float * fb_row = state.mel_filterbank.data() + (size_t) m * (size_t) n_fft_bins;
                for (int k = 0; k < n_fft_bins; ++k) {
                    sum += fb_row[k] * magnitude[k];
                }
                mel[(size_t) m * (size_t) n_frames + (size_t) f] = logf(std::max(sum, 1e-5f));
            }
        }
    }

    return true;
}

} // namespace qwen3_tts
