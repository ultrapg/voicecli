#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace qwen3_tts {

struct transformer_token_prob {
    int32_t id = 0;
    float prob = 0.0f;
};

struct transformer_sampling_state {
    int64_t seed = 0;
    int64_t subseq = 0;
};

struct transformer_philox4 {
    uint32_t x, y, z, w;
};

static inline int64_t resolve_sampling_seed(int64_t seed) {
    if (seed >= 0) {
        return seed;
    }
    std::random_device rd;
    const uint64_t raw = ((uint64_t) rd() << 32) ^ (uint64_t) rd();
    // Clear sign bit so resolved random seeds are always >= 0 and never re-trigger random_device
    return (int64_t) (raw & 0x7FFFFFFFFFFFFFFFULL);
}

static inline void transformer_mulhilo32(uint32_t a, uint32_t b, uint32_t * hi, uint32_t * lo) {
    const uint64_t prod = (uint64_t) a * (uint64_t) b;
    *lo = (uint32_t) prod;
    *hi = (uint32_t) (prod >> 32);
}

static inline transformer_philox4 transformer_philox_round(transformer_philox4 ctr,
                                                           uint32_t k0,
                                                           uint32_t k1) {
    uint32_t hi0, lo0, hi1, lo1;
    transformer_mulhilo32(0xD2511F53u, ctr.x, &hi0, &lo0);
    transformer_mulhilo32(0xCD9E8D57u, ctr.z, &hi1, &lo1);
    return {
        hi1 ^ ctr.y ^ k0,
        lo1,
        hi0 ^ ctr.w ^ k1,
        lo0,
    };
}

static inline transformer_philox4 transformer_philox4x32_10(transformer_philox4 ctr,
                                                            uint32_t seed_lo,
                                                            uint32_t seed_hi) {
    uint32_t k0 = seed_lo;
    uint32_t k1 = seed_hi;
    for (int i = 0; i < 10; ++i) {
        ctr = transformer_philox_round(ctr, k0, k1);
        if (i != 9) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
    }
    return ctr;
}

static inline float transformer_philox_uniform(int64_t seed, int64_t subseq) {
    const uint32_t seed_lo = (uint32_t) seed;
    const uint32_t seed_hi = (uint32_t) ((uint64_t) seed >> 32);
    const uint64_t s = (uint64_t) subseq;
    const transformer_philox4 ctr = {0u, 0u, (uint32_t) s, (uint32_t) (s >> 32)};
    const transformer_philox4 r = transformer_philox4x32_10(ctr, seed_lo, seed_hi);
    return ((float) r.x + 0.5f) * 2.3283064365386963e-10f;
}

static inline int32_t transformer_argmax(const float * logits, int32_t n) {
    int32_t max_idx = 0;
    float max_val = logits[0];
    for (int32_t i = 1; i < n; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

static inline void transformer_apply_repetition_penalty(float * logits,
                                                        int32_t vocab_size,
                                                        const int32_t * history,
                                                        int32_t n_history,
                                                        float penalty) {
    if (penalty == 1.0f || n_history <= 0) {
        return;
    }

    static thread_local std::vector<uint8_t> seen;
    seen.assign((size_t) vocab_size, 0);
    for (int32_t i = 0; i < n_history; ++i) {
        const int32_t tok = history[i];
        if (tok < 0 || tok >= vocab_size || seen[(size_t) tok]) {
            continue;
        }
        seen[(size_t) tok] = 1;
        const float score = logits[tok];
        logits[tok] = score < 0.0f ? score * penalty : score / penalty;
    }
}

static inline int32_t transformer_sample_top_k_p(float * logits,
                                                 int32_t vocab_size,
                                                 float temperature,
                                                 int32_t top_k,
                                                 float top_p,
                                                 float repetition_penalty,
                                                 const int32_t * history,
                                                 int32_t n_history,
                                                 transformer_sampling_state & sampling) {
    transformer_apply_repetition_penalty(logits, vocab_size, history, n_history, repetition_penalty);

    if (temperature <= 0.0f) {
        return transformer_argmax(logits, vocab_size);
    }

    const float inv_temp = 1.0f / temperature;
    static thread_local std::vector<float> topk_buf;
    static thread_local std::vector<transformer_token_prob> active_buf;
    static thread_local std::vector<transformer_token_prob> sorted_buf;

    // Collect only finite scaled logits so top-k, softmax, and top-p operate on active
    // candidates (typically 50 after top_k instead of 2048-3072 full-vocab expf passes),
    // while preserving ascending token-ID accumulation order for deterministic seed parity.
    active_buf.clear();
    if ((int32_t) active_buf.capacity() < vocab_size) {
        active_buf.reserve((size_t) vocab_size);
    }

    for (int32_t i = 0; i < vocab_size; ++i) {
        const float v = logits[i] * inv_temp;
        logits[i] = v;
        if (std::isfinite(v)) {
            active_buf.push_back({i, v});
        }
    }

    if (active_buf.empty()) {
        return 0;
    }

    if (top_k > 0 && top_k < (int32_t) active_buf.size()) {
        topk_buf.resize(active_buf.size());
        for (size_t i = 0; i < active_buf.size(); ++i) {
            topk_buf[i] = active_buf[i].prob;
        }
        std::nth_element(topk_buf.begin(),
                         topk_buf.begin() + (top_k - 1),
                         topk_buf.end(),
                         std::greater<float>());
        const float threshold = topk_buf[(size_t) (top_k - 1)];
        size_t write_idx = 0;
        for (size_t i = 0; i < active_buf.size(); ++i) {
            if (active_buf[i].prob >= threshold) {
                active_buf[write_idx++] = active_buf[i];
            } else {
                logits[active_buf[i].id] = -INFINITY;
            }
        }
        active_buf.resize(write_idx);
    }

    if (top_p > 0.0f && top_p < 1.0f && active_buf.size() > 1) {
        float max_logit = active_buf[0].prob;
        for (size_t i = 1; i < active_buf.size(); ++i) {
            if (active_buf[i].prob > max_logit) {
                max_logit = active_buf[i].prob;
            }
        }

        float sum_exp = 0.0f;
        for (size_t i = 0; i < active_buf.size(); ++i) {
            sum_exp += expf(active_buf[i].prob - max_logit);
        }
        const float inv_sum = 1.0f / sum_exp;
        const float cutoff = max_logit - 16.0f;

        sorted_buf.clear();
        for (size_t i = 0; i < active_buf.size(); ++i) {
            const int32_t id = active_buf[i].id;
            const float lg = active_buf[i].prob;
            if (lg >= cutoff) {
                sorted_buf.push_back({id, expf(lg - max_logit) * inv_sum});
            } else {
                logits[id] = -INFINITY;
            }
        }

        std::sort(sorted_buf.begin(), sorted_buf.end(),
                  [](const transformer_token_prob & a, const transformer_token_prob & b) {
                      return a.prob > b.prob;
                  });

        float cumulative = 0.0f;
        for (size_t i = 0; i < sorted_buf.size(); ++i) {
            if (i > 0 && cumulative >= top_p) {
                logits[sorted_buf[i].id] = -INFINITY;
            }
            cumulative += sorted_buf[i].prob;
        }

        size_t write_idx = 0;
        for (size_t i = 0; i < active_buf.size(); ++i) {
            if (std::isfinite(logits[active_buf[i].id])) {
                active_buf[write_idx++] = active_buf[i];
            }
        }
        active_buf.resize(write_idx);
    }

    float max_val = active_buf[0].prob;
    for (size_t i = 1; i < active_buf.size(); ++i) {
        if (active_buf[i].prob > max_val) {
            max_val = active_buf[i].prob;
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < active_buf.size(); ++i) {
        const float w = expf(active_buf[i].prob - max_val);
        active_buf[i].prob = w;
        sum += w;
    }

    const float u = transformer_philox_uniform(sampling.seed, sampling.subseq++);
    const float target = u * sum;
    float acc = 0.0f;
    for (size_t i = 0; i < active_buf.size(); ++i) {
        acc += active_buf[i].prob;
        if (acc >= target) {
            return active_buf[i].id;
        }
    }
    return active_buf.back().id;
}

} // namespace qwen3_tts
