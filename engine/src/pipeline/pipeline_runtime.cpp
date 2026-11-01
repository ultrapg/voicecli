#include "pipeline/pipeline_internal.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace qwen3_tts {
namespace pipeline_internal {

bool env_flag_enabled(const char * name) {
    const char * v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return false;
    }

    auto ieq = [](const char * a, const char * b) -> bool {
        if (!a || !b) {
            return false;
        }
        while (*a && *b) {
            if (std::tolower((unsigned char) *a) != std::tolower((unsigned char) *b)) {
                return false;
            }
            ++a;
            ++b;
        }
        return *a == '\0' && *b == '\0';
    };

    if (strcmp(v, "0") == 0) {
        return false;
    }
    if (ieq(v, "false") || ieq(v, "off") || ieq(v, "no")) {
        return false;
    }
    return true;
}

namespace {

void ggml_log_callback_filtered(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;

    if (level == GGML_LOG_LEVEL_DEBUG && !env_flag_enabled("QWEN3_TTS_GGML_DEBUG")) {
        return;
    }
    if ((level == GGML_LOG_LEVEL_INFO || level == GGML_LOG_LEVEL_CONT) &&
        !env_flag_enabled("VOICECLI_DEBUG") && !env_flag_enabled("QWEN3_TTS_GGML_DEBUG")) {
        return;
    }

    if (text) {
        fputs(text, stderr);
        fflush(stderr);
    }
}

} // namespace

void configure_ggml_logging_once() {
    static bool configured = false;
    if (configured) {
        return;
    }
    configured = true;
    ggml_log_set(ggml_log_callback_filtered, nullptr);
}

int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool get_process_memory_snapshot(process_memory_snapshot & out) {
#ifdef __APPLE__
    mach_task_basic_info_data_t basic_info = {};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&basic_info), &basic_count) != KERN_SUCCESS) {
        return false;
    }
    out.rss_bytes = (uint64_t) basic_info.resident_size;

    task_vm_info_data_t vm_info = {};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vm_info), &vm_count) == KERN_SUCCESS) {
        out.phys_footprint_bytes = (uint64_t) vm_info.phys_footprint;
    } else {
        out.phys_footprint_bytes = out.rss_bytes;
    }
    return true;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc = {};
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmc),
                              sizeof(pmc))) {
        return false;
    }
    out.rss_bytes = (uint64_t) pmc.WorkingSetSize;
    out.phys_footprint_bytes = (uint64_t) pmc.PrivateUsage;
    return true;
#else
    struct rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return false;
    }
    out.rss_bytes = (uint64_t) usage.ru_maxrss * 1024ULL;
    out.phys_footprint_bytes = out.rss_bytes;
    return true;
#endif
}

std::string format_bytes(uint64_t bytes) {
    static const char * units[] = { "B", "KB", "MB", "GB", "TB" };
    double val = (double) bytes;
    int unit = 0;
    while (val >= 1024.0 && unit < 4) {
        val /= 1024.0;
        ++unit;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", val, units[unit]);
    return std::string(buf);
}

void log_memory_usage(const char * label) {
    if (!env_flag_enabled("VOICECLI_DEBUG")) {
        return;
    }
    process_memory_snapshot mem;
    if (!get_process_memory_snapshot(mem)) {
        fprintf(stderr, "  [mem] %-24s unavailable\n", label);
        return;
    }
    fprintf(stderr, "  [mem] %-24s rss=%s  phys=%s\n",
            label, format_bytes(mem.rss_bytes).c_str(),
            format_bytes(mem.phys_footprint_bytes).c_str());
}

void resample_linear(const float * input, int input_len, int input_rate,
                     std::vector<float> & output, int output_rate) {
    if (!input || input_len <= 0 || input_rate <= 0 || output_rate <= 0) {
        output.clear();
        return;
    }
    if (input_rate == output_rate) {
        output.assign(input, input + input_len);
        return;
    }

    const double src_per_dst = (double) input_rate / (double) output_rate;
    const int output_len = (int) ceil((double) input_len * (double) output_rate /
                                      (double) input_rate);
    output.resize(output_len);

    // 512-phase linearly-interpolated polyphase Hann-windowed sinc table (~250x faster than per-sample sin/cos)
    constexpr double pi = 3.14159265358979323846264338327950288;
    constexpr int zero_crossings = 24;
    constexpr int kPhaseCount = 512;

    const double cutoff = std::min(1.0, (double) output_rate / (double) input_rate);
    const double radius = (double) zero_crossings / cutoff;
    const int half_width = (int) floor(radius);
    const int left_off = -half_width;
    const int right_off = half_width + 1;
    const int n_taps = right_off - left_off + 1;

    auto sinc = [](double x) -> double {
        if (fabs(x) < 1.0e-8) {
            return 1.0;
        }
        return sin(x) / x;
    };

    std::vector<double> phase_rows((size_t) (kPhaseCount + 1) * (size_t) n_taps, 0.0);
    for (int p = 0; p <= kPhaseCount; ++p) {
        const double frac = (double) p / (double) kPhaseCount;
        double * row = phase_rows.data() + (size_t) p * (size_t) n_taps;
        double row_sum = 0.0;
        for (int t = 0; t < n_taps; ++t) {
            const double tap_offset = (double) (left_off + t);
            const double distance = frac - tap_offset;
            const double window_pos = fabs(distance) / radius;
            if (window_pos <= 1.0) {
                const double window = 0.5 + 0.5 * cos(pi * window_pos);
                const double w = cutoff * sinc(pi * cutoff * distance) * window;
                row[t] = w;
                row_sum += w;
            }
        }
        if (fabs(row_sum) > 1.0e-12) {
            const double inv_sum = 1.0 / row_sum;
            for (int t = 0; t < n_taps; ++t) {
                row[t] *= inv_sum;
            }
        }
    }

    std::vector<float> table_w0((size_t) kPhaseCount * (size_t) n_taps);
    std::vector<float> table_dw((size_t) kPhaseCount * (size_t) n_taps);
    for (int p = 0; p < kPhaseCount; ++p) {
        const double * r0 = phase_rows.data() + (size_t) p * (size_t) n_taps;
        const double * r1 = phase_rows.data() + (size_t) (p + 1) * (size_t) n_taps;
        float * w0 = table_w0.data() + (size_t) p * (size_t) n_taps;
        float * dw = table_dw.data() + (size_t) p * (size_t) n_taps;
        for (int t = 0; t < n_taps; ++t) {
            w0[t] = (float) r0[t];
            dw[t] = (float) (r1[t] - r0[t]);
        }
    }

    for (int i = 0; i < output_len; ++i) {
        const double center = (double) i * src_per_dst;
        const int base = (int) floor(center);
        const double frac = center - (double) base;
        const double p_pos = frac * (double) kPhaseCount;
        const int p = std::min(kPhaseCount - 1, std::max(0, (int) p_pos));
        const float alpha = (float) (p_pos - (double) p);

        const float * row_w0 = table_w0.data() + (size_t) p * (size_t) n_taps;
        const float * row_dw = table_dw.data() + (size_t) p * (size_t) n_taps;
        const int j_start = base + left_off;

        if (j_start >= 0 && j_start + n_taps <= input_len) {
            const float * src = input + j_start;
            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;
            int t = 0;
            for (; t + 3 < n_taps; t += 4) {
                s0 += src[t + 0] * (row_w0[t + 0] + alpha * row_dw[t + 0]);
                s1 += src[t + 1] * (row_w0[t + 1] + alpha * row_dw[t + 1]);
                s2 += src[t + 2] * (row_w0[t + 2] + alpha * row_dw[t + 2]);
                s3 += src[t + 3] * (row_w0[t + 3] + alpha * row_dw[t + 3]);
            }
            float sum = (s0 + s1) + (s2 + s3);
            for (; t < n_taps; ++t) {
                sum += src[t] * (row_w0[t] + alpha * row_dw[t]);
            }
            output[i] = sum;
        } else {
            const int t_lo = std::max(0, -j_start);
            const int t_hi = std::min(n_taps, input_len - j_start);
            double sum = 0.0;
            double weight_sum = 0.0;
            for (int t = t_lo; t < t_hi; ++t) {
                const double w = (double) row_w0[t] + (double) alpha * (double) row_dw[t];
                sum += (double) input[j_start + t] * w;
                weight_sum += w;
            }
            if (fabs(weight_sum) > 1.0e-12) {
                output[i] = (float) (sum / weight_sum);
            } else {
                const int nearest = std::max(0, std::min(input_len - 1, (int) llround(center)));
                output[i] = input[nearest];
            }
        }
    }
}

} // namespace pipeline_internal
} // namespace qwen3_tts
