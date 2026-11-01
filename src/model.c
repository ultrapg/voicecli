#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "qwen3_tts_c.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

extern const char* voicecli_get_active_simd_name(void);
extern int voicecli_get_active_simd_tier(void);

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <process.h>
#define isatty _isatty
#define fileno _fileno
#define getpid _getpid
#define PATH_SEP '\\'
#else
#include <unistd.h>
#define PATH_SEP '/'
#endif

typedef struct {
    const char* output_path;
    const char* model_override;
    const char* language;
    float speed;
    float temperature;
    float top_p;
    int32_t top_k;
    int32_t max_tokens;
    float repetition_penalty;
    int32_t threads;
    int32_t stream;
    float stream_chunk_sec;
    int32_t normalize;
    int32_t play;
    const char* ref_text;
    const char* voice_profile;
    const char* save_profile;
} voicecli_options_t;

// Fast monotonic nanosecond clock (userspace vDSO on Linux, zero kernel syscalls)
static int64_t monotonic_now_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (int64_t)((double)counter.QuadPart * 1e9 / (double)freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
#endif
}

static bool is_ascii_vowel(unsigned char c) {
    unsigned char lc = (unsigned char)(c | 0x20);
    return lc == 'a' || lc == 'e' || lc == 'i' || lc == 'o' || lc == 'u' || lc == 'y';
}

// Fast O(N) syllable, digit, CJK, and prosodic-pause codec frame estimator (12.5 Hz = 80ms/frame).
static int estimate_speech_codec_frames(const char* text, float rate_scale) {
    if (!text || text[0] == '\0') return 20;

    double frames = 10.0; // Acoustic onset + trailing silence margin (~800ms)
    int word_alpha_len = 0;
    int word_vowel_groups = 0;
    bool prev_vowel = false;
    unsigned char prev_alpha = 0;
    unsigned char prev_prev_alpha = 0;

    const unsigned char* p = (const unsigned char*)text;
    while (*p) {
        unsigned char b = *p;
        if (b < 0x80) {
            if ((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || b == '\'') {
                if (b != '\'') {
                    unsigned char lc = (unsigned char)(b | 0x20);
                    bool v = is_ascii_vowel(lc);
                    if (v && !prev_vowel) {
                        word_vowel_groups++;
                    }
                    prev_vowel = v;
                    prev_prev_alpha = prev_alpha;
                    prev_alpha = lc;
                    word_alpha_len++;
                }
                p++;
                continue;
            }

            // Flush finished ASCII alphabetic word
            if (word_alpha_len > 0) {
                int syl = word_vowel_groups;
                // Discount silent trailing '-e' (e.g., "make", "time", "voice", but keep "-le" as in "table")
                if (word_alpha_len >= 4 && prev_alpha == 'e' && !is_ascii_vowel(prev_prev_alpha) && prev_prev_alpha != 'l' && syl > 1) {
                    syl--;
                }
                if (syl < 1) syl = 1;
                frames += (double)syl * 2.95;
                word_alpha_len = 0;
                word_vowel_groups = 0;
                prev_vowel = false;
                prev_alpha = 0;
                prev_prev_alpha = 0;
            }

            if (b >= '0' && b <= '9') {
                frames += 3.5; // Spoken digits average ~1.3 syllables (~280ms) each
            } else if (b == ',' || b == ';' || b == ':') {
                frames += 2.5; // Clause pause (~200ms)
            } else if (b == '.' || b == '!' || b == '?') {
                // Skip decimal points (3.14) and repeated ellipsis dots (...)
                bool is_decimal = (b == '.' && p > (const unsigned char*)text &&
                                   p[-1] >= '0' && p[-1] <= '9' && p[1] >= '0' && p[1] <= '9');
                bool is_repeat_dot = (b == '.' && p[1] == '.');
                if (!is_decimal && !is_repeat_dot) {
                    frames += 4.5; // Sentence boundary pause (~360ms)
                }
            } else if (b == '\n') {
                frames += 3.5;
            }
            p++;
        } else {
            // Flush any pending ASCII word
            if (word_alpha_len > 0) {
                int syl = word_vowel_groups > 0 ? word_vowel_groups : 1;
                frames += (double)syl * 2.95;
                word_alpha_len = 0;
                word_vowel_groups = 0;
                prev_vowel = false;
                prev_alpha = 0;
                prev_prev_alpha = 0;
            }

            // Decode UTF-8 codepoint
            uint32_t cp = 0;
            int adv = 1;
            if ((b & 0xE0) == 0xC0 && (p[1] & 0xC0) == 0x80) {
                cp = ((uint32_t)(b & 0x1F) << 6) | (uint32_t)(p[1] & 0x3F);
                adv = 2;
            } else if ((b & 0xF0) == 0xE0 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80) {
                cp = ((uint32_t)(b & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6) | (uint32_t)(p[2] & 0x3F);
                adv = 3;
            } else if ((b & 0xF8) == 0xF0 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80 && (p[3] & 0xC0) == 0x80) {
                cp = ((uint32_t)(b & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12) |
                     ((uint32_t)(p[2] & 0x3F) << 6) | (uint32_t)(p[3] & 0x3F);
                adv = 4;
            }

            if ((cp >= 0x3040 && cp <= 0x30FF) || // Hiragana & Katakana
                (cp >= 0x3400 && cp <= 0x9FFF) || // CJK Ideographs
                (cp >= 0xAC00 && cp <= 0xD7AF)) { // Hangul Syllables
                frames += 3.4;
            } else if (cp == 0x3002 || cp == 0xFF01 || cp == 0xFF1F) { // 。！？
                frames += 4.5;
            } else if (cp == 0x3001 || cp == 0xFF0C || cp == 0x2013 || cp == 0x2014) { // 、，–—
                frames += 2.5;
            } else if (cp >= 0x00C0 && cp <= 0x052F) { // Accented Latin / Greek / Cyrillic letters
                frames += 0.95;
            }
            p += adv;
        }
    }

    if (word_alpha_len > 0) {
        int syl = word_vowel_groups;
        if (word_alpha_len >= 4 && prev_alpha == 'e' && !is_ascii_vowel(prev_prev_alpha) && prev_prev_alpha != 'l' && syl > 1) {
            syl--;
        }
        if (syl < 1) syl = 1;
        frames += (double)syl * 2.95;
    }

    if (rate_scale > 0.1f) {
        frames *= (double)rate_scale;
    }
    int est = (int)(frames + 0.5);
    if (est < 12) est = 12;
    if (est > 4096) est = 4096;
    return est;
}

static double g_measured_voc_sec_per_frame = 0.0;

static double estimate_vocode_to_gen_ratio(const char* model_name) {
    if (!model_name) return 0.50;
    const bool is_06b = (strstr(model_name, "0.6b") != NULL || strstr(model_name, "0.6B") != NULL);
    const bool is_q4  = (strstr(model_name, "Q4") != NULL || strstr(model_name, "q4") != NULL);
    const bool is_q8  = (strstr(model_name, "Q8") != NULL || strstr(model_name, "q8") != NULL);
    if (is_06b) {
        if (is_q4) return 1.12;
        if (is_q8) return 0.88;
        return 0.62;
    }
    if (is_q4) return 0.62;
    if (is_q8) return 0.48;
    return 0.34;
}

typedef struct {
    int raw_estimated_frames;
    int estimated_frames;
    int last_tokens;
    int actual_total_frames;
    int is_tty;
    int is_streaming;
    int64_t start_ns;
    int64_t gen_start_ns;
    int64_t vocode_start_ns;
    int64_t last_render_ns;
    double last_pct;
    double vocode_start_pct;
    double vocode_to_gen_ratio;
    uint32_t spin_step;
} tts_progress_t;

static void init_tts_progress(tts_progress_t* prog, const char* text, float rate_scale, int is_streaming, const char* model_name) {
    memset(prog, 0, sizeof(*prog));
    prog->is_tty = isatty(fileno(stderr));
    prog->is_streaming = is_streaming;
    prog->raw_estimated_frames = estimate_speech_codec_frames(text, 1.0f);
    prog->estimated_frames = estimate_speech_codec_frames(text, rate_scale);
    prog->vocode_to_gen_ratio = estimate_vocode_to_gen_ratio(model_name);
    prog->start_ns = monotonic_now_ns();
}

static void render_progress_bar(char* bar, int bar_width, double pct) {
    int filled = (int)((pct * (double)bar_width) / 100.0 + 0.5);
    if (filled < 0) filled = 0;
    if (filled > bar_width) filled = bar_width;
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) bar[i] = '=';
        else if (i == filled && filled < bar_width) bar[i] = '>';
        else bar[i] = ' ';
    }
    bar[bar_width] = '\0';
}

// Accurate 3-phase (Prefill -> Generate -> Vocode -> Done) progress callback.
// Throttled to <= 20 FPS (50ms) via vDSO clock_gettime so CPU overhead is ~0.00%.
static void on_phase_progress(int32_t phase, int32_t current, int32_t total, void* user_data) {
    tts_progress_t* prog = (tts_progress_t*)user_data;
    if (!prog) return;

    if (!prog->is_tty) {
        if (phase == QWEN3_TTS_PHASE_GENERATE) {
            prog->last_tokens = current;
            if (current == 1 || current % 25 == 0) {
                float audio_sec = (float)current / 12.5f;
                fprintf(stderr, "[*] Synthesizing: %d frames (~%.1fs)...\n", current, audio_sec);
                fflush(stderr);
            }
        } else if (phase == QWEN3_TTS_PHASE_VOCODE && current == 0 && total > 0) {
            prog->actual_total_frames = total;
            fprintf(stderr, "[*] Vocoding %d frames (%.1fs audio)...\n", total, (float)total / 12.5f);
            fflush(stderr);
        } else if (phase == QWEN3_TTS_PHASE_DONE && total > 0) {
            prog->actual_total_frames = total;
        }
        return;
    }

    const int64_t now_ns = monotonic_now_ns();
    if (prog->start_ns == 0) prog->start_ns = now_ns;

    // 50ms throttle during steady-state token/tile loops; never throttle phase transitions
    const bool is_steady_gen = (phase == QWEN3_TTS_PHASE_GENERATE && current > 1);
    const bool is_steady_voc = (phase == QWEN3_TTS_PHASE_VOCODE && current > 0 && current < total);
    if ((is_steady_gen || is_steady_voc) && (now_ns - prog->last_render_ns) < 50000000LL) {
        if (phase == QWEN3_TTS_PHASE_GENERATE) prog->last_tokens = current;
        return;
    }
    prog->last_render_ns = now_ns;

    static const char* spinner[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
    const char* spin = spinner[(prog->spin_step++) % 10];
    enum { BAR_WIDTH = 16 };
    char bar[BAR_WIDTH + 1];

    if (phase == QWEN3_TTS_PHASE_PREFILL) {
        double pct = 2.0;
        if (pct < prog->last_pct) pct = prog->last_pct;
        prog->last_pct = pct;
        render_progress_bar(bar, BAR_WIDTH, pct);
        fprintf(stderr, "\r\033[K[*] Synthesizing [%s] [%s] %3d%% | Prefill...",
                spin, bar, (int)(pct + 0.5));
        fflush(stderr);
        return;
    }

    if (phase == QWEN3_TTS_PHASE_GENERATE) {
        if (prog->gen_start_ns == 0) prog->gen_start_ns = now_ns;
        prog->last_tokens = current;

        const bool streaming = (total < 0) || (prog->is_streaming != 0);
        double voc_ratio = prog->vocode_to_gen_ratio > 0.05 ? prog->vocode_to_gen_ratio : 0.50;
        double sec_per_frame = 0.0;
        const bool have_gen_rate = (current >= 3 && (now_ns - prog->gen_start_ns) > 50000000LL);
        if (have_gen_rate) {
            sec_per_frame = ((double)(now_ns - prog->gen_start_ns) * 1e-9) / (double)(current - 1);
            if (g_measured_voc_sec_per_frame > 0.001 && sec_per_frame > 0.001) {
                voc_ratio = g_measured_voc_sec_per_frame / sec_per_frame;
            }
        }

        const double span_lo = 3.0;
        const double span_hi = streaming
            ? 99.0
            : (span_lo + 96.0 * (1.0 / (1.0 + voc_ratio)));
        const double est = (double)(prog->estimated_frames > 8 ? prog->estimated_frames : 8);
        const double r = (double)current / est;

        // C1-smooth monotonic curve: linear up to 82% of estimate, smooth rational asymptote beyond
        const double frac = (r <= 0.82)
            ? r
            : (0.82 + 0.18 * (r - 0.82) / ((r - 0.82) + 0.22));
        double pct = span_lo + (span_hi - span_lo) * frac;
        if (pct > span_hi) pct = span_hi;
        if (pct < prog->last_pct) pct = prog->last_pct;
        prog->last_pct = pct;

        render_progress_bar(bar, BAR_WIDTH, pct);
        const double audio_sec = (double)current / 12.5;
        const double wall_sec = (double)(now_ns - prog->start_ns) * 1e-9;
        const double rtf = wall_sec > 0.05 ? (audio_sec / wall_sec) : 0.0;

        if (have_gen_rate) {
            const double rem_gen_frames = (current < prog->estimated_frames)
                ? (double)(prog->estimated_frames - current)
                : fmax(3.0, 0.15 * (double)current);
            const double expected_total = (double)current + rem_gen_frames;
            const double voc_spf = (g_measured_voc_sec_per_frame > 0.001)
                ? g_measured_voc_sec_per_frame
                : (voc_ratio * sec_per_frame);
            const double est_voc_sec = streaming ? 0.0 : (voc_spf * expected_total);
            const double eta_sec = rem_gen_frames * sec_per_frame + est_voc_sec;
            fprintf(stderr, "\r\033[K[*] Synthesizing [%s] [%s] %3d%% | Gen %.1fs | %.2fx RT | ETA %.1fs",
                    spin, bar, (int)(pct + 0.5), audio_sec, rtf, eta_sec);
        } else {
            fprintf(stderr, "\r\033[K[*] Synthesizing [%s] [%s] %3d%% | Gen %.1fs | %.2fx RT",
                    spin, bar, (int)(pct + 0.5), audio_sec, rtf);
        }
        fflush(stderr);
        return;
    }

    if (phase == QWEN3_TTS_PHASE_VOCODE) {
        if (total > 0) prog->actual_total_frames = total;
        if (prog->vocode_start_ns == 0) {
            prog->vocode_start_ns = now_ns;
            const double default_voc_start = 3.0 + 96.0 * (1.0 / (1.0 + prog->vocode_to_gen_ratio));
            prog->vocode_start_pct = prog->last_pct < (default_voc_start - 8.0)
                ? (default_voc_start - 8.0)
                : prog->last_pct;
        }
        const double v_frac = (total > 0) ? ((double)current / (double)total) : 0.0;
        double pct = prog->vocode_start_pct + (99.4 - prog->vocode_start_pct) * v_frac;
        if (pct > 99.4) pct = 99.4;
        if (pct < prog->last_pct) pct = prog->last_pct;
        prog->last_pct = pct;

        render_progress_bar(bar, BAR_WIDTH, pct);
        const double done_sec = (double)current / 12.5;
        const double total_sec = (double)(total > 0 ? total : current) / 12.5;

        if (current >= 4 && total > current && (now_ns - prog->vocode_start_ns) > 50000000LL) {
            const double voc_dt = (double)(now_ns - prog->vocode_start_ns) * 1e-9;
            const double eta_sec = (voc_dt / (double)current) * (double)(total - current);
            fprintf(stderr, "\r\033[K[*] Synthesizing [%s] [%s] %3d%% | Vocode %.1f/%.1fs | ETA %.1fs",
                    spin, bar, (int)(pct + 0.5), done_sec, total_sec, eta_sec);
        } else {
            fprintf(stderr, "\r\033[K[*] Synthesizing [%s] [%s] %3d%% | Vocode %.1f/%.1fs",
                    spin, bar, (int)(pct + 0.5), done_sec, total_sec);
        }
        fflush(stderr);
        return;
    }

    if (phase == QWEN3_TTS_PHASE_DONE) {
        if (total > 0) prog->actual_total_frames = total;
        if (prog->vocode_start_ns > 0 && prog->actual_total_frames >= 8) {
            const double voc_sec = (double)(now_ns - prog->vocode_start_ns) * 1e-9;
            const double obs_spf = voc_sec / (double)prog->actual_total_frames;
            if (obs_spf > 0.001 && obs_spf < 2.0) {
                g_measured_voc_sec_per_frame = (g_measured_voc_sec_per_frame > 0.001)
                    ? (0.65 * g_measured_voc_sec_per_frame + 0.35 * obs_spf)
                    : obs_spf;
            }
        }
        prog->last_pct = 100.0;
        render_progress_bar(bar, BAR_WIDTH, 100.0);
        const double total_sec = (double)(total > 0 ? total : prog->last_tokens) / 12.5;
        fprintf(stderr, "\r\033[K[*] Synthesizing [✓] [%s] 100%% | Done (%.1fs)",
                bar, total_sec);
        fflush(stderr);
    }
}

static int dir_exists(const char *path) {
    struct stat st;
    return (stat(path, &st) == 0 && (st.st_mode & S_IFDIR));
}

static void ensure_dir_exists(const char *path) {
    if (!path || strlen(path) == 0 || strcmp(path, ".") == 0) return;
    if (dir_exists(path)) return;
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0755);
#endif
}

// Cache settings.json in memory once per process to avoid repeated file opens per parameter/chunk
static const char* get_cached_settings_json(void) {
    static int s_loaded = 0;
    static char s_buf[8192];

    if (s_loaded) {
        return s_buf[0] != '\0' ? s_buf : NULL;
    }
    s_loaded = 1;
    s_buf[0] = '\0';

    FILE *f = fopen("settings.json", "rb");
    if (!f) {
        const char *base_dir = getenv("VOICECLI_BASE_DIR");
        if (base_dir && base_dir[0] != '\0') {
            char p[4096];
            snprintf(p, sizeof(p), "%s%csettings.json", base_dir, PATH_SEP);
            f = fopen(p, "rb");
        }
    }
    if (f) {
        size_t n = fread(s_buf, 1, sizeof(s_buf) - 1, f);
        s_buf[n] = '\0';
        fclose(f);
        return s_buf;
    }
    return NULL;
}

static void format_env_key(const char *key, char *env_key, size_t env_key_size) {
    snprintf(env_key, env_key_size, "VOICECLI_%s", key);
    for (size_t i = 0; env_key[i]; i++) {
        if (env_key[i] >= 'a' && env_key[i] <= 'z') env_key[i] -= 32;
    }
}

static void get_setting_string(const char *key, char *out, size_t out_size, const char *default_val) {
    char env_key[128];
    format_env_key(key, env_key, sizeof(env_key));
    const char *env_val = getenv(env_key);
    if (env_val && env_val[0] != '\0') {
        strncpy(out, env_val, out_size - 1);
        out[out_size - 1] = '\0';
        return;
    }

    const char *json = get_cached_settings_json();
    if (json) {
        char search_key[128];
        snprintf(search_key, sizeof(search_key), "\"%s\"", key);
        const char *pos = strstr(json, search_key);
        if (pos) {
            const char *colon = strchr(pos + strlen(search_key), ':');
            if (colon) {
                const char *start = strchr(colon + 1, '"');
                if (start) {
                    start++;
                    const char *end = strchr(start, '"');
                    if (end) {
                        size_t len = (size_t)(end - start);
                        if (len >= out_size) len = out_size - 1;
                        memcpy(out, start, len);
                        out[len] = '\0';
                        return;
                    }
                }
            }
        }
    }

    strncpy(out, default_val, out_size - 1);
    out[out_size - 1] = '\0';
}

static float get_setting_float(const char *key, float default_val) {
    char env_key[128];
    format_env_key(key, env_key, sizeof(env_key));
    const char *env_val = getenv(env_key);
    if (env_val && env_val[0] != '\0') {
        char *endptr = NULL;
        float v = strtof(env_val, &endptr);
        if (endptr != env_val) return v;
    }

    const char *json = get_cached_settings_json();
    if (json) {
        char search_key[128];
        snprintf(search_key, sizeof(search_key), "\"%s\"", key);
        const char *pos = strstr(json, search_key);
        if (pos) {
            const char *colon = strchr(pos + strlen(search_key), ':');
            if (colon) {
                const char *val_start = colon + 1;
                while (*val_start == ' ' || *val_start == '\t') val_start++;
                char *endptr = NULL;
                float v = strtof(val_start, &endptr);
                if (endptr != val_start) return v;
            }
        }
    }
    return default_val;
}

static int32_t get_setting_int(const char *key, int32_t default_val) {
    char env_key[128];
    format_env_key(key, env_key, sizeof(env_key));
    const char *env_val = getenv(env_key);
    if (env_val && env_val[0] != '\0') {
        char *endptr = NULL;
        long v = strtol(env_val, &endptr, 10);
        if (endptr != env_val) return (int32_t)v;
    }

    const char *json = get_cached_settings_json();
    if (json) {
        char search_key[128];
        snprintf(search_key, sizeof(search_key), "\"%s\"", key);
        const char *pos = strstr(json, search_key);
        if (pos) {
            const char *colon = strchr(pos + strlen(search_key), ':');
            if (colon) {
                const char *val_start = colon + 1;
                while (*val_start == ' ' || *val_start == '\t') val_start++;
                char *endptr = NULL;
                long v = strtol(val_start, &endptr, 10);
                if (endptr != val_start) return (int32_t)v;
            }
        }
    }
    return default_val;
}

static int32_t resolve_language_id(const char *lang) {
    char buf[64];
    if (!lang || strlen(lang) == 0) {
        get_setting_string("language", buf, sizeof(buf), "auto");
    } else {
        strncpy(buf, lang, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
    }
    for (int i = 0; buf[i]; i++) {
        buf[i] = (char)tolower((unsigned char)buf[i]);
    }
    if (strcmp(buf, "auto") == 0) return -1;
    if (strcmp(buf, "en") == 0 || strcmp(buf, "english") == 0) return 2050;
    if (strcmp(buf, "de") == 0 || strcmp(buf, "german") == 0 || strcmp(buf, "deutsch") == 0) return 2053;
    if (strcmp(buf, "es") == 0 || strcmp(buf, "spanish") == 0) return 2054;
    if (strcmp(buf, "zh") == 0 || strcmp(buf, "chinese") == 0) return 2055;
    if (strcmp(buf, "ja") == 0 || strcmp(buf, "japanese") == 0) return 2058;
    if (strcmp(buf, "fr") == 0 || strcmp(buf, "french") == 0) return 2061;
    if (strcmp(buf, "ko") == 0 || strcmp(buf, "korean") == 0) return 2064;
    if (strcmp(buf, "ru") == 0 || strcmp(buf, "russian") == 0) return 2069;
    if (strcmp(buf, "it") == 0 || strcmp(buf, "italian") == 0) return 2070;
    if (strcmp(buf, "pt") == 0 || strcmp(buf, "portuguese") == 0) return 2071;
    return -1;
}

static void resolve_model_preset(const char *mode, const char *override_str, char *out, size_t out_size) {
    if (override_str && strlen(override_str) > 0) {
        char lower[256];
        strncpy(lower, override_str, sizeof(lower) - 1);
        lower[sizeof(lower) - 1] = '\0';
        for (int i = 0; lower[i]; i++) lower[i] = (char)tolower((unsigned char)lower[i]);

        // If exact .gguf filename provided
        if (strstr(lower, ".gguf") != NULL) {
            strncpy(out, override_str, out_size - 1);
            out[out_size - 1] = '\0';
            return;
        }

        const char *size_part = "1.7b";
        if (strstr(lower, "0.6b") != NULL ||
            strcmp(lower, "fast") == 0 ||
            strcmp(lower, "small") == 0 ||
            strcmp(lower, "tiny") == 0) {
            size_part = "0.6b";
        }
        const char *quant_part = "BF16";
        if (strstr(lower, "q4") != NULL || strcmp(lower, "tiny") == 0) {
            quant_part = "Q4_K_M";
        } else if (strstr(lower, "q8") != NULL || strcmp(lower, "fast") == 0) {
            quant_part = "Q8_0";
        } else if (strstr(lower, "f32") != NULL) {
            quant_part = "F32";
        }

        const char *variant = "base";
        if (strcmp(mode, "style") == 0) {
            variant = "voicedesign";
            size_part = "1.7b"; // voicedesign is 1.7b
        } else if (strcmp(mode, "speak") == 0) {
            variant = "customvoice";
        }

        snprintf(out, out_size, "qwen-talker-%s-%s-%s.gguf", size_part, variant, quant_part);
        return;
    }

    if (strcmp(mode, "style") == 0) {
        get_setting_string("model_name_style", out, out_size, "qwen-talker-1.7b-voicedesign-BF16.gguf");
    } else if (strcmp(mode, "speak") == 0) {
        get_setting_string("model_name_speak", out, out_size, "qwen-talker-1.7b-customvoice-BF16.gguf");
    } else {
        get_setting_string("model_name_clone", out, out_size, "qwen-talker-1.7b-base-BF16.gguf");
    }
}

static const char* resolve_model_dir(void) {
    static char model_dir[4096];
    const char* custom_model_dir = getenv("VOICECLI_MODEL_DIR");
    if (custom_model_dir && strlen(custom_model_dir) > 0) {
        ensure_dir_exists(custom_model_dir);
        strncpy(model_dir, custom_model_dir, sizeof(model_dir) - 1);
        model_dir[sizeof(model_dir) - 1] = '\0';
        return model_dir;
    }

    if (dir_exists("models")) {
        return "models";
    }

    const char* base_dir = getenv("VOICECLI_BASE_DIR");
    if (base_dir && strlen(base_dir) > 0) {
        char sub[4096];
        snprintf(sub, sizeof(sub), "%s%cmodels", base_dir, PATH_SEP);
        if (dir_exists(sub)) {
            strncpy(model_dir, sub, sizeof(model_dir) - 1);
            model_dir[sizeof(model_dir) - 1] = '\0';
            return model_dir;
        }

        char parent_models[4096];
        snprintf(parent_models, sizeof(parent_models), "%s%c..%c..%cmodels", base_dir, PATH_SEP, PATH_SEP, PATH_SEP);
        if (dir_exists(parent_models)) {
            strncpy(model_dir, parent_models, sizeof(model_dir) - 1);
            model_dir[sizeof(model_dir) - 1] = '\0';
            return model_dir;
        }

        strncpy(model_dir, base_dir, sizeof(model_dir) - 1);
        model_dir[sizeof(model_dir) - 1] = '\0';
        return model_dir;
    }

    return ".";
}

static int ensure_file_exists(const char *dir, const char *filename) {
    ensure_dir_exists(dir);
    char filepath[4096];
    snprintf(filepath, sizeof(filepath), "%s%c%s", dir, PATH_SEP, filename);
    struct stat st;
    if (stat(filepath, &st) == 0 && st.st_size > 1000000) {
        return 0;
    }
    char local_check[4096];
    snprintf(local_check, sizeof(local_check), "models%c%s", PATH_SEP, filename);
    if (stat(local_check, &st) == 0 && st.st_size > 1000000) {
        return 0;
    }
    if (stat(filename, &st) == 0 && st.st_size > 1000000) {
        return 0;
    }

    fprintf(stderr, "[*] Auto-Setup: Model '%s' not found locally.\n", filename);
    fprintf(stderr, "[*] Downloading from Hugging Face (Serveurperso/Qwen3-TTS-GGUF)...\n");
    char part_path[4112];
    snprintf(part_path, sizeof(part_path), "%s.part", filepath);
    char dl_cmd[8192];
    snprintf(dl_cmd, sizeof(dl_cmd),
        "curl -f -L --progress-bar --retry 3 --retry-delay 2 -C - \"https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/main/%s\" -o \"%s\"",
        filename, part_path);
    if (system(dl_cmd) != 0) {
        fprintf(stderr, "[-] Error: Failed to download '%s'. Please check your internet connection.\n", filename);
        return -1;
    }
    if (rename(part_path, filepath) != 0) {
        fprintf(stderr, "[-] Error: Failed to finalize downloaded model '%s': %s\n", filepath, strerror(errno));
        return -1;
    }
    fprintf(stderr, "[+] Download of '%s' completed successfully!\n", filename);
    return 0;
}

static void apply_peak_normalization(float *samples, size_t num_samples) {
    if (!samples || num_samples == 0) return;

    // 1. Remove DC offset
    double sum = 0.0;
    for (size_t i = 0; i < num_samples; i++) {
        sum += (double)samples[i];
    }
    float dc_offset = (float)(sum / (double)num_samples);
    if (fabsf(dc_offset) > 1e-5f) {
        for (size_t i = 0; i < num_samples; i++) {
            samples[i] -= dc_offset;
        }
    }

    // 2. Find peak amplitude
    float max_abs = 0.0f;
    for (size_t i = 0; i < num_samples; i++) {
        float a = fabsf(samples[i]);
        if (a > max_abs) max_abs = a;
    }

    // 3. Scale to -1.0 dBFS (~0.89125) if signal is non-silent
    const float target_peak = 0.89125f;
    if (max_abs > 1e-4f && fabsf(max_abs - target_peak) > 0.01f) {
        float gain = target_peak / max_abs;
        // Cap gain at +20 dB (10x) to avoid amplifying pure background noise
        if (gain > 10.0f) gain = 10.0f;
        for (size_t i = 0; i < num_samples; i++) {
            samples[i] *= gain;
        }
    }
}

static int save_wav_file(const char *filename, const float *samples, int num_samples, int sample_rate) {
    if (!filename || !samples || num_samples <= 0) return -1;
    bool to_stdout = (strcmp(filename, "-") == 0);
    FILE *f = NULL;
    if (to_stdout) {
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        f = stdout;
    } else {
        f = fopen(filename, "wb");
        if (!f) return -1;
    }

    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t sr = (uint32_t)sample_rate;
    uint32_t byte_rate = sr * num_channels * (bits_per_sample / 8);
    uint16_t block_align = num_channels * (bits_per_sample / 8);
    uint32_t data_size = (uint32_t)num_samples * sizeof(int16_t);
    uint32_t chunk_size = 36 + data_size;
    uint32_t subchunk1_size = 16;
    uint16_t audio_format = 1; // PCM

    fwrite("RIFF", 1, 4, f);
    fwrite(&chunk_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    fwrite(&subchunk1_size, 4, 1, f);
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);

    // Buffered batch conversion and writing (64 KB chunks = 32768 samples)
    enum { PCM_BUF_SAMPLES = 32768 };
    int16_t *pcm_buf = (int16_t*)malloc(PCM_BUF_SAMPLES * sizeof(int16_t));
    if (!pcm_buf) {
        if (!to_stdout) fclose(f);
        return -1;
    }

    int idx = 0;
    while (idx < num_samples) {
        int batch = num_samples - idx;
        if (batch > PCM_BUF_SAMPLES) batch = PCM_BUF_SAMPLES;
        for (int i = 0; i < batch; i++) {
            float s = samples[idx + i];
            if (s > 1.0f) s = 1.0f;
            else if (s < -1.0f) s = -1.0f;
            pcm_buf[i] = (int16_t)lrintf(s * 32767.0f);
        }
        if (fwrite(pcm_buf, sizeof(int16_t), (size_t)batch, f) != (size_t)batch) {
            free(pcm_buf);
            if (!to_stdout) fclose(f);
            return -1;
        }
        idx += batch;
    }

    free(pcm_buf);
    fflush(f);
    if (!to_stdout) fclose(f);
    return 0;
}

static void play_audio_file(const char *filepath) {
    if (!filepath || strcmp(filepath, "-") == 0) return;
    fprintf(stderr, "[*] Playing generated audio (%s)...\n", filepath);
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "(pw-play \"%s\" || paplay \"%s\" || aplay -q \"%s\" || ffplay -nodisp -autoexit -loglevel quiet \"%s\") >/dev/null 2>&1",
        filepath, filepath, filepath, filepath);
    if (system(cmd) != 0) {
        fprintf(stderr, "[-] Warning: Could not find a working audio player (tried pw-play, paplay, aplay, ffplay).\n");
    }
}

static char* prepare_reference_audio(const char *audio_path, int *is_temp) {
    *is_temp = 0;
    if (!audio_path) return NULL;

    // Check if the file is already a valid RIFF WAVE file
    FILE *f = fopen(audio_path, "rb");
    if (f) {
        char header[12];
        size_t n = fread(header, 1, 12, f);
        fclose(f);
        if (n == 12 && memcmp(header, "RIFF", 4) == 0 && memcmp(header + 8, "WAVE", 4) == 0) {
            return strdup(audio_path);
        }
    }

    // Not a standard RIFF WAVE file (e.g. MP3, FLAC, OGG, M4A) — auto-convert via ffmpeg
    fprintf(stderr, "[*] Reference audio '%s' is not a raw WAV file; auto-converting to 24kHz mono WAV via ffmpeg...\n", audio_path);
    char temp_wav[4096];
#ifdef _WIN32
    snprintf(temp_wav, sizeof(temp_wav), "voicecli_ref_temp_%d.wav", (int)getpid());
#else
    snprintf(temp_wav, sizeof(temp_wav), "/tmp/voicecli_ref_%d.wav", (int)getpid());
#endif
    char cmd[8192];
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -v error -i \"%s\" -ar 24000 -ac 1 -c:a pcm_s16le \"%s\"",
        audio_path, temp_wav);
    if (system(cmd) == 0) {
        *is_temp = 1;
        return strdup(temp_wav);
    }

    fprintf(stderr, "[-] Warning: ffmpeg conversion failed; attempting direct load of '%s'.\n", audio_path);
    return strdup(audio_path);
}

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
static float sola_dot_product_avx2(const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int j = 0;
    for (; j + 15 < n; j += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + j),     _mm256_loadu_ps(b + j),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + j + 8), _mm256_loadu_ps(b + j + 8), acc1);
    }
    for (; j + 7 < n; j += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j), acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sums = _mm_add_ps(sum4, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float dot = _mm_cvtss_f32(sums);
    for (; j < n; j++) {
        dot += a[j] * b[j];
    }
    return dot;
}
#endif

static float sola_dot_product_base(const float* a, const float* b, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    int j = 0;
    for (; j + 7 < n; j += 8) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a + j),     _mm_loadu_ps(b + j)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(a + j + 4), _mm_loadu_ps(b + j + 4)));
    }
    acc0 = _mm_add_ps(acc0, acc1);
    __m128 shuf = _mm_shuffle_ps(acc0, acc0, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(acc0, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float dot = _mm_cvtss_f32(sums);
    for (; j < n; j++) {
        dot += a[j] * b[j];
    }
    return dot;
#elif defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int j = 0;
    for (; j + 7 < n; j += 8) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + j),     vld1q_f32(b + j));
        acc1 = vmlaq_f32(acc1, vld1q_f32(a + j + 4), vld1q_f32(b + j + 4));
    }
    float dot = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; j < n; j++) {
        dot += a[j] * b[j];
    }
    return dot;
#else
    float dot = 0.0f;
    for (int j = 0; j < n; j++) {
        dot += a[j] * b[j];
    }
    return dot;
#endif
}

static float* apply_sola_timestretch(const float* in_samples, int in_len, int sample_rate, float speed, int* out_len) {
    if (!in_samples || in_len <= 0) {
        *out_len = 0;
        return NULL;
    }
    if (fabsf(speed - 1.0f) < 0.01f) {
        float* copy = (float*)malloc((size_t)in_len * sizeof(float));
        if (!copy) {
            *out_len = 0;
            return NULL;
        }
        memcpy(copy, in_samples, (size_t)in_len * sizeof(float));
        *out_len = in_len;
        return copy;
    }

    int W = (int)(0.025f * sample_rate); // 25ms = 600 samples at 24kHz
    if (W < 64) W = 64;
    int L = W / 2;                        // 50% overlap = 300 samples
    int S_s = W - L;                      // synthesis step = 300 samples
    int S_a = (int)roundf(S_s * speed);   // analysis step
    if (S_a < 1) S_a = 1;
    int delta = (int)(0.005f * sample_rate); // 5ms search window = 120 samples
    if (delta < 8) delta = 8;

    if (in_len <= W + delta) {
        float* copy = (float*)malloc((size_t)in_len * sizeof(float));
        if (!copy) { *out_len = 0; return NULL; }
        memcpy(copy, in_samples, (size_t)in_len * sizeof(float));
        *out_len = in_len;
        return copy;
    }

    // Precompute O(n) prefix sum of squares for O(1) candidate window energy lookup
    double* prefix_sq = (double*)malloc(((size_t)in_len + 1) * sizeof(double));
    if (!prefix_sq) {
        *out_len = 0;
        return NULL;
    }
    prefix_sq[0] = 0.0;
    for (int i = 0; i < in_len; i++) {
        double v = (double)in_samples[i];
        prefix_sq[i + 1] = prefix_sq[i] + v * v;
    }

    // Precompute raised-cosine (Hann) crossfade weights for smooth phase transitions
    float* fade_in_table = (float*)malloc((size_t)L * sizeof(float));
    if (!fade_in_table) {
        free(prefix_sq);
        *out_len = 0;
        return NULL;
    }
    for (int j = 0; j < L; j++) {
        fade_in_table[j] = 0.5f * (1.0f - cosf((float)M_PI * (float)j / (float)L));
    }

    size_t est_out = (size_t)((float)in_len / speed) + (size_t)W * 4;
    float* y = (float*)calloc(est_out, sizeof(float));
    if (!y) {
        free(fade_in_table);
        free(prefix_sq);
        *out_len = 0;
        return NULL;
    }

    float (*dot_fn)(const float*, const float*, int) = sola_dot_product_base;
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    if (voicecli_get_active_simd_tier() >= 2) {
        dot_fn = sola_dot_product_avx2;
    }
#endif

    memcpy(y, in_samples, (size_t)W * sizeof(float));
    int out_pos = S_s;
    int in_pos = S_a;

    while (in_pos + W + delta < in_len && (size_t)(out_pos + W) < est_out) {
        int best_k = 0;
        float best_corr = -1e30f;
        const float* y_ptr = y + out_pos;

        // Coarse search (step = 2) followed by fine refinement around best match
        for (int k = -delta; k <= delta; k += 2) {
            int actual_idx = in_pos + k;
            if (actual_idx < 0 || actual_idx + L > in_len) continue;

            float dot = dot_fn(y_ptr, in_samples + actual_idx, L);
            float norm_cand = (float)(prefix_sq[actual_idx + L] - prefix_sq[actual_idx]);
            float score = dot / (sqrtf(fmaxf(0.0f, norm_cand)) + 1e-6f);
            if (score > best_corr) {
                best_corr = score;
                best_k = k;
            }
        }

        // Fine search around best_k (k - 1 and k + 1)
        for (int dk = -1; dk <= 1; dk += 2) {
            int k = best_k + dk;
            if (k < -delta || k > delta) continue;
            int actual_idx = in_pos + k;
            if (actual_idx < 0 || actual_idx + L > in_len) continue;

            float dot = dot_fn(y_ptr, in_samples + actual_idx, L);
            float norm_cand = (float)(prefix_sq[actual_idx + L] - prefix_sq[actual_idx]);
            float score = dot / (sqrtf(fmaxf(0.0f, norm_cand)) + 1e-6f);
            if (score > best_corr) {
                best_corr = score;
                best_k = k;
            }
        }

        int actual_in = in_pos + best_k;
        if (actual_in < 0) actual_in = 0;
        if (actual_in + W > in_len) actual_in = in_len - W;

        const float* x_best = in_samples + actual_in;
        for (int j = 0; j < L; j++) {
            float fade = fade_in_table[j];
            y[out_pos + j] = (1.0f - fade) * y[out_pos + j] + fade * x_best[j];
        }
        memcpy(y + out_pos + L, x_best + L, (size_t)(W - L) * sizeof(float));

        out_pos += S_s;
        in_pos += S_a;
    }

    free(fade_in_table);
    free(prefix_sq);

    int final_len = out_pos + L;
    if (final_len > (int)est_out) final_len = (int)est_out;
    *out_len = final_len;
    return y;
}

static void configure_runtime_threads(int32_t requested_threads) {
    if (requested_threads > 0) {
        qwen3_tts_set_cpu_threads(requested_threads);
    } else {
        int32_t cfg_threads = get_setting_int("threads", 0);
        if (cfg_threads > 0) {
            qwen3_tts_set_cpu_threads(cfg_threads);
        }
    }
}

static qwen3_tts_params_t build_tts_params(const voicecli_options_t* opts, const char* instruction, const char* speaker) {
    qwen3_tts_params_t params;
    memset(&params, 0, sizeof(params));

    params.max_audio_tokens = (opts && opts->max_tokens > 0)
        ? opts->max_tokens
        : get_setting_int("max_tokens", 4096);
    params.temperature = (opts && opts->temperature >= 0.0f)
        ? opts->temperature
        : get_setting_float("temperature", 0.9f);
    params.top_p = (opts && opts->top_p > 0.0f)
        ? opts->top_p
        : get_setting_float("top_p", 1.0f);
    params.top_k = (opts && opts->top_k >= 0)
        ? opts->top_k
        : get_setting_int("top_k", 50);
    params.repetition_penalty = (opts && opts->repetition_penalty > 0.0f)
        ? opts->repetition_penalty
        : get_setting_float("repetition_penalty", 1.05f);
    params.n_threads = (opts && opts->threads > 0) ? opts->threads : qwen3_tts_get_cpu_threads();
    params.print_progress = 0;
    params.print_timing = 0;
    params.language_id = resolve_language_id(opts ? opts->language : NULL);
    params.instruction = (instruction && strlen(instruction) > 0) ? instruction : NULL;
    params.speaker = (speaker && strlen(speaker) > 0) ? speaker : NULL;
    params.vocoder_left_context_sec = 0.0f;
    return params;
}

static void build_style_prompt(const char* prompt, float speed, char* final_prompt, size_t max_len) {
    if (prompt && strlen(prompt) > 0) {
        strncpy(final_prompt, prompt, max_len - 1);
        final_prompt[max_len - 1] = '\0';
    } else {
        final_prompt[0] = '\0';
    }

    if (strstr(final_prompt, "speed") == NULL && strstr(final_prompt, "Speed") == NULL) {
        if (speed >= 1.25f && strlen(final_prompt) + 22 < max_len) {
            if (strlen(final_prompt) > 0) strcat(final_prompt, ". ");
            strcat(final_prompt, "speed: Fast-paced.");
        } else if (speed <= 0.85f && strlen(final_prompt) + 22 < max_len) {
            if (strlen(final_prompt) > 0) strcat(final_prompt, ". ");
            strcat(final_prompt, "speed: Slow-paced.");
        }
    }
}

static float update_adaptive_speech_rate(float current_scale, const tts_progress_t* prog) {
    int actual = prog->actual_total_frames > 0 ? prog->actual_total_frames : prog->last_tokens;
    if (actual >= 10 && prog->raw_estimated_frames >= 10) {
        float observed = (float)actual / (float)prog->raw_estimated_frames;
        if (observed < 0.45f) observed = 0.45f;
        if (observed > 2.20f) observed = 2.20f;
        return 0.35f * current_scale + 0.65f * observed;
    }
    return current_scale;
}

static void append_chunk_audio(
    float** full_audio,
    size_t* total_samples,
    size_t* audio_capacity,
    const float* chunk_audio,
    int chunk_len,
    int sample_rate,
    int is_last_chunk
) {
    if (!chunk_audio || chunk_len <= 0) return;

    // 1. Trim leading and trailing vocoder silence / low-level room noise
    float threshold = 0.003f;
    int margin_start = (int)(sample_rate * 0.015f); // 15ms onset margin
    int margin_end   = (int)(sample_rate * 0.025f); // 25ms decay margin

    int start = 0;
    while (start < chunk_len && fabsf(chunk_audio[start]) < threshold) {
        start++;
    }
    start = (start > margin_start) ? start - margin_start : 0;

    int end = chunk_len - 1;
    while (end > start && fabsf(chunk_audio[end]) < threshold) {
        end--;
    }
    end = (end + margin_end < chunk_len) ? end + margin_end : chunk_len - 1;

    int clean_len = end - start + 1;
    if (clean_len <= 0) {
        clean_len = chunk_len;
        start = 0;
    }

    // Natural 150ms breathing pause between chunks
    int silence_samples = is_last_chunk ? 0 : (int)(sample_rate * 0.15f);

    size_t needed = *total_samples + (size_t)clean_len + (size_t)silence_samples;
    if (needed > *audio_capacity) {
        size_t new_cap = needed * 2;
        float* new_buf = (float*)realloc(*full_audio, new_cap * sizeof(float));
        if (!new_buf) {
            fprintf(stderr, "[-] Error: Failed to reallocate audio buffer\n");
            return;
        }
        *full_audio = new_buf;
        *audio_capacity = new_cap;
    }

    // Copy directly into destination buffer and apply Hann fades in-place (zero intermediate malloc)
    float* dst = *full_audio + *total_samples;
    memcpy(dst, chunk_audio + start, (size_t)clean_len * sizeof(float));

    int fade_in = (int)(sample_rate * 0.015f);
    if (fade_in > clean_len / 2) fade_in = clean_len / 2;
    for (int j = 0; j < fade_in; j++) {
        float factor = 0.5f * (1.0f - cosf((float)M_PI * (float)j / (float)fade_in));
        dst[j] *= factor;
    }

    int fade_out = (int)(sample_rate * 0.020f);
    if (fade_out > clean_len / 2) fade_out = clean_len / 2;
    for (int j = 0; j < fade_out; j++) {
        float factor = 0.5f * (1.0f + cosf((float)M_PI * (float)j / (float)fade_out));
        dst[clean_len - fade_out + j] *= factor;
    }

    *total_samples += (size_t)clean_len;

    if (silence_samples > 0) {
        memset(*full_audio + *total_samples, 0, (size_t)silence_samples * sizeof(float));
        *total_samples += (size_t)silence_samples;
    }
}

typedef struct {
    int chunks_emitted;
    int total_samples;
} stream_ctx_t;

static int32_t on_audio_stream_chunk(const qwen3_tts_audio_chunk_t* chunk, void* user_data) {
    stream_ctx_t* sc = (stream_ctx_t*)user_data;
    if (sc && chunk) {
        sc->chunks_emitted++;
        sc->total_samples += chunk->n_samples;
    }
    return 1; // continue streaming
}

static int finalize_and_save_audio(
    float* audio_data,
    int audio_len,
    int sample_rate,
    int64_t t_total_ms,
    int is_tty,
    const voicecli_options_t* opts
) {
    float speed = opts ? opts->speed : 1.0f;
    const char* output_path = (opts && opts->output_path) ? opts->output_path : "output.wav";
    float* stretched = NULL;
    float* working_audio = audio_data;
    int final_samples = audio_len;

    if (fabsf(speed - 1.0f) >= 0.01f) {
        int stretched_len = 0;
        stretched = apply_sola_timestretch(audio_data, audio_len, sample_rate, speed, &stretched_len);
        if (stretched && stretched_len > 0) {
            working_audio = stretched;
            final_samples = stretched_len;
        }
    }

    if (opts && opts->normalize) {
        apply_peak_normalization(working_audio, (size_t)final_samples);
    }

    double audio_sec = sample_rate > 0 ? (double)final_samples / (double)sample_rate : 0.0;
    double wall_sec = (double)t_total_ms / 1000.0;
    double rtf = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;
    if (is_tty) {
        fprintf(stderr, "\r\033[K[+] Synthesis complete: %.2fs audio generated in %.2fs (%.2fx real-time)\n",
                audio_sec, wall_sec, rtf);
    } else {
        fprintf(stderr, "[+] Synthesis complete: %.2fs audio generated in %.2fs (%.2fx real-time)\n",
                audio_sec, wall_sec, rtf);
    }

    fprintf(stderr, "[*] Saving audio to %s (Sample Rate: %d Hz)...\n",
            strcmp(output_path, "-") == 0 ? "stdout" : output_path, sample_rate);
    int rc = 0;
    if (save_wav_file(output_path, working_audio, final_samples, sample_rate) == 0) {
        fprintf(stderr, "[+] Audio generation complete!\n");
        if (opts && opts->play && strcmp(output_path, "-") != 0) {
            play_audio_file(output_path);
        }
    } else {
        fprintf(stderr, "[-] Error: Failed to write audio file %s\n", output_path);
        rc = 1;
    }

    if (stretched) free(stretched);
    return rc;
}

int c_generate_style_ex(const char* text, const char* prompt, const voicecli_options_t* opts) {
    configure_runtime_threads(opts ? opts->threads : 0);

    char model_name[256];
    resolve_model_preset("style", opts ? opts->model_override : NULL, model_name, sizeof(model_name));

    fprintf(stderr, "[*] Loading model: %s ...\n", model_name);
    fprintf(stderr, "[*] Target device: CPU (In-Process Native C++ / GGML SIMD: %s)\n", voicecli_get_active_simd_name());

    const char* model_dir = resolve_model_dir();
    char tok_name[256];
    get_setting_string("tokenizer_model", tok_name, sizeof(tok_name), "qwen-tokenizer-12hz-BF16.gguf");

    if (ensure_file_exists(model_dir, tok_name) != 0 || ensure_file_exists(model_dir, model_name) != 0) {
        return 1;
    }

    qwen3_tts_context_t* ctx = qwen3_tts_init();
    if (!ctx) {
        fprintf(stderr, "[-] Error: Failed to initialize native Qwen3-TTS context\n");
        return 1;
    }

    fprintf(stderr, "[*] Loading model weights into memory...\n");
    if (!qwen3_tts_load_models_with_name(ctx, model_dir, model_name)) {
        fprintf(stderr, "[-] Error: Failed to load models '%s' from '%s'\n", model_name, model_dir);
        qwen3_tts_free(ctx);
        return 1;
    }
    fprintf(stderr, "[+] Model loaded successfully.\n");

    float speed = opts ? opts->speed : 1.0f;
    char final_prompt[1024];
    build_style_prompt(prompt, speed, final_prompt, sizeof(final_prompt));

    fprintf(stderr, "[*] Generating speech for text: \"%s\"\n", text);
    if (strlen(final_prompt) > 0) {
        fprintf(stderr, "[*] Acoustic style description: \"%s\"\n", final_prompt);
    }
    if (fabsf(speed - 1.0f) >= 0.01f) {
        fprintf(stderr, "[*] Speech speed multiplier: %.2fx\n", speed);
    }

    tts_progress_t prog;
    init_tts_progress(&prog, text, 1.0f, (opts && opts->stream) ? 1 : 0, model_name);
    qwen3_tts_set_phase_progress_callback(ctx, on_phase_progress, &prog);

    qwen3_tts_params_t params = build_tts_params(opts, final_prompt, NULL);
    qwen3_tts_result_t result;

    if (opts && opts->stream) {
        qwen3_tts_streaming_params_t sparams;
        memset(&sparams, 0, sizeof(sparams));
        sparams.generation = params;
        sparams.chunk_sec = opts->stream_chunk_sec > 0.0f ? opts->stream_chunk_sec : 1.0f;
        sparams.left_context_sec = 2.0f;
        sparams.collect_audio = 1;
        stream_ctx_t sctx = {0, 0};
        result = qwen3_tts_synthesize_streaming(ctx, text, sparams, on_audio_stream_chunk, &sctx);
    } else {
        result = qwen3_tts_synthesize(ctx, text, params);
    }

    int rc = 0;
    if (!result.success || result.audio_len <= 0) {
        if (prog.is_tty) fprintf(stderr, "\r\033[K");
        fprintf(stderr, "[-] Native synthesis failed: %s\n", result.error_msg ? result.error_msg : "unknown error");
        rc = 1;
    } else {
        rc = finalize_and_save_audio(result.audio, result.audio_len, result.sample_rate, result.t_total_ms, prog.is_tty, opts);
    }

    qwen3_tts_free_result(result);
    qwen3_tts_free(ctx);
    return rc;
}

static void format_chunk_preview(const char* text, char* preview, size_t preview_size) {
    if (!text || preview_size < 5) {
        if (preview_size > 0) preview[0] = '\0';
        return;
    }
    size_t max_bytes = preview_size - 4;
    size_t len = strlen(text);
    if (len <= max_bytes) {
        memcpy(preview, text, len);
        preview[len] = '\0';
    } else {
        size_t cut = max_bytes;
        // Backtrack if we landed inside a UTF-8 continuation byte (10xxxxxx)
        while (cut > 0 && ((unsigned char)text[cut] & 0xC0) == 0x80) {
            cut--;
        }
        memcpy(preview, text, cut);
        memcpy(preview + cut, "...", 4);
    }
    for (size_t c = 0; preview[c]; c++) {
        if (preview[c] == '\n' || preview[c] == '\r' || preview[c] == '\t') {
            preview[c] = ' ';
        }
    }
}

int c_generate_style_chunked_ex(const char** chunks, int num_chunks, const char* prompt, const voicecli_options_t* opts) {
    if (!chunks || num_chunks <= 0) return 1;
    configure_runtime_threads(opts ? opts->threads : 0);

    char model_name[256];
    resolve_model_preset("style", opts ? opts->model_override : NULL, model_name, sizeof(model_name));

    fprintf(stderr, "[*] [Auto-Chunk] Loading model: %s ...\n", model_name);
    fprintf(stderr, "[*] Target device: CPU (In-Process Native C++ / GGML SIMD: %s)\n", voicecli_get_active_simd_name());

    const char* model_dir = resolve_model_dir();
    char tok_name[256];
    get_setting_string("tokenizer_model", tok_name, sizeof(tok_name), "qwen-tokenizer-12hz-BF16.gguf");

    if (ensure_file_exists(model_dir, tok_name) != 0 || ensure_file_exists(model_dir, model_name) != 0) {
        return 1;
    }

    qwen3_tts_context_t* ctx = qwen3_tts_init();
    if (!ctx) {
        fprintf(stderr, "[-] Error: Failed to initialize native Qwen3-TTS context\n");
        return 1;
    }

    fprintf(stderr, "[*] Loading model weights into memory...\n");
    if (!qwen3_tts_load_models_with_name(ctx, model_dir, model_name)) {
        fprintf(stderr, "[-] Error: Failed to load models '%s' from '%s'\n", model_name, model_dir);
        qwen3_tts_free(ctx);
        return 1;
    }
    fprintf(stderr, "[+] Model loaded successfully. Processing %d chunks in memory...\n", num_chunks);

    float speed = opts ? opts->speed : 1.0f;
    char final_prompt[1024];
    build_style_prompt(prompt, speed, final_prompt, sizeof(final_prompt));

    if (fabsf(speed - 1.0f) >= 0.01f) {
        fprintf(stderr, "[*] Speech speed multiplier: %.2fx\n", speed);
    }

    size_t total_samples = 0;
    size_t audio_capacity = 24000 * 60;
    float* full_audio = (float*)malloc(audio_capacity * sizeof(float));
    if (!full_audio) {
        fprintf(stderr, "[-] Error: Failed to allocate audio buffer\n");
        qwen3_tts_free(ctx);
        return 1;
    }
    int sample_rate = 24000;
    float speech_rate_scale = 1.0f;
    qwen3_tts_params_t params = build_tts_params(opts, final_prompt, NULL);

    for (int i = 0; i < num_chunks; i++) {
        char preview[64];
        format_chunk_preview(chunks[i], preview, sizeof(preview));

        fprintf(stderr, "\n[*] Chunk [%d/%d] (%zu chars): \"%s\"\n", i + 1, num_chunks, strlen(chunks[i]), preview);

        tts_progress_t prog;
        init_tts_progress(&prog, chunks[i], speech_rate_scale, 0, model_name);
        qwen3_tts_set_phase_progress_callback(ctx, on_phase_progress, &prog);

        qwen3_tts_result_t result = qwen3_tts_synthesize(ctx, chunks[i], params);
        if (!result.success || result.audio_len <= 0) {
            if (prog.is_tty) fprintf(stderr, "\r\033[K");
            fprintf(stderr, "[-] Warning: Chunk %d synthesis failed: %s\n", i + 1,
                    result.error_msg ? result.error_msg : "unknown error");
            qwen3_tts_free_result(result);
            continue;
        }

        speech_rate_scale = update_adaptive_speech_rate(speech_rate_scale, &prog);
        sample_rate = result.sample_rate;
        double chunk_sec = (double)result.audio_len / (double)result.sample_rate;
        double wall_sec = (double)result.t_total_ms / 1000.0;
        double rtf = wall_sec > 0.0 ? chunk_sec / wall_sec : 0.0;
        if (prog.is_tty) {
            fprintf(stderr, "\r\033[K[+] Chunk [%d/%d] synthesized: %.2fs audio (%.2fx real-time)\n",
                    i + 1, num_chunks, chunk_sec, rtf);
        } else {
            fprintf(stderr, "[+] Chunk [%d/%d] synthesized: %.2fs audio (%.2fx real-time)\n",
                    i + 1, num_chunks, chunk_sec, rtf);
        }

        append_chunk_audio(
            &full_audio,
            &total_samples,
            &audio_capacity,
            result.audio,
            result.audio_len,
            sample_rate,
            i == num_chunks - 1
        );

        qwen3_tts_free_result(result);
    }

    int rc = 0;
    const char* output_path = (opts && opts->output_path) ? opts->output_path : "output.wav";
    if (total_samples > 0) {
        if (fabsf(speed - 1.0f) >= 0.01f) {
            int stretched_len = 0;
            float* stretched = apply_sola_timestretch(full_audio, (int)total_samples, sample_rate, speed, &stretched_len);
            if (stretched && stretched_len > 0) {
                double orig_sec = (double)total_samples / (double)sample_rate;
                double new_sec = (double)stretched_len / (double)sample_rate;
                fprintf(stderr, "[*] Applied speed multiplier %.2fx: duration %.2fs -> %.2fs\n", speed, orig_sec, new_sec);
                free(full_audio);
                full_audio = stretched;
                total_samples = (size_t)stretched_len;
            }
        }

        if (opts && opts->normalize) {
            apply_peak_normalization(full_audio, total_samples);
        }

        double total_audio_sec = (double)total_samples / (double)sample_rate;
        fprintf(stderr, "\n============================================================\n");
        fprintf(stderr, "[+] All %d chunks synthesized successfully!\n", num_chunks);
        fprintf(stderr, "[+] Total stitched audio duration: %.2fs (~%.1f min)\n",
               total_audio_sec, total_audio_sec / 60.0);
        fprintf(stderr, "[*] Saving combined audio to %s (Sample Rate: %d Hz)...\n",
                strcmp(output_path, "-") == 0 ? "stdout" : output_path, sample_rate);
        if (save_wav_file(output_path, full_audio, (int)total_samples, sample_rate) == 0) {
            fprintf(stderr, "[+] Audio generation complete!\n");
            if (opts && opts->play && strcmp(output_path, "-") != 0) {
                play_audio_file(output_path);
            }
        } else {
            fprintf(stderr, "[-] Error: Failed to write audio file %s\n", output_path);
            rc = 1;
        }
        fprintf(stderr, "============================================================\n");
    } else {
        fprintf(stderr, "[-] Error: No audio samples were generated.\n");
        rc = 1;
    }

    free(full_audio);
    qwen3_tts_free(ctx);
    return rc;
}

static qwen3_tts_result_t execute_clone_synthesis(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* effective_audio_path,
    const char* inline_ref_text,
    const char* active_profile_path,
    int is_icl_profile,
    qwen3_tts_params_t params,
    int use_stream,
    float stream_chunk_sec
) {
    if (use_stream) {
        qwen3_tts_streaming_params_t sparams;
        memset(&sparams, 0, sizeof(sparams));
        sparams.generation = params;
        sparams.chunk_sec = stream_chunk_sec > 0.0f ? stream_chunk_sec : 1.0f;
        sparams.left_context_sec = 2.0f;
        sparams.collect_audio = 1;
        stream_ctx_t sctx = {0, 0};

        if (active_profile_path && strlen(active_profile_path) > 0) {
            if (is_icl_profile) {
                return qwen3_tts_synthesize_with_icl_prompt_streaming(ctx, text, active_profile_path, sparams, on_audio_stream_chunk, &sctx);
            } else {
                return qwen3_tts_synthesize_with_speaker_embedding_streaming(ctx, text, active_profile_path, sparams, on_audio_stream_chunk, &sctx);
            }
        }
        if (inline_ref_text && strlen(inline_ref_text) > 0) {
            return qwen3_tts_synthesize_with_voice_icl_streaming(ctx, text, effective_audio_path, inline_ref_text, sparams, on_audio_stream_chunk, &sctx);
        }
        return qwen3_tts_synthesize_with_voice_streaming(ctx, text, effective_audio_path, sparams, on_audio_stream_chunk, &sctx);
    } else {
        if (active_profile_path && strlen(active_profile_path) > 0) {
            if (is_icl_profile) {
                return qwen3_tts_synthesize_with_icl_prompt(ctx, text, active_profile_path, params);
            } else {
                return qwen3_tts_synthesize_with_speaker_embedding(ctx, text, active_profile_path, params);
            }
        }
        if (inline_ref_text && strlen(inline_ref_text) > 0) {
            return qwen3_tts_synthesize_with_voice_icl(ctx, text, effective_audio_path, inline_ref_text, params);
        }
        return qwen3_tts_synthesize_with_voice(ctx, text, effective_audio_path, params);
    }
}

static int detect_icl_profile_file(const char* filepath) {
    if (!filepath) return 0;
    FILE* f = fopen(filepath, "rb");
    if (!f) return 0;
    char buf[1024];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    return (strstr(buf, "\"reference_codes\"") != NULL || strstr(buf, "\"reference_text\"") != NULL) ? 1 : 0;
}

int c_generate_clone_ex(const char* text, const char* audio_path, const voicecli_options_t* opts) {
    configure_runtime_threads(opts ? opts->threads : 0);

    char model_name[256];
    resolve_model_preset("clone", opts ? opts->model_override : NULL, model_name, sizeof(model_name));

    fprintf(stderr, "[*] Loading model: %s ...\n", model_name);
    fprintf(stderr, "[*] Target device: CPU (In-Process Native C++ / GGML SIMD: %s)\n", voicecli_get_active_simd_name());

    const char* model_dir = resolve_model_dir();
    char tok_name[256];
    get_setting_string("tokenizer_model", tok_name, sizeof(tok_name), "qwen-tokenizer-12hz-BF16.gguf");

    if (ensure_file_exists(model_dir, tok_name) != 0 || ensure_file_exists(model_dir, model_name) != 0) {
        return 1;
    }

    qwen3_tts_context_t* ctx = qwen3_tts_init();
    if (!ctx) {
        fprintf(stderr, "[-] Error: Failed to initialize native Qwen3-TTS context\n");
        return 1;
    }

    fprintf(stderr, "[*] Loading model weights into memory...\n");
    if (!qwen3_tts_load_models_with_name(ctx, model_dir, model_name)) {
        fprintf(stderr, "[-] Error: Failed to load models '%s' from '%s'\n", model_name, model_dir);
        qwen3_tts_free(ctx);
        return 1;
    }
    fprintf(stderr, "[+] Model loaded successfully.\n");

    int is_temp_wav = 0;
    char* prepared_audio = NULL;
    const char* active_profile = (opts && opts->voice_profile && strlen(opts->voice_profile) > 0) ? opts->voice_profile : NULL;
    const char* inline_ref_text = NULL;
    int is_icl_profile = 0;

    if (active_profile) {
        is_icl_profile = detect_icl_profile_file(active_profile);
        fprintf(stderr, "[*] Using precomputed %s profile: %s\n",
                is_icl_profile ? "ICL voice" : "speaker embedding", active_profile);
    } else if (audio_path && strlen(audio_path) > 0) {
        prepared_audio = prepare_reference_audio(audio_path, &is_temp_wav);
        fprintf(stderr, "[*] Cloning voice from reference audio: %s\n", audio_path);

        if (opts && opts->ref_text && strlen(opts->ref_text) > 0) {
            if (opts->save_profile && strlen(opts->save_profile) > 0) {
                fprintf(stderr, "[*] Extracting and saving full ICL voice profile to: %s\n", opts->save_profile);
                if (qwen3_tts_extract_icl_prompt(ctx, prepared_audio, opts->ref_text, opts->save_profile)) {
                    active_profile = opts->save_profile;
                    is_icl_profile = 1;
                    fprintf(stderr, "[+] Saved reusable ICL voice profile to: %s\n", opts->save_profile);
                } else {
                    fprintf(stderr, "[-] Warning: ICL profile save failed; using in-memory ICL prompt.\n");
                    inline_ref_text = opts->ref_text;
                }
            } else {
                // Zero-disk-I/O in-memory Full-ICL extraction via Qwen3TTS voice_prompt_cache_
                inline_ref_text = opts->ref_text;
            }
        } else if (opts && opts->save_profile && strlen(opts->save_profile) > 0) {
            fprintf(stderr, "[*] Extracting and saving speaker embedding profile to: %s\n", opts->save_profile);
            if (qwen3_tts_extract_speaker_embedding(ctx, prepared_audio, opts->save_profile)) {
                active_profile = opts->save_profile;
                is_icl_profile = 0;
                fprintf(stderr, "[+] Speaker embedding profile saved to: %s\n", opts->save_profile);
            }
        }
    }

    float speed = opts ? opts->speed : 1.0f;
    fprintf(stderr, "[*] Generating speech for text: \"%s\"\n", text);
    if (fabsf(speed - 1.0f) >= 0.01f) {
        fprintf(stderr, "[*] Speech speed multiplier: %.2fx\n", speed);
    }

    int use_stream = (opts && opts->stream) ? 1 : 0;
    float stream_chunk_sec = (opts && opts->stream_chunk_sec > 0.0f) ? opts->stream_chunk_sec : 1.0f;

    tts_progress_t prog;
    init_tts_progress(&prog, text, 1.0f, use_stream, model_name);
    qwen3_tts_set_phase_progress_callback(ctx, on_phase_progress, &prog);

    qwen3_tts_params_t params = build_tts_params(opts, NULL, NULL);
    qwen3_tts_result_t result = execute_clone_synthesis(
        ctx, text, prepared_audio, inline_ref_text, active_profile, is_icl_profile,
        params, use_stream, stream_chunk_sec
    );

    int rc = 0;
    if (!result.success || result.audio_len <= 0) {
        if (prog.is_tty) fprintf(stderr, "\r\033[K");
        fprintf(stderr, "[-] Native voice cloning failed: %s\n", result.error_msg ? result.error_msg : "unknown error");
        rc = 1;
    } else {
        rc = finalize_and_save_audio(result.audio, result.audio_len, result.sample_rate, result.t_total_ms, prog.is_tty, opts);
    }

    if (is_temp_wav && prepared_audio) remove(prepared_audio);
    if (prepared_audio) free(prepared_audio);
    qwen3_tts_free_result(result);
    qwen3_tts_free(ctx);
    return rc;
}

int c_generate_clone_chunked_ex(const char** chunks, int num_chunks, const char* audio_path, const voicecli_options_t* opts) {
    if (!chunks || num_chunks <= 0) return 1;
    configure_runtime_threads(opts ? opts->threads : 0);

    char model_name[256];
    resolve_model_preset("clone", opts ? opts->model_override : NULL, model_name, sizeof(model_name));

    fprintf(stderr, "[*] [Auto-Chunk] Loading model: %s ...\n", model_name);
    fprintf(stderr, "[*] Target device: CPU (In-Process Native C++ / GGML SIMD: %s)\n", voicecli_get_active_simd_name());

    const char* model_dir = resolve_model_dir();
    char tok_name[256];
    get_setting_string("tokenizer_model", tok_name, sizeof(tok_name), "qwen-tokenizer-12hz-BF16.gguf");

    if (ensure_file_exists(model_dir, tok_name) != 0 || ensure_file_exists(model_dir, model_name) != 0) {
        return 1;
    }

    qwen3_tts_context_t* ctx = qwen3_tts_init();
    if (!ctx) {
        fprintf(stderr, "[-] Error: Failed to initialize native Qwen3-TTS context\n");
        return 1;
    }

    fprintf(stderr, "[*] Loading model weights into memory...\n");
    if (!qwen3_tts_load_models_with_name(ctx, model_dir, model_name)) {
        fprintf(stderr, "[-] Error: Failed to load models '%s' from '%s'\n", model_name, model_dir);
        qwen3_tts_free(ctx);
        return 1;
    }
    fprintf(stderr, "[+] Model loaded successfully. Processing %d chunks in memory...\n", num_chunks);

    int is_temp_wav = 0;
    char* prepared_audio = NULL;
    const char* active_profile = (opts && opts->voice_profile && strlen(opts->voice_profile) > 0) ? opts->voice_profile : NULL;
    const char* inline_ref_text = NULL;
    int is_icl_profile = 0;

    if (active_profile) {
        is_icl_profile = detect_icl_profile_file(active_profile);
        fprintf(stderr, "[*] Using precomputed %s profile: %s\n",
                is_icl_profile ? "ICL voice" : "speaker embedding", active_profile);
    } else if (audio_path && strlen(audio_path) > 0) {
        prepared_audio = prepare_reference_audio(audio_path, &is_temp_wav);
        fprintf(stderr, "[*] Cloning voice from reference audio: %s\n", audio_path);

        if (opts && opts->ref_text && strlen(opts->ref_text) > 0) {
            if (opts->save_profile && strlen(opts->save_profile) > 0) {
                fprintf(stderr, "[*] Extracting and saving full ICL voice profile to: %s\n", opts->save_profile);
                if (qwen3_tts_extract_icl_prompt(ctx, prepared_audio, opts->ref_text, opts->save_profile)) {
                    active_profile = opts->save_profile;
                    is_icl_profile = 1;
                    fprintf(stderr, "[+] Saved reusable ICL voice profile to: %s\n", opts->save_profile);
                } else {
                    inline_ref_text = opts->ref_text;
                }
            } else {
                // Cached in RAM across all chunks via Qwen3TTS voice_prompt_cache_
                inline_ref_text = opts->ref_text;
            }
        } else if (opts && opts->save_profile && strlen(opts->save_profile) > 0) {
            fprintf(stderr, "[*] Extracting and saving speaker embedding profile to: %s\n", opts->save_profile);
            if (qwen3_tts_extract_speaker_embedding(ctx, prepared_audio, opts->save_profile)) {
                active_profile = opts->save_profile;
                is_icl_profile = 0;
                fprintf(stderr, "[+] Speaker embedding profile saved to: %s\n", opts->save_profile);
            }
        }
    }

    float speed = opts ? opts->speed : 1.0f;
    if (fabsf(speed - 1.0f) >= 0.01f) {
        fprintf(stderr, "[*] Speech speed multiplier: %.2fx\n", speed);
    }

    size_t total_samples = 0;
    size_t audio_capacity = 24000 * 60;
    float* full_audio = (float*)malloc(audio_capacity * sizeof(float));
    if (!full_audio) {
        fprintf(stderr, "[-] Error: Failed to allocate audio buffer\n");
        if (is_temp_wav && prepared_audio) remove(prepared_audio);
        if (prepared_audio) free(prepared_audio);
        qwen3_tts_free(ctx);
        return 1;
    }
    int sample_rate = 24000;
    float speech_rate_scale = 1.0f;
    qwen3_tts_params_t params = build_tts_params(opts, NULL, NULL);

    for (int i = 0; i < num_chunks; i++) {
        char preview[64];
        format_chunk_preview(chunks[i], preview, sizeof(preview));

        fprintf(stderr, "\n[*] Chunk [%d/%d] (%zu chars): \"%s\"\n", i + 1, num_chunks, strlen(chunks[i]), preview);

        tts_progress_t prog;
        init_tts_progress(&prog, chunks[i], speech_rate_scale, 0, model_name);
        qwen3_tts_set_phase_progress_callback(ctx, on_phase_progress, &prog);

        qwen3_tts_result_t result = execute_clone_synthesis(
            ctx, chunks[i], prepared_audio, inline_ref_text, active_profile, is_icl_profile,
            params, 0, 1.0f
        );

        if (!result.success || result.audio_len <= 0) {
            if (prog.is_tty) fprintf(stderr, "\r\033[K");
            fprintf(stderr, "[-] Warning: Chunk %d voice cloning failed: %s\n", i + 1,
                    result.error_msg ? result.error_msg : "unknown error");
            qwen3_tts_free_result(result);
            continue;
        }

        speech_rate_scale = update_adaptive_speech_rate(speech_rate_scale, &prog);
        sample_rate = result.sample_rate;
        double chunk_sec = (double)result.audio_len / (double)result.sample_rate;
        double wall_sec = (double)result.t_total_ms / 1000.0;
        double rtf = wall_sec > 0.0 ? chunk_sec / wall_sec : 0.0;
        if (prog.is_tty) {
            fprintf(stderr, "\r\033[K[+] Chunk [%d/%d] synthesized: %.2fs audio (%.2fx real-time)\n",
                    i + 1, num_chunks, chunk_sec, rtf);
        } else {
            fprintf(stderr, "[+] Chunk [%d/%d] synthesized: %.2fs audio (%.2fx real-time)\n",
                    i + 1, num_chunks, chunk_sec, rtf);
        }

        append_chunk_audio(
            &full_audio,
            &total_samples,
            &audio_capacity,
            result.audio,
            result.audio_len,
            sample_rate,
            i == num_chunks - 1
        );

        qwen3_tts_free_result(result);
    }

    int rc = 0;
    const char* output_path = (opts && opts->output_path) ? opts->output_path : "clone_output.wav";
    if (total_samples > 0) {
        if (fabsf(speed - 1.0f) >= 0.01f) {
            int stretched_len = 0;
            float* stretched = apply_sola_timestretch(full_audio, (int)total_samples, sample_rate, speed, &stretched_len);
            if (stretched && stretched_len > 0) {
                double orig_sec = (double)total_samples / (double)sample_rate;
                double new_sec = (double)stretched_len / (double)sample_rate;
                fprintf(stderr, "[*] Applied speed multiplier %.2fx: duration %.2fs -> %.2fs\n", speed, orig_sec, new_sec);
                free(full_audio);
                full_audio = stretched;
                total_samples = (size_t)stretched_len;
            }
        }

        if (opts && opts->normalize) {
            apply_peak_normalization(full_audio, total_samples);
        }

        double total_audio_sec = (double)total_samples / (double)sample_rate;
        fprintf(stderr, "\n============================================================\n");
        fprintf(stderr, "[+] All %d chunks synthesized successfully!\n", num_chunks);
        fprintf(stderr, "[+] Total stitched audio duration: %.2fs (~%.1f min)\n",
               total_audio_sec, total_audio_sec / 60.0);
        fprintf(stderr, "[*] Saving combined audio to %s (Sample Rate: %d Hz)...\n",
                strcmp(output_path, "-") == 0 ? "stdout" : output_path, sample_rate);
        if (save_wav_file(output_path, full_audio, (int)total_samples, sample_rate) == 0) {
            fprintf(stderr, "[+] Audio generation complete!\n");
            if (opts && opts->play && strcmp(output_path, "-") != 0) {
                play_audio_file(output_path);
            }
        } else {
            fprintf(stderr, "[-] Error: Failed to write audio file %s\n", output_path);
            rc = 1;
        }
        fprintf(stderr, "============================================================\n");
    } else {
        fprintf(stderr, "[-] Error: No audio samples were generated.\n");
        rc = 1;
    }

    if (is_temp_wav && prepared_audio) remove(prepared_audio);
    if (prepared_audio) free(prepared_audio);
    free(full_audio);
    qwen3_tts_free(ctx);
    return rc;
}

int c_generate_speak_ex(const char* text, const char* speaker, const char* prompt, const voicecli_options_t* opts) {
    configure_runtime_threads(opts ? opts->threads : 0);

    char model_name[256];
    resolve_model_preset("speak", opts ? opts->model_override : NULL, model_name, sizeof(model_name));
    const char* model_dir = resolve_model_dir();

    // Fast path for --list-speakers: inspect GGUF metadata header in <5ms without loading weights or vocoder
    if (!speaker || strlen(speaker) == 0 || strcmp(speaker, "list") == 0) {
        if (ensure_file_exists(model_dir, model_name) != 0) {
            return 1;
        }
        char* speakers = qwen3_tts_list_speakers_from_model(model_dir, model_name);
        if (!speakers) {
            fprintf(stderr, "[-] Error: Failed to inspect model metadata '%s' in '%s'\n", model_name, model_dir);
            return 1;
        }
        printf("Available built-in speakers in '%s':\n", model_name);
        if (speakers[0] != '\0') {
            printf("%s\n", speakers);
        } else {
            printf("  (No named speakers found in model metadata)\n");
        }
        qwen3_tts_free_string(speakers);
        return 0;
    }

    fprintf(stderr, "[*] Loading CustomVoice model: %s ...\n", model_name);
    fprintf(stderr, "[*] Target device: CPU (In-Process Native C++ / GGML SIMD: %s)\n", voicecli_get_active_simd_name());

    char tok_name[256];
    get_setting_string("tokenizer_model", tok_name, sizeof(tok_name), "qwen-tokenizer-12hz-BF16.gguf");

    if (ensure_file_exists(model_dir, tok_name) != 0 || ensure_file_exists(model_dir, model_name) != 0) {
        return 1;
    }

    qwen3_tts_context_t* ctx = qwen3_tts_init();
    if (!ctx) {
        fprintf(stderr, "[-] Error: Failed to initialize native Qwen3-TTS context\n");
        return 1;
    }

    fprintf(stderr, "[*] Loading model weights into memory...\n");
    if (!qwen3_tts_load_models_with_name(ctx, model_dir, model_name)) {
        fprintf(stderr, "[-] Error: Failed to load models '%s' from '%s'\n", model_name, model_dir);
        qwen3_tts_free(ctx);
        return 1;
    }
    fprintf(stderr, "[+] Model loaded successfully.\n");

    float speed = opts ? opts->speed : 1.0f;
    char final_prompt[1024];
    build_style_prompt(prompt, speed, final_prompt, sizeof(final_prompt));

    fprintf(stderr, "[*] Using built-in speaker: \"%s\"\n", speaker);
    fprintf(stderr, "[*] Generating speech for text: \"%s\"\n", text);
    if (strlen(final_prompt) > 0) {
        fprintf(stderr, "[*] Voice steering instruction: \"%s\"\n", final_prompt);
    }

    int use_stream = (opts && opts->stream) ? 1 : 0;
    tts_progress_t prog;
    init_tts_progress(&prog, text, 1.0f, use_stream, model_name);
    qwen3_tts_set_phase_progress_callback(ctx, on_phase_progress, &prog);

    qwen3_tts_params_t params = build_tts_params(opts, final_prompt, speaker);
    qwen3_tts_result_t result;
    if (use_stream) {
        qwen3_tts_streaming_params_t sparams;
        memset(&sparams, 0, sizeof(sparams));
        sparams.generation = params;
        sparams.chunk_sec = (opts && opts->stream_chunk_sec > 0.0f) ? opts->stream_chunk_sec : 1.0f;
        sparams.left_context_sec = 2.0f;
        sparams.collect_audio = 1;
        stream_ctx_t sctx = {0, 0};
        result = qwen3_tts_synthesize_streaming(ctx, text, sparams, on_audio_stream_chunk, &sctx);
    } else {
        result = qwen3_tts_synthesize(ctx, text, params);
    }

    int rc = 0;
    if (!result.success || result.audio_len <= 0) {
        if (prog.is_tty) fprintf(stderr, "\r\033[K");
        fprintf(stderr, "[-] CustomVoice synthesis failed: %s\n", result.error_msg ? result.error_msg : "unknown error");
        rc = 1;
    } else {
        rc = finalize_and_save_audio(result.audio, result.audio_len, result.sample_rate, result.t_total_ms, prog.is_tty, opts);
    }

    qwen3_tts_free_result(result);
    qwen3_tts_free(ctx);
    return rc;
}

int c_generate_speak_chunked_ex(const char** chunks, int num_chunks, const char* speaker, const char* prompt, const voicecli_options_t* opts) {
    if (!chunks || num_chunks <= 0) return 1;
    configure_runtime_threads(opts ? opts->threads : 0);

    char model_name[256];
    resolve_model_preset("speak", opts ? opts->model_override : NULL, model_name, sizeof(model_name));

    fprintf(stderr, "[*] [Auto-Chunk] Loading CustomVoice model: %s ...\n", model_name);
    const char* model_dir = resolve_model_dir();
    char tok_name[256];
    get_setting_string("tokenizer_model", tok_name, sizeof(tok_name), "qwen-tokenizer-12hz-BF16.gguf");

    if (ensure_file_exists(model_dir, tok_name) != 0 || ensure_file_exists(model_dir, model_name) != 0) {
        return 1;
    }

    qwen3_tts_context_t* ctx = qwen3_tts_init();
    if (!ctx) return 1;

    if (!qwen3_tts_load_models_with_name(ctx, model_dir, model_name)) {
        fprintf(stderr, "[-] Error: Failed to load models '%s' from '%s'\n", model_name, model_dir);
        qwen3_tts_free(ctx);
        return 1;
    }
    fprintf(stderr, "[+] Model loaded successfully. Processing %d chunks with speaker '%s'...\n", num_chunks, speaker);

    float speed = opts ? opts->speed : 1.0f;
    char final_prompt[1024];
    build_style_prompt(prompt, speed, final_prompt, sizeof(final_prompt));

    size_t total_samples = 0;
    size_t audio_capacity = 24000 * 60;
    float* full_audio = (float*)malloc(audio_capacity * sizeof(float));
    if (!full_audio) {
        qwen3_tts_free(ctx);
        return 1;
    }
    int sample_rate = 24000;
    float speech_rate_scale = 1.0f;
    qwen3_tts_params_t params = build_tts_params(opts, final_prompt, speaker);

    for (int i = 0; i < num_chunks; i++) {
        char preview[64];
        format_chunk_preview(chunks[i], preview, sizeof(preview));
        fprintf(stderr, "\n[*] Chunk [%d/%d] (%zu chars): \"%s\"\n", i + 1, num_chunks, strlen(chunks[i]), preview);

        tts_progress_t prog;
        init_tts_progress(&prog, chunks[i], speech_rate_scale, 0, model_name);
        qwen3_tts_set_phase_progress_callback(ctx, on_phase_progress, &prog);

        qwen3_tts_result_t result = qwen3_tts_synthesize(ctx, chunks[i], params);
        if (!result.success || result.audio_len <= 0) {
            if (prog.is_tty) fprintf(stderr, "\r\033[K");
            fprintf(stderr, "[-] Warning: Chunk %d synthesis failed: %s\n", i + 1,
                    result.error_msg ? result.error_msg : "unknown error");
            qwen3_tts_free_result(result);
            continue;
        }

        speech_rate_scale = update_adaptive_speech_rate(speech_rate_scale, &prog);
        sample_rate = result.sample_rate;
        double chunk_sec = (double)result.audio_len / (double)result.sample_rate;
        double wall_sec = (double)result.t_total_ms / 1000.0;
        double rtf = wall_sec > 0.0 ? chunk_sec / wall_sec : 0.0;
        if (prog.is_tty) {
            fprintf(stderr, "\r\033[K[+] Chunk [%d/%d] synthesized: %.2fs audio (%.2fx real-time)\n",
                    i + 1, num_chunks, chunk_sec, rtf);
        } else {
            fprintf(stderr, "[+] Chunk [%d/%d] synthesized: %.2fs audio (%.2fx real-time)\n",
                    i + 1, num_chunks, chunk_sec, rtf);
        }
        append_chunk_audio(&full_audio, &total_samples, &audio_capacity, result.audio, result.audio_len, sample_rate, i == num_chunks - 1);
        qwen3_tts_free_result(result);
    }

    int rc = 0;
    const char* output_path = (opts && opts->output_path) ? opts->output_path : "speak_output.wav";
    if (total_samples > 0) {
        if (fabsf(speed - 1.0f) >= 0.01f) {
            int stretched_len = 0;
            float* stretched = apply_sola_timestretch(full_audio, (int)total_samples, sample_rate, speed, &stretched_len);
            if (stretched && stretched_len > 0) {
                free(full_audio);
                full_audio = stretched;
                total_samples = (size_t)stretched_len;
            }
        }
        if (opts && opts->normalize) {
            apply_peak_normalization(full_audio, total_samples);
        }
        if (save_wav_file(output_path, full_audio, (int)total_samples, sample_rate) == 0) {
            fprintf(stderr, "[+] Audio generation complete!\n");
            if (opts && opts->play && strcmp(output_path, "-") != 0) {
                play_audio_file(output_path);
            }
        } else {
            rc = 1;
        }
    } else {
        rc = 1;
    }

    free(full_audio);
    qwen3_tts_free(ctx);
    return rc;
}

int c_extract_profile(const char* audio_path, const char* ref_text, const char* output_profile_path, const char* model_override) {
    if (!audio_path || !output_profile_path) return 1;
    configure_runtime_threads(0);

    char model_name[256];
    resolve_model_preset("clone", model_override, model_name, sizeof(model_name));
    const char* model_dir = resolve_model_dir();
    const bool need_icl = (ref_text && strlen(ref_text) > 0);

    if (ensure_file_exists(model_dir, model_name) != 0) {
        return 1;
    }
    if (need_icl) {
        char tok_name[256];
        get_setting_string("tokenizer_model", tok_name, sizeof(tok_name), "qwen-tokenizer-12hz-BF16.gguf");
        if (ensure_file_exists(model_dir, tok_name) != 0) {
            return 1;
        }
    }

    qwen3_tts_context_t* ctx = qwen3_tts_init();
    if (!ctx) return 1;

    int is_temp_wav = 0;
    char* prepared_audio = prepare_reference_audio(audio_path, &is_temp_wav);
    int rc = 0;

    if (need_icl) {
        fprintf(stderr, "[*] Loading ICL prompt encoders from '%s'...\n", model_name);
        if (!qwen3_tts_load_icl_prompt_encoder_with_name(ctx, model_dir, model_name)) {
            fprintf(stderr, "[-] Error: Failed to load ICL prompt encoder\n");
            rc = 1;
        } else if (!qwen3_tts_extract_icl_prompt(ctx, prepared_audio, ref_text, output_profile_path)) {
            char* err = qwen3_tts_get_last_error(ctx);
            fprintf(stderr, "[-] Error: Failed to extract ICL prompt: %s\n", err ? err : "unknown");
            qwen3_tts_free_string(err);
            rc = 1;
        } else {
            fprintf(stderr, "[+] Full ICL voice profile saved to: %s\n", output_profile_path);
        }
    } else {
        // Fast speaker-encoder-only load (~7.5 MB instead of full 4.1 GB transformer + vocoder)
        fprintf(stderr, "[*] Loading speaker encoder from '%s'...\n", model_name);
        if (!qwen3_tts_load_speaker_encoder_with_name(ctx, model_dir, model_name)) {
            fprintf(stderr, "[-] Error: Failed to load speaker encoder from '%s'\n", model_name);
            rc = 1;
        } else if (!qwen3_tts_extract_speaker_embedding(ctx, prepared_audio, output_profile_path)) {
            char* err = qwen3_tts_get_last_error(ctx);
            fprintf(stderr, "[-] Error: Failed to extract speaker embedding: %s\n", err ? err : "unknown");
            qwen3_tts_free_string(err);
            rc = 1;
        } else {
            fprintf(stderr, "[+] Speaker embedding profile saved to: %s\n", output_profile_path);
        }
    }

    if (is_temp_wav && prepared_audio) remove(prepared_audio);
    if (prepared_audio) free(prepared_audio);
    qwen3_tts_free(ctx);
    return rc;
}

int c_download_models(const char* preset) {
    const char* model_dir = resolve_model_dir();
    char tok_name[256];
    get_setting_string("tokenizer_model", tok_name, sizeof(tok_name), "qwen-tokenizer-12hz-BF16.gguf");

    if (ensure_file_exists(model_dir, tok_name) != 0) return 1;

    char style_model[256];
    char clone_model[256];
    resolve_model_preset("style", preset, style_model, sizeof(style_model));
    resolve_model_preset("clone", preset, clone_model, sizeof(clone_model));

    if (ensure_file_exists(model_dir, style_model) != 0) return 1;
    if (ensure_file_exists(model_dir, clone_model) != 0) return 1;
    return 0;
}

void c_print_system_info(void) {
    const char* model_dir = resolve_model_dir();
    int32_t threads = qwen3_tts_get_cpu_threads();
    int32_t backend_mask = qwen3_tts_get_compiled_backend_mask();

    char style_model[256];
    char clone_model[256];
    char speak_model[256];
    char tok_model[256];
    get_setting_string("model_name_style", style_model, sizeof(style_model), "qwen-talker-1.7b-voicedesign-BF16.gguf");
    get_setting_string("model_name_clone", clone_model, sizeof(clone_model), "qwen-talker-1.7b-base-BF16.gguf");
    get_setting_string("model_name_speak", speak_model, sizeof(speak_model), "qwen-talker-1.7b-customvoice-BF16.gguf");
    get_setting_string("tokenizer_model", tok_model, sizeof(tok_model), "qwen-tokenizer-12hz-BF16.gguf");

    printf("============================================================\n");
    printf("              voicecli - System & Model Status              \n");
    printf("============================================================\n");
    printf("  Model Directory     : %s\n", model_dir);
    printf("  Default CPU Threads : %d\n", threads);
    printf("  Active CPU SIMD     : %s\n", voicecli_get_active_simd_name());
    printf("  Compiled Backends   : CPU (Multi-ISA Runtime Dispatch)%s\n", (backend_mask & 2) ? ", CUDA" : "");
    printf("  Configured Models   :\n");
    printf("    - Style (Design)  : %s\n", style_model);
    printf("    - Clone (Base)    : %s\n", clone_model);
    printf("    - Speak (Custom)  : %s\n", speak_model);
    printf("    - Audio Codec     : %s\n", tok_model);
    printf("\n  Available Presets (--model <PRESET>):\n");
    printf("    - bf16            : Full 1.7B BF16 precision (default, highest quality)\n");
    printf("    - q8              : 1.7B Q8_0 quantized (~50%% less RAM, fast CPU inference)\n");
    printf("    - q4              : 1.7B Q4_K_M quantized (~70%% less RAM)\n");
    printf("    - 0.6b / fast     : 0.6B Q8_0 lightweight model (ultra-fast voice cloning)\n");
    printf("    - 0.6b-bf16       : 0.6B BF16 lightweight model\n");
    printf("    - 0.6b-q4 / tiny  : 0.6B Q4_K_M ultra-compact model (<600 MB)\n");
    printf("============================================================\n");
}

// Legacy ABI wrappers for backwards compatibility
void c_generate_style(const char* text, const char* prompt, const char* output_path, float speed) {
    voicecli_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.output_path = output_path;
    opts.speed = speed;
    opts.temperature = -1.0f;
    opts.top_k = -1;
    c_generate_style_ex(text, prompt, &opts);
}

void c_generate_clone(const char* text, const char* audio_path, const char* output_path, float speed) {
    voicecli_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.output_path = output_path;
    opts.speed = speed;
    opts.temperature = -1.0f;
    opts.top_k = -1;
    c_generate_clone_ex(text, audio_path, &opts);
}

void c_generate_style_chunked(const char** chunks, int num_chunks, const char* prompt, const char* output_path, float speed) {
    voicecli_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.output_path = output_path;
    opts.speed = speed;
    opts.temperature = -1.0f;
    opts.top_k = -1;
    c_generate_style_chunked_ex(chunks, num_chunks, prompt, &opts);
}

void c_generate_clone_chunked(const char** chunks, int num_chunks, const char* audio_path, const char* output_path, float speed) {
    voicecli_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.output_path = output_path;
    opts.speed = speed;
    opts.temperature = -1.0f;
    opts.top_k = -1;
    c_generate_clone_chunked_ex(chunks, num_chunks, audio_path, &opts);
}
