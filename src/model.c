#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <errno.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "qwen3_tts_c.h"

extern const char* voicecli_get_active_simd_name(void);

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#define isatty _isatty
#define fileno _fileno
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

typedef struct {
    int estimated_frames;
    int last_tokens;
    int is_tty;
} tts_progress_t;

static void on_progress(int tokens, int max_tokens, void* user_data) {
    (void)max_tokens;
    tts_progress_t* prog = (tts_progress_t*)user_data;
    if (!prog) return;

    static const char* spinner[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
    const int num_spinner = 10;
    const char* spin = spinner[tokens % num_spinner];

    int target = prog->estimated_frames > 0 ? prog->estimated_frames : 40;
    if (tokens >= target) {
        target = tokens + 8;
        prog->estimated_frames = target;
    }
    int percent = (tokens * 100) / target;
    if (percent > 99) percent = 99;
    if (percent < 1) percent = 1;

    int bar_width = 24;
    int filled = (percent * bar_width) / 100;
    char bar[32];
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) bar[i] = '=';
        else if (i == filled) bar[i] = '>';
        else bar[i] = ' ';
    }
    bar[bar_width] = '\0';

    float audio_sec = (float)tokens / 12.0f; // 12Hz speech tokenizer frame rate

    if (prog->is_tty) {
        fprintf(stderr, "\r\033[K[*] Synthesizing [%s] [%s] %2d%% (%d frames, ~%.1fs)",
                spin, bar, percent, tokens, audio_sec);
        fflush(stderr);
    } else {
        if (tokens == 1 || tokens % 20 == 0) {
            fprintf(stderr, "[*] Synthesizing: %d frames (~%.1fs)...\n", tokens, audio_sec);
            fflush(stderr);
        }
    }
    prog->last_tokens = tokens;
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

static FILE* open_settings_file(void) {
    FILE *f = fopen("settings.json", "r");
    if (!f) {
        const char *base_dir = getenv("VOICECLI_BASE_DIR");
        if (base_dir && strlen(base_dir) > 0) {
            char p[4096];
            snprintf(p, sizeof(p), "%s%csettings.json", base_dir, PATH_SEP);
            f = fopen(p, "r");
        }
    }
    return f;
}

static void get_setting_string(const char *key, char *out, size_t out_size, const char *default_val) {
    char env_key[128];
    snprintf(env_key, sizeof(env_key), "VOICECLI_%s", key);
    for (int i = 0; env_key[i]; i++) {
        if (env_key[i] >= 'a' && env_key[i] <= 'z') env_key[i] -= 32;
    }
    const char *env_val = getenv(env_key);
    if (env_val && strlen(env_val) > 0) {
        strncpy(out, env_val, out_size - 1);
        out[out_size - 1] = '\0';
        return;
    }

    FILE *f = open_settings_file();
    if (f) {
        char line[512];
        char search_key[128];
        snprintf(search_key, sizeof(search_key), "\"%s\"", key);
        while (fgets(line, sizeof(line), f)) {
            char *pos = strstr(line, search_key);
            if (pos) {
                char *colon = strchr(pos + strlen(search_key), ':');
                if (colon) {
                    char *start = strchr(colon + 1, '"');
                    if (start) {
                        start++;
                        char *end = strchr(start, '"');
                        if (end) {
                            *end = '\0';
                            strncpy(out, start, out_size - 1);
                            out[out_size - 1] = '\0';
                            fclose(f);
                            return;
                        }
                    }
                }
            }
        }
        fclose(f);
    }

    strncpy(out, default_val, out_size - 1);
    out[out_size - 1] = '\0';
}

static float get_setting_float(const char *key, float default_val) {
    char env_key[128];
    snprintf(env_key, sizeof(env_key), "VOICECLI_%s", key);
    for (int i = 0; env_key[i]; i++) {
        if (env_key[i] >= 'a' && env_key[i] <= 'z') env_key[i] -= 32;
    }
    const char *env_val = getenv(env_key);
    if (env_val && strlen(env_val) > 0) {
        char *endptr = NULL;
        float v = strtof(env_val, &endptr);
        if (endptr != env_val) return v;
    }

    FILE *f = open_settings_file();
    if (f) {
        char line[512];
        char search_key[128];
        snprintf(search_key, sizeof(search_key), "\"%s\"", key);
        while (fgets(line, sizeof(line), f)) {
            char *pos = strstr(line, search_key);
            if (pos) {
                char *colon = strchr(pos + strlen(search_key), ':');
                if (colon) {
                    char *val_start = colon + 1;
                    while (*val_start && (*val_start == ' ' || *val_start == '\t')) val_start++;
                    char *endptr = NULL;
                    float v = strtof(val_start, &endptr);
                    if (endptr != val_start) {
                        fclose(f);
                        return v;
                    }
                }
            }
        }
        fclose(f);
    }
    return default_val;
}

static int32_t get_setting_int(const char *key, int32_t default_val) {
    char env_key[128];
    snprintf(env_key, sizeof(env_key), "VOICECLI_%s", key);
    for (int i = 0; env_key[i]; i++) {
        if (env_key[i] >= 'a' && env_key[i] <= 'z') env_key[i] -= 32;
    }
    const char *env_val = getenv(env_key);
    if (env_val && strlen(env_val) > 0) {
        char *endptr = NULL;
        long v = strtol(env_val, &endptr, 10);
        if (endptr != env_val) return (int32_t)v;
    }

    FILE *f = open_settings_file();
    if (f) {
        char line[512];
        char search_key[128];
        snprintf(search_key, sizeof(search_key), "\"%s\"", key);
        while (fgets(line, sizeof(line), f)) {
            char *pos = strstr(line, search_key);
            if (pos) {
                char *colon = strchr(pos + strlen(search_key), ':');
                if (colon) {
                    char *val_start = colon + 1;
                    while (*val_start && (*val_start == ' ' || *val_start == '\t')) val_start++;
                    char *endptr = NULL;
                    long v = strtol(val_start, &endptr, 10);
                    if (endptr != val_start) {
                        fclose(f);
                        return (int32_t)v;
                    }
                }
            }
        }
        fclose(f);
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
        if (strstr(lower, "0.6b") != NULL || strcmp(lower, "fast") == 0 || strcmp(lower, "small") == 0) {
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

            const float* x_ptr = in_samples + actual_idx;
            float dot = 0.0f;
            for (int j = 0; j < L; j++) {
                dot += y_ptr[j] * x_ptr[j];
            }
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

            const float* x_ptr = in_samples + actual_idx;
            float dot = 0.0f;
            for (int j = 0; j < L; j++) {
                dot += y_ptr[j] * x_ptr[j];
            }
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

    float* temp = (float*)malloc((size_t)clean_len * sizeof(float));
    if (!temp) return;
    memcpy(temp, chunk_audio + start, (size_t)clean_len * sizeof(float));

    // Smooth Hann fade-in (15ms)
    int fade_in = (int)(sample_rate * 0.015f);
    if (fade_in > clean_len / 2) fade_in = clean_len / 2;
    for (int j = 0; j < fade_in; j++) {
        float factor = 0.5f * (1.0f - cosf((float)M_PI * (float)j / (float)fade_in));
        temp[j] *= factor;
    }

    // Smooth Hann fade-out (20ms)
    int fade_out = (int)(sample_rate * 0.020f);
    if (fade_out > clean_len / 2) fade_out = clean_len / 2;
    for (int j = 0; j < fade_out; j++) {
        float factor = 0.5f * (1.0f + cosf((float)M_PI * (float)j / (float)fade_out));
        temp[clean_len - fade_out + j] *= factor;
    }

    // Natural 150ms breathing pause between chunks
    int silence_samples = is_last_chunk ? 0 : (int)(sample_rate * 0.15f);

    size_t needed = *total_samples + (size_t)clean_len + (size_t)silence_samples;
    if (needed > *audio_capacity) {
        size_t new_cap = needed * 2;
        float* new_buf = (float*)realloc(*full_audio, new_cap * sizeof(float));
        if (!new_buf) {
            fprintf(stderr, "[-] Error: Failed to reallocate audio buffer\n");
            free(temp);
            return;
        }
        *full_audio = new_buf;
        *audio_capacity = new_cap;
    }

    memcpy(*full_audio + *total_samples, temp, (size_t)clean_len * sizeof(float));
    *total_samples += (size_t)clean_len;

    if (silence_samples > 0) {
        memset(*full_audio + *total_samples, 0, (size_t)silence_samples * sizeof(float));
        *total_samples += (size_t)silence_samples;
    }

    free(temp);
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
    prog.is_tty = isatty(fileno(stderr));
    prog.last_tokens = 0;
    int text_len = text ? (int)strlen(text) : 20;
    prog.estimated_frames = text_len < 24 ? 24 : text_len;
    qwen3_tts_set_progress_callback(ctx, on_progress, &prog);

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

    for (int i = 0; i < num_chunks; i++) {
        char preview[64];
        format_chunk_preview(chunks[i], preview, sizeof(preview));

        fprintf(stderr, "\n[*] Chunk [%d/%d] (%zu chars): \"%s\"\n", i + 1, num_chunks, strlen(chunks[i]), preview);

        tts_progress_t prog;
        prog.is_tty = isatty(fileno(stderr));
        prog.last_tokens = 0;
        int text_len = (int)strlen(chunks[i]);
        prog.estimated_frames = text_len < 24 ? 24 : text_len;
        qwen3_tts_set_progress_callback(ctx, on_progress, &prog);

        qwen3_tts_params_t params = build_tts_params(opts, final_prompt, NULL);
        qwen3_tts_result_t result = qwen3_tts_synthesize(ctx, chunks[i], params);
        if (!result.success || result.audio_len <= 0) {
            if (prog.is_tty) fprintf(stderr, "\r\033[K");
            fprintf(stderr, "[-] Warning: Chunk %d synthesis failed: %s\n", i + 1,
                    result.error_msg ? result.error_msg : "unknown error");
            qwen3_tts_free_result(result);
            continue;
        }

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
    const char* active_profile_path,
    int is_icl_profile,
    qwen3_tts_params_t params,
    const voicecli_options_t* opts
) {
    if (opts && opts->stream) {
        qwen3_tts_streaming_params_t sparams;
        memset(&sparams, 0, sizeof(sparams));
        sparams.generation = params;
        sparams.chunk_sec = opts->stream_chunk_sec > 0.0f ? opts->stream_chunk_sec : 1.0f;
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
        return qwen3_tts_synthesize_with_voice_streaming(ctx, text, effective_audio_path, sparams, on_audio_stream_chunk, &sctx);
    } else {
        if (active_profile_path && strlen(active_profile_path) > 0) {
            if (is_icl_profile) {
                return qwen3_tts_synthesize_with_icl_prompt(ctx, text, active_profile_path, params);
            } else {
                return qwen3_tts_synthesize_with_speaker_embedding(ctx, text, active_profile_path, params);
            }
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
    char temp_icl_profile[4096] = {0};
    int is_temp_icl = 0;
    int is_icl_profile = 0;

    if (active_profile) {
        is_icl_profile = detect_icl_profile_file(active_profile);
        fprintf(stderr, "[*] Using precomputed %s profile: %s\n",
                is_icl_profile ? "ICL voice" : "speaker embedding", active_profile);
    } else if (audio_path && strlen(audio_path) > 0) {
        prepared_audio = prepare_reference_audio(audio_path, &is_temp_wav);
        fprintf(stderr, "[*] Cloning voice from reference audio: %s\n", audio_path);

        if (opts && opts->ref_text && strlen(opts->ref_text) > 0) {
            const char* out_prof = (opts->save_profile && strlen(opts->save_profile) > 0) ? opts->save_profile : NULL;
            if (!out_prof) {
#ifdef _WIN32
                snprintf(temp_icl_profile, sizeof(temp_icl_profile), "voicecli_icl_%d.json", (int)getpid());
#else
                snprintf(temp_icl_profile, sizeof(temp_icl_profile), "/tmp/voicecli_icl_%d.json", (int)getpid());
#endif
                out_prof = temp_icl_profile;
                is_temp_icl = 1;
            }
            fprintf(stderr, "[*] Extracting full ICL prompt (speaker embedding + reference speech codes)...\n");
            if (qwen3_tts_extract_icl_prompt(ctx, prepared_audio, opts->ref_text, out_prof)) {
                active_profile = out_prof;
                is_icl_profile = 1;
                if (!is_temp_icl) {
                    fprintf(stderr, "[+] Saved reusable ICL voice profile to: %s\n", out_prof);
                }
            } else {
                fprintf(stderr, "[-] Warning: ICL prompt extraction failed; falling back to speaker embedding.\n");
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

    tts_progress_t prog;
    prog.is_tty = isatty(fileno(stderr));
    prog.last_tokens = 0;
    int text_len = text ? (int)strlen(text) : 20;
    prog.estimated_frames = text_len < 24 ? 24 : text_len;
    qwen3_tts_set_progress_callback(ctx, on_progress, &prog);

    qwen3_tts_params_t params = build_tts_params(opts, NULL, NULL);
    qwen3_tts_result_t result = execute_clone_synthesis(
        ctx, text, prepared_audio, active_profile, is_icl_profile, params, opts
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
    if (is_temp_icl && temp_icl_profile[0]) remove(temp_icl_profile);
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
    char temp_icl_profile[4096] = {0};
    int is_temp_icl = 0;
    int is_icl_profile = 0;

    if (active_profile) {
        is_icl_profile = detect_icl_profile_file(active_profile);
        fprintf(stderr, "[*] Using precomputed %s profile: %s\n",
                is_icl_profile ? "ICL voice" : "speaker embedding", active_profile);
    } else if (audio_path && strlen(audio_path) > 0) {
        prepared_audio = prepare_reference_audio(audio_path, &is_temp_wav);
        fprintf(stderr, "[*] Cloning voice from reference audio: %s\n", audio_path);

        if (opts && opts->ref_text && strlen(opts->ref_text) > 0) {
            const char* out_prof = (opts->save_profile && strlen(opts->save_profile) > 0) ? opts->save_profile : NULL;
            if (!out_prof) {
#ifdef _WIN32
                snprintf(temp_icl_profile, sizeof(temp_icl_profile), "voicecli_icl_%d.json", (int)getpid());
#else
                snprintf(temp_icl_profile, sizeof(temp_icl_profile), "/tmp/voicecli_icl_%d.json", (int)getpid());
#endif
                out_prof = temp_icl_profile;
                is_temp_icl = 1;
            }
            fprintf(stderr, "[*] Extracting full ICL prompt once for all %d chunks...\n", num_chunks);
            if (qwen3_tts_extract_icl_prompt(ctx, prepared_audio, opts->ref_text, out_prof)) {
                active_profile = out_prof;
                is_icl_profile = 1;
                if (!is_temp_icl) {
                    fprintf(stderr, "[+] Saved reusable ICL voice profile to: %s\n", out_prof);
                }
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
        if (is_temp_icl && temp_icl_profile[0]) remove(temp_icl_profile);
        if (prepared_audio) free(prepared_audio);
        qwen3_tts_free(ctx);
        return 1;
    }
    int sample_rate = 24000;

    for (int i = 0; i < num_chunks; i++) {
        char preview[64];
        format_chunk_preview(chunks[i], preview, sizeof(preview));

        fprintf(stderr, "\n[*] Chunk [%d/%d] (%zu chars): \"%s\"\n", i + 1, num_chunks, strlen(chunks[i]), preview);

        tts_progress_t prog;
        prog.is_tty = isatty(fileno(stderr));
        prog.last_tokens = 0;
        int text_len = (int)strlen(chunks[i]);
        prog.estimated_frames = text_len < 24 ? 24 : text_len;
        qwen3_tts_set_progress_callback(ctx, on_progress, &prog);

        qwen3_tts_params_t params = build_tts_params(opts, NULL, NULL);
        qwen3_tts_result_t result = execute_clone_synthesis(
            ctx, chunks[i], prepared_audio, active_profile, is_icl_profile, params, NULL
        );

        if (!result.success || result.audio_len <= 0) {
            if (prog.is_tty) fprintf(stderr, "\r\033[K");
            fprintf(stderr, "[-] Warning: Chunk %d voice cloning failed: %s\n", i + 1,
                    result.error_msg ? result.error_msg : "unknown error");
            qwen3_tts_free_result(result);
            continue;
        }

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
    if (is_temp_icl && temp_icl_profile[0]) remove(temp_icl_profile);
    if (prepared_audio) free(prepared_audio);
    free(full_audio);
    qwen3_tts_free(ctx);
    return rc;
}

int c_generate_speak_ex(const char* text, const char* speaker, const char* prompt, const voicecli_options_t* opts) {
    configure_runtime_threads(opts ? opts->threads : 0);

    char model_name[256];
    resolve_model_preset("speak", opts ? opts->model_override : NULL, model_name, sizeof(model_name));

    fprintf(stderr, "[*] Loading CustomVoice model: %s ...\n", model_name);
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

    // If speaker is empty or "list", list available speakers
    if (!speaker || strlen(speaker) == 0 || strcmp(speaker, "list") == 0) {
        char* speakers = qwen3_tts_get_available_speakers(ctx);
        printf("Available built-in speakers in '%s':\n", model_name);
        if (speakers && strlen(speakers) > 0) {
            printf("%s\n", speakers);
        } else {
            printf("  (No named speakers found in model metadata)\n");
        }
        qwen3_tts_free_string(speakers);
        qwen3_tts_free(ctx);
        return 0;
    }

    float speed = opts ? opts->speed : 1.0f;
    char final_prompt[1024];
    build_style_prompt(prompt, speed, final_prompt, sizeof(final_prompt));

    fprintf(stderr, "[*] Using built-in speaker: \"%s\"\n", speaker);
    fprintf(stderr, "[*] Generating speech for text: \"%s\"\n", text);
    if (strlen(final_prompt) > 0) {
        fprintf(stderr, "[*] Voice steering instruction: \"%s\"\n", final_prompt);
    }

    tts_progress_t prog;
    prog.is_tty = isatty(fileno(stderr));
    prog.last_tokens = 0;
    int text_len = text ? (int)strlen(text) : 20;
    prog.estimated_frames = text_len < 24 ? 24 : text_len;
    qwen3_tts_set_progress_callback(ctx, on_progress, &prog);

    qwen3_tts_params_t params = build_tts_params(opts, final_prompt, speaker);
    qwen3_tts_result_t result = qwen3_tts_synthesize(ctx, text, params);

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

    for (int i = 0; i < num_chunks; i++) {
        char preview[64];
        format_chunk_preview(chunks[i], preview, sizeof(preview));
        fprintf(stderr, "\n[*] Chunk [%d/%d] (%zu chars): \"%s\"\n", i + 1, num_chunks, strlen(chunks[i]), preview);

        tts_progress_t prog;
        prog.is_tty = isatty(fileno(stderr));
        prog.last_tokens = 0;
        int text_len = (int)strlen(chunks[i]);
        prog.estimated_frames = text_len < 24 ? 24 : text_len;
        qwen3_tts_set_progress_callback(ctx, on_progress, &prog);

        qwen3_tts_params_t params = build_tts_params(opts, final_prompt, speaker);
        qwen3_tts_result_t result = qwen3_tts_synthesize(ctx, chunks[i], params);
        if (!result.success || result.audio_len <= 0) {
            if (prog.is_tty) fprintf(stderr, "\r\033[K");
            fprintf(stderr, "[-] Warning: Chunk %d synthesis failed: %s\n", i + 1,
                    result.error_msg ? result.error_msg : "unknown error");
            qwen3_tts_free_result(result);
            continue;
        }
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

    char model_name[256];
    resolve_model_preset("clone", model_override, model_name, sizeof(model_name));
    const char* model_dir = resolve_model_dir();
    char tok_name[256];
    get_setting_string("tokenizer_model", tok_name, sizeof(tok_name), "qwen-tokenizer-12hz-BF16.gguf");

    if (ensure_file_exists(model_dir, tok_name) != 0 || ensure_file_exists(model_dir, model_name) != 0) {
        return 1;
    }

    qwen3_tts_context_t* ctx = qwen3_tts_init();
    if (!ctx) return 1;

    int is_temp_wav = 0;
    char* prepared_audio = prepare_reference_audio(audio_path, &is_temp_wav);
    int rc = 0;

    if (ref_text && strlen(ref_text) > 0) {
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
        fprintf(stderr, "[*] Loading speaker encoder from '%s'...\n", model_name);
        if (!qwen3_tts_load_models_with_name(ctx, model_dir, model_name)) {
            fprintf(stderr, "[-] Error: Failed to load model '%s'\n", model_name);
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
