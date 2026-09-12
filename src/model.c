#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <stdint.h>
#include "qwen3_tts_c.h"

#ifdef _WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#define PATH_SEP '\\'
#else
#include <unistd.h>
#define PATH_SEP '/'
#endif

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

    FILE *f = fopen("settings.json", "r");
    if (!f) {
        const char *base_dir = getenv("VOICECLI_BASE_DIR");
        if (base_dir) {
            char p[4096];
            snprintf(p, sizeof(p), "%s%csettings.json", base_dir, PATH_SEP);
            f = fopen(p, "r");
        }
    }
    if (f) {
        char line[512];
        char search_key[128];
        snprintf(search_key, sizeof(search_key), "\"%s\":", key);
        while (fgets(line, sizeof(line), f)) {
            char *pos = strstr(line, search_key);
            if (pos) {
                char *start = strchr(pos + strlen(search_key), '"');
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
        fclose(f);
    }

    strncpy(out, default_val, out_size - 1);
    out[out_size - 1] = '\0';
}

static const char* resolve_model_dir(void) {
    static char model_dir[4096];
    const char* custom_model_dir = getenv("VOICECLI_MODEL_DIR");
    if (custom_model_dir && strlen(custom_model_dir) > 0) {
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

static void ensure_file_exists(const char *dir, const char *filename) {
    char filepath[4096];
    snprintf(filepath, sizeof(filepath), "%s%c%s", dir, PATH_SEP, filename);
    struct stat st;
    if (stat(filepath, &st) == 0 && st.st_size > 1000000) {
        return;
    }
    char local_check[4096];
    snprintf(local_check, sizeof(local_check), "models/%s", filename);
    if (stat(local_check, &st) == 0 && st.st_size > 1000000) {
        return;
    }
    if (stat(filename, &st) == 0 && st.st_size > 1000000) {
        return;
    }

    printf("[*] Auto-Setup: Model '%s' not found locally.\n", filename);
    printf("[*] Downloading from Hugging Face (Serveurperso/Qwen3-TTS-GGUF)...\n");
    char dl_cmd[8192];
    snprintf(dl_cmd, sizeof(dl_cmd),
        "curl -f -L --progress-bar -C - \"https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/main/%s\" -o \"%s\"",
        filename, filepath);
    if (system(dl_cmd) != 0) {
        fprintf(stderr, "[-] Error: Failed to download '%s'. Please check your internet connection.\n", filename);
    } else {
        printf("[+] Download of '%s' completed successfully!\n", filename);
    }
}

static int save_wav_file(const char *filename, const float *samples, int num_samples, int sample_rate) {
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;

    int num_channels = 1;
    int bits_per_sample = 16;
    int byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
    int block_align = num_channels * (bits_per_sample / 8);
    int data_size = num_samples * sizeof(int16_t);
    int chunk_size = 36 + data_size;

    fwrite("RIFF", 1, 4, f);
    fwrite(&chunk_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    int subchunk1_size = 16;
    int16_t audio_format = 1; // PCM
    fwrite(&subchunk1_size, 4, 1, f);
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);

    for (int i = 0; i < num_samples; i++) {
        float s = samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        int16_t pcm = (int16_t)(s * 32767.0f);
        fwrite(&pcm, sizeof(int16_t), 1, f);
    }
    fclose(f);
    return 0;
}

void c_generate_style(const char* text, const char* prompt, const char* output_path) {
    char model_name[256];
    get_setting_string("model_name_style", model_name, sizeof(model_name), "qwen-talker-1.7b-voicedesign-BF16.gguf");

    printf("[*] Loading model: %s (FP16/BF16) ...\n", model_name);
    printf("[*] Target device: CPU (In-Process Native C++ / GGML FP16 SIMD)\n");

    const char* model_dir = resolve_model_dir();
    char tok_name[256];
    get_setting_string("tokenizer_model", tok_name, sizeof(tok_name), "qwen-tokenizer-12hz-BF16.gguf");

    ensure_file_exists(model_dir, tok_name);
    ensure_file_exists(model_dir, model_name);

    qwen3_tts_context_t* ctx = qwen3_tts_init();
    if (!ctx) {
        fprintf(stderr, "[-] Error: Failed to initialize native Qwen3-TTS context\n");
        return;
    }

    printf("[*] Loading model weights into memory...\n");
    fflush(stdout);

    if (!qwen3_tts_load_models_with_name(ctx, model_dir, model_name)) {
        fprintf(stderr, "[-] Error: Failed to load models '%s' from '%s'\n", model_name, model_dir);
        qwen3_tts_free(ctx);
        return;
    }
    printf("[+] Model loaded successfully.\n");

    printf("[*] Generating speech for text: \"%s\"\n", text);
    if (prompt && strlen(prompt) > 0) {
        printf("[*] Acoustic style description: \"%s\"\n", prompt);
    }

    tts_progress_t prog;
    prog.is_tty = isatty(fileno(stderr));
    prog.last_tokens = 0;
    int text_len = text ? (int)strlen(text) : 20;
    prog.estimated_frames = (int)(text_len * 1.0f);
    if (prog.estimated_frames < 24) prog.estimated_frames = 24;

    qwen3_tts_set_progress_callback(ctx, on_progress, &prog);

    qwen3_tts_params_t params;
    memset(&params, 0, sizeof(params));
    params.max_audio_tokens = 4096;
    params.temperature = 0.9f;
    params.top_p = 1.0f;
    params.top_k = 50;
    params.n_threads = 0; // auto-detect cores
    params.print_progress = 0;
    params.print_timing = 0;
    params.repetition_penalty = 1.05f;
    params.instruction = (prompt && strlen(prompt) > 0) ? prompt : NULL;

    qwen3_tts_result_t result = qwen3_tts_synthesize(ctx, text, params);
    if (!result.success || result.audio_len <= 0) {
        if (prog.is_tty) fprintf(stderr, "\r\033[K");
        fprintf(stderr, "[-] Native synthesis failed: %s\n", result.error_msg ? result.error_msg : "unknown error");
    } else {
        double audio_sec = result.sample_rate > 0 ? (double)result.audio_len / (double)result.sample_rate : 0.0;
        double wall_sec = (double)result.t_total_ms / 1000.0;
        double rtf = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;
        if (prog.is_tty) {
            fprintf(stderr, "\r\033[K[+] Synthesis complete: %.2fs audio generated in %.2fs (%.2fx real-time)\n",
                    audio_sec, wall_sec, rtf);
        } else {
            fprintf(stderr, "[+] Synthesis complete: %.2fs audio generated in %.2fs (%.2fx real-time)\n",
                    audio_sec, wall_sec, rtf);
        }
        printf("[*] Saving audio to %s (Sample Rate: %d Hz)...\n", output_path, result.sample_rate);
        if (save_wav_file(output_path, result.audio, result.audio_len, result.sample_rate) == 0) {
            printf("[+] Audio generation complete!\n");
        } else {
            fprintf(stderr, "[-] Error: Failed to write audio file %s\n", output_path);
        }
    }

    qwen3_tts_free_result(result);
    qwen3_tts_free(ctx);
}

void c_generate_clone(const char* text, const char* audio_path, const char* output_path) {
    char model_name[256];
    get_setting_string("model_name_clone", model_name, sizeof(model_name), "qwen-talker-1.7b-base-BF16.gguf");

    printf("[*] Loading model: %s (FP16/BF16) ...\n", model_name);
    printf("[*] Target device: CPU (In-Process Native C++ / GGML FP16 SIMD)\n");

    const char* model_dir = resolve_model_dir();
    char tok_name[256];
    get_setting_string("tokenizer_model", tok_name, sizeof(tok_name), "qwen-tokenizer-12hz-BF16.gguf");

    ensure_file_exists(model_dir, tok_name);
    ensure_file_exists(model_dir, model_name);

    qwen3_tts_context_t* ctx = qwen3_tts_init();
    if (!ctx) {
        fprintf(stderr, "[-] Error: Failed to initialize native Qwen3-TTS context\n");
        return;
    }

    printf("[*] Loading model weights into memory...\n");
    fflush(stdout);

    if (!qwen3_tts_load_models_with_name(ctx, model_dir, model_name)) {
        fprintf(stderr, "[-] Error: Failed to load models '%s' from '%s'\n", model_name, model_dir);
        qwen3_tts_free(ctx);
        return;
    }
    printf("[+] Model loaded successfully.\n");

    printf("[*] Cloning voice from reference audio: %s\n", audio_path);
    printf("[*] Generating speech for text: \"%s\"\n", text);
    printf("[*] Extracting speaker embedding and reference audio codes...\n");
    fflush(stdout);

    tts_progress_t prog;
    prog.is_tty = isatty(fileno(stderr));
    prog.last_tokens = 0;
    int text_len = text ? (int)strlen(text) : 20;
    prog.estimated_frames = (int)(text_len * 1.0f);
    if (prog.estimated_frames < 24) prog.estimated_frames = 24;

    qwen3_tts_set_progress_callback(ctx, on_progress, &prog);

    qwen3_tts_params_t params;
    memset(&params, 0, sizeof(params));
    params.max_audio_tokens = 4096;
    params.temperature = 0.9f;
    params.top_p = 1.0f;
    params.top_k = 50;
    params.n_threads = 0;
    params.print_progress = 0;
    params.print_timing = 0;
    params.repetition_penalty = 1.05f;

    qwen3_tts_result_t result = qwen3_tts_synthesize_with_voice(ctx, text, audio_path, params);
    if (!result.success || result.audio_len <= 0) {
        if (prog.is_tty) fprintf(stderr, "\r\033[K");
        fprintf(stderr, "[-] Native voice cloning failed: %s\n", result.error_msg ? result.error_msg : "unknown error");
    } else {
        double audio_sec = result.sample_rate > 0 ? (double)result.audio_len / (double)result.sample_rate : 0.0;
        double wall_sec = (double)result.t_total_ms / 1000.0;
        double rtf = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;
        if (prog.is_tty) {
            fprintf(stderr, "\r\033[K[+] Synthesis complete: %.2fs audio generated in %.2fs (%.2fx real-time)\n",
                    audio_sec, wall_sec, rtf);
        } else {
            fprintf(stderr, "[+] Synthesis complete: %.2fs audio generated in %.2fs (%.2fx real-time)\n",
                    audio_sec, wall_sec, rtf);
        }
        printf("[*] Saving audio to %s (Sample Rate: %d Hz)...\n", output_path, result.sample_rate);
        if (save_wav_file(output_path, result.audio, result.audio_len, result.sample_rate) == 0) {
            printf("[+] Audio generation complete!\n");
        } else {
            fprintf(stderr, "[-] Error: Failed to write audio file %s\n", output_path);
        }
    }

    qwen3_tts_free_result(result);
    qwen3_tts_free(ctx);
}

void c_generate_style_chunked(const char** chunks, int num_chunks, const char* prompt, const char* output_path) {
    if (!chunks || num_chunks <= 0) return;

    char model_name[256];
    get_setting_string("model_name_style", model_name, sizeof(model_name), "qwen-talker-1.7b-voicedesign-BF16.gguf");

    printf("[*] [Auto-Chunk] Loading model: %s (FP16/BF16) ...\n", model_name);
    printf("[*] Target device: CPU (In-Process Native C++ / GGML FP16 SIMD)\n");

    const char* model_dir = resolve_model_dir();
    char tok_name[256];
    get_setting_string("tokenizer_model", tok_name, sizeof(tok_name), "qwen-tokenizer-12hz-BF16.gguf");

    ensure_file_exists(model_dir, tok_name);
    ensure_file_exists(model_dir, model_name);

    qwen3_tts_context_t* ctx = qwen3_tts_init();
    if (!ctx) {
        fprintf(stderr, "[-] Error: Failed to initialize native Qwen3-TTS context\n");
        return;
    }

    printf("[*] Loading model weights into memory...\n");
    fflush(stdout);

    if (!qwen3_tts_load_models_with_name(ctx, model_dir, model_name)) {
        fprintf(stderr, "[-] Error: Failed to load models '%s' from '%s'\n", model_name, model_dir);
        qwen3_tts_free(ctx);
        return;
    }
    printf("[+] Model loaded successfully. Processing %d chunks in memory...\n", num_chunks);

    size_t total_samples = 0;
    size_t audio_capacity = 24000 * 60; // 1 minute initial capacity
    float* full_audio = (float*)malloc(audio_capacity * sizeof(float));
    if (!full_audio) {
        fprintf(stderr, "[-] Error: Failed to allocate audio buffer\n");
        qwen3_tts_free(ctx);
        return;
    }
    int sample_rate = 24000;

    for (int i = 0; i < num_chunks; i++) {
        char preview[64];
        strncpy(preview, chunks[i], sizeof(preview) - 4);
        preview[sizeof(preview) - 4] = '\0';
        if (strlen(chunks[i]) > sizeof(preview) - 4) {
            strcat(preview, "...");
        }
        for (int c = 0; preview[c]; c++) {
            if (preview[c] == '\n' || preview[c] == '\r') preview[c] = ' ';
        }

        printf("\n[*] Chunk [%d/%d] (%zu chars): \"%s\"\n", i + 1, num_chunks, strlen(chunks[i]), preview);
        fflush(stdout);

        tts_progress_t prog;
        prog.is_tty = isatty(fileno(stderr));
        prog.last_tokens = 0;
        int text_len = (int)strlen(chunks[i]);
        prog.estimated_frames = (int)(text_len * 1.0f);
        if (prog.estimated_frames < 24) prog.estimated_frames = 24;

        qwen3_tts_set_progress_callback(ctx, on_progress, &prog);

        qwen3_tts_params_t params;
        memset(&params, 0, sizeof(params));
        params.max_audio_tokens = 4096;
        params.temperature = 0.9f;
        params.top_p = 1.0f;
        params.top_k = 50;
        params.n_threads = 0;
        params.print_progress = 0;
        params.print_timing = 0;
        params.repetition_penalty = 1.05f;
        params.instruction = (prompt && strlen(prompt) > 0) ? prompt : NULL;

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

        int silence_samples = (i < num_chunks - 1) ? (int)(sample_rate * 0.25f) : 0;
        size_t needed = total_samples + result.audio_len + silence_samples;
        if (needed > audio_capacity) {
            audio_capacity = needed * 2;
            float* new_buf = (float*)realloc(full_audio, audio_capacity * sizeof(float));
            if (!new_buf) {
                fprintf(stderr, "[-] Error: Failed to reallocate audio buffer\n");
                qwen3_tts_free_result(result);
                break;
            }
            full_audio = new_buf;
        }

        memcpy(full_audio + total_samples, result.audio, result.audio_len * sizeof(float));
        total_samples += result.audio_len;

        if (silence_samples > 0) {
            memset(full_audio + total_samples, 0, silence_samples * sizeof(float));
            total_samples += silence_samples;
        }

        qwen3_tts_free_result(result);
    }

    if (total_samples > 0) {
        double total_audio_sec = (double)total_samples / (double)sample_rate;
        printf("\n============================================================\n");
        printf("[+] All %d chunks synthesized successfully!\n", num_chunks);
        printf("[+] Total stitched audio duration: %.2fs (~%.1f min)\n",
               total_audio_sec, total_audio_sec / 60.0);
        printf("[*] Saving combined audio to %s (Sample Rate: %d Hz)...\n", output_path, sample_rate);
        if (save_wav_file(output_path, full_audio, (int)total_samples, sample_rate) == 0) {
            printf("[+] Audio generation complete!\n");
        } else {
            fprintf(stderr, "[-] Error: Failed to write audio file %s\n", output_path);
        }
        printf("============================================================\n");
    } else {
        fprintf(stderr, "[-] Error: No audio samples were generated.\n");
    }

    free(full_audio);
    qwen3_tts_free(ctx);
}

void c_generate_clone_chunked(const char** chunks, int num_chunks, const char* audio_path, const char* output_path) {
    if (!chunks || num_chunks <= 0) return;

    char model_name[256];
    get_setting_string("model_name_clone", model_name, sizeof(model_name), "qwen-talker-1.7b-base-BF16.gguf");

    printf("[*] [Auto-Chunk] Loading model: %s (FP16/BF16) ...\n", model_name);
    printf("[*] Target device: CPU (In-Process Native C++ / GGML FP16 SIMD)\n");

    const char* model_dir = resolve_model_dir();
    char tok_name[256];
    get_setting_string("tokenizer_model", tok_name, sizeof(tok_name), "qwen-tokenizer-12hz-BF16.gguf");

    ensure_file_exists(model_dir, tok_name);
    ensure_file_exists(model_dir, model_name);

    qwen3_tts_context_t* ctx = qwen3_tts_init();
    if (!ctx) {
        fprintf(stderr, "[-] Error: Failed to initialize native Qwen3-TTS context\n");
        return;
    }

    printf("[*] Loading model weights into memory...\n");
    fflush(stdout);

    if (!qwen3_tts_load_models_with_name(ctx, model_dir, model_name)) {
        fprintf(stderr, "[-] Error: Failed to load models '%s' from '%s'\n", model_name, model_dir);
        qwen3_tts_free(ctx);
        return;
    }
    printf("[+] Model loaded successfully. Processing %d chunks in memory...\n", num_chunks);

    printf("[*] Cloning voice from reference audio: %s\n", audio_path);
    printf("[*] Extracting speaker embedding...\n");
    fflush(stdout);

    size_t total_samples = 0;
    size_t audio_capacity = 24000 * 60;
    float* full_audio = (float*)malloc(audio_capacity * sizeof(float));
    if (!full_audio) {
        fprintf(stderr, "[-] Error: Failed to allocate audio buffer\n");
        qwen3_tts_free(ctx);
        return;
    }
    int sample_rate = 24000;

    for (int i = 0; i < num_chunks; i++) {
        char preview[64];
        strncpy(preview, chunks[i], sizeof(preview) - 4);
        preview[sizeof(preview) - 4] = '\0';
        if (strlen(chunks[i]) > sizeof(preview) - 4) {
            strcat(preview, "...");
        }
        for (int c = 0; preview[c]; c++) {
            if (preview[c] == '\n' || preview[c] == '\r') preview[c] = ' ';
        }

        printf("\n[*] Chunk [%d/%d] (%zu chars): \"%s\"\n", i + 1, num_chunks, strlen(chunks[i]), preview);
        fflush(stdout);

        tts_progress_t prog;
        prog.is_tty = isatty(fileno(stderr));
        prog.last_tokens = 0;
        int text_len = (int)strlen(chunks[i]);
        prog.estimated_frames = (int)(text_len * 1.0f);
        if (prog.estimated_frames < 24) prog.estimated_frames = 24;

        qwen3_tts_set_progress_callback(ctx, on_progress, &prog);

        qwen3_tts_params_t params;
        memset(&params, 0, sizeof(params));
        params.max_audio_tokens = 4096;
        params.temperature = 0.9f;
        params.top_p = 1.0f;
        params.top_k = 50;
        params.n_threads = 0;
        params.print_progress = 0;
        params.print_timing = 0;
        params.repetition_penalty = 1.05f;

        qwen3_tts_result_t result = qwen3_tts_synthesize_with_voice(ctx, chunks[i], audio_path, params);
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

        int silence_samples = (i < num_chunks - 1) ? (int)(sample_rate * 0.25f) : 0;
        size_t needed = total_samples + result.audio_len + silence_samples;
        if (needed > audio_capacity) {
            audio_capacity = needed * 2;
            float* new_buf = (float*)realloc(full_audio, audio_capacity * sizeof(float));
            if (!new_buf) {
                fprintf(stderr, "[-] Error: Failed to reallocate audio buffer\n");
                qwen3_tts_free_result(result);
                break;
            }
            full_audio = new_buf;
        }

        memcpy(full_audio + total_samples, result.audio, result.audio_len * sizeof(float));
        total_samples += result.audio_len;

        if (silence_samples > 0) {
            memset(full_audio + total_samples, 0, silence_samples * sizeof(float));
            total_samples += silence_samples;
        }

        qwen3_tts_free_result(result);
    }

    if (total_samples > 0) {
        double total_audio_sec = (double)total_samples / (double)sample_rate;
        printf("\n============================================================\n");
        printf("[+] All %d chunks synthesized successfully!\n", num_chunks);
        printf("[+] Total stitched audio duration: %.2fs (~%.1f min)\n",
               total_audio_sec, total_audio_sec / 60.0);
        printf("[*] Saving combined audio to %s (Sample Rate: %d Hz)...\n", output_path, sample_rate);
        if (save_wav_file(output_path, full_audio, (int)total_samples, sample_rate) == 0) {
            printf("[+] Audio generation complete!\n");
        } else {
            fprintf(stderr, "[-] Error: Failed to write audio file %s\n", output_path);
        }
        printf("============================================================\n");
    } else {
        fprintf(stderr, "[-] Error: No audio samples were generated.\n");
    }

    free(full_audio);
    qwen3_tts_free(ctx);
}
