#include "qwen3_tts.h"
#include "gguf_loader.h"
#include "pipeline/pipeline_internal.h"
#include "transformer/transformer_internal.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace qwen3_tts {
namespace fs = std::filesystem;
using pipeline_internal::configure_ggml_logging_once;
using pipeline_internal::env_flag_enabled;
using pipeline_internal::get_time_ms;
using pipeline_internal::log_memory_usage;

namespace {

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    return value;
}

bool has_speaker_encoder_tensors_in_ctx(const struct gguf_context * ctx) {
    if (!ctx) {
        return false;
    }
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        if (name && strncmp(name, "spk_enc.", 8) == 0) {
            return true;
        }
    }
    return false;
}

bool has_speaker_encoder_tensors(const std::string & model_path) {
    GGUFLoader loader;
    if (!loader.open(model_path)) {
        return false;
    }
    return has_speaker_encoder_tensors_in_ctx(loader.get_ctx());
}

std::string choose_tts_model_path(const std::string & model_dir,
                                  const std::string & model_name,
                                  std::vector<fs::path> & tts_candidates) {
    std::string tts_model_path;
    tts_candidates.clear();

    if (fs::exists(model_dir) && fs::is_directory(model_dir)) {
        for (const auto & entry : fs::directory_iterator(model_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string filename = entry.path().filename().string();
            std::string filename_lower = to_lower(filename);
            std::string ext = to_lower(entry.path().extension().string());

            if (ext == ".gguf" &&
                filename_lower.find("tokenizer") == std::string::npos &&
                (filename_lower.find("qwen-talker") != std::string::npos ||
                 filename_lower.find("full") != std::string::npos)) {
                tts_candidates.push_back(entry.path());
                if (!model_name.empty()) {
                    if (filename.find(model_name) != std::string::npos) {
                        tts_model_path = entry.path().string();
                    }
                } else if (tts_model_path.empty() || filename.find("0.6b") != std::string::npos) {
                    tts_model_path = entry.path().string();
                }
            }
        }
    }

    if (tts_model_path.empty()) {
        if (!model_name.empty()) {
            tts_model_path = model_dir + "/" + model_name;
        } else {
            tts_model_path = model_dir + "/qwen-talker-0.6b-base-Q8_0.gguf";
        }
    }

    return tts_model_path;
}

std::string choose_tokenizer_model_path(const std::string & model_dir,
                                        const std::string & tts_model_path) {
    std::vector<fs::path> tokenizer_candidates;
    if (fs::exists(model_dir) && fs::is_directory(model_dir)) {
        for (const auto & entry : fs::directory_iterator(model_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string filename_lower = to_lower(entry.path().filename().string());
            std::string ext = to_lower(entry.path().extension().string());
            if (ext == ".gguf" && filename_lower.find("tokenizer") != std::string::npos) {
                tokenizer_candidates.push_back(entry.path());
            }
        }
    }

    const std::vector<std::string> preferred_exact = {
        "qwen-tokenizer-12hz-f32.gguf",
        "qwen-tokenizer-12hz-f16.gguf",
        "qwen-tokenizer-12hz-bf16.gguf",
        "qwen-tokenizer-12hz-q8_0.gguf",
    };
    for (const auto & preferred : preferred_exact) {
        for (const auto & path : tokenizer_candidates) {
            if (to_lower(path.filename().string()) == preferred) {
                return path.string();
            }
        }
    }

    const bool prefer_qwen_talker =
        to_lower(fs::path(tts_model_path).filename().string()).find("qwen-talker") != std::string::npos;
    if (prefer_qwen_talker) {
        for (const auto & path : tokenizer_candidates) {
            if (to_lower(path.filename().string()).find("qwen-tokenizer-12hz") != std::string::npos) {
                return path.string();
            }
        }
    }

    if (!tokenizer_candidates.empty()) {
        return tokenizer_candidates.front().string();
    }
    return model_dir + "/qwen-tokenizer-12hz-Q8_0.gguf";
}

std::string find_speaker_encoder_model(const std::string & tts_model_path,
                                       const std::vector<fs::path> & tts_candidates,
                                       int32_t expected_embedding_dim,
                                       int self_has_spk_enc_hint = -1) {
    const bool self_has = (self_has_spk_enc_hint >= 0)
        ? (self_has_spk_enc_hint != 0)
        : has_speaker_encoder_tensors(tts_model_path);
    if (self_has) {
        return tts_model_path;
    }

    const std::string size_marker = expected_embedding_dim >= 2048 ? "1.7b" : "0.6b";
    int best_score = -1;
    std::string best_path;

    for (const auto & candidate : tts_candidates) {
        const std::string filename = to_lower(candidate.filename().string());
        if (filename.find("tokenizer") != std::string::npos ||
            filename.find(size_marker) == std::string::npos ||
            filename.find("base") == std::string::npos) {
            continue;
        }

        const std::string path = candidate.string();
        if (path == tts_model_path) {
            continue;
        }
        if (!has_speaker_encoder_tensors(path)) {
            continue;
        }

        int score = 0;
        if (filename.find("qwen-talker") != std::string::npos) score += 40;
        if (filename.find("q8_0") != std::string::npos) score += 30;
        if (filename.find("bf16") != std::string::npos) score += 20;
        if (filename.find("f16") != std::string::npos) score += 10;
        if (score > best_score) {
            best_score = score;
            best_path = path;
        }
    }

    return best_path;
}

} // namespace

bool Qwen3TTS::list_speakers_from_gguf(const std::string & model_dir,
                                       const std::string & model_name,
                                       std::vector<std::string> & speakers,
                                       std::string * error_msg) {
    configure_ggml_logging_once();
    speakers.clear();

    std::vector<fs::path> tts_candidates;
    const std::string tts_model_path = choose_tts_model_path(model_dir, model_name, tts_candidates);

    GGUFLoader loader;
    if (!loader.open(tts_model_path)) {
        if (error_msg) {
            *error_msg = "Failed to open TTS model: " + loader.get_error();
        }
        return false;
    }

    struct gguf_context * ctx = loader.get_ctx();
    std::map<std::string, int32_t> speaker_id_map;

    const int64_t spk_count_idx = gguf_find_key(ctx, "qwen3-tts.speaker.count");
    int32_t spk_count = 0;
    if (spk_count_idx >= 0) {
        const enum gguf_type spk_count_type = gguf_get_kv_type(ctx, spk_count_idx);
        if (spk_count_type == GGUF_TYPE_UINT32) {
            spk_count = (int32_t) gguf_get_val_u32(ctx, spk_count_idx);
        } else if (spk_count_type == GGUF_TYPE_INT32) {
            spk_count = gguf_get_val_i32(ctx, spk_count_idx);
        }
    }
    if (spk_count > 0) {
        for (int32_t i = 0; i < spk_count; ++i) {
            char name_key[64];
            char id_key[64];
            snprintf(name_key, sizeof(name_key), "qwen3-tts.speaker.%d.name", i);
            snprintf(id_key, sizeof(id_key), "qwen3-tts.speaker.%d.id", i);
            const int64_t name_idx = gguf_find_key(ctx, name_key);
            const int64_t id_idx = gguf_find_key(ctx, id_key);
            if (name_idx < 0 || id_idx < 0 || gguf_get_kv_type(ctx, name_idx) != GGUF_TYPE_STRING) {
                continue;
            }
            const char * raw_name = gguf_get_val_str(ctx, name_idx);
            if (!raw_name || raw_name[0] == '\0') {
                continue;
            }

            int32_t spk_id = -1;
            const enum gguf_type id_type = gguf_get_kv_type(ctx, id_idx);
            if (id_type == GGUF_TYPE_UINT32) {
                spk_id = (int32_t) gguf_get_val_u32(ctx, id_idx);
            } else if (id_type == GGUF_TYPE_INT32) {
                spk_id = gguf_get_val_i32(ctx, id_idx);
            } else {
                continue;
            }
            if (spk_id >= 0) {
                speaker_id_map[transformer_internal::normalize_speaker_name(raw_name)] = spk_id;
            }
        }
    }

    if (speaker_id_map.empty()) {
        const int64_t names_idx = gguf_find_key(ctx, "qwen3-tts.codec.speaker_names");
        const int64_t ids_idx = gguf_find_key(ctx, "qwen3-tts.codec.speaker_ids");
        if (names_idx >= 0 && ids_idx >= 0 &&
            gguf_get_kv_type(ctx, names_idx) == GGUF_TYPE_ARRAY &&
            gguf_get_kv_type(ctx, ids_idx) == GGUF_TYPE_ARRAY &&
            gguf_get_arr_type(ctx, names_idx) == GGUF_TYPE_STRING) {
            const size_t n_names = gguf_get_arr_n(ctx, names_idx);
            const size_t n_ids = gguf_get_arr_n(ctx, ids_idx);
            const size_t n = std::min(n_names, n_ids);
            const enum gguf_type id_arr_type = gguf_get_arr_type(ctx, ids_idx);
            const void * id_data = gguf_get_arr_data(ctx, ids_idx);

            for (size_t i = 0; i < n; ++i) {
                const char * raw_name = gguf_get_arr_str(ctx, names_idx, i);
                if (!raw_name || raw_name[0] == '\0') {
                    continue;
                }
                int32_t spk_id = -1;
                if (id_arr_type == GGUF_TYPE_INT32 && id_data) {
                    spk_id = ((const int32_t *) id_data)[i];
                } else if (id_arr_type == GGUF_TYPE_UINT32 && id_data) {
                    spk_id = (int32_t) ((const uint32_t *) id_data)[i];
                } else {
                    continue;
                }
                if (spk_id >= 0) {
                    speaker_id_map[transformer_internal::normalize_speaker_name(raw_name)] = spk_id;
                }
            }
        }
    }

    speakers.reserve(speaker_id_map.size());
    for (const auto & kv : speaker_id_map) {
        speakers.push_back(kv.first);
    }
    return true;
}

bool Qwen3TTS::load_models(const std::string & model_dir,
                           const std::string & model_name,
                           const std::string & tokenizer_model_override) {
    configure_ggml_logging_once();
    const bool debug_log = env_flag_enabled("VOICECLI_DEBUG");

    int64_t t_start = get_time_ms();
    log_memory_usage("load/start");

    transformer_.unload_model();
    audio_encoder_.unload_model();
    speech_encoder_.unload_model();
    audio_decoder_.unload_model();
    streaming_audio_decoder_.unload_model();
    voice_prompt_cache_ = {};
    text_tokenizer_loaded_ = false;
    encoder_loaded_ = false;
    transformer_loaded_ = false;
    speech_encoder_loaded_ = false;
    speech_encoder_cuda_enabled_ = false;
    decoder_loaded_ = false;
    streaming_decoder_loaded_ = false;

    std::vector<fs::path> tts_candidates;
    const std::string tts_model_path = choose_tts_model_path(model_dir, model_name, tts_candidates);
    const std::string tokenizer_model_path = tokenizer_model_override.empty()
        ? choose_tokenizer_model_path(model_dir, tts_model_path)
        : tokenizer_model_override;
    if (!fs::exists(tokenizer_model_path)) {
        error_msg_ = "Tokenizer/codec model not found: " + tokenizer_model_path;
        return false;
    }

    if (debug_log) {
        fprintf(stderr, "  TTS model path:       %s\n", tts_model_path.c_str());
        fprintf(stderr, "  Tokenizer model path: %s\n", tokenizer_model_path.c_str());
    }

    tts_model_path_ = tts_model_path;
    speaker_encoder_model_path_.clear();
    tokenizer_model_path_ = tokenizer_model_path;
    decoder_model_path_ = tokenizer_model_path;

    const char * low_mem_env = std::getenv("QWEN3_TTS_LOW_MEM");
    low_mem_mode_ = low_mem_env && low_mem_env[0] != '\0' && low_mem_env[0] != '0';
    if (low_mem_mode_ && debug_log) {
        fprintf(stderr, "  Low-memory mode enabled (lazy decoder + component unloads)\n");
    }

    if (debug_log) {
        fprintf(stderr, "Loading TTS model from %s...\n", tts_model_path.c_str());
    }

    bool self_has_spk_enc = false;
    {
        // Parse GGUF metadata once and share across TextTokenizer, TTSTransformer, and spk_enc check
        GGUFLoader loader;
        if (!loader.open(tts_model_path)) {
            error_msg_ = "Failed to open TTS model: " + loader.get_error();
            return false;
        }

        int64_t t_tokenizer_start = get_time_ms();
        if (!tokenizer_.load_from_gguf(loader.get_ctx())) {
            error_msg_ = "Failed to load text tokenizer: " + tokenizer_.get_error();
            return false;
        }
        text_tokenizer_loaded_ = true;
        if (debug_log) {
            fprintf(stderr, "  Text tokenizer loaded: vocab_size=%d (%lld ms)\n",
                    tokenizer_.get_config().vocab_size,
                    (long long) (get_time_ms() - t_tokenizer_start));
            log_memory_usage("load/after-tokenizer");
            fprintf(stderr, "  Speaker encoder: deferred (lazy load)\n");
        }

        self_has_spk_enc = has_speaker_encoder_tensors_in_ctx(loader.get_ctx());

        int64_t t_transformer_start = get_time_ms();
        if (!transformer_.load_model_from_gguf(tts_model_path, loader.get_ctx())) {
            error_msg_ = "Failed to load TTS transformer: " + transformer_.get_error();
            return false;
        }
        transformer_loaded_ = true;
        if (debug_log) {
            fprintf(stderr, "  TTS transformer loaded: hidden_size=%d, n_layers=%d (%lld ms)\n",
                    transformer_.get_config().hidden_size, transformer_.get_config().n_layers,
                    (long long) (get_time_ms() - t_transformer_start));
        }
    }

    speaker_encoder_model_path_ = find_speaker_encoder_model(
        tts_model_path, tts_candidates, transformer_.get_config().hidden_size, self_has_spk_enc ? 1 : 0);
    if (debug_log && !speaker_encoder_model_path_.empty() && speaker_encoder_model_path_ != tts_model_path) {
        fprintf(stderr, "  Speaker encoder provider: %s\n", speaker_encoder_model_path_.c_str());
    }
    log_memory_usage("load/after-transformer");

    if (!low_mem_mode_) {
        if (debug_log) {
            fprintf(stderr, "Loading vocoder from %s...\n", tokenizer_model_path.c_str());
        }
        int64_t t_decoder_start = get_time_ms();
        if (!audio_decoder_.load_model(tokenizer_model_path)) {
            error_msg_ = "Failed to load vocoder: " + audio_decoder_.get_error();
            return false;
        }
        decoder_loaded_ = true;
        if (debug_log) {
            fprintf(stderr, "  Vocoder loaded: sample_rate=%d, n_codebooks=%d (%lld ms)\n",
                    audio_decoder_.get_config().sample_rate, audio_decoder_.get_config().n_codebooks,
                    (long long) (get_time_ms() - t_decoder_start));
        }
        log_memory_usage("load/after-vocoder");
    } else if (debug_log) {
        fprintf(stderr, "  Vocoder: deferred (lazy load)\n");
    }

    models_loaded_ = true;

    int64_t t_end = get_time_ms();
    if (debug_log) {
        fprintf(stderr, "All models loaded in %lld ms\n", (long long) (t_end - t_start));
    }
    log_memory_usage("load/end");

    return true;
}

bool Qwen3TTS::load_speaker_encoder_only(const std::string & model_dir, const std::string & model_name) {
    configure_ggml_logging_once();
    const bool debug_log = env_flag_enabled("VOICECLI_DEBUG");

    int64_t t_start = get_time_ms();
    log_memory_usage("speaker-load/start");

    transformer_.unload_model();
    audio_encoder_.unload_model();
    speech_encoder_.unload_model();
    audio_decoder_.unload_model();
    streaming_audio_decoder_.unload_model();
    voice_prompt_cache_ = {};
    models_loaded_ = false;
    encoder_loaded_ = false;
    transformer_loaded_ = false;
    speech_encoder_loaded_ = false;
    speech_encoder_cuda_enabled_ = false;
    decoder_loaded_ = false;
    streaming_decoder_loaded_ = false;

    std::vector<fs::path> tts_candidates;
    tts_model_path_ = choose_tts_model_path(model_dir, model_name, tts_candidates);
    int32_t expected_dim = 1024;
    bool self_has_spk_enc = false;
    {
        GGUFLoader loader;
        if (loader.open(tts_model_path_)) {
            expected_dim = loader.get_u32("qwen3-tts.talker.embedding_length",
                                          loader.get_u32("qwen3-tts.embedding_length", 1024));
            self_has_spk_enc = has_speaker_encoder_tensors_in_ctx(loader.get_ctx());
        }
    }
    speaker_encoder_model_path_ = find_speaker_encoder_model(
        tts_model_path_, tts_candidates, expected_dim, self_has_spk_enc ? 1 : 0);

    if (speaker_encoder_model_path_.empty()) {
        error_msg_ = "No speaker encoder tensors found for model: " + tts_model_path_;
        return false;
    }

    if (debug_log) {
        fprintf(stderr, "  TTS model path:              %s\n", tts_model_path_.c_str());
        fprintf(stderr, "  Speaker encoder model path:  %s\n", speaker_encoder_model_path_.c_str());
    }

    int64_t t_encoder_load_start = get_time_ms();
    if (!audio_encoder_.load_model(speaker_encoder_model_path_)) {
        error_msg_ = "Failed to load speaker encoder: " + audio_encoder_.get_error();
        return false;
    }
    encoder_loaded_ = true;

    if (debug_log) {
        fprintf(stderr, "  Speaker encoder loaded: embedding_dim=%d (%lld ms)\n",
                audio_encoder_.get_config().embedding_dim,
                (long long) (get_time_ms() - t_encoder_load_start));
        fprintf(stderr, "Speaker encoder-only load complete in %lld ms\n",
                (long long) (get_time_ms() - t_start));
    }
    log_memory_usage("speaker-load/end");

    return true;
}

bool Qwen3TTS::load_icl_prompt_encoder_only(const std::string & model_dir,
                                            const std::string & model_name,
                                            const std::string & tokenizer_model_override) {
    configure_ggml_logging_once();
    const bool debug_log = env_flag_enabled("VOICECLI_DEBUG");

    const int64_t t_start = get_time_ms();
    log_memory_usage("icl-load/start");

    transformer_.unload_model();
    audio_encoder_.unload_model();
    speech_encoder_.unload_model();
    audio_decoder_.unload_model();
    streaming_audio_decoder_.unload_model();
    voice_prompt_cache_ = {};
    models_loaded_ = false;
    text_tokenizer_loaded_ = false;
    encoder_loaded_ = false;
    transformer_loaded_ = false;
    speech_encoder_loaded_ = false;
    speech_encoder_cuda_enabled_ = false;
    decoder_loaded_ = false;
    streaming_decoder_loaded_ = false;

    std::vector<fs::path> tts_candidates;
    tts_model_path_ = choose_tts_model_path(model_dir, model_name, tts_candidates);
    tokenizer_model_path_ = tokenizer_model_override.empty()
        ? choose_tokenizer_model_path(model_dir, tts_model_path_)
        : tokenizer_model_override;
    if (!fs::exists(tokenizer_model_path_)) {
        error_msg_ = "Tokenizer/codec model not found: " + tokenizer_model_path_;
        return false;
    }
    decoder_model_path_ = tokenizer_model_path_;

    int32_t expected_dim = 1024;
    bool self_has_spk_enc = false;
    {
        GGUFLoader loader;
        if (loader.open(tts_model_path_)) {
            expected_dim = loader.get_u32("qwen3-tts.talker.embedding_length",
                                          loader.get_u32("qwen3-tts.embedding_length", 1024));
            self_has_spk_enc = has_speaker_encoder_tensors_in_ctx(loader.get_ctx());
        }
    }
    speaker_encoder_model_path_ = find_speaker_encoder_model(
        tts_model_path_, tts_candidates, expected_dim, self_has_spk_enc ? 1 : 0);
    if (speaker_encoder_model_path_.empty()) {
        error_msg_ = "No speaker encoder tensors found for model: " + tts_model_path_;
        return false;
    }

    if (debug_log) {
        fprintf(stderr, "  TTS model path:              %s\n", tts_model_path_.c_str());
        fprintf(stderr, "  Tokenizer model path:        %s\n", tokenizer_model_path_.c_str());
        fprintf(stderr, "  Speaker encoder model path:  %s\n", speaker_encoder_model_path_.c_str());
        fprintf(stderr, "  Text tokenizer: deferred (loaded after speaker encode)\n");
    }

    const int64_t t_encoder_start = get_time_ms();
    if (!audio_encoder_.load_model(speaker_encoder_model_path_)) {
        error_msg_ = "Failed to load speaker encoder: " + audio_encoder_.get_error();
        return false;
    }
    encoder_loaded_ = true;
    if (debug_log) {
        fprintf(stderr, "  Speaker encoder loaded: embedding_dim=%d (%lld ms)\n",
                audio_encoder_.get_config().embedding_dim,
                (long long) (get_time_ms() - t_encoder_start));
        fprintf(stderr, "  Speech tokenizer encoder: deferred (loaded after speaker encode)\n");
        fprintf(stderr, "ICL prompt encoder-only load complete in %lld ms\n",
                (long long) (get_time_ms() - t_start));
    }
    log_memory_usage("icl-load/end");
    return true;
}

std::vector<std::string> Qwen3TTS::get_available_speakers() const {
    std::vector<std::string> speakers;
    const auto & speaker_map = transformer_.get_config().speaker_id_map;
    speakers.reserve(speaker_map.size());
    for (const auto & it : speaker_map) {
        speakers.push_back(it.first);
    }
    return speakers;
}

tts_model_capabilities Qwen3TTS::get_model_capabilities() const {
    tts_model_capabilities caps;
    caps.loaded = models_loaded_;
    if (!models_loaded_) {
        return caps;
    }

    const auto & cfg = transformer_.get_config();
    caps.model_type = cfg.tts_model_type;
    caps.speaker_embedding_dim = cfg.hidden_size;
    caps.speaker_count = (int32_t) cfg.speaker_id_map.size();
    caps.supports_named_speakers = caps.speaker_count > 0;
    caps.supports_voice_clone = !speaker_encoder_model_path_.empty();

    if (cfg.has_supports_instruction) {
        caps.supports_instruction = cfg.supports_instruction;
    } else if (cfg.tts_model_type == "custom_voice") {
        caps.supports_instruction = cfg.hidden_size >= 2048;
    } else if (cfg.tts_model_type == "voice_design") {
        caps.supports_instruction = true;
    } else {
        caps.supports_instruction = false;
    }

    return caps;
}

} // namespace qwen3_tts
