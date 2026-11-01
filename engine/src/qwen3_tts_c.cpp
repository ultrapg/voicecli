#include "qwen3_tts_c.h"
#include "qwen3_tts.h"
#include "gguf_loader.h"
#include "pipeline/pipeline_internal.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

#ifdef _WIN32
#define strdup _strdup
#endif

struct file_stamp {
    std::string path;
    int64_t mtime = -1;
    int64_t size = -1;

    bool matches(const char * query_path, int64_t q_mtime, int64_t q_size) const {
        return query_path && path == query_path && mtime == q_mtime && size == q_size;
    }
};

static bool query_file_stamp(const char * path, int64_t & out_mtime, int64_t & out_size) {
    if (!path || path[0] == '\0') {
        return false;
    }
    struct stat st;
    if (stat(path, &st) != 0) {
        return false;
    }
    out_mtime = (int64_t) st.st_mtime;
    out_size = (int64_t) st.st_size;
    return true;
}

struct qwen3_tts_context {
    qwen3_tts::Qwen3TTS tts;
    qwen3_tts_progress_callback progress_callback = nullptr;
    void* user_data = nullptr;
    qwen3_tts_phase_progress_callback phase_progress_callback = nullptr;
    void* phase_user_data = nullptr;
    std::string last_error;

    // Per-context file caches so multi-chunk synthesis never re-reads/resamples/parses files per chunk
    file_stamp cached_ref_audio_stamp;
    std::vector<float> cached_ref_samples_24k;

    file_stamp cached_spk_emb_stamp;
    std::vector<float> cached_spk_emb;

    file_stamp cached_icl_stamp;
    qwen3_tts::icl_prompt cached_icl_prompt;
};

static void clear_last_error(qwen3_tts_context_t* ctx) {
    if (ctx) {
        ctx->last_error.clear();
    }
}

static void set_last_error(qwen3_tts_context_t* ctx, const std::string & message) {
    if (ctx) {
        ctx->last_error = message;
    }
}

static int32_t to_model_kind(const std::string & model_type) {
    if (model_type == "base") return QWEN3_TTS_MODEL_KIND_BASE;
    if (model_type == "custom_voice") return QWEN3_TTS_MODEL_KIND_CUSTOM_VOICE;
    if (model_type == "voice_design") return QWEN3_TTS_MODEL_KIND_VOICE_DESIGN;
    return QWEN3_TTS_MODEL_KIND_UNKNOWN;
}

static qwen3_tts::tts_params convert_params(qwen3_tts_params_t params) {
    qwen3_tts::tts_params p;
    p.max_audio_tokens = params.max_audio_tokens;
    p.temperature = params.temperature;
    p.top_p = params.top_p;
    p.top_k = params.top_k;
    p.n_threads = params.n_threads;
    p.print_progress = params.print_progress != 0;
    p.print_timing = params.print_timing != 0;
    p.repetition_penalty = params.repetition_penalty;
    p.language_id = params.language_id;
    if (params.vocoder_left_context_sec > 0.0f) {
        p.vocoder_left_context_sec = params.vocoder_left_context_sec;
    }
    if (params.instruction) {
        p.instruction = params.instruction;
    }
    if (params.speaker) {
        p.speaker = params.speaker;
    }
    return p;
}

static qwen3_tts::tts_streaming_params convert_streaming_params(qwen3_tts_streaming_params_t params) {
    qwen3_tts::tts_streaming_params p;
    p.generation = convert_params(params.generation);
    p.chunk_sec = params.chunk_sec > 0.0f ? params.chunk_sec : p.chunk_sec;
    p.left_context_sec = params.left_context_sec >= 0.0f ? params.left_context_sec : p.left_context_sec;
    p.collect_audio = params.collect_audio != 0;
    return p;
}

static qwen3_tts_result_t convert_result(const qwen3_tts::tts_result& res) {
    qwen3_tts_result_t r;
    r.audio_len = static_cast<int32_t>(res.audio.size());
    if (r.audio_len > 0) {
        r.audio = (float*)malloc((size_t)r.audio_len * sizeof(float));
        std::memcpy(r.audio, res.audio.data(), (size_t)r.audio_len * sizeof(float));
    } else {
        r.audio = nullptr;
    }
    r.sample_rate = res.sample_rate;
    r.success = res.success ? 1 : 0;
    if (!res.error_msg.empty()) {
        r.error_msg = strdup(res.error_msg.c_str());
    } else {
        r.error_msg = nullptr;
    }
    r.t_total_ms = res.t_total_ms;
    return r;
}

static qwen3_tts_result_t make_error_result(const char * message) {
    qwen3_tts_result_t res = {0};
    res.success = 0;
    res.error_msg = strdup(message ? message : "Unknown error");
    return res;
}

static const std::vector<float> * get_cached_ref_audio_24k(
    qwen3_tts_context_t * ctx,
    const char * reference_audio,
    qwen3_tts_result_t & error_result
) {
    int64_t mtime = -1;
    int64_t size = -1;
    const bool has_stamp = query_file_stamp(reference_audio, mtime, size);
    if (has_stamp && !ctx->cached_ref_samples_24k.empty() &&
        ctx->cached_ref_audio_stamp.matches(reference_audio, mtime, size)) {
        return &ctx->cached_ref_samples_24k;
    }

    std::vector<float> samples;
    int sr = 0;
    if (!qwen3_tts::load_audio_file(reference_audio, samples, sr)) {
        std::string msg = std::string("Failed to load reference audio: ") + reference_audio;
        error_result = make_error_result(msg.c_str());
        return nullptr;
    }
    if (sr != 24000) {
        std::vector<float> resampled;
        qwen3_tts::pipeline_internal::resample_linear(
            samples.data(), (int) samples.size(), sr, resampled, 24000);
        samples = std::move(resampled);
    }

    ctx->cached_ref_samples_24k = std::move(samples);
    if (has_stamp) {
        ctx->cached_ref_audio_stamp.path = reference_audio;
        ctx->cached_ref_audio_stamp.mtime = mtime;
        ctx->cached_ref_audio_stamp.size = size;
    } else {
        ctx->cached_ref_audio_stamp = {};
    }
    return &ctx->cached_ref_samples_24k;
}

static const std::vector<float> * get_cached_speaker_embedding(
    qwen3_tts_context_t * ctx,
    const char * speaker_embedding_file,
    qwen3_tts_result_t & error_result
) {
    int64_t mtime = -1;
    int64_t size = -1;
    const bool has_stamp = query_file_stamp(speaker_embedding_file, mtime, size);
    if (has_stamp && !ctx->cached_spk_emb.empty() &&
        ctx->cached_spk_emb_stamp.matches(speaker_embedding_file, mtime, size)) {
        return &ctx->cached_spk_emb;
    }

    std::vector<float> emb;
    if (!qwen3_tts::load_speaker_embedding_file(speaker_embedding_file, emb)) {
        error_result = make_error_result("Failed to load speaker embedding file");
        return nullptr;
    }
    ctx->cached_spk_emb = std::move(emb);
    if (has_stamp) {
        ctx->cached_spk_emb_stamp.path = speaker_embedding_file;
        ctx->cached_spk_emb_stamp.mtime = mtime;
        ctx->cached_spk_emb_stamp.size = size;
    } else {
        ctx->cached_spk_emb_stamp = {};
    }
    return &ctx->cached_spk_emb;
}

static const qwen3_tts::icl_prompt * get_cached_icl_prompt(
    qwen3_tts_context_t * ctx,
    const char * icl_prompt_file,
    qwen3_tts::tts_params & params,
    qwen3_tts_result_t & error_result
) {
    int64_t mtime = -1;
    int64_t size = -1;
    const bool has_stamp = query_file_stamp(icl_prompt_file, mtime, size);
    if (!has_stamp || ctx->cached_icl_prompt.speaker_embedding.empty() ||
        !ctx->cached_icl_stamp.matches(icl_prompt_file, mtime, size)) {
        qwen3_tts::icl_prompt prompt;
        if (!qwen3_tts::load_icl_prompt_file(icl_prompt_file, prompt)) {
            error_result = make_error_result("Failed to load ICL prompt file");
            return nullptr;
        }
        ctx->cached_icl_prompt = std::move(prompt);
        if (has_stamp) {
            ctx->cached_icl_stamp.path = icl_prompt_file;
            ctx->cached_icl_stamp.mtime = mtime;
            ctx->cached_icl_stamp.size = size;
        } else {
            ctx->cached_icl_stamp = {};
        }
    }

    params.reference_text = ctx->cached_icl_prompt.reference_text;
    params.reference_token_ids = ctx->cached_icl_prompt.reference_token_ids;
    params.reference_codes = ctx->cached_icl_prompt.reference_codes;
    return &ctx->cached_icl_prompt;
}

static qwen3_tts_audio_chunk_t convert_audio_chunk(const qwen3_tts::tts_audio_chunk & chunk) {
    qwen3_tts_audio_chunk_t out;
    out.samples = chunk.samples;
    out.n_samples = chunk.n_samples;
    out.sample_rate = chunk.sample_rate;
    out.start_sample = chunk.start_sample;
    out.end_sample = chunk.end_sample;
    out.start_frame = chunk.start_frame;
    out.end_frame = chunk.end_frame;
    out.start_text_byte = chunk.start_text_byte;
    out.end_text_byte = chunk.end_text_byte;
    out.text_alignment_kind = chunk.text_alignment_kind;
    out.confidence = chunk.confidence;
    return out;
}

qwen3_tts_context_t* qwen3_tts_init() {
    return new qwen3_tts_context();
}

void qwen3_tts_free(qwen3_tts_context_t* ctx) {
    delete ctx;
}

int32_t qwen3_tts_set_backend_preference(int32_t preference) {
    qwen3_tts::backend_preference native_preference = qwen3_tts::backend_preference::auto_select;
    if (preference == QWEN3_TTS_BACKEND_CPU) {
        native_preference = qwen3_tts::backend_preference::cpu;
    } else if (preference == QWEN3_TTS_BACKEND_CUDA) {
        native_preference = qwen3_tts::backend_preference::cuda;
    }
    return qwen3_tts::set_backend_preference(native_preference) ? 1 : 0;
}

int32_t qwen3_tts_set_cpu_threads(int32_t n_threads) {
    return qwen3_tts::set_cpu_thread_count(n_threads) ? 1 : 0;
}

int32_t qwen3_tts_get_cpu_threads() {
    return qwen3_tts::get_cpu_thread_count();
}

int32_t qwen3_tts_get_compiled_backend_mask() {
    return qwen3_tts::get_compiled_backend_mask();
}

char* qwen3_tts_get_active_backend_name() {
    const std::string name = qwen3_tts::get_active_backend_name();
    return strdup(name.c_str());
}

int32_t qwen3_tts_load_models(qwen3_tts_context_t* ctx, const char* model_dir) {
    return qwen3_tts_load_models_with_name(ctx, model_dir, nullptr);
}

int32_t qwen3_tts_load_models_with_name(
    qwen3_tts_context_t* ctx,
    const char* model_dir,
    const char* model_name
) {
    if (!ctx || !model_dir) return 0;
    return ctx->tts.load_models(model_dir, model_name ? model_name : "") ? 1 : 0;
}

int32_t qwen3_tts_load_speaker_encoder_with_name(
    qwen3_tts_context_t* ctx,
    const char* model_dir,
    const char* model_name
) {
    if (!ctx || !model_dir) return 0;
    return ctx->tts.load_speaker_encoder_only(model_dir, model_name ? model_name : "") ? 1 : 0;
}

int32_t qwen3_tts_load_icl_prompt_encoder(
    qwen3_tts_context_t* ctx,
    const char* model_dir
) {
    return qwen3_tts_load_icl_prompt_encoder_with_name(ctx, model_dir, nullptr);
}

int32_t qwen3_tts_load_icl_prompt_encoder_with_name(
    qwen3_tts_context_t* ctx,
    const char* model_dir,
    const char* model_name
) {
    if (!ctx || !model_dir) return 0;
    return ctx->tts.load_icl_prompt_encoder_only(model_dir, model_name ? model_name : "") ? 1 : 0;
}

char* qwen3_tts_list_speakers_from_model(
    const char* model_dir,
    const char* model_name
) {
    if (!model_dir) return nullptr;
    std::vector<std::string> speakers;
    if (!qwen3_tts::Qwen3TTS::list_speakers_from_gguf(
            model_dir, model_name ? model_name : "", speakers, nullptr)) {
        return nullptr;
    }
    std::string joined;
    for (size_t i = 0; i < speakers.size(); ++i) {
        if (i != 0) joined.push_back('\n');
        joined += speakers[i];
    }
    return strdup(joined.c_str());
}

qwen3_tts_result_t qwen3_tts_synthesize(
    qwen3_tts_context_t* ctx, 
    const char* text, 
    qwen3_tts_params_t params
) {
    if (!ctx || !text) {
        return make_error_result("Invalid context or text");
    }
    auto result = ctx->tts.synthesize(text, convert_params(params));
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_with_voice(
    qwen3_tts_context_t* ctx, 
    const char* text, 
    const char* reference_audio, 
    qwen3_tts_params_t params
) {
    if (!ctx || !text || !reference_audio) {
        return make_error_result("Invalid context, text, or reference audio");
    }
    qwen3_tts_result_t error_result = {0};
    const std::vector<float> * samples = get_cached_ref_audio_24k(ctx, reference_audio, error_result);
    if (!samples) {
        return error_result;
    }
    auto result = ctx->tts.synthesize_with_voice(
        text, samples->data(), (int32_t) samples->size(), convert_params(params));
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_with_voice_icl(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* reference_audio,
    const char* reference_text,
    qwen3_tts_params_t params
) {
    if (!ctx || !text || !reference_audio) {
        return make_error_result("Invalid context, text, or reference audio");
    }
    qwen3_tts_result_t error_result = {0};
    const std::vector<float> * samples = get_cached_ref_audio_24k(ctx, reference_audio, error_result);
    if (!samples) {
        return error_result;
    }
    qwen3_tts::tts_params native_params = convert_params(params);
    if (reference_text && reference_text[0] != '\0') {
        native_params.reference_text = reference_text;
    }
    auto result = ctx->tts.synthesize_with_voice(
        text, samples->data(), (int32_t) samples->size(), native_params);
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_with_speaker_embedding(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* speaker_embedding_file,
    qwen3_tts_params_t params
) {
    if (!ctx || !text || !speaker_embedding_file) {
        return make_error_result("Invalid context, text, or speaker embedding file");
    }

    qwen3_tts_result_t error_result = {0};
    const std::vector<float> * speaker_embedding =
        get_cached_speaker_embedding(ctx, speaker_embedding_file, error_result);
    if (!speaker_embedding) {
        return error_result;
    }

    auto result = ctx->tts.synthesize_with_speaker_embedding(text, *speaker_embedding, convert_params(params));
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_with_icl_prompt(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* icl_prompt_file,
    qwen3_tts_params_t params
) {
    if (!ctx || !text || !icl_prompt_file) {
        return make_error_result("Invalid context, text, or ICL prompt file");
    }

    qwen3_tts::tts_params native_params = convert_params(params);
    qwen3_tts_result_t error_result = {0};
    const qwen3_tts::icl_prompt * prompt =
        get_cached_icl_prompt(ctx, icl_prompt_file, native_params, error_result);
    if (!prompt) {
        return error_result;
    }

    auto result = ctx->tts.synthesize_with_speaker_embedding(text, prompt->speaker_embedding, native_params);
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
) {
    if (!ctx || !text || !callback) {
        return make_error_result("Invalid context, text, or streaming callback");
    }

    qwen3_tts::tts_audio_chunk_callback_t cb =
        [callback, user_data](const qwen3_tts::tts_audio_chunk & chunk) {
            qwen3_tts_audio_chunk_t c_chunk = convert_audio_chunk(chunk);
            return callback(&c_chunk, user_data) != 0;
        };
    auto result = ctx->tts.synthesize_streaming(text, cb, convert_streaming_params(params));
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_with_voice_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* reference_audio,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
) {
    if (!ctx || !text || !reference_audio || !callback) {
        return make_error_result("Invalid context, text, reference audio, or streaming callback");
    }

    qwen3_tts_result_t error_result = {0};
    const std::vector<float> * samples = get_cached_ref_audio_24k(ctx, reference_audio, error_result);
    if (!samples) {
        return error_result;
    }

    qwen3_tts::tts_audio_chunk_callback_t cb =
        [callback, user_data](const qwen3_tts::tts_audio_chunk & chunk) {
            qwen3_tts_audio_chunk_t c_chunk = convert_audio_chunk(chunk);
            return callback(&c_chunk, user_data) != 0;
        };
    auto result = ctx->tts.synthesize_with_voice_streaming(
        text, samples->data(), (int32_t) samples->size(), cb, convert_streaming_params(params));
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_with_voice_icl_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* reference_audio,
    const char* reference_text,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
) {
    if (!ctx || !text || !reference_audio || !callback) {
        return make_error_result("Invalid context, text, reference audio, or streaming callback");
    }

    qwen3_tts_result_t error_result = {0};
    const std::vector<float> * samples = get_cached_ref_audio_24k(ctx, reference_audio, error_result);
    if (!samples) {
        return error_result;
    }

    qwen3_tts::tts_streaming_params native_params = convert_streaming_params(params);
    if (reference_text && reference_text[0] != '\0') {
        native_params.generation.reference_text = reference_text;
    }

    qwen3_tts::tts_audio_chunk_callback_t cb =
        [callback, user_data](const qwen3_tts::tts_audio_chunk & chunk) {
            qwen3_tts_audio_chunk_t c_chunk = convert_audio_chunk(chunk);
            return callback(&c_chunk, user_data) != 0;
        };
    auto result = ctx->tts.synthesize_with_voice_streaming(
        text, samples->data(), (int32_t) samples->size(), cb, native_params);
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_with_speaker_embedding_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* speaker_embedding_file,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
) {
    if (!ctx || !text || !speaker_embedding_file || !callback) {
        return make_error_result("Invalid context, text, speaker embedding file, or streaming callback");
    }

    qwen3_tts_result_t error_result = {0};
    const std::vector<float> * speaker_embedding =
        get_cached_speaker_embedding(ctx, speaker_embedding_file, error_result);
    if (!speaker_embedding) {
        return error_result;
    }

    qwen3_tts::tts_audio_chunk_callback_t cb =
        [callback, user_data](const qwen3_tts::tts_audio_chunk & chunk) {
            qwen3_tts_audio_chunk_t c_chunk = convert_audio_chunk(chunk);
            return callback(&c_chunk, user_data) != 0;
        };
    auto result = ctx->tts.synthesize_with_speaker_embedding_streaming(
        text, *speaker_embedding, cb, convert_streaming_params(params));
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_with_icl_prompt_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* icl_prompt_file,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
) {
    if (!ctx || !text || !icl_prompt_file || !callback) {
        return make_error_result("Invalid context, text, ICL prompt file, or streaming callback");
    }

    qwen3_tts::tts_streaming_params native_params = convert_streaming_params(params);
    qwen3_tts_result_t error_result = {0};
    const qwen3_tts::icl_prompt * prompt =
        get_cached_icl_prompt(ctx, icl_prompt_file, native_params.generation, error_result);
    if (!prompt) {
        return error_result;
    }

    qwen3_tts::tts_audio_chunk_callback_t cb =
        [callback, user_data](const qwen3_tts::tts_audio_chunk & chunk) {
            qwen3_tts_audio_chunk_t c_chunk = convert_audio_chunk(chunk);
            return callback(&c_chunk, user_data) != 0;
        };
    auto result = ctx->tts.synthesize_with_speaker_embedding_streaming(
        text, prompt->speaker_embedding, cb, native_params);
    return convert_result(result);
}

int32_t qwen3_tts_extract_speaker_embedding(
    qwen3_tts_context_t* ctx,
    const char* reference_audio,
    const char* output_path
) {
    if (!ctx || !reference_audio || !output_path) {
        set_last_error(ctx, "Invalid context, reference audio, or output path");
        return 0;
    }
    clear_last_error(ctx);

    std::vector<float> speaker_embedding;
    if (!ctx->tts.extract_speaker_embedding(reference_audio, speaker_embedding, nullptr)) {
        set_last_error(ctx, ctx->tts.get_error());
        return 0;
    }

    if (!qwen3_tts::save_speaker_embedding_file(output_path, speaker_embedding)) {
        set_last_error(ctx, std::string("Failed to save speaker embedding file: ") + output_path);
        return 0;
    }
    return 1;
}

int32_t qwen3_tts_extract_icl_prompt(
    qwen3_tts_context_t* ctx,
    const char* reference_audio,
    const char* reference_text,
    const char* output_path
) {
    if (!ctx || !reference_audio || !reference_text || !output_path) {
        set_last_error(ctx, "Invalid context, reference audio, reference text, or output path");
        return 0;
    }
    clear_last_error(ctx);

    qwen3_tts::icl_prompt prompt;
    if (!ctx->tts.extract_icl_prompt(reference_audio, reference_text, prompt, nullptr)) {
        set_last_error(ctx, ctx->tts.get_error());
        return 0;
    }

    if (!qwen3_tts::save_icl_prompt_file(output_path, prompt)) {
        set_last_error(ctx, std::string("Failed to save ICL prompt file: ") + output_path);
        return 0;
    }
    return 1;
}

qwen3_tts_model_capabilities_t qwen3_tts_get_model_capabilities(qwen3_tts_context_t* ctx) {
    qwen3_tts_model_capabilities_t out = {0};
    out.model_kind = QWEN3_TTS_MODEL_KIND_UNKNOWN;
    if (!ctx) {
        return out;
    }

    const qwen3_tts::tts_model_capabilities caps = ctx->tts.get_model_capabilities();
    out.loaded = caps.loaded ? 1 : 0;
    out.supports_voice_clone = caps.supports_voice_clone ? 1 : 0;
    out.supports_named_speakers = caps.supports_named_speakers ? 1 : 0;
    out.supports_instruction = caps.supports_instruction ? 1 : 0;
    out.speaker_embedding_dim = caps.speaker_embedding_dim;
    out.speaker_count = caps.speaker_count;
    out.model_kind = to_model_kind(caps.model_type);
    return out;
}

char* qwen3_tts_get_available_speakers(qwen3_tts_context_t* ctx) {
    if (!ctx) {
        return strdup("");
    }

    const std::vector<std::string> speakers = ctx->tts.get_available_speakers();
    std::string joined;
    for (size_t i = 0; i < speakers.size(); ++i) {
        if (i != 0) {
            joined.push_back('\n');
        }
        joined += speakers[i];
    }

    return strdup(joined.c_str());
}

char* qwen3_tts_get_last_error(qwen3_tts_context_t* ctx) {
    if (!ctx) {
        return strdup("Invalid context");
    }
    const std::string & error = !ctx->last_error.empty() ? ctx->last_error : ctx->tts.get_error();
    return strdup(error.c_str());
}

void qwen3_tts_free_string(char* value) {
    if (value) {
        free(value);
    }
}

void qwen3_tts_free_result(qwen3_tts_result_t result) {
    if (result.audio) free(result.audio);
    if (result.error_msg) free(result.error_msg);
}

void qwen3_tts_set_progress_callback(
    qwen3_tts_context_t* ctx, 
    qwen3_tts_progress_callback callback, 
    void* user_data
) {
    if (!ctx) return;
    ctx->progress_callback = callback;
    ctx->user_data = user_data;
    
    if (callback) {
        ctx->tts.set_progress_callback([ctx](int tokens, int max) {
            ctx->progress_callback(tokens, max, ctx->user_data);
        });
    } else {
        ctx->tts.set_progress_callback(nullptr);
    }
}

void qwen3_tts_set_phase_progress_callback(
    qwen3_tts_context_t* ctx,
    qwen3_tts_phase_progress_callback callback,
    void* user_data
) {
    if (!ctx) return;
    ctx->phase_progress_callback = callback;
    ctx->phase_user_data = user_data;

    if (callback) {
        ctx->tts.set_phase_progress_callback([ctx](qwen3_tts::tts_phase phase, int32_t current, int32_t total) {
            ctx->phase_progress_callback((int32_t) phase, current, total, ctx->phase_user_data);
        });
    } else {
        ctx->tts.set_phase_progress_callback(nullptr);
    }
}
