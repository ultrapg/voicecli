#include "qwen3_tts.h"

#include <utility>

namespace qwen3_tts {

Qwen3TTS::Qwen3TTS() = default;

Qwen3TTS::~Qwen3TTS() = default;

void Qwen3TTS::set_progress_callback(tts_progress_callback_t callback) {
    progress_callback_ = std::move(callback);
}

void Qwen3TTS::set_phase_progress_callback(tts_phase_progress_callback_t callback) {
    phase_progress_callback_ = std::move(callback);
}

} // namespace qwen3_tts
