#pragma once

#include "gguf.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace qwen3_tts {

// BPE tokenizer configuration
struct tokenizer_config {
    int32_t vocab_size = 151936;
    int32_t pad_token_id = 151643;
    int32_t eos_token_id = 151645;  // <|im_end|>
    int32_t bos_token_id = 151644;  // <|im_start|>
};

// Text tokenizer class (BPE-based, GPT-2 style with Qwen2 pre-tokenization)
class TextTokenizer {
public:
    TextTokenizer();
    ~TextTokenizer();

    // Load tokenizer from GGUF file
    bool load_from_gguf(struct gguf_context * ctx);

    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string & text) const;

    // Encode with TTS text format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    std::vector<int32_t> encode_for_tts(const std::string & text) const;

    // Encode reference text for ICL voice cloning:
    // <|im_start|>assistant\n{text}<|im_end|>\n
    std::vector<int32_t> encode_reference_for_tts(const std::string & text) const;

    // Encode instruction for TTS (e.g. voice steering)
    std::vector<int32_t> encode_instruct(const std::string & instruct) const;

    // Decode token IDs to text
    std::string decode(const std::vector<int32_t> & tokens) const;

    // Decode single token
    std::string decode_token(int32_t token_id) const;

    // Get configuration
    const tokenizer_config & get_config() const { return config_; }

    // Get error message
    const std::string & get_error() const { return error_msg_; }

    // Check if loaded
    bool is_loaded() const { return loaded_; }

    // Get special token IDs
    int32_t bos_token_id() const { return config_.bos_token_id; }
    int32_t eos_token_id() const { return config_.eos_token_id; }
    int32_t pad_token_id() const { return config_.pad_token_id; }

private:
    struct vocab_slot {
        uint32_t hash = 0;
        int32_t id = -1;
    };

    struct merge_slot {
        uint64_t pair_key = ~uint64_t(0);
        int32_t rank = -1;
        int32_t merged_id = -1;
    };

    tokenizer_config config_;
    std::string error_msg_;
    bool loaded_ = false;

    // Contiguous arena storing all real (0..vocab_size-1) and synthetic (>= vocab_size) BPE symbols
    std::string token_arena_;
    std::vector<uint32_t> token_offsets_;

    // Open-addressed string -> symbol ID hash table
    std::vector<vocab_slot> vocab_slots_;
    uint32_t vocab_mask_ = 0;

    // Open-addressed (id_a, id_b) -> (rank, merged_id) hash table
    std::vector<merge_slot> merge_slots_;
    uint32_t merge_mask_ = 0;

    // Special tokens for "assistant", "user" and newline
    int32_t assistant_token_id_ = 77091;
    int32_t user_token_id_ = 872;
    int32_t newline_token_id_ = 198;  // '\n' encoded

    // Helper: convert bytes to unicode (GPT-2 style byte encoding)
    static std::string bytes_to_unicode(const std::string & text);
    static std::string unicode_to_bytes(std::string_view text);

    // Helper: get UTF-8 character length
    static size_t utf8_len(char c);

    std::string_view symbol_view(int32_t id) const;
    int32_t find_symbol_id(std::string_view text) const;
    int32_t intern_symbol(std::string_view text);
    void rehash_vocab(size_t min_capacity);
    bool lookup_merge(int32_t id_a, int32_t id_b, int32_t & rank_out, int32_t & merged_id_out) const;
    void encode_word_bpe(std::string_view word, std::vector<int32_t> & out_tokens) const;
};

} // namespace qwen3_tts
