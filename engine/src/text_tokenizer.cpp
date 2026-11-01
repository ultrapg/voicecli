#include "text_tokenizer.h"
#include "tokenizer_unicode.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace qwen3_tts {

static const std::vector<std::string> GPT2_REGEX_EXPRS = {
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
};

// GPT-2 byte-to-unicode mapping
// Maps bytes 0-255 to unicode characters to avoid control characters
static const char * BYTE_TO_UNICODE[256] = {
    "Ā", "ā", "Ă", "ă", "Ą", "ą", "Ć", "ć", "Ĉ", "ĉ", "Ċ", "ċ", "Č", "č", "Ď", "ď",
    "Đ", "đ", "Ē", "ē", "Ĕ", "ĕ", "Ė", "ė", "Ę", "ę", "Ě", "ě", "Ĝ", "ĝ", "Ğ", "ğ",
    "Ġ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?",
    "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_",
    "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "ġ",
    "Ģ", "ģ", "Ĥ", "ĥ", "Ħ", "ħ", "Ĩ", "ĩ", "Ī", "ī", "Ĭ", "ĭ", "Į", "į", "İ", "ı",
    "Ĳ", "ĳ", "Ĵ", "ĵ", "Ķ", "ķ", "ĸ", "Ĺ", "ĺ", "Ļ", "ļ", "Ľ", "ľ", "Ŀ", "ŀ", "Ł",
    "ł", "¡", "¢", "£", "¤", "¥", "¦", "§", "¨", "©", "ª", "«", "¬", "Ń", "®", "¯",
    "°", "±", "²", "³", "´", "µ", "¶", "·", "¸", "¹", "º", "»", "¼", "½", "¾", "¿",
    "À", "Á", "Â", "Ã", "Ä", "Å", "Æ", "Ç", "È", "É", "Ê", "Ë", "Ì", "Í", "Î", "Ï",
    "Ð", "Ñ", "Ò", "Ó", "Ô", "Õ", "Ö", "×", "Ø", "Ù", "Ú", "Û", "Ü", "Ý", "Þ", "ß",
    "à", "á", "â", "ã", "ä", "å", "æ", "ç", "è", "é", "ê", "ë", "ì", "í", "î", "ï",
    "ð", "ñ", "ò", "ó", "ô", "õ", "ö", "÷", "ø", "ù", "ú", "û", "ü", "ý", "þ", "ÿ"
};

namespace {

// Direct O(1) lookup table for GPT-2 unicode codepoint (0..323) -> byte (0..255)
struct UnicodeToByteTable {
    int16_t cpt_to_byte[512];

    UnicodeToByteTable() {
        for (int i = 0; i < 512; ++i) {
            cpt_to_byte[i] = -1;
        }
        for (int b = 0; b < 256; ++b) {
            const unsigned char * s = reinterpret_cast<const unsigned char *>(BYTE_TO_UNICODE[b]);
            uint32_t cpt = 0;
            if ((s[0] & 0x80) == 0) {
                cpt = s[0];
            } else if ((s[0] & 0xE0) == 0xC0) {
                cpt = ((uint32_t)(s[0] & 0x1F) << 6) | (uint32_t)(s[1] & 0x3F);
            }
            if (cpt < 512) {
                cpt_to_byte[cpt] = (int16_t) b;
            }
        }
    }
};

static const UnicodeToByteTable g_unicode_to_byte_table;

inline uint32_t hash_bytes(std::string_view sv) noexcept {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : sv) {
        h ^= (uint64_t) c;
        h *= 1099511628211ULL;
    }
    h ^= h >> 32;
    uint32_t res = (uint32_t) h;
    return res == 0 ? 1u : res;
}

inline uint32_t hash_pair_key(uint64_t key) noexcept {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (uint32_t) key;
}

size_t next_pow2(size_t n) noexcept {
    size_t p = 1024;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

} // namespace

TextTokenizer::TextTokenizer() = default;

TextTokenizer::~TextTokenizer() = default;

size_t TextTokenizer::utf8_len(char c) {
    const unsigned char uc = (unsigned char) c;
    if ((uc & 0x80) == 0) return 1;
    if ((uc & 0xE0) == 0xC0) return 2;
    if ((uc & 0xF0) == 0xE0) return 3;
    if ((uc & 0xF8) == 0xF0) return 4;
    return 1;
}

std::string TextTokenizer::bytes_to_unicode(const std::string & text) {
    std::string result;
    result.reserve(text.size() * 2);
    for (unsigned char c : text) {
        result += BYTE_TO_UNICODE[c];
    }
    return result;
}

std::string TextTokenizer::unicode_to_bytes(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        const unsigned char c0 = (unsigned char) text[i];
        if ((c0 & 0x80) == 0) {
            const int16_t mapped = g_unicode_to_byte_table.cpt_to_byte[c0];
            if (mapped >= 0) {
                result.push_back((char)(uint8_t) mapped);
            } else {
                result.push_back((char) c0);
            }
            i += 1;
        } else if ((c0 & 0xE0) == 0xC0 && i + 1 < text.size()) {
            const unsigned char c1 = (unsigned char) text[i + 1];
            const uint32_t cpt = ((uint32_t)(c0 & 0x1F) << 6) | (uint32_t)(c1 & 0x3F);
            if (cpt < 512 && g_unicode_to_byte_table.cpt_to_byte[cpt] >= 0) {
                result.push_back((char)(uint8_t) g_unicode_to_byte_table.cpt_to_byte[cpt]);
            } else {
                result.append(text.data() + i, 2);
            }
            i += 2;
        } else {
            size_t len = utf8_len(text[i]);
            if (i + len > text.size()) {
                len = text.size() - i;
            }
            result.append(text.data() + i, len);
            i += len;
        }
    }
    return result;
}

std::string_view TextTokenizer::symbol_view(int32_t id) const {
    if (id < 0 || (size_t)(id + 1) >= token_offsets_.size()) {
        return {};
    }
    const uint32_t start = token_offsets_[(size_t) id];
    const uint32_t end = token_offsets_[(size_t) id + 1];
    return std::string_view(token_arena_.data() + start, end - start);
}

int32_t TextTokenizer::find_symbol_id(std::string_view text) const {
    if (vocab_slots_.empty()) {
        return -1;
    }
    const uint32_t h = hash_bytes(text);
    uint32_t slot = h & vocab_mask_;
    while (true) {
        const vocab_slot & entry = vocab_slots_[slot];
        if (entry.id < 0) {
            return -1;
        }
        if (entry.hash == h && symbol_view(entry.id) == text) {
            return entry.id;
        }
        slot = (slot + 1) & vocab_mask_;
    }
}

void TextTokenizer::rehash_vocab(size_t min_capacity) {
    const size_t cap = next_pow2(min_capacity);
    vocab_slots_.assign(cap, vocab_slot{});
    vocab_mask_ = (uint32_t)(cap - 1);
    const int32_t n_symbols = (int32_t) token_offsets_.size() - 1;
    for (int32_t id = 0; id < n_symbols; ++id) {
        const std::string_view sv = symbol_view(id);
        // Skip empty unpopulated token slots except if legitimately empty at id 0
        if (sv.empty() && id > 0) {
            continue;
        }
        const uint32_t h = hash_bytes(sv);
        uint32_t slot = h & vocab_mask_;
        while (vocab_slots_[slot].id >= 0) {
            if (vocab_slots_[slot].hash == h && symbol_view(vocab_slots_[slot].id) == sv) {
                break;
            }
            slot = (slot + 1) & vocab_mask_;
        }
        if (vocab_slots_[slot].id < 0) {
            vocab_slots_[slot].hash = h;
            vocab_slots_[slot].id = id;
        }
    }
}

int32_t TextTokenizer::intern_symbol(std::string_view text) {
    const int32_t existing = find_symbol_id(text);
    if (existing >= 0) {
        return existing;
    }
    const int32_t new_id = (int32_t) token_offsets_.size() - 1;
    token_arena_.append(text.data(), text.size());
    token_offsets_.push_back((uint32_t) token_arena_.size());

    if ((size_t)(new_id + 1) * 10 >= vocab_slots_.size() * 7) {
        rehash_vocab(vocab_slots_.size() * 2);
        return new_id;
    }

    const uint32_t h = hash_bytes(text);
    uint32_t slot = h & vocab_mask_;
    while (vocab_slots_[slot].id >= 0) {
        slot = (slot + 1) & vocab_mask_;
    }
    vocab_slots_[slot].hash = h;
    vocab_slots_[slot].id = new_id;
    return new_id;
}

bool TextTokenizer::lookup_merge(int32_t id_a, int32_t id_b,
                                 int32_t & rank_out, int32_t & merged_id_out) const {
    if (merge_slots_.empty() || id_a < 0 || id_b < 0) {
        return false;
    }
    const uint64_t key = ((uint64_t)(uint32_t) id_a << 32) | (uint64_t)(uint32_t) id_b;
    uint32_t slot = hash_pair_key(key) & merge_mask_;
    while (true) {
        const merge_slot & entry = merge_slots_[slot];
        if (entry.pair_key == ~uint64_t(0)) {
            return false;
        }
        if (entry.pair_key == key) {
            rank_out = entry.rank;
            merged_id_out = entry.merged_id;
            return true;
        }
        slot = (slot + 1) & merge_mask_;
    }
}

bool TextTokenizer::load_from_gguf(struct gguf_context * ctx) {
    if (!ctx) {
        error_msg_ = "GGUF context is null";
        return false;
    }

    loaded_ = false;
    token_arena_.clear();
    token_offsets_.clear();
    vocab_slots_.clear();
    merge_slots_.clear();
    vocab_mask_ = 0;
    merge_mask_ = 0;

    // Get vocabulary
    const int64_t tokens_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (tokens_key < 0) {
        error_msg_ = "tokenizer.ggml.tokens not found in GGUF";
        return false;
    }

    const size_t n_vocab = gguf_get_arr_n(ctx, tokens_key);
    if (n_vocab == 0) {
        error_msg_ = "Empty vocabulary";
        return false;
    }

    config_.vocab_size = (int32_t) n_vocab;

    size_t total_bytes = 0;
    for (size_t i = 0; i < n_vocab; ++i) {
        const char * token = gguf_get_arr_str(ctx, tokens_key, i);
        if (token) {
            total_bytes += std::strlen(token);
        }
    }

    token_arena_.reserve(total_bytes + 4096);
    token_offsets_.resize(n_vocab + 1);
    token_offsets_[0] = 0;

    const size_t init_vocab_cap = next_pow2((n_vocab + 2048) * 2);
    vocab_slots_.assign(init_vocab_cap, vocab_slot{});
    vocab_mask_ = (uint32_t)(init_vocab_cap - 1);

    for (size_t i = 0; i < n_vocab; ++i) {
        const char * token = gguf_get_arr_str(ctx, tokens_key, i);
        if (token) {
            const size_t len = std::strlen(token);
            token_arena_.append(token, len);
            token_offsets_[i + 1] = (uint32_t) token_arena_.size();

            const std::string_view sv(token, len);
            const uint32_t h = hash_bytes(sv);
            uint32_t slot = h & vocab_mask_;
            while (vocab_slots_[slot].id >= 0) {
                if (vocab_slots_[slot].hash == h && symbol_view(vocab_slots_[slot].id) == sv) {
                    break;
                }
                slot = (slot + 1) & vocab_mask_;
            }
            // Match std::unordered_map overwrite semantics (`vocab_[token] = i`)
            vocab_slots_[slot].hash = h;
            vocab_slots_[slot].id = (int32_t) i;
        } else {
            token_offsets_[i + 1] = (uint32_t) token_arena_.size();
        }
    }

    // Get merges and build flat (id_a, id_b) -> (rank, merged_id) hash table
    const int64_t merges_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (merges_key >= 0) {
        const size_t n_merges = gguf_get_arr_n(ctx, merges_key);
        if (n_merges > 0) {
            const size_t merge_cap = next_pow2(n_merges * 2 + 1024);
            merge_slots_.assign(merge_cap, merge_slot{});
            merge_mask_ = (uint32_t)(merge_cap - 1);

            std::string joined_scratch;
            joined_scratch.reserve(128);

            for (size_t i = 0; i < n_merges; ++i) {
                const char * merge = gguf_get_arr_str(ctx, merges_key, i);
                if (!merge) {
                    continue;
                }
                const char * space = std::strchr(merge, ' ');
                if (!space) {
                    continue;
                }
                const size_t first_len = (size_t)(space - merge);
                const char * second_ptr = space + 1;
                const size_t second_len = std::strlen(second_ptr);

                const std::string_view first_sv(merge, first_len);
                const std::string_view second_sv(second_ptr, second_len);

                const int32_t id_a = intern_symbol(first_sv);
                const int32_t id_b = intern_symbol(second_sv);

                joined_scratch.assign(first_sv.data(), first_sv.size());
                joined_scratch.append(second_sv.data(), second_sv.size());
                const int32_t merged_id = intern_symbol(joined_scratch);

                const uint64_t key = ((uint64_t)(uint32_t) id_a << 32) | (uint64_t)(uint32_t) id_b;
                uint32_t slot = hash_pair_key(key) & merge_mask_;
                while (merge_slots_[slot].pair_key != ~uint64_t(0) &&
                       merge_slots_[slot].pair_key != key) {
                    slot = (slot + 1) & merge_mask_;
                }
                merge_slots_[slot].pair_key = key;
                merge_slots_[slot].rank = (int32_t) i;
                merge_slots_[slot].merged_id = merged_id;
            }
        }
    }

    // Get special token IDs (optional, use defaults if not found)
    const int64_t bos_key = gguf_find_key(ctx, "tokenizer.ggml.bos_token_id");
    if (bos_key >= 0) {
        config_.bos_token_id = (int32_t) gguf_get_val_u32(ctx, bos_key);
    }

    const int64_t eos_key = gguf_find_key(ctx, "tokenizer.ggml.eos_token_id");
    if (eos_key >= 0) {
        config_.eos_token_id = (int32_t) gguf_get_val_u32(ctx, eos_key);
    }

    const int64_t pad_key = gguf_find_key(ctx, "tokenizer.ggml.padding_token_id");
    if (pad_key >= 0) {
        config_.pad_token_id = (int32_t) gguf_get_val_u32(ctx, pad_key);
    }

    const int64_t im_start_key = gguf_find_key(ctx, "qwen3-tts.text.im_start_id");
    if (im_start_key >= 0) {
        config_.bos_token_id = (int32_t) gguf_get_val_u32(ctx, im_start_key);
    }

    const int64_t im_end_key = gguf_find_key(ctx, "qwen3-tts.text.im_end_id");
    if (im_end_key >= 0) {
        config_.eos_token_id = (int32_t) gguf_get_val_u32(ctx, im_end_key);
    }

    auto find_real_token = [this](std::string_view text) -> int32_t {
        const int32_t id = find_symbol_id(text);
        return (id >= 0 && id < config_.vocab_size) ? id : -1;
    };

    assistant_token_id_ = find_real_token("assistant");
    if (assistant_token_id_ < 0) {
        assistant_token_id_ = find_real_token("Ġassistant");
    }

    user_token_id_ = find_real_token("user");
    if (user_token_id_ < 0) {
        user_token_id_ = find_real_token("Ġuser");
    }

    const int32_t im_start_id = find_real_token("<|im_start|>");
    if (im_start_id >= 0) {
        config_.bos_token_id = im_start_id;
    }

    const int32_t im_end_id = find_real_token("<|im_end|>");
    if (im_end_id >= 0) {
        config_.eos_token_id = im_end_id;
    }

    newline_token_id_ = find_real_token("Ċ");
    if (newline_token_id_ < 0) {
        newline_token_id_ = find_real_token("\n");
    }

    loaded_ = true;
    return true;
}

void TextTokenizer::encode_word_bpe(std::string_view word, std::vector<int32_t> & out_tokens) const {
    if (word.empty()) {
        return;
    }

    int32_t stack_ids[128];
    std::vector<int32_t> heap_ids;
    int32_t * ids = stack_ids;
    size_t count = 0;

    // Fast scan to count UTF-8 characters
    size_t scan = 0;
    size_t n_chars = 0;
    while (scan < word.size()) {
        size_t len = utf8_len(word[scan]);
        if (scan + len > word.size()) {
            len = word.size() - scan;
        }
        scan += len;
        ++n_chars;
    }

    if (n_chars > 128) {
        heap_ids.resize(n_chars);
        ids = heap_ids.data();
    }

    size_t pos = 0;
    while (pos < word.size()) {
        size_t len = utf8_len(word[pos]);
        if (pos + len > word.size()) {
            len = word.size() - pos;
        }
        ids[count++] = find_symbol_id(word.substr(pos, len));
        pos += len;
    }

    if (count > 1 && !merge_slots_.empty()) {
        while (count > 1) {
            int32_t min_rank = std::numeric_limits<int32_t>::max();
            int32_t best_a = -1;
            int32_t best_b = -1;
            int32_t best_merged = -1;

            for (size_t i = 0; i + 1 < count; ++i) {
                int32_t rank = 0;
                int32_t merged_id = -1;
                if (lookup_merge(ids[i], ids[i + 1], rank, merged_id) && rank < min_rank) {
                    min_rank = rank;
                    best_a = ids[i];
                    best_b = ids[i + 1];
                    best_merged = merged_id;
                }
            }

            if (min_rank == std::numeric_limits<int32_t>::max()) {
                break;
            }

            // In-place left-to-right compaction (out_idx <= j always holds)
            size_t out_idx = 0;
            size_t j = 0;
            while (j < count) {
                if (j + 1 < count && ids[j] == best_a && ids[j + 1] == best_b) {
                    ids[out_idx++] = best_merged;
                    j += 2;
                } else {
                    ids[out_idx++] = ids[j++];
                }
            }
            count = out_idx;
        }
    }

    for (size_t i = 0; i < count; ++i) {
        const int32_t id = ids[i];
        if (id >= 0 && id < config_.vocab_size) {
            out_tokens.push_back(id);
        } else if (id >= config_.vocab_size) {
            // Synthetic intermediate merge piece not in vocabulary: fall back to byte-level pieces
            const std::string_view sv = symbol_view(id);
            size_t p = 0;
            while (p < sv.size()) {
                size_t len = utf8_len(sv[p]);
                if (p + len > sv.size()) {
                    len = sv.size() - p;
                }
                const int32_t byte_id = find_symbol_id(sv.substr(p, len));
                if (byte_id >= 0 && byte_id < config_.vocab_size) {
                    out_tokens.push_back(byte_id);
                }
                p += len;
            }
        }
    }
}

std::vector<int32_t> TextTokenizer::encode(const std::string & text) const {
    if (!loaded_) {
        return {};
    }

    // Check for special tokens first to avoid splitting them
    if (text == "<|im_start|>") return {config_.bos_token_id};
    if (text == "<|im_end|>")   return {config_.eos_token_id};
    if (text == "<|im_pad|>")   return {config_.pad_token_id};

    std::vector<int32_t> tokens;
    tokens.reserve(text.size() / 2 + 8);

    // unicode_regex_split() returns byte-encoded GPT-style pieces already.
    std::vector<std::string> words = unicode_regex_split(text, GPT2_REGEX_EXPRS);
    if (words.empty() && !text.empty()) {
        words.push_back(bytes_to_unicode(text));
    }

    for (const auto & word : words) {
        encode_word_bpe(word, tokens);
    }

    return tokens;
}

std::vector<int32_t> TextTokenizer::encode_instruct(const std::string & instruct) const {
    if (!loaded_ || instruct.empty()) {
        return {};
    }

    // Format: <|im_start|>user\n{instruct}<|im_end|>\n
    auto text_tokens = encode(instruct);
    std::vector<int32_t> tokens;
    tokens.reserve(text_tokens.size() + 5);

    tokens.push_back(config_.bos_token_id);
    tokens.push_back(user_token_id_);
    tokens.push_back(newline_token_id_);
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    tokens.push_back(config_.eos_token_id);
    tokens.push_back(newline_token_id_);

    return tokens;
}

std::vector<int32_t> TextTokenizer::encode_for_tts(const std::string & text) const {
    if (!loaded_) {
        return {};
    }

    // Format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    auto text_tokens = encode(text);
    std::vector<int32_t> tokens;
    tokens.reserve(text_tokens.size() + 8);

    tokens.push_back(config_.bos_token_id);
    tokens.push_back(assistant_token_id_);
    tokens.push_back(newline_token_id_);
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    tokens.push_back(config_.eos_token_id);
    tokens.push_back(newline_token_id_);
    tokens.push_back(config_.bos_token_id);
    tokens.push_back(assistant_token_id_);
    tokens.push_back(newline_token_id_);

    return tokens;
}

std::vector<int32_t> TextTokenizer::encode_reference_for_tts(const std::string & text) const {
    if (!loaded_) {
        return {};
    }

    // Format: <|im_start|>assistant\n{text}<|im_end|>\n
    auto text_tokens = encode(text);
    std::vector<int32_t> tokens;
    tokens.reserve(text_tokens.size() + 5);

    tokens.push_back(config_.bos_token_id);
    tokens.push_back(assistant_token_id_);
    tokens.push_back(newline_token_id_);
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    tokens.push_back(config_.eos_token_id);
    tokens.push_back(newline_token_id_);
    return tokens;
}

std::string TextTokenizer::decode(const std::vector<int32_t> & tokens) const {
    std::string result;
    result.reserve(tokens.size() * 4);
    for (int32_t token : tokens) {
        result += decode_token(token);
    }
    return result;
}

std::string TextTokenizer::decode_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= config_.vocab_size) {
        return "";
    }
    return unicode_to_bytes(symbol_view(token_id));
}

} // namespace qwen3_tts
