#include "tokenizer_unicode.h"
#include "tokenizer_unicode_data.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static void append_cpt_utf8(std::string & out, uint32_t cpt) {
    if (cpt <= 0x7f) {
        out.push_back((char) cpt);
    } else if (cpt <= 0x7ff) {
        out.push_back((char)(0xc0 | ((cpt >> 6) & 0x1f)));
        out.push_back((char)(0x80 | (cpt & 0x3f)));
    } else if (cpt <= 0xffff) {
        out.push_back((char)(0xe0 | ((cpt >> 12) & 0x0f)));
        out.push_back((char)(0x80 | ((cpt >> 6) & 0x3f)));
        out.push_back((char)(0x80 | (cpt & 0x3f)));
    } else if (cpt <= 0x10ffff) {
        out.push_back((char)(0xf0 | ((cpt >> 18) & 0x07)));
        out.push_back((char)(0x80 | ((cpt >> 12) & 0x3f)));
        out.push_back((char)(0x80 | ((cpt >> 6) & 0x3f)));
        out.push_back((char)(0x80 | (cpt & 0x3f)));
    } else {
        throw std::invalid_argument("invalid codepoint");
    }
}

uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}

static std::vector<unicode_cpt_flags> unicode_cpt_flags_array() {
    std::vector<unicode_cpt_flags> cpt_flags(MAX_CODEPOINTS, unicode_cpt_flags::UNDEFINED);

    assert (unicode_ranges_flags.begin()[0].first == 0);
    assert (unicode_ranges_flags.begin()[unicode_ranges_flags.size()-1].first == MAX_CODEPOINTS);
    for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
        const auto range_ini = unicode_ranges_flags.begin()[i-1];  // codepoint_ini, flags
        const auto range_end = unicode_ranges_flags.begin()[i];    // codepoint_end, flags
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    for (auto cpt : unicode_set_whitespace) {
        cpt_flags[cpt].is_whitespace = true;
    }

    for (auto p : unicode_map_lowercase) {
        cpt_flags[p.second].is_lowercase = true;
    }

    for (auto p : unicode_map_uppercase) {
        cpt_flags[p.second].is_uppercase = true;
    }

    for (auto &range : unicode_ranges_nfd) {  // start, last, nfd
        cpt_flags[range.nfd].is_nfd = true;
    }

    return cpt_flags;
}

static const std::array<std::string, 256> & unicode_byte_to_utf8_table() {
    static const std::array<std::string, 256> table = []() {
        std::array<std::string, 256> arr;
        std::array<bool, 256> assigned{};
        for (int ch = 0x21; ch <= 0x7E; ++ch) {
            arr[(size_t) ch] = unicode_cpt_to_utf8((uint32_t) ch);
            assigned[(size_t) ch] = true;
        }
        for (int ch = 0xA1; ch <= 0xAC; ++ch) {
            arr[(size_t) ch] = unicode_cpt_to_utf8((uint32_t) ch);
            assigned[(size_t) ch] = true;
        }
        for (int ch = 0xAE; ch <= 0xFF; ++ch) {
            arr[(size_t) ch] = unicode_cpt_to_utf8((uint32_t) ch);
            assigned[(size_t) ch] = true;
        }
        int n = 0;
        for (int ch = 0; ch < 256; ++ch) {
            if (!assigned[(size_t) ch]) {
                arr[(size_t) ch] = unicode_cpt_to_utf8((uint32_t)(256 + n));
                ++n;
            }
        }
        return arr;
    }();
    return table;
}

static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    const auto & table = unicode_byte_to_utf8_table();
    map.reserve(256);
    for (int ch = 0; ch < 256; ++ch) {
        map[table[(size_t) ch]] = (uint8_t) ch;
    }
    return map;
}

// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static std::vector<size_t> unicode_regex_split_custom_gpt2_cpts(const std::vector<uint32_t> & cpts, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size() * 2 + 8);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            if (cpt == '\'' && pos + 1 < offset_end) {
                uint32_t cpt_next = _get_cpt(pos + 1);
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos + 2);
                    continue;
                }
                if (pos + 2 < offset_end) {
                    uint32_t cpt_next_next = _get_cpt(pos + 2);
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos + 3);
                        continue;
                    }
                }
            }

            auto flags2 = (cpt == ' ' ? _get_flags(pos + 1) : flags);
            if (flags2.is_letter) {
                pos += (cpt == ' ');
                while (flags2.is_letter) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }
            if (flags2.is_number) {
                pos += (cpt == ' ');
                while (flags2.is_number) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            while (_get_flags(pos + num_whitespaces).is_whitespace) {
                num_whitespaces++;
            }

            if (num_whitespaces > 1 && _get_cpt(pos + num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

static std::vector<size_t> unicode_regex_split_custom_llama3_cpts(const std::vector<uint32_t> & cpts, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size() * 2 + 8);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            if (cpt == '\'' && pos + 1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos + 1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos + 2);
                    continue;
                }
                if (pos + 2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos + 2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos + 3);
                        continue;
                    }
                }
            }

            if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
                if (flags.is_letter || _get_flags(pos + 1).is_letter) {
                    pos++;
                    while (_get_flags(pos).is_letter) {
                        pos++;
                    }
                    _add_token(pos);
                    continue;
                }
            }

            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            auto flags2 = (cpt == ' ' ? _get_flags(pos + 1) : flags);
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos + num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos + num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            if (num_whitespaces > 1 && _get_cpt(pos + num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

template <typename CharT>
static std::vector<size_t> unicode_regex_split_stl(const std::basic_string<CharT> & text, const std::basic_string<CharT> & regex, const std::vector<size_t> & offsets) {
    using BidirIt = typename std::basic_string<CharT>::const_iterator;
#ifdef _MSC_VER
    constexpr auto regex_flags = std::regex_constants::ECMAScript;
#else
    constexpr auto regex_flags = std::regex_constants::optimize | std::regex_constants::nosubs;
#endif
    std::basic_regex<CharT> expr(regex, regex_flags);
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());
    size_t start = 0;
    for (auto offset : offsets) {
        std::regex_iterator<BidirIt> it(text.begin() + start, text.begin() + start + offset, expr);
        std::regex_iterator<BidirIt> end;

        int64_t start_idx = 0;
        while (it != end) {
            std::match_results<BidirIt> match = *it;
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

static std::vector<size_t> unicode_regex_split_custom_kimi_k2_cpts(const std::vector<uint32_t> & cpts, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            if (unicode_cpt_is_han(cpt)) {
                while (unicode_cpt_is_han(_get_cpt(pos))) {
                    pos++;
                }
                _add_token(pos);
                continue;
            }

            bool is_letter_pattern = (flags.is_letter && !unicode_cpt_is_han(cpt)) ||
                                     (!(cpt == '\r' || cpt == '\n' || flags.is_letter || flags.is_number) &&
                                      _get_flags(pos + 1).is_letter && !unicode_cpt_is_han(_get_cpt(pos + 1)));

            if (is_letter_pattern) {
                bool has_leading_char = false;
                if (!(cpt == '\r' || cpt == '\n' || flags.is_letter || flags.is_number)) {
                    has_leading_char = true;
                    pos++;
                }

                bool has_letters = false;
                while (_get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos))) {
                    has_letters = true;
                    pos++;
                }

                if (has_letters || (!has_leading_char && _get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos)))) {
                    if (!has_letters) pos++;

                    while (_get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos))) {
                        pos++;
                    }

                    if (_get_cpt(pos) == '\'' && pos + 1 < offset_end) {
                        uint32_t cpt_next = unicode_tolower(_get_cpt(pos + 1));
                        if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                            pos += 2;
                        } else if (pos + 2 < offset_end) {
                            uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos + 2));
                            if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                                (cpt_next == 'v' && cpt_next_next == 'e') ||
                                (cpt_next == 'l' && cpt_next_next == 'l')) {
                                pos += 3;
                            }
                        }
                    }

                    _add_token(pos);
                    continue;
                } else if (has_leading_char) {
                    pos--;
                }
            }

            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            auto flags2 = (cpt == ' ' ? _get_flags(pos + 1) : flags);
            if (!(flags2.is_whitespace || flags2.is_letter || flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace || flags2.is_letter || flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos + num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos + num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            if (num_whitespaces > 1 && _get_cpt(pos + num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

static std::vector<size_t> unicode_regex_split_custom_afmoe_cpts(const std::vector<uint32_t> & cpts, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; ) {
            const auto flags = _get_flags(pos);

            if (flags.is_number) {
                size_t digit_start = pos;
                size_t digit_count = 0;

                while (_get_flags(pos).is_number && pos < offset_end) {
                    digit_count++;
                    pos++;
                }

                size_t remainder = digit_count % 3;
                size_t current = digit_start;

                if (remainder > 0) {
                    _add_token(current + remainder);
                    current += remainder;
                }

                while (current < digit_start + digit_count) {
                    _add_token(current + 3);
                    current += 3;
                }
                continue;
            }

            pos++;
        }

        if (_prev_end < offset_end) {
            _add_token(offset_end);
        }
    }

    return bpe_offsets;
}

static std::vector<size_t> unicode_regex_split_custom_cpts(const std::vector<uint32_t> & cpts, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    if (regex_expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
        return unicode_regex_split_custom_gpt2_cpts(cpts, offsets);
    }
    if (regex_expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
        regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        return unicode_regex_split_custom_llama3_cpts(cpts, offsets);
    }
    if (regex_expr == "\\p{Han}+") {
        return unicode_regex_split_custom_kimi_k2_cpts(cpts, offsets);
    }
    if (regex_expr == "\\p{AFMoE_digits}" || regex_expr == "\\d{1,3}(?=(?:\\d{3})*\\b)") {
        return unicode_regex_split_custom_afmoe_cpts(cpts, offsets);
    }
    return {};
}

//
// interface
//

std::string unicode_cpt_to_utf8(uint32_t cpt) {
    std::string result;
    append_cpt_utf8(result, cpt);
    return result;
}

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    auto comp = [] (const uint32_t cpt, const range_nfd & range) {
        return cpt < range.first;
    };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        auto it = std::upper_bound(unicode_ranges_nfd.begin(), unicode_ranges_nfd.end(), cpt, comp) - 1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        const unsigned char c0 = (unsigned char) utf8[offset];
        if ((c0 & 0x80) == 0) {
            result.push_back((uint32_t) c0);
            offset += 1;
            continue;
        }
        try {
            result.push_back(unicode_cpt_from_utf8(utf8, offset));
        } catch (const std::invalid_argument &) {
            ++offset;
            result.emplace_back(0xFFFD);
        }
    }
    return result;
}

unicode_cpt_flags unicode_cpt_flags_from_cpt(const uint32_t cpt) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
}

unicode_cpt_flags unicode_cpt_flags_from_utf8(const std::string & utf8) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    if (utf8.empty()) {
        return undef;
    }
    size_t offset = 0;
    return unicode_cpt_flags_from_cpt(unicode_cpt_from_utf8(utf8, offset));
}

std::string unicode_byte_to_utf8(uint8_t byte) {
    return unicode_byte_to_utf8_table()[(size_t) byte];
}

uint8_t unicode_utf8_to_byte(const std::string & utf8) {
    static const std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    return map.at(utf8);
}

uint32_t unicode_tolower(uint32_t cpt) {
    auto it = std::lower_bound(unicode_map_lowercase.begin(), unicode_map_lowercase.end(), cpt,
        [](const std::pair<uint32_t, uint32_t> & pair, uint32_t value) {
            return pair.first < value;
        });
    if (it != unicode_map_lowercase.end() && it->first == cpt) {
        return it->second;
    }
    return cpt;
}

bool unicode_cpt_is_han(uint32_t cpt) {
    if (cpt >= 0x4E00 && cpt <= 0x9FFF) return true;
    if (cpt >= 0x3400 && cpt <= 0x4DBF) return true;
    if (cpt >= 0x20000 && cpt <= 0x2A6DF) return true;
    if (cpt >= 0x2A700 && cpt <= 0x2B73F) return true;
    if (cpt >= 0x2B740 && cpt <= 0x2B81F) return true;
    if (cpt >= 0x2B820 && cpt <= 0x2CEAF) return true;
    if (cpt >= 0x2CEB0 && cpt <= 0x2EBEF) return true;
    if (cpt >= 0xF900 && cpt <= 0xFAFF) return true;
    if (cpt >= 0x2F800 && cpt <= 0x2FA1F) return true;
    return false;
}

std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs) {
    static const std::map<std::string, int> k_ucat_enum = {
        { "\\p{N}", unicode_cpt_flags::NUMBER },
        { "\\p{L}", unicode_cpt_flags::LETTER },
        { "\\p{P}", unicode_cpt_flags::PUNCTUATION },
        { "\\p{M}", unicode_cpt_flags::ACCENT_MARK },
        { "\\p{S}", unicode_cpt_flags::SYMBOL },
        { "\\p{Lu}", unicode_cpt_flags::LETTER },
        { "\\p{Ll}", unicode_cpt_flags::LETTER },
        { "\\p{Lt}", unicode_cpt_flags::LETTER },
        { "\\p{Lm}", unicode_cpt_flags::LETTER },
        { "\\p{Lo}", unicode_cpt_flags::LETTER },
    };

    static const std::map<int, int> k_ucat_cpt = {
        { unicode_cpt_flags::NUMBER,      0xD1 },
        { unicode_cpt_flags::LETTER,      0xD2 },
        { unicode_cpt_flags::PUNCTUATION, 0xD3 },
        { unicode_cpt_flags::ACCENT_MARK, 0xD4 },
        { unicode_cpt_flags::SYMBOL,      0xD5 },
    };

    static const std::map<int, std::string> k_ucat_map = {
        { unicode_cpt_flags::NUMBER,      "\x30-\x39" },
        { unicode_cpt_flags::LETTER,      "\x41-\x5A\x61-\x7A" },
        { unicode_cpt_flags::PUNCTUATION, "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D" },
        { unicode_cpt_flags::ACCENT_MARK, "" },
        { unicode_cpt_flags::SYMBOL,      "\\\x24\\\x2B\x3C-\x3E\x5E\x60\\\x7C" },
    };

    const auto cpts = unicode_cpts_from_utf8(text);
    std::vector<size_t> bpe_offsets = { cpts.size() };
    std::string text_collapsed;
    bool collapsed_built = false;

    for (const auto & regex_expr : regex_exprs) {
        // Fast path: custom C++ splitter directly over decoded `cpts` without re-decoding UTF-8 or building `text_collapsed`
        auto tmp = unicode_regex_split_custom_cpts(cpts, regex_expr, bpe_offsets);
        if (!tmp.empty()) {
            bpe_offsets = std::move(tmp);
            continue;
        }

        try {
            bool use_collapsed = false;
            for (const auto & ucat : k_ucat_enum) {
                if (std::string::npos != regex_expr.find(ucat.first)) {
                    use_collapsed = true;
                    break;
                }
            }
            const auto cpts_regex = unicode_cpts_from_utf8(regex_expr);

            if (use_collapsed) {
                if (!collapsed_built) {
                    text_collapsed.resize(cpts.size());
                    for (size_t i = 0; i < cpts.size(); ++i) {
                        if (cpts[i] < 128) {
                            text_collapsed[i] = (char) cpts[i];
                            continue;
                        }
                        const auto flags = unicode_cpt_flags_from_cpt(cpts[i]);
                        if (flags.is_whitespace) {
                            text_collapsed[i] = (char) 0x0B;
                        } else if (k_ucat_cpt.find(flags.category_flag()) != k_ucat_cpt.end()) {
                            text_collapsed[i] = (char) k_ucat_cpt.at(flags.category_flag());
                        } else {
                            text_collapsed[i] = (char) 0xD0;
                        }
                    }
                    collapsed_built = true;
                }

                for (size_t i = 0; i < cpts_regex.size(); ++i) {
                    if (cpts_regex[i] >= 128) {
                        throw std::runtime_error("Regex includes both unicode categories and non-ASCII characters - not supported");
                    }
                }

                std::string regex_expr_collapsed;
                bool inside = false;
                for (size_t i = 0; i < regex_expr.size(); ++i) {
                    if (regex_expr[i] == '[' && (i == 0 || regex_expr[i - 1] != '\\')) {
                        regex_expr_collapsed += '[';
                        inside = true;
                        continue;
                    }
                    if (inside && regex_expr[i] == ']' && regex_expr[i - 1] != '\\') {
                        regex_expr_collapsed += ']';
                        inside = false;
                        continue;
                    }
                    if (regex_expr[i + 0] == '\\' && i + 3 < regex_expr.size() &&
                        regex_expr[i + 1] == 'p' &&
                        regex_expr[i + 2] == '{') {
                        size_t closing_brace = regex_expr.find('}', i + 3);
                        if (closing_brace != std::string::npos && closing_brace <= i + 10) {
                            const std::string pat = regex_expr.substr(i, closing_brace - i + 1);
                            if (k_ucat_enum.find(pat) != k_ucat_enum.end()) {
                                if (!inside) {
                                    regex_expr_collapsed += '[';
                                }
                                regex_expr_collapsed += (char) k_ucat_cpt.at(k_ucat_enum.at(pat));
                                regex_expr_collapsed += k_ucat_map.at(k_ucat_enum.at(pat));
                                if (!inside) {
                                    regex_expr_collapsed += ']';
                                }
                                i = closing_brace;
                                continue;
                            }
                        }
                    }
                    regex_expr_collapsed += regex_expr[i];
                }

                bpe_offsets = unicode_regex_split_stl(text_collapsed, regex_expr_collapsed, bpe_offsets);
            } else {
                std::wstring wregex_expr(cpts_regex.begin(), cpts_regex.end());
                std::wstring wtext(cpts.begin(), cpts.end());
                for (size_t i = 0; i < wtext.size(); ++i) {
                    if (wtext[i] > 0x7F && unicode_cpt_flags_from_cpt(wtext[i]).is_whitespace) {
                        wtext[i] = 0x0B;
                    }
                }
                bpe_offsets = unicode_regex_split_stl(wtext, wregex_expr, bpe_offsets);
            }
        } catch (std::regex_error & e) {
            fprintf(stderr, "Failed to process regex: '%s'\n", regex_expr.c_str());
            fprintf(stderr, "Regex error: %s\n", e.what());
            throw std::runtime_error("Failed to process regex");
        }
    }

    // Fuse UTF-8 emission + GPT-2 byte encoding directly into output words (zero intermediate word copies!)
    const auto & byte_table = unicode_byte_to_utf8_table();
    std::vector<std::string> bpe_encoded_words;
    bpe_encoded_words.reserve(bpe_offsets.size());

    size_t start = 0;
    for (size_t offset : bpe_offsets) {
        std::string encoded_token;
        encoded_token.reserve(offset * 2);
        for (size_t i = start; i < start + offset; ++i) {
            const uint32_t cpt = cpts[i];
            if (cpt <= 0x7f) {
                encoded_token += byte_table[(size_t) cpt];
            } else if (cpt <= 0x7ff) {
                encoded_token += byte_table[(size_t)(0xc0 | ((cpt >> 6) & 0x1f))];
                encoded_token += byte_table[(size_t)(0x80 | (cpt & 0x3f))];
            } else if (cpt <= 0xffff) {
                encoded_token += byte_table[(size_t)(0xe0 | ((cpt >> 12) & 0x0f))];
                encoded_token += byte_table[(size_t)(0x80 | ((cpt >> 6) & 0x3f))];
                encoded_token += byte_table[(size_t)(0x80 | (cpt & 0x3f))];
            } else if (cpt <= 0x10ffff) {
                encoded_token += byte_table[(size_t)(0xf0 | ((cpt >> 18) & 0x07))];
                encoded_token += byte_table[(size_t)(0x80 | ((cpt >> 12) & 0x3f))];
                encoded_token += byte_table[(size_t)(0x80 | ((cpt >> 6) & 0x3f))];
                encoded_token += byte_table[(size_t)(0x80 | (cpt & 0x3f))];
            }
        }
        bpe_encoded_words.emplace_back(std::move(encoded_token));
        start += offset;
    }

    return bpe_encoded_words;
}
