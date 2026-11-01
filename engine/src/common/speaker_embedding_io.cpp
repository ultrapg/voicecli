#include "qwen3_tts.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

namespace qwen3_tts {

static std::string get_lower_extension(const std::string & path) {
    const size_t pos = path.find_last_of('.');
    if (pos == std::string::npos) {
        return "";
    }
    std::string ext = path.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return ext;
}

static bool has_json_extension(const std::string & path) {
    return get_lower_extension(path) == ".json";
}

static bool has_binary_extension(const std::string & path) {
    const std::string ext = get_lower_extension(path);
    return ext == ".bin" || ext == ".emb" || ext == ".raw";
}

static bool looks_like_ascii_embedding_text(const std::string & data) {
    bool saw_digit = false;
    for (unsigned char c : data) {
        if (c == '\0' || (c < 0x20 && c != '\t' && c != '\n' && c != '\r') || c >= 0x7F) {
            return false;
        }
        if (c >= '0' && c <= '9') {
            saw_digit = true;
        }
    }
    return saw_digit;
}

static bool parse_embedding_text(const std::string & text, std::vector<float> & embedding) {
    embedding.clear();
    embedding.reserve(2048);
    const char * p = text.c_str();
    const char * end = p + text.size();
    while (p < end) {
        while (p < end && (std::isspace((unsigned char) *p) || *p == '[' || *p == ']' || *p == ',' || *p == ';')) {
            ++p;
        }
        if (p >= end) {
            break;
        }
        char * next = nullptr;
        const float value = std::strtof(p, &next);
        if (next == p) {
            return false;
        }
        embedding.push_back(value);
        p = next;
    }
    return !embedding.empty();
}

bool load_speaker_embedding_file(const std::string & path,
                                 std::vector<float> & embedding) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open speaker embedding file: %s\n", path.c_str());
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        std::fclose(f);
        fprintf(stderr, "ERROR: Speaker embedding file is empty: %s\n", path.c_str());
        return false;
    }

    std::string data((size_t) sz, '\0');
    if (std::fread(&data[0], 1, (size_t) sz, f) != (size_t) sz) {
        std::fclose(f);
        fprintf(stderr, "ERROR: Failed to read speaker embedding file: %s\n", path.c_str());
        return false;
    }
    std::fclose(f);

    const bool is_json = has_json_extension(path);
    const bool is_bin  = has_binary_extension(path);
    if (is_json || (!is_bin && looks_like_ascii_embedding_text(data))) {
        if (!parse_embedding_text(data, embedding)) {
            fprintf(stderr, "ERROR: Failed to parse speaker embedding JSON/text: %s\n", path.c_str());
            return false;
        }
        return true;
    }

    if (data.size() % sizeof(float) != 0) {
        fprintf(stderr, "ERROR: Speaker embedding binary size is not a multiple of 4 bytes: %s\n", path.c_str());
        return false;
    }

    embedding.resize(data.size() / sizeof(float));
    std::memcpy(embedding.data(), data.data(), data.size());
    return true;
}

bool save_speaker_embedding_file(const std::string & path,
                                 const std::vector<float> & embedding) {
    if (embedding.empty()) {
        fprintf(stderr, "ERROR: Refusing to save empty speaker embedding\n");
        return false;
    }

    if (has_json_extension(path)) {
        std::ofstream out(path, std::ios::out | std::ios::trunc);
        if (!out) {
            fprintf(stderr, "ERROR: Cannot create speaker embedding JSON file: %s\n", path.c_str());
            return false;
        }
        out << std::setprecision(std::numeric_limits<float>::max_digits10);
        out << "[\n";
        for (size_t i = 0; i < embedding.size(); ++i) {
            out << "  " << embedding[i];
            if (i + 1 != embedding.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "]\n";
        return true;
    }

    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot create speaker embedding binary file: %s\n", path.c_str());
        return false;
    }
    const size_t nbytes = embedding.size() * sizeof(float);
    const bool ok = std::fwrite(embedding.data(), 1, nbytes, f) == nbytes;
    std::fclose(f);
    return ok;
}

} // namespace qwen3_tts
