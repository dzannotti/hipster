// Byte-level BPE tokenizer for the Qwen3.8 GGUFs (tokenizer.ggml.model = gpt2, pre = qwen35): the pre-tokenizer regex
// is hand-written over Unicode categories (front/unicode_tables.h), merges ranked from the GGUF, special tokens matched
// longest-first when parse_special. Validated against llama.cpp's /tokenize (docs/ref/tok-tests.json).
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace hip { class GGUF; }
namespace tok {

class Tokenizer {
public:
    explicit Tokenizer(const hip::GGUF& g);
    std::vector<int> encode(const std::string& text, bool parse_special) const;
    std::string decode(const std::vector<int>& ids) const;   // special tokens rendered as their text
    std::string piece(int id) const;                          // one token's bytes
    int eos() const { return eos_; }
    int special(const std::string& s) const { auto it = special_.find(s); return it == special_.end() ? -1 : it->second; }
    size_t size() const { return tokens_.size(); }

private:
    std::vector<std::string> pretokenize(const std::string& text) const;   // regex chunks of a plain (special-free) span
    void bpe(const std::string& chunk, std::vector<int>& out) const;
    std::vector<std::string> tokens_;                 // vocab in byte-level (Ġ-style) encoding
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<std::string, int> merges_;     // "a b" -> rank
    std::unordered_map<std::string, int> special_;    // special token text -> id
    std::vector<std::string> special_list_, userdef_list_;   // control (parse_special only) / user-defined (always), longest first
    int eos_ = -1;
    std::string byte_to_unicode_[256]; std::unordered_map<std::string, uint8_t> unicode_to_byte_;
};

}  // namespace tok
