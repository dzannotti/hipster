#include "tokenizer.h"
#include "unicode_tables.h"
#include "../engine/src/gguf.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace tok {
namespace {
// ---- UTF-8 <-> code points ----
static uint32_t decode_cp(const std::string& s, size_t& i) {
    const unsigned char c = s[i];
    if (c < 0x80) { ++i; return c; }
    int n = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
    uint32_t cp = n == 3 ? (c & 7) : n == 2 ? (c & 15) : n == 1 ? (c & 31) : c;
    ++i;
    for (int k = 0; k < n && i < s.size(); ++k, ++i) cp = (cp << 6) | ((unsigned char)s[i] & 0x3F);
    return cp;
}
static void append_cp(std::string& s, uint32_t cp) {
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
    else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
}
static bool in_table(const Range* t, int n, uint32_t cp) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) { const int m = (lo + hi) / 2; if (cp < t[m].lo) hi = m - 1; else if (cp > t[m].hi) lo = m + 1; else return true; }
    return false;
}
static bool is_L(uint32_t c) { return in_table(LETTER, N_LETTER, c); }
static bool is_N(uint32_t c) { return in_table(NUMBER, N_NUMBER, c); }
static bool is_M(uint32_t c) { return in_table(MARK, N_MARK, c); }
static bool is_S(uint32_t c) { return in_table(SPACE, N_SPACE, c); }
static bool is_CRLF(uint32_t c) { return c == '\r' || c == '\n'; }
static bool ieq(uint32_t c, char a) { return c == (uint32_t)a || c == (uint32_t)(a - 32); }

// GPT-2 byte-level mapping: printable bytes map to themselves, the rest to U+0100..
static void build_byte_map(std::string* b2u, std::unordered_map<std::string, uint8_t>& u2b) {
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        std::string u; append_cp(u, printable ? (uint32_t)b : (uint32_t)(256 + n++));
        b2u[b] = u; u2b[u] = (uint8_t)b;
    }
}
}  // namespace

Tokenizer::Tokenizer(const hip::GGUF& g) {
    build_byte_map(byte_to_unicode_, unicode_to_byte_);
    const auto* toks = g.kv("tokenizer.ggml.tokens"); const auto* types = g.kv("tokenizer.ggml.token_type"); const auto* merges = g.kv("tokenizer.ggml.merges");
    if (!toks || !merges) throw std::runtime_error("tokenizer: missing vocab in the GGUF");
    tokens_.reserve(toks->arr.size());
    for (size_t i = 0; i < toks->arr.size(); ++i) { tokens_.push_back(toks->arr[i].s); vocab_[toks->arr[i].s] = (int)i; }
    for (size_t i = 0; i < merges->arr.size(); ++i) merges_[merges->arr[i].s] = (int)i;
    if (types) for (size_t i = 0; i < types->arr.size(); ++i) {
        const int64_t ty = types->arr[i].i ? types->arr[i].i : (int64_t)types->arr[i].u;
        if (ty == 3 || ty == 4) { special_[tokens_[i]] = (int)i; (ty == 4 ? userdef_list_ : special_list_).push_back(tokens_[i]); }   // control / user-defined
    }
    // llama.cpp matches user-defined tokens (<think>, <tool_call>, ...) always, control tokens (<|im_start|>, ...) only with parse_special
    auto longest = [](const std::string& a, const std::string& b) { return a.size() > b.size(); };
    std::sort(special_list_.begin(), special_list_.end(), longest); std::sort(userdef_list_.begin(), userdef_list_.end(), longest);
    eos_ = (int)g.kv_u32("tokenizer.ggml.eos_token_id");
}

// The qwen35 pre-tokenizer, leftmost-first over the 7 alternatives of
// (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N} | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    std::vector<uint32_t> cp; std::vector<size_t> off;   // code points and their byte offsets
    for (size_t i = 0; i < text.size();) { off.push_back(i); cp.push_back(decode_cp(text, i)); }
    off.push_back(text.size());
    const size_t n = cp.size();
    std::vector<std::string> out;
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        // 1. contractions
        if (cp[i] == '\'' && i + 1 < n) {
            const uint32_t a = cp[i + 1], b = i + 2 < n ? cp[i + 2] : 0;
            if (ieq(a, 's') || ieq(a, 't') || ieq(a, 'm') || ieq(a, 'd')) j = i + 2;
            else if ((ieq(a, 'r') && ieq(b, 'e')) || (ieq(a, 'v') && ieq(b, 'e')) || (ieq(a, 'l') && ieq(b, 'l'))) j = i + 3;
        }
        if (j == i) {   // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            size_t k = i;
            if (!is_CRLF(cp[k]) && !is_L(cp[k]) && !is_N(cp[k]) && k + 1 < n && (is_L(cp[k + 1]) || is_M(cp[k + 1]))) ++k;
            if (k < n && (is_L(cp[k]) || is_M(cp[k]))) { while (k < n && (is_L(cp[k]) || is_M(cp[k]))) ++k; j = k; }
        }
        if (j == i && is_N(cp[i])) j = i + 1;   // 3. \p{N}
        if (j == i) {   // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            size_t k = i; if (cp[k] == ' ') ++k;
            size_t k0 = k;
            while (k < n && !is_S(cp[k]) && !is_L(cp[k]) && !is_M(cp[k]) && !is_N(cp[k])) ++k;
            if (k > k0) { while (k < n && is_CRLF(cp[k])) ++k; j = k; }
        }
        if (j == i) {   // 5. \s*[\r\n]+
            size_t k = i; while (k < n && is_S(cp[k]) && !is_CRLF(cp[k])) ++k;
            if (k < n && is_CRLF(cp[k])) { while (k < n && is_CRLF(cp[k])) ++k; j = k; }
        }
        if (j == i && is_S(cp[i])) {   // 6. \s+(?!\S)  then 7. \s+
            size_t k = i; while (k < n && is_S(cp[k])) ++k;
            if (k < n && k - i > 1) j = k - 1;   // followed by a non-space: leave the last whitespace to the next chunk
            else j = k;                          // (a single space before a non-space falls to alternative 7 = itself... regex: \s+(?!\S) fails, \s+ takes one)
        }
        if (j == i) j = i + 1;   // should not happen
        out.emplace_back(text.substr(off[i], off[j] - off[i]));
        i = j;
    }
    return out;
}

void Tokenizer::bpe(const std::string& chunk, std::vector<int>& out) const {
    // bytes -> byte-level unicode symbols
    std::vector<std::string> sym; sym.reserve(chunk.size());
    for (unsigned char c : chunk) sym.push_back(byte_to_unicode_[c]);
    auto whole = [&] { std::string w; for (auto& s : sym) w += s; return w; };
    { auto it = vocab_.find(whole()); if (it != vocab_.end()) { out.push_back(it->second); return; } }
    while (sym.size() > 1) {
        int best = INT32_MAX; size_t bi = 0;
        for (size_t i = 0; i + 1 < sym.size(); ++i) { auto it = merges_.find(sym[i] + " " + sym[i + 1]); if (it != merges_.end() && it->second < best) { best = it->second; bi = i; } }
        if (best == INT32_MAX) break;
        sym[bi] += sym[bi + 1]; sym.erase(sym.begin() + bi + 1);
    }
    for (auto& s : sym) { auto it = vocab_.find(s); if (it == vocab_.end()) throw std::runtime_error("tokenizer: unknown symbol " + s); out.push_back(it->second); }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int> out;
    auto encode_plain = [&](const std::string& span) { for (auto& ch : pretokenize(span)) bpe(ch, out); };
    size_t start = 0, i = 0;
    auto try_list = [&](const std::vector<std::string>& lst) { for (auto& sp : lst) if (text.compare(i, sp.size(), sp) == 0) { if (i > start) encode_plain(text.substr(start, i - start)); out.push_back(special_.at(sp)); i += sp.size(); start = i; return true; } return false; };
    while (i < text.size()) { if (try_list(userdef_list_) || (parse_special && try_list(special_list_))) continue; ++i; }
    if (start < text.size()) encode_plain(text.substr(start));
    return out;
}
std::string Tokenizer::piece(int id) const {
    if (id < 0 || id >= (int)tokens_.size()) return "";
    const std::string& t = tokens_[id];
    if (special_.count(t)) return t;
    std::string bytes;
    for (size_t i = 0; i < t.size();) { size_t j = i; decode_cp(t, j); auto it = unicode_to_byte_.find(t.substr(i, j - i)); if (it == unicode_to_byte_.end()) bytes += t.substr(i, j - i); else bytes += (char)it->second; i = j; }
    return bytes;
}
std::string Tokenizer::decode(const std::vector<int>& ids) const { std::string s; for (int id : ids) s += piece(id); return s; }
}  // namespace tok
