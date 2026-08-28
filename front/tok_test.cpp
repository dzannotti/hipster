// tok_test <gguf> <tok-tests.json> [ref.json...] : our tokenizer vs llama.cpp's /tokenize (ids with and without special parsing)
#include "tokenizer.h"
#include "../engine/src/gguf.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
static std::string unescape(const std::string& s) {   // JSON string body -> bytes (handles \n \t \" \\ \/ \uXXXX incl. surrogates)
    std::string o; for (size_t i = 0; i < s.size(); ++i) { if (s[i] != '\\') { o += s[i]; continue; } ++i; char c = s[i];
        if (c == 'n') o += '\n'; else if (c == 't') o += '\t'; else if (c == 'r') o += '\r'; else if (c == 'b') o += '\b'; else if (c == 'f') o += '\f'; else if (c == 'u') {
            unsigned cp = std::stoul(s.substr(i + 1, 4), nullptr, 16); i += 4;
            if (cp >= 0xD800 && cp < 0xDC00 && s.compare(i + 1, 2, "\\u") == 0) { unsigned lo = std::stoul(s.substr(i + 3, 4), nullptr, 16); cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); i += 6; }
            if (cp < 0x80) o += (char)cp; else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 0x3F)); }
            else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
            else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 0x3F)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
        } else o += c; }
    return o; }
static std::string get_str(const std::string& j, size_t from, const char* key) {   // "key": "..." after `from`
    size_t p = j.find(std::string("\"") + key + "\"", from); p = j.find('"', j.find(':', p) + 1) + 1; size_t e = p;
    while (j[e] != '"') { if (j[e] == '\\') ++e; ++e; } return unescape(j.substr(p, e - p)); }
static std::vector<int> get_ids(const std::string& j, size_t from, const char* key) {
    size_t p = j.find(std::string("\"") + key + "\"", from); p = j.find('[', p); size_t e = j.find(']', p);
    std::vector<int> v; std::stringstream ss(j.substr(p + 1, e - p - 1)); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(atoi(t.c_str())); return v; }
static std::string show(const std::vector<int>& v) { std::string s; for (size_t i = 0; i < v.size() && i < 12; ++i) s += std::to_string(v[i]) + " "; return s; }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s gguf tok-tests.json [ref.json...]\n", argv[0]); return 2; }
    hip::GGUF g(argv[1]); tok::Tokenizer T(g);
    printf("vocab %zu, eos %d, <|im_start|> = %d\n", T.size(), T.eos(), T.special("<|im_start|>"));
    int bad = 0, total = 0;
    { std::ifstream f(argv[2]); std::stringstream b; b << f.rdbuf(); const std::string j = b.str();
      for (size_t p = j.find("\"text\""); p != std::string::npos; p = j.find("\"text\"", p + 1)) {
          const std::string text = get_str(j, p, "text"); const auto ids = get_ids(j, p, "ids"), idsp = get_ids(j, p, "ids_special");
          for (int sp = 0; sp < 2; ++sp) { const auto ours = T.encode(text, sp); const auto& ref = sp ? idsp : ids; ++total;
              if (ours != ref) { ++bad; printf("MISMATCH (special=%d) %.40s...\n  ours %zu: %s\n  ref  %zu: %s\n", sp, text.c_str(), ours.size(), show(ours).c_str(), ref.size(), show(ref).c_str()); } }
          if (T.decode(ids) != text) { ++bad; printf("DECODE MISMATCH %.40s\n", text.c_str()); } ++total;
      } }
    for (int a = 3; a < argc; ++a) { std::ifstream f(argv[a]); std::stringstream b; b << f.rdbuf(); const std::string j = b.str(); if (j.find("\"text\"") == std::string::npos) continue;
        const std::string text = get_str(j, 0, "text"); const auto ref = get_ids(j, 0, "prompt_ids"); const auto ours = T.encode(text, true); ++total;
        if (ours != ref) { ++bad; printf("MISMATCH %s: ours %zu ref %zu\n  %s\n  %s\n", argv[a], ours.size(), ref.size(), show(ours).c_str(), show(ref).c_str()); } }
    printf("%s: %d/%d cases identical to llama.cpp\n", bad ? "MISMATCH" : "EXACT", total - bad, total);
    return bad ? 1 : 0;
}
