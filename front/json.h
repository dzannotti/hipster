// Minimal JSON value + parser + serializer (enough for the OpenAI chat API): objects keep insertion order.
#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <cstdio>
#include <cmath>

namespace js {
struct Value {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ } type = NUL;
    bool b = false; double n = 0; std::string s; std::vector<Value> a; std::vector<std::pair<std::string, Value>> o;
    Value() {}
    Value(bool v) : type(BOOL), b(v) {}
    Value(int v) : type(NUM), n(v) {}
    Value(double v) : type(NUM), n(v) {}
    Value(const char* v) : type(STR), s(v) {}
    Value(const std::string& v) : type(STR), s(v) {}
    static Value array() { Value v; v.type = ARR; return v; }
    static Value object() { Value v; v.type = OBJ; return v; }
    bool is_null() const { return type == NUL; }
    const Value* get(const std::string& k) const { if (type != OBJ) return nullptr; for (auto& kv : o) if (kv.first == k) return &kv.second; return nullptr; }
    Value& operator[](const std::string& k) { if (type == NUL) type = OBJ; for (auto& kv : o) if (kv.first == k) return kv.second; o.push_back({k, Value()}); return o.back().second; }
    Value& push(const Value& v) { if (type == NUL) type = ARR; a.push_back(v); return a.back(); }
    std::string str(const std::string& k, const std::string& d = "") const { auto* v = get(k); return v && v->type == STR ? v->s : d; }
    double num(const std::string& k, double d) const { auto* v = get(k); return v && v->type == NUM ? v->n : d; }
    bool boolean(const std::string& k, bool d) const { auto* v = get(k); return v && v->type == BOOL ? v->b : d; }
};
inline void escape(const std::string& s, std::string& o) {
    o += '"';
    for (unsigned char c : s) { switch (c) { case '"': o += "\\\""; break; case '\\': o += "\\\\"; break; case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
        default: if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char)c; } }
    o += '"';
}
inline void dump(const Value& v, std::string& o) {
    switch (v.type) {
        case Value::NUL: o += "null"; break; case Value::BOOL: o += v.b ? "true" : "false"; break;
        case Value::NUM: { char b[32]; if (v.n == std::floor(v.n) && std::fabs(v.n) < 1e15) snprintf(b, sizeof b, "%.0f", v.n); else snprintf(b, sizeof b, "%.10g", v.n); o += b; break; }
        case Value::STR: escape(v.s, o); break;
        case Value::ARR: o += '['; for (size_t i = 0; i < v.a.size(); ++i) { if (i) o += ','; dump(v.a[i], o); } o += ']'; break;
        case Value::OBJ: o += '{'; for (size_t i = 0; i < v.o.size(); ++i) { if (i) o += ','; escape(v.o[i].first, o); o += ':'; dump(v.o[i].second, o); } o += '}'; break;
    }
}
inline std::string dump(const Value& v) { std::string o; dump(v, o); return o; }
struct Parser {
    const std::string& s; size_t i = 0;
    Parser(const std::string& str) : s(str) {}
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; }
    [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("json: ") + m + " at " + std::to_string(i)); }
    static void cp(std::string& o, unsigned c) { if (c < 0x80) o += (char)c; else if (c < 0x800) { o += (char)(0xC0 | (c >> 6)); o += (char)(0x80 | (c & 0x3F)); } else if (c < 0x10000) { o += (char)(0xE0 | (c >> 12)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); } else { o += (char)(0xF0 | (c >> 18)); o += (char)(0x80 | ((c >> 12) & 0x3F)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); } }
    std::string string() {
        if (s[i] != '"') fail("expected string"); ++i; std::string o;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') { ++i; char c = s[i++]; switch (c) { case 'n': o += '\n'; break; case 't': o += '\t'; break; case 'r': o += '\r'; break; case 'b': o += '\b'; break; case 'f': o += '\f'; break;
                case 'u': { unsigned c1 = std::stoul(s.substr(i, 4), nullptr, 16); i += 4; if (c1 >= 0xD800 && c1 < 0xDC00 && s.compare(i, 2, "\\u") == 0) { unsigned c2 = std::stoul(s.substr(i + 2, 4), nullptr, 16); i += 6; c1 = 0x10000 + ((c1 - 0xD800) << 10) + (c2 - 0xDC00); } cp(o, c1); break; }
                default: o += c; } }
            else o += s[i++];
        }
        if (i >= s.size()) fail("unterminated string"); ++i; return o;
    }
    Value value() {
        ws(); if (i >= s.size()) fail("unexpected end");
        char c = s[i];
        if (c == '{') { Value v = Value::object(); ++i; ws(); if (s[i] == '}') { ++i; return v; } while (true) { ws(); std::string k = string(); ws(); if (s[i] != ':') fail("expected :"); ++i; v.o.push_back({k, value()}); ws(); if (s[i] == ',') { ++i; continue; } if (s[i] == '}') { ++i; return v; } fail("expected , or }"); } }
        if (c == '[') { Value v = Value::array(); ++i; ws(); if (s[i] == ']') { ++i; return v; } while (true) { v.a.push_back(value()); ws(); if (s[i] == ',') { ++i; continue; } if (s[i] == ']') { ++i; return v; } fail("expected , or ]"); } }
        if (c == '"') return Value(string());
        if (s.compare(i, 4, "true") == 0) { i += 4; return Value(true); } if (s.compare(i, 5, "false") == 0) { i += 5; return Value(false); } if (s.compare(i, 4, "null") == 0) { i += 4; return Value(); }
        size_t j = i; while (j < s.size() && (isdigit((unsigned char)s[j]) || s[j] == '-' || s[j] == '+' || s[j] == '.' || s[j] == 'e' || s[j] == 'E')) ++j;
        if (j == i) fail("unexpected character"); Value v(std::stod(s.substr(i, j - i))); i = j; return v;
    }
};
inline Value parse(const std::string& s) { Parser p(s); Value v = p.value(); return v; }
}  // namespace js
