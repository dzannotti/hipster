#include "chat.h"
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace chat {
namespace {
std::string trim(const std::string& s) { size_t a = 0, b = s.size(); while (a < b && isspace((unsigned char)s[a])) ++a; while (b > a && isspace((unsigned char)s[b - 1])) --b; return s.substr(a, b - a); }
// content may be a string or an array of {type:text,text:...} parts (images/videos are not supported here)
std::string content_text(const js::Value* c) {
    if (!c) return ""; if (c->type == js::Value::STR) return c->s;
    std::string o; if (c->type == js::Value::ARR) for (auto& it : c->a) { if (it.str("type") == "text" || it.get("text")) o += it.str("text"); else throw std::runtime_error("only text content is supported"); }
    return o;
}
}

std::string render(const js::Value& messages, const js::Value* tools, const Opts& opts) {
    std::string out;
    // leading system/developer messages merge into one
    size_t num_sys = 0; std::string merged;
    for (auto& m : messages.a) { const std::string role = m.str("role"); if (role != "system" && role != "developer") break; const std::string c = trim(content_text(m.get("content"))); if (!c.empty()) { if (!merged.empty()) merged += "\n"; merged += c; } ++num_sys; }
    std::string reasoning;
    if (opts.enable_thinking) {
        std::string e = opts.reasoning_effort; if (e == "high") e = "xhigh";
        if (e != "xhigh" && e != "medium" && e != "low") throw std::runtime_error("unexpected reasoning effort " + opts.reasoning_effort);
        if (e == "xhigh") reasoning = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
        else if (e == "low") reasoning = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
    }
    const bool has_tools = tools && tools->type == js::Value::ARR && !tools->a.empty();
    if (has_tools) {
        out += "<|im_start|>system\n"; if (!reasoning.empty()) out += reasoning + "\n\n";
        out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (auto& t : tools->a) { out += "\n"; out += js::dump(t); }
        out += "\n</tools>";
        out += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
        if (!merged.empty()) out += "\n\n" + merged;
        out += "<|im_end|>\n";
    } else if (!merged.empty()) out += "<|im_start|>system\n" + (reasoning.empty() ? "" : reasoning + "\n\n") + merged + "<|im_end|>\n";
    else if (!reasoning.empty()) out += "<|im_start|>system\n" + reasoning + "<|im_end|>\n";
    // last real user query (tool responses don't count): assistant reasoning before it is dropped (preserve_thinking undefined -> kept... the
    // template keeps thinking when preserve_thinking is undefined; we follow that: reasoning_content rendered when present)
    const size_t n = messages.a.size();
    for (size_t i = num_sys; i < n; ++i) {
        const auto& m = messages.a[i]; const std::string role = m.str("role"); const std::string content = trim(content_text(m.get("content")));
        if (role == "system" || role == "developer") throw std::runtime_error("System message must be at the beginning.");
        if (role == "user") out += "<|im_start|>user\n" + content + "<|im_end|>\n";
        else if (role == "assistant") {
            std::string rc; if (auto* r = m.get("reasoning_content")) if (r->type == js::Value::STR) rc = trim(r->s);
            out += "<|im_start|>assistant\n<think>\n" + rc + "\n</think>\n\n" + content;
            if (auto* tcs = m.get("tool_calls")) if (tcs->type == js::Value::ARR) {
                bool first = true;
                for (auto& tc0 : tcs->a) {
                    const js::Value& tc = tc0.get("function") ? *tc0.get("function") : tc0;
                    const std::string name = tc.str("name"); if (name.empty()) throw std::runtime_error("Tool call is missing a function name.");
                    out += (first ? (content.empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=") : "\n<tool_call>\n<function=") + name + ">\n"; first = false;
                    const js::Value* args = tc.get("arguments"); js::Value parsed;
                    if (args && args->type == js::Value::STR) { if (!trim(args->s).empty()) parsed = js::parse(args->s); args = parsed.type == js::Value::OBJ ? &parsed : nullptr; }
                    if (args && args->type == js::Value::OBJ) for (auto& kv : args->o) out += "<parameter=" + kv.first + ">\n" + (kv.second.type == js::Value::STR ? kv.second.s : js::dump(kv.second)) + "\n</parameter>\n";
                    out += "</function>\n</tool_call>";
                }
            }
            out += "<|im_end|>\n";
        } else if (role == "tool") {
            if (i == 0 || messages.a[i - 1].str("role") != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (i + 1 >= n || messages.a[i + 1].str("role") != "tool") out += "<|im_end|>\n";
        } else throw std::runtime_error("Unexpected message role.");
    }
    if (opts.add_generation_prompt) out += opts.enable_thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return out;
}

Parsed parse_output(const std::string& text0, bool open) {
    Parsed p; std::string text = text0;
    if (open) {   // the prompt ended with "<think>\n": everything up to </think> is reasoning
        size_t e = text.find("</think>");
        if (e == std::string::npos) { p.reasoning = trim(text); return p; }
        p.reasoning = trim(text.substr(0, e)); text = text.substr(e + 8);
    }
    // tool calls: <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>...</function>\n</tool_call>
    std::string content;
    size_t pos = 0;
    while (true) {
        size_t a = text.find("<tool_call>", pos); if (a == std::string::npos) { content += text.substr(pos); break; }
        content += text.substr(pos, a - pos);
        size_t b = text.find("</tool_call>", a); std::string body = text.substr(a + 11, b == std::string::npos ? std::string::npos : b - a - 11);
        size_t f = body.find("<function="); if (f != std::string::npos) {
            size_t fe = body.find('>', f); std::string name = body.substr(f + 10, fe - f - 10);
            js::Value args = js::Value::object(); size_t q = fe + 1;
            while (true) { size_t pa = body.find("<parameter=", q); if (pa == std::string::npos) break; size_t pe = body.find('>', pa); std::string k = body.substr(pa + 11, pe - pa - 11);
                size_t ve = body.find("</parameter>", pe); std::string v = body.substr(pe + 1, ve == std::string::npos ? std::string::npos : ve - pe - 1);
                if (!v.empty() && v[0] == '\n') v.erase(0, 1); if (!v.empty() && v.back() == '\n') v.pop_back();
                js::Value jv; try { jv = js::parse(v); if (jv.type == js::Value::STR) jv = js::Value(v); } catch (...) { jv = js::Value(v); }   // numbers/objects/bools parsed, else string
                args[k] = jv; q = ve == std::string::npos ? body.size() : ve + 12; }
            js::Value call = js::Value::object(); call["id"] = "call_" + std::to_string(p.tool_calls.a.size()); call["type"] = "function";
            js::Value fn = js::Value::object(); fn["name"] = name; fn["arguments"] = js::dump(args); call["function"] = fn; p.tool_calls.push(call);
        }
        if (b == std::string::npos) break; pos = b + 12;
    }
    p.content = trim(content);
    return p;
}
}  // namespace chat
