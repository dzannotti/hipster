// Qwen3.8 chat template (the GGUF's Jinja, text parts only), rendered directly: system/developer merge, tools block,
// user/assistant/tool turns, assistant tool calls in the <tool_call><function=...> form, generation prompt with
// <think> (thinking on) or <think>\n\n</think>\n\n (off). Also the inverse: parse the generated text into
// reasoning / content / tool calls.
#pragma once
#include "json.h"
#include <string>
#include <vector>

namespace chat {
struct Opts { bool enable_thinking = true; std::string reasoning_effort = "xhigh"; bool add_generation_prompt = true; };
// messages / tools: the OpenAI request arrays
std::string render(const js::Value& messages, const js::Value* tools, const Opts& opts);
struct Parsed { std::string reasoning, content; js::Value tool_calls = js::Value::array(); };
Parsed parse_output(const std::string& text, bool thinking_prefix_open);   // thinking_prefix_open: the prompt ended with "<think>\n"
}  // namespace chat
