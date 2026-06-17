#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <nlohmann/json.hpp>

#include "inference.h"
#include "llama.h"
#include "llama_weights.h"
#include "tokenizer_bridge.h"


namespace
{

struct PromptOptions
{
    std::string weights_path = "model.safetensors";
    std::string tokenizer_script = "python/tokenizer.py";
    std::string tokenizer_model = "meta-llama/Llama-3.2-1B-Instruct";
    std::string python_executable = "python3";
    std::string prompt;
    std::size_t max_new_tokens = 16;
    bool dump_token_ids = false;
    std::size_t dump_top_k = 0;
    bool dump_top_k_fp32 = false;
    bool dump_final_norm = false;
    std::size_t dump_final_norm_step = 0;
    bool dump_layer_hidden = false;
    std::size_t dump_layer_hidden_step = 0;
};

bool file_exists(const std::string& path)
{
    std::ifstream file(path);
    return file.good();
}

std::string resolve_default_path(const std::string& primary, const std::string& parent_relative)
{
    if (file_exists(primary)) {
        return primary;
    }
    if (file_exists(parent_relative)) {
        return parent_relative;
    }
    return primary;
}

void print_usage(const char* program)
{
    std::cout
        << "Usage: " << program << " --prompt <text> [options]\n"
        << "  --prompt <text>          single user prompt to run through the chat template\n"
        << "  --weights <path>         safetensors weight file (default: model.safetensors)\n"
        << "  --tokenizer-script <p>   tokenizer.py path (default: python/tokenizer.py)\n"
        << "  --tokenizer-model <name> HuggingFace tokenizer id/path\n"
        << "  --python <path>          Python executable (default: python3)\n"
        << "  --max-new-tokens <n>     generation limit for alignment runs (default: 16)\n"
        << "  --dump-token-ids         print generated token IDs as JSON instead of decoded text\n"
        << "  --dump-top-k <n>         print generated token IDs and per-step top-k logits as JSON\n"
        << "  --dump-top-k-fp32 <n>    like --dump-top-k, but recompute traced logits as FP32\n"
        << "  --dump-final-norm-step <n> include final RMSNorm state for generation step n\n"
        << "  --dump-layer-hidden-step <n> include per-layer hidden states for generation step n\n";
}

std::size_t parse_size(const std::string& value, const std::string& option_name)
{
    std::size_t parsed_chars = 0;
    const unsigned long long parsed = std::stoull(value, &parsed_chars);
    if (parsed_chars != value.size()) {
        throw std::runtime_error(option_name + " must be an integer");
    }
    return static_cast<std::size_t>(parsed);
}

PromptOptions parse_options(int argc, char** argv)
{
    PromptOptions options;
    options.weights_path = resolve_default_path("model.safetensors", "../model.safetensors");
    options.tokenizer_script = resolve_default_path("python/tokenizer.py", "../python/tokenizer.py");

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(name + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--prompt") {
            options.prompt = require_value(arg);
        } else if (arg == "--weights") {
            options.weights_path = require_value(arg);
        } else if (arg == "--tokenizer-script") {
            options.tokenizer_script = require_value(arg);
        } else if (arg == "--tokenizer-model") {
            options.tokenizer_model = require_value(arg);
        } else if (arg == "--python") {
            options.python_executable = require_value(arg);
        } else if (arg == "--max-new-tokens") {
            options.max_new_tokens = parse_size(require_value(arg), arg);
        } else if (arg == "--dump-token-ids") {
            options.dump_token_ids = true;
        } else if (arg == "--dump-top-k") {
            options.dump_top_k = parse_size(require_value(arg), arg);
        } else if (arg == "--dump-top-k-fp32") {
            options.dump_top_k = parse_size(require_value(arg), arg);
            options.dump_top_k_fp32 = true;
        } else if (arg == "--dump-final-norm-step") {
            options.dump_final_norm_step = parse_size(require_value(arg), arg);
            options.dump_final_norm = true;
        } else if (arg == "--dump-layer-hidden-step") {
            options.dump_layer_hidden_step = parse_size(require_value(arg), arg);
            options.dump_layer_hidden = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (options.prompt.empty()) {
        throw std::runtime_error("--prompt is required");
    }

    return options;
}

std::vector<std::int32_t> encode_prompt(
    const TokenizerBridge& tokenizer,
    const std::string& prompt
) {
    std::vector<ChatMessage> messages = {{"user", prompt}};
    std::vector<std::int32_t> token_ids = tokenizer.encode_chat(messages);
    if (token_ids.empty() || token_ids.size() >= MAX_TOKEN_LEN) {
        throw std::runtime_error("prompt does not fit the model token window");
    }
    return token_ids;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const PromptOptions options = parse_options(argc, argv);

        Llama3_2 weights = load_llama_weights(options.weights_path);
        InferenceContext context;
        TokenizerBridge tokenizer(
            options.tokenizer_script,
            options.tokenizer_model,
            options.python_executable
        );

        std::vector<std::int32_t> token_ids = encode_prompt(tokenizer, options.prompt);
        const std::size_t prompt_token_count = token_ids.size();
        InferenceTrace trace;
        trace.top_k = options.dump_top_k;
        trace.fp32_logits = options.dump_top_k_fp32;
        trace.capture_final_norm = options.dump_final_norm;
        trace.final_norm_step = options.dump_final_norm_step;
        trace.capture_layer_hidden_states = options.dump_layer_hidden;
        trace.layer_hidden_step = options.dump_layer_hidden_step;
        inference(
            token_ids,
            weights,
            context,
            options.max_new_tokens,
            (options.dump_top_k > 0 || options.dump_final_norm || options.dump_layer_hidden) ? &trace : nullptr
        );

        std::vector<std::int32_t> generated_ids(
            token_ids.begin() + static_cast<std::ptrdiff_t>(prompt_token_count),
            token_ids.end()
        );

        if (options.dump_top_k > 0 || options.dump_final_norm || options.dump_layer_hidden) {
            nlohmann::json trace_json = nlohmann::json::array();
            for (const InferenceStepTrace& step_trace: trace.steps) {
                nlohmann::json top_logits = nlohmann::json::array();
                for (const LogitTopKEntry& entry: step_trace.top_logits) {
                    top_logits.push_back({
                        {"token_id", entry.token_id},
                        {"logit", entry.logit},
                    });
                }
                nlohmann::json step_json = {
                    {"step", step_trace.step},
                    {"top_logits", std::move(top_logits)},
                };
                if (!step_trace.final_norm.empty()) {
                    step_json["final_norm"] = step_trace.final_norm;
                }
                if (!step_trace.layer_hidden_states.empty()) {
                    step_json["layer_hidden_states"] = step_trace.layer_hidden_states;
                }
                trace_json.push_back(std::move(step_json));
            }

            std::cout << nlohmann::json({
                {"generated_token_ids", generated_ids},
                {"logit_dtype", options.dump_top_k_fp32 ? "fp32" : "bf16"},
                {"top_k", options.dump_top_k},
                {"trace", std::move(trace_json)},
            }).dump() << '\n';
        } else if (options.dump_token_ids) {
            std::cout << nlohmann::json(generated_ids).dump() << '\n';
        } else {
            std::cout << tokenizer.decode(generated_ids) << '\n';
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
