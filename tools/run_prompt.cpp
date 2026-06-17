#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
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
        << "  --dump-token-ids         print generated token IDs as JSON instead of decoded text\n";
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
        inference(token_ids, weights, context, options.max_new_tokens);

        std::vector<std::int32_t> generated_ids(
            token_ids.begin() + static_cast<std::ptrdiff_t>(prompt_token_count),
            token_ids.end()
        );

        if (options.dump_token_ids) {
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
