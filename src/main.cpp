#include <iostream>
#include <cstdlib>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "macro.h"
#include "llama_weights.h"
#include "inference.h"
#include "tokenizer_bridge.h"


namespace
{

struct RuntimeOptions
{
    std::string weights_path = "model.safetensors";
    std::string tokenizer_script = "python/tokenizer.py";
    std::string tokenizer_model = "meta-llama/Llama-3.2-1B-Instruct";
    std::string python_executable = "python3";
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
        << "Usage: " << program << " [options]\n"
        << "  --weights <path>          safetensors weight file (default: model.safetensors)\n"
        << "  --tokenizer-script <path> tokenizer.py path (default: python/tokenizer.py)\n"
        << "  --tokenizer-model <name>  HuggingFace tokenizer id/path\n"
        << "  --python <path>           Python executable (default: python3)\n";
}

RuntimeOptions parse_options(int argc, char** argv)
{
    RuntimeOptions options;
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

        if (arg == "--weights") {
            options.weights_path = require_value(arg);
        } else if (arg == "--tokenizer-script") {
            options.tokenizer_script = require_value(arg);
        } else if (arg == "--tokenizer-model") {
            options.tokenizer_model = require_value(arg);
        } else if (arg == "--python") {
            options.python_executable = require_value(arg);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    return options;
}

bool should_exit(const std::string& input)
{
    return input == "/exit" || input == "/quit";
}

std::vector<std::int32_t> encode_prompt_with_budget(
    const TokenizerBridge& tokenizer,
    std::vector<ChatMessage>& messages
) {
    std::vector<std::int32_t> token_ids = tokenizer.encode_chat(messages);
    bool dropped_history = false;

    while (token_ids.size() >= MAX_TOKEN_LEN && messages.size() > 1) {
        messages.erase(messages.begin());
        dropped_history = true;
        token_ids = tokenizer.encode_chat(messages);
    }

    if (dropped_history) {
        std::cout << "[context] Older turns were dropped to fit the 512 token window.\n";
    }

    return token_ids;
}

} // namespace


int check_gpu_status()
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA device found.\n";
        return -1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    constexpr double bytes_per_mib = double(1LL << 20);
    constexpr double bytes_per_gib = double(1LL << 30);
    std::cout << "Global memory: " << prop.totalGlobalMem / bytes_per_mib << "MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "Free memory: " << free_mem / bytes_per_gib << "GB, total memory: " << total_mem / bytes_per_gib << "GB" << std::endl;
    return 0;
}

int main(int argc, char** argv)
{
    try {
        const RuntimeOptions options = parse_options(argc, argv);

        if (check_gpu_status() == -1) {
            return -1;
        }
        std::cout << "\n\n\n";

        std::cout << "Loading weights from " << options.weights_path << "...\n";
        Llama3_2 llama = load_llama_weights(options.weights_path);
        InferenceContext context;
        TokenizerBridge tokenizer(
            options.tokenizer_script,
            options.tokenizer_model,
            options.python_executable
        );

        std::cout << "Ready. Type /exit to quit, /reset to clear the conversation.\n";

        std::vector<ChatMessage> messages;
        std::string user_input;
        while (true) {
            std::cout << "\nuser> " << std::flush;
            if (!std::getline(std::cin, user_input)) {
                break;
            }

            if (should_exit(user_input)) {
                break;
            }

            if (user_input == "/reset") {
                messages.clear();
                std::cout << "[context] Conversation reset.\n";
                continue;
            }

            if (user_input.empty()) {
                continue;
            }

            messages.push_back({"user", user_input});
            std::vector<ChatMessage> prompt_messages = messages;
            std::vector<std::int32_t> token_ids = encode_prompt_with_budget(tokenizer, prompt_messages);

            if (token_ids.empty() || token_ids.size() >= MAX_TOKEN_LEN) {
                messages.pop_back();
                std::cout << "Input is too long for the 512 token window.\n";
                continue;
            }

            const std::size_t prompt_token_count = token_ids.size();
            std::cout << "assistant> " << std::flush;

            inference(token_ids, llama, context);

            std::vector<std::int32_t> generated_ids(
                token_ids.begin() + static_cast<std::ptrdiff_t>(prompt_token_count),
                token_ids.end()
            );
            const std::string assistant_text = tokenizer.decode(generated_ids);
            std::cout << assistant_text << '\n';

            prompt_messages.push_back({"assistant", assistant_text});
            messages = std::move(prompt_messages);
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

}
