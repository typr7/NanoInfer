#include <iostream>
#include <cstdlib>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

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
    bool benchmark = false;
    bool benchmark_json = false;
    std::string benchmark_prompt = "Explain what a GPU kernel is in one paragraph.";
    std::size_t max_new_tokens = 64;
    std::size_t warmup_runs = 1;
    std::size_t runs = 3;
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

std::string read_text_file(const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open file: " + path);
    }
    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

std::size_t parse_positive_size(const std::string& name, const std::string& value)
{
    if (value.empty() || value.front() == '-') {
        throw std::runtime_error(name + " must be a positive integer");
    }
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || parsed == 0) {
        throw std::runtime_error(name + " must be a positive integer");
    }
    return static_cast<std::size_t>(parsed);
}

std::size_t parse_non_negative_size(const std::string& name, const std::string& value)
{
    if (value.empty() || value.front() == '-') {
        throw std::runtime_error(name + " must be a non-negative integer");
    }
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size()) {
        throw std::runtime_error(name + " must be a non-negative integer");
    }
    return static_cast<std::size_t>(parsed);
}

void print_usage(const char* program)
{
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --weights <path>          safetensors weight file (default: model.safetensors)\n"
        << "  --tokenizer-script <path> tokenizer.py path (default: python/tokenizer.py)\n"
        << "  --tokenizer-model <name>  HuggingFace tokenizer id/path\n"
        << "  --python <path>           Python executable (default: python3)\n"
        << "  --benchmark               run a fixed-prompt TTFT/TPOT benchmark instead of chat\n"
        << "  --benchmark-json          print benchmark result as JSON to stdout\n"
        << "  --prompt <text>           benchmark prompt text\n"
        << "  --prompt-file <path>      read benchmark prompt text from a file\n"
        << "  --max-new-tokens <n>      generated-token cap for benchmark (default: 64)\n"
        << "  --warmup-runs <n>         untimed benchmark warmup runs (default: 1)\n"
        << "  --runs <n>                measured benchmark runs (default: 3)\n";
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
        } else if (arg == "--benchmark") {
            options.benchmark = true;
        } else if (arg == "--benchmark-json") {
            options.benchmark = true;
            options.benchmark_json = true;
        } else if (arg == "--prompt") {
            options.benchmark_prompt = require_value(arg);
        } else if (arg == "--prompt-file") {
            options.benchmark_prompt = read_text_file(require_value(arg));
        } else if (arg == "--max-new-tokens") {
            options.max_new_tokens = parse_positive_size(arg, require_value(arg));
        } else if (arg == "--warmup-runs") {
            options.warmup_runs = parse_non_negative_size(arg, require_value(arg));
        } else if (arg == "--runs") {
            options.runs = parse_positive_size(arg, require_value(arg));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    return options;
}

nlohmann::json benchmark_result_to_json(const InferenceBenchmarkResult& result)
{
    return {
        {"prompt_tokens", result.prompt_tokens},
        {"generated_tokens", result.generated_tokens},
        {"decode_tokens", result.decode_tokens},
        {"ttft_ms", result.ttft_ms},
        {"tpot_ms", result.tpot_ms},
        {"decode_ms_total", result.decode_ms_total},
        {"stopped_eos", result.stopped_eos},
    };
}

int run_benchmark(
    const RuntimeOptions& options,
    const Llama3_2& llama,
    InferenceContext& context,
    const TokenizerBridge& tokenizer,
    std::ostream& log
) {
    std::vector<ChatMessage> messages = {
        {"user", options.benchmark_prompt},
    };
    const std::vector<std::int32_t> prompt_token_ids = tokenizer.encode_chat(messages);

    if (prompt_token_ids.empty() || prompt_token_ids.size() >= MAX_TOKEN_LEN) {
        throw std::runtime_error("benchmark prompt is empty or exceeds the token window");
    }

    log << "Benchmark prompt tokens: " << prompt_token_ids.size() << "\n";
    log << "Warmup runs: " << options.warmup_runs
        << ", measured runs: " << options.runs
        << ", max new tokens: " << options.max_new_tokens << "\n";

    for (std::size_t run_idx = 0; run_idx < options.warmup_runs; run_idx++) {
        std::vector<std::int32_t> token_ids = prompt_token_ids;
        (void)inference_benchmark(token_ids, llama, context, options.max_new_tokens);
    }

    std::vector<InferenceBenchmarkResult> results;
    results.reserve(options.runs);

    double ttft_ms_total = 0.0;
    double decode_ms_total = 0.0;
    std::size_t decode_tokens_total = 0;
    std::size_t generated_tokens_total = 0;

    for (std::size_t run_idx = 0; run_idx < options.runs; run_idx++) {
        std::vector<std::int32_t> token_ids = prompt_token_ids;
        InferenceBenchmarkResult result = inference_benchmark(
            token_ids,
            llama,
            context,
            options.max_new_tokens
        );
        ttft_ms_total += result.ttft_ms;
        decode_ms_total += result.decode_ms_total;
        decode_tokens_total += result.decode_tokens;
        generated_tokens_total += result.generated_tokens;
        results.push_back(result);

        if (!options.benchmark_json) {
            log << "Run " << (run_idx + 1)
                << ": TTFT " << std::fixed << std::setprecision(3) << result.ttft_ms
                << " ms, TPOT " << result.tpot_ms
                << " ms, generated " << result.generated_tokens << " tokens\n";
        }
    }

    nlohmann::json output;
    output["backend"] = "nanoinfer";
    output["prompt_tokens"] = prompt_token_ids.size();
    output["max_new_tokens"] = options.max_new_tokens;
    output["warmup_runs"] = options.warmup_runs;
    output["runs"] = nlohmann::json::array();
    for (const InferenceBenchmarkResult& result: results) {
        output["runs"].push_back(benchmark_result_to_json(result));
    }
    output["summary"] = {
        {"runs", options.runs},
        {"ttft_ms_avg", ttft_ms_total / static_cast<double>(options.runs)},
        {"tpot_ms_avg", decode_tokens_total == 0
            ? 0.0
            : decode_ms_total / static_cast<double>(decode_tokens_total)},
        {"decode_ms_total", decode_ms_total},
        {"decode_tokens_total", decode_tokens_total},
        {"generated_tokens_avg", generated_tokens_total / static_cast<double>(options.runs)},
    };

    if (options.benchmark_json) {
        std::cout << output.dump(2) << '\n';
    } else {
        const auto& summary = output["summary"];
        log << "Summary: TTFT "
            << std::fixed << std::setprecision(3)
            << summary["ttft_ms_avg"].get<double>()
            << " ms, TPOT "
            << summary["tpot_ms_avg"].get<double>()
            << " ms/token\n";
    }

    return 0;
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
        std::cout << "[context] Older turns were dropped to fit the "
                  << MAX_TOKEN_LEN << " token window.\n";
    }

    return token_ids;
}

} // namespace


int check_gpu_status(std::ostream& output)
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA device found.\n";
        return -1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    output << "Device: " << prop.name << "\n";
    output << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    constexpr double bytes_per_mib = double(1LL << 20);
    constexpr double bytes_per_gib = double(1LL << 30);
    output << "Global memory: " << prop.totalGlobalMem / bytes_per_mib << "MB\n";
    output << "SM count: " << prop.multiProcessorCount << "\n";
    output << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    output << "Free memory: " << free_mem / bytes_per_gib
           << "GB, total memory: " << total_mem / bytes_per_gib << "GB" << std::endl;
    return 0;
}

int main(int argc, char** argv)
{
    try {
        const RuntimeOptions options = parse_options(argc, argv);
        std::ostream& log = options.benchmark_json ? std::cerr : std::cout;

        if (check_gpu_status(log) == -1) {
            return -1;
        }
        log << "\n\n\n";

        log << "Loading weights from " << options.weights_path << "...\n";
        Llama3_2 llama = load_llama_weights(options.weights_path);
        InferenceContext context;
        TokenizerBridge tokenizer(
            options.tokenizer_script,
            options.tokenizer_model,
            options.python_executable
        );

        if (options.benchmark) {
            return run_benchmark(options, llama, context, tokenizer, log);
        }

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
                std::cout << "Input is too long for the "
                          << MAX_TOKEN_LEN << " token window.\n";
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
