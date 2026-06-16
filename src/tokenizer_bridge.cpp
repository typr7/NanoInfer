#include "tokenizer_bridge.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <sys/wait.h>
#include <unistd.h>

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <nlohmann/json.hpp>


namespace
{

using json = nlohmann::json;

class TempFile
{
public:
    explicit TempFile(const std::string& stem)
    {
        std::string pattern = "/tmp/" + stem + ".XXXXXX";
        std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
        mutable_pattern.push_back('\0');

        const int fd = mkstemp(mutable_pattern.data());
        if (fd == -1) {
            throw std::system_error(errno, std::generic_category(), "mkstemp failed");
        }
        close(fd);
        path = mutable_pattern.data();
    }

    ~TempFile()
    {
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    const std::string& filename() const
    {
        return path;
    }

    void write(const std::string& content) const
    {
        std::ofstream file(path);
        if (!file) {
            throw std::runtime_error("failed to open temporary file: " + path);
        }
        file << content;
        if (!file) {
            throw std::runtime_error("failed to write temporary file: " + path);
        }
    }

private:
    std::string path;
};

std::string shell_quote(const std::string& value)
{
    std::string quoted = "'";
    for (char ch: value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::string run_command(const std::string& command)
{
    std::array<char, 4096> buffer = {};
    std::string output;

    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("failed to start tokenizer subprocess");
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

    const int status = pclose(pipe);
    if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw std::runtime_error("tokenizer subprocess failed with command: " + command);
    }

    return output;
}

std::string strip_single_trailing_newline(std::string text)
{
    if (!text.empty() && text.back() == '\n') {
        text.pop_back();
        if (!text.empty() && text.back() == '\r') {
            text.pop_back();
        }
    }
    return text;
}

} // namespace

TokenizerBridge::TokenizerBridge(
    std::string script_path_,
    std::string model_id_,
    std::string python_executable_
):
    script_path(std::move(script_path_)),
    model_id(std::move(model_id_)),
    python_executable(std::move(python_executable_))
{
}

std::vector<std::int32_t> TokenizerBridge::encode_chat(
    const std::vector<ChatMessage>& messages
) const {
    json message_array = json::array();
    for (const ChatMessage& message: messages) {
        message_array.push_back({
            {"role", message.role},
            {"content", message.content},
        });
    }

    TempFile messages_file("tvllm_messages");
    messages_file.write(message_array.dump());

    const std::string command = shell_quote(python_executable)
                              + " " + shell_quote(script_path)
                              + " --model " + shell_quote(model_id)
                              + " --chat-template"
                              + " --messages-file " + shell_quote(messages_file.filename());

    const std::string output = run_command(command);
    const json token_json = json::parse(output);

    std::vector<std::int32_t> token_ids;
    token_ids.reserve(token_json.size());
    for (const auto& token: token_json) {
        token_ids.push_back(token.get<std::int32_t>());
    }
    return token_ids;
}

std::string TokenizerBridge::decode(const std::vector<std::int32_t>& token_ids) const
{
    if (token_ids.empty()) {
        return "";
    }

    TempFile ids_file("tvllm_token_ids");
    ids_file.write(json(token_ids).dump());

    const std::string command = shell_quote(python_executable)
                              + " " + shell_quote(script_path)
                              + " --model " + shell_quote(model_id)
                              + " --decode"
                              + " --skip-special-tokens"
                              + " --ids-file " + shell_quote(ids_file.filename());

    return strip_single_trailing_newline(run_command(command));
}
