#pragma once

#include <cstdint>
#include <string>
#include <vector>


struct ChatMessage
{
    std::string role;
    std::string content;
};

class TokenizerBridge
{
public:
    TokenizerBridge(
        std::string script_path,
        std::string model_id,
        std::string python_executable = "python3"
    );

    std::vector<std::int32_t> encode_chat(const std::vector<ChatMessage>& messages) const;
    std::string decode(const std::vector<std::int32_t>& token_ids) const;

private:
    std::string script_path;
    std::string model_id;
    std::string python_executable;
};
