#pragma once

#include <string>

#include "llama.h"


Llama3_2 load_llama_weights(const std::string& model_path);
