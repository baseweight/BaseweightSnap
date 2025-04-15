//
// Created by bowserj on 4/15/25.
//

#include "SmolVLMTokenizer.h"

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <memory>


// Helper to tokenize text
std::vector<int> SmolVLMTokenizer::encode(const std::string& text) {
    // NOTE: This is a placeholder. You'll need a proper tokenizer implementation.
    // For now, we'll use a very naive approach that won't work properly
    std::vector<int> tokens;
    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        if (token_to_id.find(word) != token_to_id.end()) {
            tokens.push_back(token_to_id[word]);
        } else {
            // Unknown token handling
            tokens.push_back(0); // Unknown token ID
        }
    }

    return tokens;
}

SmolVLMTokenizer::SmolVLMTokenizer(const std::string &vocab_file) {
    // Load vocabulary from file
    loadVocabulary(vocab_file);

    // Set default tokens
    image_token_id = 32001; // Placeholder - you need the actual value
    eos_token_id = 2; // Placeholder - you need the actual value
}

void SmolVLMTokenizer::loadVocabulary(const std::string& vocab_file) {
    // Load vocabulary from file
    std::ifstream file(vocab_file);
    std::string line;
    int id = 0;

    while (std::getline(file, line)) {
        // Parse the vocabulary line
        // Format might differ based on your actual vocabulary file
        token_to_id[line] = id;
        id_to_token[id] = line;
        id++;
    }
}

std::vector<int> SmolVLMTokenizer::applyTemplate(const std::string& user_text, bool has_image) {
    // Apply chat template similar to processor.apply_chat_template
    std::string prompt = "<|user|>\n";

    if (has_image) {
        prompt += "<image>\n";
    }

    prompt += user_text + "\n<|assistant|>\n";

    // Tokenize the prompt
    std::vector<int> tokens = encode(prompt);

    // If there's an image, replace the image token placeholder
    if (has_image) {
        for (size_t i = 0; i < tokens.size(); i++) {
            if (tokens[i] == token_to_id["<image>"]) {
                tokens[i] = image_token_id;
            }
        }
    }

    return tokens;
}

std::string SmolVLMTokenizer::decode(const std::vector<int>& tokens) {
    std::string result;
    for (int token : tokens) {
        if (id_to_token.find(token) != id_to_token.end()) {
            result += id_to_token[token] + " ";
        }
    }
    return result;
}

