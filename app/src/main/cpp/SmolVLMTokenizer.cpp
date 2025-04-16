//
// Created by bowserj on 4/15/25.
//

#include "SmolVLMTokenizer.h"
#include <android/log.h>

#define LOG_TAG "SmolVLMTokenizer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

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

SmolVLMTokenizer::SmolVLMTokenizer(const std::string& vocab_path, const std::string& config_path) {
    loadVocab(vocab_path);
    loadConfig(config_path);
}

void SmolVLMTokenizer::loadVocab(const std::string& vocab_path) {
    std::ifstream ifs(vocab_path);
    if (!ifs.is_open()) {
        LOGE("Failed to open vocabulary file: %s", vocab_path.c_str());
        throw std::runtime_error("Failed to open vocabulary file");
    }

    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document doc;
    doc.ParseStream(isw);

    if (doc.HasParseError()) {
        LOGE("Failed to parse vocabulary JSON");
        throw std::runtime_error("Failed to parse vocabulary JSON");
    }

    // Load vocabulary
    for (auto it = doc.MemberBegin(); it != doc.MemberEnd(); ++it) {
        std::string token = it->name.GetString();
        int id = it->value.GetInt();
        vocab[token] = id;
        inv_vocab[id] = token;
    }

    LOGI("Loaded vocabulary with %zu tokens", vocab.size());
}

void SmolVLMTokenizer::loadConfig(const std::string& config_path) {
    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {
        LOGE("Failed to open config file: %s", config_path.c_str());
        throw std::runtime_error("Failed to open config file");
    }

    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document doc;
    doc.ParseStream(isw);

    if (doc.HasParseError()) {
        LOGE("Failed to parse config JSON");
        throw std::runtime_error("Failed to parse config JSON");
    }

    // Load special tokens
    bos_token_id = doc["bos_token_id"].GetInt();
    eos_token_id = doc["eos_token_id"].GetInt();
    pad_token_id = doc["pad_token_id"].GetInt();
    unk_token_id = doc["unk_token_id"].GetInt();
    image_token_id = doc["image_token_id"].GetInt();

    // Load BPE merges
    const rapidjson::Value& merges_array = doc["merges"];
    for (const auto& merge : merges_array.GetArray()) {
        std::string first = merge[0].GetString();
        std::string second = merge[1].GetString();
        merges.emplace_back(first, second);
    }

    LOGI("Loaded tokenizer configuration");
}

std::string SmolVLMTokenizer::cleanText(const std::string& text) {
    std::string cleaned = text;
    // Remove control characters
    cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(), 
        [](unsigned char c) { return std::iscntrl(c); }), cleaned.end());
    // Normalize whitespace
    std::replace_if(cleaned.begin(), cleaned.end(), 
        [](unsigned char c) { return std::isspace(c); }, ' ');
    return cleaned;
}

std::vector<std::string> SmolVLMTokenizer::whitespaceTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::string> SmolVLMTokenizer::bpe(const std::string& token) {
    std::vector<std::string> word;
    for (char c : token) {
        word.push_back(std::string(1, c));
    }

    for (const auto& merge : merges) {
        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            if (i < word.size() - 1 && word[i] == merge.first && word[i + 1] == merge.second) {
                new_word.push_back(word[i] + word[i + 1]);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }
        word = new_word;
    }

    return word;
}

std::vector<int> SmolVLMTokenizer::encode(const std::string& text) {
    std::vector<int> token_ids;
    std::string cleaned_text = cleanText(text);
    std::vector<std::string> tokens = whitespaceTokenize(cleaned_text);

    for (const auto& token : tokens) {
        std::vector<std::string> bpe_tokens = bpe(token);
        for (const auto& bpe_token : bpe_tokens) {
            auto it = vocab.find(bpe_token);
            if (it != vocab.end()) {
                token_ids.push_back(it->second);
            } else {
                token_ids.push_back(unk_token_id);
            }
        }
    }

    return token_ids;
}

std::string SmolVLMTokenizer::decode(const std::vector<int>& token_ids) {
    std::string text;
    for (size_t i = 0; i < token_ids.size(); ++i) {
        auto it = inv_vocab.find(token_ids[i]);
        if (it != inv_vocab.end()) {
            text += it->second;
        } else {
            text += "<unk>";
        }
    }
    return text;
}

std::vector<int> SmolVLMTokenizer::applyTemplate(const std::string& text, bool has_image) {
    std::vector<int> token_ids;
    
    // Add BOS token
    token_ids.push_back(bos_token_id);
    
    // Add image token if present
    if (has_image) {
        token_ids.push_back(image_token_id);
    }
    
    // Add text tokens
    std::vector<int> text_tokens = encode(text);
    token_ids.insert(token_ids.end(), text_tokens.begin(), text_tokens.end());
    
    // Add EOS token
    token_ids.push_back(eos_token_id);
    
    return token_ids;
}

