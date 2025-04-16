//
// Created by bowserj on 4/15/25.
//

#include "SmolVLMTokenizer.h"
#include <android/log.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <unordered_set>

#define LOG_TAG "SmolVLMTokenizer"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
#include <cmath>
#include <memory>


// Helper function to clean text
std::string SmolVLMTokenizer::cleanText(const std::string& text) {
    std::string cleaned = text;
    
    // Replace control characters with space
    std::replace_if(cleaned.begin(), cleaned.end(),
        [](unsigned char c) { return std::iscntrl(c); }, ' ');
    
    // Normalize whitespace
    cleaned = std::regex_replace(cleaned, std::regex("\\s+"), " ");
    
    // Trim
    cleaned = std::regex_replace(cleaned, std::regex("^\\s+|\\s+$"), "");
    
    LOGV("Cleaned text: '%s' -> '%s'", text.c_str(), cleaned.c_str());
    return cleaned;
}

// Helper function to split text into words
std::vector<std::string> SmolVLMTokenizer::whitespaceTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    LOGV("Tokenized into %zu words", tokens.size());
    return tokens;
}

// Helper function to split word into characters
std::vector<std::string> SmolVLMTokenizer::splitIntoChars(const std::string& word) {
    std::vector<std::string> chars;
    for (char c : word) {
        chars.push_back(std::string(1, c));
    }
    LOGV("Split word '%s' into %zu characters", word.c_str(), chars.size());
    return chars;
}

// Helper function to get BPE pairs
std::vector<std::pair<std::string, std::string>> SmolVLMTokenizer::getPairs(const std::vector<std::string>& word) {
    std::vector<std::pair<std::string, std::string>> pairs;
    if (word.size() < 2) return pairs;
    
    for (size_t i = 0; i < word.size() - 1; i++) {
        pairs.emplace_back(word[i], word[i + 1]);
    }
    LOGV("Found %zu BPE pairs", pairs.size());
    return pairs;
}

// BPE encoding for a single word
std::vector<std::string> SmolVLMTokenizer::bpe(const std::string& word) {
    if (word.empty()) return {};
    
    std::vector<std::string> word_chars = splitIntoChars(word);
    if (word_chars.size() == 1) return word_chars;
    
    int merge_count = 0;
    while (true) {
        auto pairs = getPairs(word_chars);
        if (pairs.empty()) break;
        
        // Find the highest priority merge
        std::pair<std::string, std::string> best_pair;
        int best_priority = -1;
        
        for (const auto& pair : pairs) {
            std::string merged = pair.first + pair.second;
            auto it = bpe_ranks.find(merged);
            if (it != bpe_ranks.end() && it->second > best_priority) {
                best_pair = pair;
                best_priority = it->second;
            }
        }
        
        if (best_priority == -1) break;
        
        // Merge the best pair
        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word_chars.size()) {
            if (i < word_chars.size() - 1 && 
                word_chars[i] == best_pair.first && 
                word_chars[i + 1] == best_pair.second) {
                new_word.push_back(best_pair.first + best_pair.second);
                i += 2;
                merge_count++;
            } else {
                new_word.push_back(word_chars[i]);
                i += 1;
            }
        }
        word_chars = new_word;
    }
    
    LOGV("BPE encoded '%s' with %d merges into %zu tokens", 
         word.c_str(), merge_count, word_chars.size());
    return word_chars;
}

// Main encoding function
std::vector<int> SmolVLMTokenizer::encode(const std::string& text) {
    std::vector<int> token_ids;
    
    // Clean and normalize text
    std::string cleaned_text = cleanText(text);
    
    // Split into words
    auto words = whitespaceTokenize(cleaned_text);
    
    // Process each word
    for (const auto& word : words) {
        // Apply BPE
        auto bpe_tokens = bpe(word);
        
        // Convert to token IDs
        for (const auto& token : bpe_tokens) {
            auto it = token_to_id.find(token);
            if (it != token_to_id.end()) {
                token_ids.push_back(it->second);
            } else {
                // Handle unknown tokens
                LOGW("Unknown token: '%s'", token.c_str());
                token_ids.push_back(unk_token_id);
            }
        }
    }
    
    LOGI("Encoded text into %zu tokens", token_ids.size());
    return token_ids;
}

// Constructor implementation
SmolVLMTokenizer::SmolVLMTokenizer(const std::string& vocab_path, const std::string& config_path) {
    loadVocab(vocab_path);
    loadConfig(config_path);
}

// Load vocabulary from JSON
void SmolVLMTokenizer::loadVocab(const std::string& vocab_path) {
    std::ifstream ifs(vocab_path);
    if (!ifs.is_open()) {
        LOGE("Failed to open vocab file: %s", vocab_path.c_str());
        return;
    }
    
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document doc;
    doc.ParseStream(isw);
    
    if (doc.HasParseError()) {
        LOGE("Failed to parse vocab JSON: %s", 
             rapidjson::GetParseError_En(doc.GetParseError()));
        return;
    }
    
    // Load token to ID mapping
    for (auto it = doc.MemberBegin(); it != doc.MemberEnd(); ++it) {
        std::string token = it->name.GetString();
        int id = it->value.GetInt();
        token_to_id[token] = id;
        id_to_token[id] = token;
    }
    
    LOGI("Loaded vocabulary with %zu tokens", token_to_id.size());
}

// Load tokenizer configuration from JSON
void SmolVLMTokenizer::loadConfig(const std::string& config_path) {
    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {
        LOGE("Failed to open config file: %s", config_path.c_str());
        return;
    }
    
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document doc;
    doc.ParseStream(isw);
    
    if (doc.HasParseError()) {
        LOGE("Failed to parse config JSON: %s", 
             rapidjson::GetParseError_En(doc.GetParseError()));
        return;
    }
    
    // Load special tokens
    if (doc.HasMember("special_tokens")) {
        const auto& special_tokens = doc["special_tokens"];
        bos_token_id = special_tokens["bos_token_id"].GetInt();
        eos_token_id = special_tokens["eos_token_id"].GetInt();
        unk_token_id = special_tokens["unk_token_id"].GetInt();
        image_token_id = special_tokens["image_token_id"].GetInt();
        LOGI("Loaded special tokens: BOS=%d, EOS=%d, UNK=%d, IMAGE=%d", 
             bos_token_id, eos_token_id, unk_token_id, image_token_id);
    } else {
        LOGE("Missing special_tokens in config");
    }
    
    // Load BPE merge rules
    if (doc.HasMember("merges")) {
        const auto& merges = doc["merges"];
        int priority = 0;
        for (const auto& merge : merges.GetArray()) {
            std::string pair = merge.GetString();
            bpe_ranks[pair] = priority++;
        }
        LOGI("Loaded %zu BPE merge rules", bpe_ranks.size());
    } else {
        LOGE("Missing merges in config");
    }
}

// Apply template for text with optional image
std::vector<int> SmolVLMTokenizer::applyTemplate(const std::string& text, bool has_image) {
    std::vector<int> token_ids;
    
    // Add BOS token
    token_ids.push_back(bos_token_id);
    
    // If there's an image, add the image token
    if (has_image) {
        token_ids.push_back(image_token_id);
    }
    
    // Add the text tokens
    std::vector<int> text_tokens = encode(text);
    token_ids.insert(token_ids.end(), text_tokens.begin(), text_tokens.end());
    
    // Add EOS token
    token_ids.push_back(eos_token_id);
    
    return token_ids;
}

// Decode tokens back to text
std::string SmolVLMTokenizer::decode(const std::vector<int>& token_ids) {
    std::string text;
    for (size_t i = 0; i < token_ids.size(); i++) {
        int token_id = token_ids[i];
        
        // Skip special tokens
        if (token_id == bos_token_id || token_id == eos_token_id || 
            token_id == image_token_id || token_id == unk_token_id) {
            continue;
        }
        
        auto it = id_to_token.find(token_id);
        if (it != id_to_token.end()) {
            text += it->second;
        }
    }
    
    LOGI("Decoded %zu tokens to text: '%s'", token_ids.size(), text.c_str());
    return text;
}

