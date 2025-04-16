//
// Created by bowserj on 4/15/25.
//

#ifndef BASEWEIGHTSNAP_SMOLVLMTokenizer_H
#define BASEWEIGHTSNAP_SMOLVLMTokenizer_H

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <memory>
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

class SmolVLMTokenizer {
private:
    // Vocabulary mapping
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;
    
    // Special tokens
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
    int unk_token_id;
    int image_token_id;
    
    // BPE merges
    std::vector<std::pair<std::string, std::string>> merges;
    
    // Load vocabulary from JSON file
    void loadVocab(const std::string& vocab_path);
    
    // Load tokenizer configuration from JSON file
    void loadConfig(const std::string& config_path);
    
    // BPE encoding helper functions
    std::vector<std::string> bpe(const std::string& token);
    std::vector<std::string> whitespaceTokenize(const std::string& text);
    std::string cleanText(const std::string& text);
    
public:
    SmolVLMTokenizer(const std::string& vocab_path, const std::string& config_path);
    
    // Tokenize text into token IDs
    std::vector<int> encode(const std::string& text);
    
    // Convert token IDs back to text
    std::string decode(const std::vector<int>& token_ids);
    
    // Apply template for image-text pairs
    std::vector<int> applyTemplate(const std::string& text, bool has_image);
    
    // Get special token IDs
    int getBosTokenId() const { return bos_token_id; }
    int getEosTokenId() const { return eos_token_id; }
    int getPadTokenId() const { return pad_token_id; }
    int getUnkTokenId() const { return unk_token_id; }
    int getImageTokenId() const { return image_token_id; }
};

#endif //BASEWEIGHTSNAP_SMOLVLMTokenizer_H
