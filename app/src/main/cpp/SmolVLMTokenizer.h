//
// Created by bowserj on 4/15/25.
//

#ifndef BASEWEIGHTSNAP_SMOLVLMTOKENIZER_H
#define BASEWEIGHTSNAP_SMOLVLMTOKENIZER_H
#import <unordered_map>
#import <string>

class SmolVLMTokenizer {
private:
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
    int image_token_id;
    int eos_token_id;

    std::vector<int> encode(const std::string & text);


public:
    SmolVLMTokenizer(const std::string& vocab_file);
    void loadVocabulary(const std::string& vocab_file);
    std::vector<int> applyTemplate(const std::string& user_text, bool has_image);
    std::string decode(const std::vector<int>& tokens);

    int getEosTokenId() const { return eos_token_id; }
    int getImageTokenId() const { return image_token_id; }

};


#endif //BASEWEIGHTSNAP_SMOLVLMTOKENIZER_H
