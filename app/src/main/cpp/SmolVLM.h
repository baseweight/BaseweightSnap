//
// Created by bowserj on 4/15/25.
//

#ifndef BASEWEIGHTSNAP_SMOLVLM_H
#define BASEWEIGHTSNAP_SMOLVLM_H
#include "SmolVLMTokenizer.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <memory>

class SmolVLM {
private:
    // ONNX Runtime session pointers
    Ort::Session vision_session;
    Ort::Session embed_session;
    Ort::Session decoder_session;

    // Tokenizer instance
    SmolVLMTokenizer tokenizer;

    // Model configuration
    int num_key_value_heads;
    int head_dim;
    int num_hidden_layers;
    int eos_token_id;
    int image_token_id;

    // ONNX Runtime environment
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;

    template <typename T>
    Ort::Value createTensor(const std::vector<T>&data, const std::vector<int64_t>& shape);

public:
    SmolVLM(const std::string& vision_model_path,
            const std::string& embed_model_path,
            const std::string& decoder_model_path,
            const std::string& vocab_path);

    // TODO: Support mutliple images???  I mean, an image is just another embedding, right?
    std::string generateText(const std::string& prompt, const cv::Mat& image, int max_new_tokens);

};


#endif //BASEWEIGHTSNAP_SMOLVLM_H
