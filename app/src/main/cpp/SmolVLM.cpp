//
// Created by bowserj on 4/15/25.
//

#include "SmolVLM.h"
#include "ImageProcessor.h"
#include <android/log.h>
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

#define LOG_TAG "SmolVLM"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Initialize the static instance pointer
std::unique_ptr<SmolVLM> SmolVLM::instance = nullptr;

// Helper function to create ONNX tensor
template <typename T>
Ort::Value SmolVLM::createTensor(const std::vector<T>& data, const std::vector<int64_t>& shape) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

    auto tensor = Ort::Value::CreateTensor<T>(
            memory_info, const_cast<T*>(data.data()), data.size(),
            shape.data(), shape.size());

    return tensor;
}

// Private constructor implementation
SmolVLM::SmolVLM(const std::string& vision_model_path,
                 const std::string& embed_model_path,
                 const std::string& decoder_model_path,
                 const std::string& vocab_path,
                 const std::string& tokenizer_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "SmolVLM"),
      tokenizer(vocab_path, tokenizer_path) {
    
    LOGI("Starting SmolVLM initialization...");
    
    try {
        // Initialize session options
        LOGI("Setting up session options...");
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Initialize vision session
        LOGI("Loading vision model from: %s", vision_model_path.c_str());
        vision_session = std::make_unique<Ort::Session>(env, vision_model_path.c_str(), session_options);
        LOGI("Vision model loaded successfully");
        
        // Initialize embed session
        LOGI("Loading embed model from: %s", embed_model_path.c_str());
        embed_session = std::make_unique<Ort::Session>(env, embed_model_path.c_str(), session_options);
        LOGI("Embed model loaded successfully");
        
        // Initialize decoder session
        LOGI("Loading decoder model from: %s", decoder_model_path.c_str());
        decoder_session = std::make_unique<Ort::Session>(env, decoder_model_path.c_str(), session_options);
        LOGI("Decoder model loaded successfully");
        
        // Get special token IDs from tokenizer
        //LOGI("Getting token IDs from tokenizer...");
        LOGI("hardcoding token IDs from Python implementation");
        LOGI("PLEASE FIX THIS");
        //eos_token_id = tokenizer.getEosTokenId();
        //image_token_id = tokenizer.getImageTokenId();
        eos_token_id = 2;
        image_token_id = 49190;

        // Initialize model configuration
        LOGI("Setting model configuration...");
        num_key_value_heads = 5;
        head_dim = 64;
        num_hidden_layers = 32;
        
        LOGI("SmolVLM initialization completed successfully");
        LOGI("Configuration:");
        LOGI("  - num_key_value_heads: %d", num_key_value_heads);
        LOGI("  - head_dim: %d", head_dim);
        LOGI("  - num_hidden_layers: %d", num_hidden_layers);
        LOGI("  - eos_token_id: %d", eos_token_id);
        LOGI("  - image_token_id: %d", image_token_id);
    } catch (const Ort::Exception& e) {
        LOGE("ONNX Runtime error during initialization: %s", e.what());
        throw std::runtime_error("Failed to initialize ONNX Runtime sessions: " + std::string(e.what()));
    } catch (const std::exception& e) {
        LOGE("Error during initialization: %s", e.what());
        throw;
    }
}

std::string SmolVLM::generateText(const std::string& prompt, const cv::Mat& image, int max_new_tokens) {
    LOGI("Generating text for prompt: '%s'", prompt.c_str());
    
    // 1. Process inputs
    bool has_image = !image.empty();
    
    // Create the chat template
    std::string chat_template = R"({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": ")" + prompt + R"("}
        ]
    })";
    
    // Tokenize the chat template
    std::vector<int> input_ids = tokenizer.applyTemplate(chat_template, has_image);
    LOGI("Tokenized input: %zu tokens", input_ids.size());

    // Prepare batch dimension
    const int batch_size = 1;

    // 2. Prepare attention mask and position IDs
    std::vector<int64_t> attention_mask(input_ids.size(), 1);
    std::vector<int64_t> position_ids(input_ids.size());
    int pos = 0;
    for (size_t i = 0; i < position_ids.size(); i++) {
        position_ids[i] = pos;
        pos += attention_mask[i];
    }

    // 3. Process image if present
    std::vector<float> image_features;
    if (has_image) {
        LOGI("Processing image with dimensions: %dx%d", image.cols, image.rows);
        
        // Ensure image is 512x512
        const int target_size = 512;
        cv::Mat resized_image;
        if (image.cols != target_size || image.rows != target_size) {
            cv::resize(image, resized_image, cv::Size(target_size, target_size), 0, 0, cv::INTER_AREA);
            LOGI("Resized image to %dx%d", resized_image.cols, resized_image.rows);
        } else {
            resized_image = image;
        }
        
        // Preprocess image
        std::vector<float> pixel_values = ImageProcessor::preprocess(resized_image, target_size, target_size);
        
        // Create attention mask array with enough values for the full tensor
        const size_t attention_mask_size = 1 * 1 * target_size * target_size;  // batch_size * sequence_length * height * width
        std::vector<uint8_t> pixel_attention_mask(attention_mask_size, 1);  // Use uint8_t instead of bool

        // Create input tensors for vision encoder
        // Add batch dimension to match expected shape [batch_size, channels, height, width]
        auto pixel_values_tensor = createTensor<float>(
                pixel_values, {1, 1, 3, target_size, target_size});  // [batch_size, sequence_length, channels, height, width]

        // Create shape array for attention mask [batch_size, sequence_length, height, width]
        const int64_t attention_mask_shape[] = {1, 1, target_size, target_size};

        // Create a bool tensor from uint8_t data
        auto pixel_attention_mask_tensor = Ort::Value::CreateTensor<bool>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
            reinterpret_cast<bool*>(pixel_attention_mask.data()),
            attention_mask_size,
            attention_mask_shape,
            4  // rank
        );

        // Define input and output names
        const char* vision_input_names[] = {"pixel_values", "pixel_attention_mask"};
        const char* vision_output_names[] = {"image_features"};

        // Run vision encoder
        std::vector<Ort::Value> vision_inputs;
        vision_inputs.push_back(std::move(pixel_values_tensor));
        vision_inputs.push_back(std::move(pixel_attention_mask_tensor));

        auto vision_outputs = vision_session->Run(
                Ort::RunOptions{nullptr},
                vision_input_names, vision_inputs.data(), vision_inputs.size(),
                vision_output_names, 1);

        // Get image features
        float* features_data = vision_outputs[0].GetTensorMutableData<float>();
        size_t features_size = vision_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        image_features.assign(features_data, features_data + features_size);
        
        LOGI("Extracted image features: %zu dimensions", features_size);
    }

    // 4. Initialize past key values
    std::unordered_map<std::string, std::vector<float>> past_key_values;
    for (int layer = 0; layer < num_hidden_layers; layer++) {
        for (const char* kv : {"key", "value"}) {
            std::string name = "past_key_values." + std::to_string(layer) + "." + kv;
            past_key_values[name] = std::vector<float>();
        }
    }

    // 5. Generation loop
    std::vector<int> generated_tokens;
    LOGI("Starting generation with max_new_tokens=%d", max_new_tokens);

    for (int i = 0; i < max_new_tokens; i++) {
        // Create input tensor for token embeddings
        std::vector<int64_t> input_tensor(input_ids.begin(), input_ids.end());
        auto input_tensor_value = createTensor<int64_t>(
                input_tensor, {batch_size, static_cast<int64_t>(input_ids.size())});

        // Define input and output names for embedding
        const char* embed_input_names[] = {"input_ids"};
        const char* embed_output_names[] = {"inputs_embeds"};

        // Run token embedding
        std::vector<Ort::Value> embed_inputs;
        embed_inputs.push_back(std::move(input_tensor_value));

        auto embed_outputs = embed_session->Run(
                Ort::RunOptions{nullptr},
                embed_input_names, embed_inputs.data(), embed_inputs.size(),
                embed_output_names, 1);

        // Get embedded tokens
        float* embeds_data = embed_outputs[0].GetTensorMutableData<float>();
        auto embeds_shape = embed_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t embeds_size = embed_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> inputs_embeds(embeds_data, embeds_data + embeds_size);

        // Replace image token embeddings with image features if necessary
        if (!image_features.empty()) {
            for (size_t j = 0; j < input_ids.size(); j++) {
                if (input_ids[j] == image_token_id) {
                    size_t embed_dim = embeds_shape[2];
                    for (size_t k = 0; k < embed_dim; k++) {
                        inputs_embeds[j * embed_dim + k] = image_features[k];
                    }
                }
            }
        }

        // Create input tensors for decoder
        auto attention_mask_tensor = createTensor<int64_t>(
                attention_mask, {batch_size, static_cast<int64_t>(attention_mask.size())});

        auto position_ids_tensor = createTensor<int64_t>(
                position_ids, {batch_size, static_cast<int64_t>(position_ids.size())});

        auto inputs_embeds_tensor = createTensor<float>(
                inputs_embeds, embeds_shape);

        // Prepare decoder inputs
        std::vector<Ort::Value> decoder_inputs;
        decoder_inputs.push_back(std::move(inputs_embeds_tensor));
        decoder_inputs.push_back(std::move(attention_mask_tensor));
        decoder_inputs.push_back(std::move(position_ids_tensor));

        // Add past key values to inputs
        std::vector<std::string> decoder_input_names = {"inputs_embeds", "attention_mask", "position_ids"};

        for (int layer = 0; layer < num_hidden_layers; layer++) {
            for (const char* kv : {"key", "value"}) {
                std::string name = "past_key_values." + std::to_string(layer) + "." + kv;
                decoder_input_names.push_back(name);

                std::vector<int64_t> kv_shape;
                if (past_key_values[name].empty()) {
                    kv_shape = {batch_size, num_key_value_heads, 0, head_dim};
                } else {
                    int seq_len = past_key_values[name].size() / (batch_size * num_key_value_heads * head_dim);
                    kv_shape = {batch_size, num_key_value_heads, seq_len, head_dim};
                }

                decoder_inputs.push_back(createTensor<float>(past_key_values[name], kv_shape));
            }
        }

        // Run decoder
        std::vector<const char*> decoder_input_names_c;
        for (const auto& name : decoder_input_names) {
            decoder_input_names_c.push_back(name.c_str());
        }

        // Get all output names
        std::vector<std::string> output_names = {"logits"};
        for (int layer = 0; layer < num_hidden_layers; layer++) {
            for (const char* kv : {"key", "value"}) {
                output_names.push_back("present." + std::to_string(layer) + "." + kv);
            }
        }

        std::vector<const char*> output_names_c;
        for (const auto& name : output_names) {
            output_names_c.push_back(name.c_str());
        }

        auto decoder_outputs = decoder_session->Run(
                Ort::RunOptions{nullptr},
                decoder_input_names_c.data(), decoder_inputs.data(), decoder_inputs.size(),
                output_names_c.data(), output_names.size());

        // Process decoder outputs
        float* logits_data = decoder_outputs[0].GetTensorMutableData<float>();
        auto logits_shape = decoder_outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        // Get the next token (argmax of last logits)
        int vocab_size = logits_shape[2];
        int next_token = 0;
        float max_logit = -std::numeric_limits<float>::infinity();

        LOGI("Processing logits for next token, vocab_size=%d", vocab_size);
        for (int v = 0; v < vocab_size; v++) {
            float logit = logits_data[(logits_shape[1] - 1) * vocab_size + v];
            if (logit > max_logit) {
                max_logit = logit;
                next_token = v;
            }
        }

        // Validate the token ID
        if (next_token < 0 || next_token >= vocab_size) {
            LOGE("Invalid token ID generated: %d (vocab_size=%d)", next_token, vocab_size);
            next_token = eos_token_id;  // Force end of generation
        }

        LOGI("Generated token %d: id=%d, logit=%.4f", i, next_token, max_logit);

        // Store updated past key values
        for (size_t j = 1; j < decoder_outputs.size(); j++) {
            std::string key = output_names[j];
            key.replace(0, 8, "past_key_values");

            float* kv_data = decoder_outputs[j].GetTensorMutableData<float>();
            size_t kv_size = decoder_outputs[j].GetTensorTypeAndShapeInfo().GetElementCount();
            past_key_values[key].assign(kv_data, kv_data + kv_size);
        }

        // Update for next iteration
        input_ids = {next_token};
        attention_mask = {1};
        position_ids = {position_ids.back() + 1};

        // Add to generated tokens
        generated_tokens.push_back(next_token);

        // Check for EOS token
        if (next_token == eos_token_id) {
            LOGI("Generated EOS token, stopping generation");
            break;
        }
    }

    std::string result = tokenizer.decode(generated_tokens);
    LOGI("Generated text: '%s'", result.c_str());
    return result;
}