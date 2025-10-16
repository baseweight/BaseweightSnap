/**
 * @file nanovlm_android.cpp
 * @brief JNI wrapper for nanoVLM ExecuTorch inference on Android
 *
 * This file provides JNI bindings for running nanoVLM inference using ExecuTorch.
 * It replaces the llama.cpp-based SmolVLM2 implementation with a more efficient
 * and compatible solution.
 */

#include <android/log.h>
#include <jni.h>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cmath>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include "nanovlm_preprocessor.h"  // Rust tokenizer FFI
#include "image_preprocessor.h"     // C++ image preprocessing
#include "config_loader.h"          // Config loading

#define TAG "nanovlm-android"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGd(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

using namespace torch::executor;
using executorch::extension::from_blob;
using executorch::extension::TensorPtr;
using executorch::extension::clone_tensor_ptr;
using executorch::runtime::EValue;

// Global instance of the inference engine
class NanoVLMAndroidInference {
private:
    std::unique_ptr<Module> vision_encoder_;
    std::unique_ptr<Module> modality_projector_;
    std::unique_ptr<Module> prefill_decoder_;
    std::unique_ptr<Module> decode_decoder_;
    std::unique_ptr<Module> token_embedding_;
    std::unique_ptr<Module> lm_head_;

    TokenizerHandle* tokenizer_;
    NanoVLMConfig config_;

    // Store preprocessed image data between calls
    nanovlm::MultiImageResult current_image_data_;
    std::vector<std::vector<float>> current_image_embeddings_;
    bool has_image_ = false;

public:
    NanoVLMAndroidInference() : tokenizer_(nullptr) {}

    ~NanoVLMAndroidInference() {
        if (tokenizer_) {
            nanovlm_free_tokenizer(tokenizer_);
        }
    }

    bool loadModels(const std::string& model_dir, const std::string& tokenizer_path) {
        try {
            LOGi("Loading nanoVLM models from %s", model_dir.c_str());

            // Load config
            config_ = load_config(model_dir + "/config.json");
            LOGi("Config loaded successfully");

            // Load ExecuTorch modules
            LOGi("Loading vision encoder from: %s", (model_dir + "/vision_encoder.pte").c_str());
            vision_encoder_ = std::make_unique<Module>(model_dir + "/vision_encoder.pte");
            LOGi("Vision encoder loaded successfully");

            // Check if module loaded correctly
            auto method_names = vision_encoder_->method_names();
            if (method_names.ok()) {
                LOGi("Vision encoder has %zu methods", method_names.get().size());
                for (const auto& name : method_names.get()) {
                    LOGi("  - Method: %s", name.c_str());
                }
            } else {
                LOGe("Failed to get vision encoder method names: error %d", (int)method_names.error());
            }

            LOGi("Loading modality projector from: %s", (model_dir + "/modality_projector.pte").c_str());
            modality_projector_ = std::make_unique<Module>(model_dir + "/modality_projector.pte");
            LOGi("Modality projector loaded successfully");

            prefill_decoder_ = std::make_unique<Module>(model_dir + "/language_decoder_prefill.pte");
            LOGi("Prefill decoder loaded");

            decode_decoder_ = std::make_unique<Module>(model_dir + "/language_decoder_decode.pte");
            LOGi("Decode decoder loaded");

            token_embedding_ = std::make_unique<Module>(model_dir + "/token_embedding.pte");
            LOGi("Token embedding loaded");

            lm_head_ = std::make_unique<Module>(model_dir + "/lm_head.pte");
            LOGi("LM head loaded");

            // Load tokenizer
            tokenizer_ = nanovlm_load_tokenizer(tokenizer_path.c_str(), config_.image_token.c_str());
            if (!tokenizer_) {
                LOGe("Failed to load tokenizer from %s", tokenizer_path.c_str());
                return false;
            }
            LOGi("Tokenizer loaded successfully");

            return true;
        } catch (const std::exception& e) {
            LOGe("Failed to load models: %s", e.what());
            return false;
        }
    }

    bool processImageFromBuffer(const uint8_t* argb_buffer, int width, int height) {
        try {
            LOGi("Processing image from buffer (%dx%d)", width, height);

            // Convert ARGB to RGB and preprocess
            current_image_data_ = nanovlm::preprocess_image_from_argb_buffer(
                argb_buffer, width, height,
                config_.max_img_size,
                config_.splitted_image_size,
                config_.resize_to_max_side_len
            );

            if (current_image_data_.images.empty()) {
                LOGe("Failed to preprocess image");
                return false;
            }

            LOGi("Image preprocessing complete: %zu images, grid %zux%zu",
                 current_image_data_.images.size(),
                 current_image_data_.grid_h,
                 current_image_data_.grid_w);

            // Run vision encoder on all images
            current_image_embeddings_.clear();
            for (size_t img_idx = 0; img_idx < current_image_data_.images.size(); img_idx++) {
                auto& img = current_image_data_.images[img_idx];

                // Create tensor
                std::vector<int32_t> image_shape = {1, (int32_t)img.channels,
                                                    (int32_t)img.height, (int32_t)img.width};

                // Sanity check: verify data is in reasonable range [0, 1]
                float min_val = *std::min_element(img.data.begin(), img.data.end());
                float max_val = *std::max_element(img.data.begin(), img.data.end());

                LOGi("Running vision encoder on image %zu: shape [%d, %d, %d, %d], data range [%.3f, %.3f]",
                     img_idx, image_shape[0], image_shape[1], image_shape[2], image_shape[3], min_val, max_val);

                auto image_tensor = from_blob(img.data.data(), image_shape, ScalarType::Float);

                // Run vision encoder
                std::vector<EValue> vision_inputs = {image_tensor};
                auto vision_result = vision_encoder_->forward(vision_inputs);
                if (!vision_result.ok()) {
                    LOGe("Vision encoder forward failed for image %zu: error code %d",
                         img_idx, (int)vision_result.error());
                    return false;
                }

                // Run modality projector
                std::vector<EValue> proj_inputs = {vision_result.get()[0]};
                auto proj_result = modality_projector_->forward(proj_inputs);
                if (!proj_result.ok()) {
                    LOGe("Modality projector forward failed");
                    return false;
                }

                // Extract embeddings
                const auto& img_emb_tensor = proj_result.get()[0].toTensor();
                size_t emb_size = config_.mp_image_token_length * config_.lm_hidden_dim;
                std::vector<float> embeddings(emb_size);
                const float* emb_ptr = img_emb_tensor.const_data_ptr<float>();
                std::memcpy(embeddings.data(), emb_ptr, emb_size * sizeof(float));

                current_image_embeddings_.push_back(embeddings);
            }

            LOGi("Vision encoding complete: %zu image embeddings", current_image_embeddings_.size());
            has_image_ = true;
            return true;

        } catch (const std::exception& e) {
            LOGe("Image processing failed: %s", e.what());
            return false;
        }
    }

    // Greedy sampling
    int64_t sample_token(const std::vector<float>& logits) {
        auto max_it = std::max_element(logits.begin(), logits.end());
        return std::distance(logits.begin(), max_it);
    }

    // Extract hidden state for a specific token
    std::vector<float> extract_token_hidden_state(const EValue& hidden_states_eval, size_t token_index) {
        const auto& tensor = hidden_states_eval.toTensor();
        const float* data_ptr = tensor.const_data_ptr<float>();
        auto sizes = tensor.sizes();
        size_t hidden_dim = sizes[2];

        std::vector<float> hidden_state(hidden_dim);
        size_t offset = token_index * hidden_dim;
        std::memcpy(hidden_state.data(), data_ptr + offset, hidden_dim * sizeof(float));
        return hidden_state;
    }

    // Run LM head
    std::vector<float> get_logits(const std::vector<float>& hidden_state) {
        std::vector<float> hidden_copy = hidden_state;
        std::vector<int32_t> hidden_shape = {1, 1, (int32_t)config_.lm_hidden_dim};
        auto hidden_tensor = from_blob(hidden_copy.data(), hidden_shape, ScalarType::Float);

        std::vector<EValue> lm_inputs = {hidden_tensor};
        auto lm_result = lm_head_->forward(lm_inputs);
        if (!lm_result.ok()) {
            throw std::runtime_error("LM head forward failed");
        }

        const auto& logits_tensor = lm_result.get()[0].toTensor();
        const float* logits_ptr = logits_tensor.const_data_ptr<float>();

        std::vector<float> logits(config_.lm_vocab_size);
        std::memcpy(logits.data(), logits_ptr, config_.lm_vocab_size * sizeof(float));
        return logits;
    }

    // Generate image string with grid tokens
    std::string get_image_string(size_t grid_h, size_t grid_w) {
        std::string image_string;

        if (grid_h > 1 || grid_w > 1) {
            image_string += config_.global_image_token;
            for (size_t i = 0; i < config_.mp_image_token_length; i++) {
                image_string += config_.image_token;
            }
        }

        for (size_t row = 0; row < grid_h; row++) {
            for (size_t col = 0; col < grid_w; col++) {
                image_string += "<row_" + std::to_string(row + 1) + "_col_" + std::to_string(col + 1) + ">";
                for (size_t i = 0; i < config_.mp_image_token_length; i++) {
                    image_string += config_.image_token;
                }
            }
        }

        return image_string;
    }

    std::string generateResponse(const std::string& prompt, int max_new_tokens) {
        if (!has_image_) {
            throw std::runtime_error("No image processed");
        }

        try {
            LOGi("Starting generation (max %d tokens)", max_new_tokens);

            // Concatenate image embeddings
            size_t total_image_tokens = current_image_embeddings_.size() * config_.mp_image_token_length;
            std::vector<float> combined_image_embeddings(total_image_tokens * config_.lm_hidden_dim);

            for (size_t i = 0; i < current_image_embeddings_.size(); i++) {
                size_t offset = i * config_.mp_image_token_length * config_.lm_hidden_dim;
                std::memcpy(combined_image_embeddings.data() + offset,
                           current_image_embeddings_[i].data(),
                           current_image_embeddings_[i].size() * sizeof(float));
            }

            // Format prompt with chat template
            std::string image_string = get_image_string(current_image_data_.grid_h, current_image_data_.grid_w);
            std::string formatted_prompt = "<|im_start|>user\n" + image_string + prompt + "<|im_end|>\n<|im_start|>assistant\n";

            // Tokenize
            TokenizationResult tok_result = nanovlm_tokenize(tokenizer_, formatted_prompt.c_str(), 0);
            if (!tok_result.token_ids) {
                throw std::runtime_error("Tokenization failed");
            }

            LOGi("Tokenization complete: %zu tokens", tok_result.num_tokens);

            // Get token embeddings
            std::vector<int32_t> token_shape = {1, (int32_t)tok_result.num_tokens};
            auto token_tensor = from_blob(tok_result.token_ids, token_shape, ScalarType::Long);

            std::vector<EValue> token_inputs = {token_tensor};
            auto token_emb_result = token_embedding_->forward(token_inputs);
            if (!token_emb_result.ok()) {
                nanovlm_free_tokenization_result(tok_result);
                throw std::runtime_error("Token embedding forward failed");
            }

            const auto& text_emb_tensor = token_emb_result.get()[0].toTensor();

            // Combine embeddings
            size_t total_tokens = tok_result.num_tokens;
            size_t hidden_dim = config_.lm_hidden_dim;
            std::vector<float> combined_embeddings(total_tokens * hidden_dim);
            const float* text_emb_ptr = text_emb_tensor.const_data_ptr<float>();
            std::memcpy(combined_embeddings.data(), text_emb_ptr, total_tokens * hidden_dim * sizeof(float));

            // Replace image tokens
            const int64_t IMAGE_TOKEN_ID = 49152;
            const int64_t GLOBAL_IMAGE_TOKEN_ID = 49153;
            size_t image_emb_idx = 0;

            for (size_t pos = 0; pos < total_tokens; pos++) {
                int64_t token_id = tok_result.token_ids[pos];
                if ((token_id == IMAGE_TOKEN_ID || token_id == GLOBAL_IMAGE_TOKEN_ID) && image_emb_idx < total_image_tokens) {
                    size_t src_offset = image_emb_idx * hidden_dim;
                    size_t dst_offset = pos * hidden_dim;
                    std::memcpy(combined_embeddings.data() + dst_offset,
                               combined_image_embeddings.data() + src_offset,
                               hidden_dim * sizeof(float));
                    image_emb_idx++;
                }
            }

            // Create attention mask and position IDs
            std::vector<int64_t> mask_data(total_tokens, 1);
            std::vector<int32_t> mask_shape = {1, (int32_t)total_tokens};
            auto attention_mask = from_blob(mask_data.data(), mask_shape, ScalarType::Long);

            std::vector<int64_t> pos_data(total_tokens);
            for (size_t i = 0; i < total_tokens; i++) pos_data[i] = i;
            auto position_ids = from_blob(pos_data.data(), mask_shape, ScalarType::Long);

            std::vector<int32_t> combined_shape = {1, (int32_t)total_tokens, (int32_t)hidden_dim};
            auto combined_tensor = from_blob(combined_embeddings.data(), combined_shape, ScalarType::Float);

            // Run prefill
            LOGi("Running prefill...");
            std::vector<EValue> prefill_inputs = {combined_tensor, attention_mask, position_ids};
            auto prefill_result = prefill_decoder_->forward(prefill_inputs);
            if (!prefill_result.ok()) {
                nanovlm_free_tokenization_result(tok_result);
                throw std::runtime_error("Prefill failed");
            }

            auto& prefill_outputs = prefill_result.get();
            const auto& prefill_hidden = prefill_outputs[0];

            // Clone KV cache (fixes reference invalidation bug!)
            std::vector<TensorPtr> kv_cache_storage;
            kv_cache_storage.reserve(prefill_outputs.size() - 1);
            for (size_t i = 1; i < prefill_outputs.size(); i++) {
                const auto& tensor = prefill_outputs[i].toTensor();
                kv_cache_storage.push_back(clone_tensor_ptr(tensor));
            }

            // Get first token
            auto last_hidden = extract_token_hidden_state(prefill_hidden, total_tokens - 1);
            auto logits = get_logits(last_hidden);
            int64_t next_token = sample_token(logits);

            std::vector<int64_t> generated_tokens;
            generated_tokens.push_back(next_token);
            size_t current_seq_len = total_tokens;

            LOGi("Starting decode loop...");
            const int64_t eos_token_id = 2;

            // Decode loop
            for (int step = 1; step < max_new_tokens; step++) {
                // Get embedding for new token
                std::vector<int64_t> token_id_vec = {next_token};
                std::vector<int32_t> single_token_shape = {1, 1};
                auto single_token_tensor = from_blob(token_id_vec.data(), single_token_shape, ScalarType::Long);

                std::vector<EValue> emb_inputs = {single_token_tensor};
                auto emb_result = token_embedding_->forward(emb_inputs);
                if (!emb_result.ok()) break;

                const auto& token_emb = emb_result.get()[0];

                // Create decode inputs
                std::vector<int64_t> decode_mask(current_seq_len + 1, 1);
                std::vector<int32_t> decode_mask_shape = {1, (int32_t)(current_seq_len + 1)};
                auto decode_attention_mask = from_blob(decode_mask.data(), decode_mask_shape, ScalarType::Long);

                std::vector<int64_t> decode_pos = {(int64_t)current_seq_len};
                std::vector<int32_t> decode_pos_shape = {1, 1};
                auto decode_position_ids = from_blob(decode_pos.data(), decode_pos_shape, ScalarType::Long);

                std::vector<EValue> decode_inputs = {token_emb, decode_attention_mask, decode_position_ids};
                for (const auto& kv_ptr : kv_cache_storage) {
                    decode_inputs.emplace_back(*kv_ptr);
                }

                auto decode_result = decode_decoder_->forward(decode_inputs);
                if (!decode_result.ok()) break;

                auto& decode_outputs = decode_result.get();
                const auto& decode_hidden = decode_outputs[0];

                // Update KV cache
                kv_cache_storage.clear();
                kv_cache_storage.reserve(decode_outputs.size() - 1);
                for (size_t i = 1; i < decode_outputs.size(); i++) {
                    const auto& tensor = decode_outputs[i].toTensor();
                    kv_cache_storage.push_back(clone_tensor_ptr(tensor));
                }

                // Sample next token
                auto hidden_vec = extract_token_hidden_state(decode_hidden, 0);
                logits = get_logits(hidden_vec);
                next_token = sample_token(logits);

                generated_tokens.push_back(next_token);
                current_seq_len++;

                if (next_token == eos_token_id) {
                    LOGi("EOS token at step %d", step);
                    break;
                }
            }

            LOGi("Generation complete: %zu tokens", generated_tokens.size());

            // Decode tokens
            char* decoded_text = nanovlm_decode(tokenizer_, generated_tokens.data(), generated_tokens.size());
            if (!decoded_text) {
                nanovlm_free_tokenization_result(tok_result);
                throw std::runtime_error("Failed to decode tokens");
            }

            std::string result(decoded_text);
            nanovlm_free_string(decoded_text);
            nanovlm_free_tokenization_result(tok_result);

            return result;

        } catch (const std::exception& e) {
            LOGe("Generation failed: %s", e.what());
            throw;
        }
    }

    bool isLoaded() const {
        return vision_encoder_ && modality_projector_ && prefill_decoder_ &&
               decode_decoder_ && token_embedding_ && lm_head_ && tokenizer_;
    }
};

// Global instance
static std::unique_ptr<NanoVLMAndroidInference> g_inference;

// JNI exports
extern "C" {

JNIEXPORT jboolean JNICALL
Java_ai_baseweight_baseweightsnap_NanoVLM_1Android_nativeLoadModels(
        JNIEnv *env,
        jobject,
        jstring model_dir_path,
        jstring tokenizer_path) {

    const char *model_dir = env->GetStringUTFChars(model_dir_path, 0);
    const char *tokenizer = env->GetStringUTFChars(tokenizer_path, 0);

    try {
        g_inference = std::make_unique<NanoVLMAndroidInference>();
        bool success = g_inference->loadModels(model_dir, tokenizer);

        env->ReleaseStringUTFChars(model_dir_path, model_dir);
        env->ReleaseStringUTFChars(tokenizer_path, tokenizer);

        return success ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        LOGe("Failed to load models: %s", e.what());
        env->ReleaseStringUTFChars(model_dir_path, model_dir);
        env->ReleaseStringUTFChars(tokenizer_path, tokenizer);
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_ai_baseweight_baseweightsnap_NanoVLM_1Android_nativeProcessImageFromBuffer(
        JNIEnv *env,
        jobject,
        jobject buffer,
        jint width,
        jint height) {

    if (!g_inference || !g_inference->isLoaded()) {
        LOGe("Models not loaded");
        return JNI_FALSE;
    }

    jbyte* buff = (jbyte*)env->GetDirectBufferAddress(buffer);
    if (!buff) {
        LOGe("Failed to get buffer address");
        return JNI_FALSE;
    }

    try {
        bool success = g_inference->processImageFromBuffer((const uint8_t*)buff, width, height);
        return success ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        LOGe("Image processing failed: %s", e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT jstring JNICALL
Java_ai_baseweight_baseweightsnap_NanoVLM_1Android_nativeGenerateResponse(
        JNIEnv *env,
        jobject,
        jstring prompt,
        jint max_tokens) {

    if (!g_inference || !g_inference->isLoaded()) {
        LOGe("Models not loaded");
        return env->NewStringUTF("");
    }

    const char* c_prompt = env->GetStringUTFChars(prompt, 0);

    try {
        std::string result = g_inference->generateResponse(c_prompt, max_tokens);
        env->ReleaseStringUTFChars(prompt, c_prompt);
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception& e) {
        LOGe("Generation failed: %s", e.what());
        env->ReleaseStringUTFChars(prompt, c_prompt);
        return env->NewStringUTF("");
    }
}

JNIEXPORT void JNICALL
Java_ai_baseweight_baseweightsnap_NanoVLM_1Android_nativeFreeModels(
        JNIEnv *,
        jobject) {
    g_inference.reset();
    LOGi("Models freed");
}

} // extern "C"
