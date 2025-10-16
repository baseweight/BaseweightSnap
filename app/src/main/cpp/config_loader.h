#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct NanoVLMConfig {
    size_t vit_img_size;
    size_t vit_hidden_dim;
    size_t lm_hidden_dim;
    size_t lm_n_heads;
    size_t lm_n_kv_heads;
    size_t lm_n_blocks;
    size_t lm_vocab_size;
    size_t mp_image_token_length;
    std::string image_token;
    std::string global_image_token;
    std::string lm_tokenizer;

    // Image splitting parameters
    size_t max_img_size;
    size_t splitted_image_size;
    bool resize_to_max_side_len;
};

inline NanoVLMConfig load_config(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }

    json j;
    try {
        file >> j;
    } catch (const json::parse_error& e) {
        throw std::runtime_error(std::string("Failed to parse JSON: ") + e.what());
    }

    NanoVLMConfig config;

    try {
        config.vit_img_size = j["vit_img_size"];
        config.vit_hidden_dim = j["vit_hidden_dim"];
        config.lm_hidden_dim = j["lm_hidden_dim"];
        config.lm_n_heads = j["lm_n_heads"];
        config.lm_n_kv_heads = j["lm_n_kv_heads"];
        config.lm_n_blocks = j["lm_n_blocks"];
        config.lm_vocab_size = j["lm_vocab_size"];
        config.mp_image_token_length = j["mp_image_token_length"];

        // Extract tokens from vlm_extra_tokens
        if (j.contains("vlm_extra_tokens")) {
            const auto& extra_tokens = j["vlm_extra_tokens"];
            config.image_token = extra_tokens.value("image_token", "<|image|>");
            config.global_image_token = extra_tokens.value("global_image_token", "<|global_image|>");
        } else {
            config.image_token = "<|image|>";
            config.global_image_token = "<|global_image|>";
        }

        config.lm_tokenizer = j["lm_tokenizer"];

        // Image splitting parameters with defaults
        config.max_img_size = j.value("max_img_size", 2048);
        config.splitted_image_size = j.value("splitted_image_size", 512);
        config.resize_to_max_side_len = j.value("resize_to_max_side_len", false);

        std::cout << "Loaded config:" << std::endl;
        std::cout << "  vit_img_size: " << config.vit_img_size << std::endl;
        std::cout << "  lm_hidden_dim: " << config.lm_hidden_dim << std::endl;
        std::cout << "  lm_n_blocks: " << config.lm_n_blocks << std::endl;
        std::cout << "  mp_image_token_length: " << config.mp_image_token_length << std::endl;
        std::cout << "  image_token: " << config.image_token << std::endl;
        std::cout << "  global_image_token: " << config.global_image_token << std::endl;
        std::cout << "  max_img_size: " << config.max_img_size << std::endl;
        std::cout << "  splitted_image_size: " << config.splitted_image_size << std::endl;
        std::cout << "  resize_to_max_side_len: " << (config.resize_to_max_side_len ? "true" : "false") << std::endl;

    } catch (const json::exception& e) {
        throw std::runtime_error(std::string("Failed to extract config values: ") + e.what());
    }

    return config;
}

#endif // CONFIG_LOADER_H
