#include "model_manager.h"
#include <android/log.h>

#define TAG "model_manager.cpp"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

ModelManager::~ModelManager() {
    cleanup();
}

void ModelManager::cleanup() {
    if (sampler) {
        common_sampler_free(sampler);
        sampler = nullptr;
    }
    if (lctx) {
        llama_free(lctx);
        lctx = nullptr;
    }
    if (model) {
        llama_free_model(model);
        model = nullptr;
    }
    if (ctx_vision) {
        mtmd_free(ctx_vision);
        ctx_vision = nullptr;
    }
    vocab = nullptr;
    n_past = 0;
    bitmaps.entries.clear();
}

bool ModelManager::loadLanguageModel(const char* model_path) {
    cleanup();  // Clean up any existing models first
    
    llama_model_params model_params = llama_model_default_params();
    model = llama_load_model_from_file(model_path, model_params);
    if (!model) {
        LOGe("Failed to load language model from %s", model_path);
        return false;
    }
    vocab = llama_model_get_vocab(model);
    return true;
}

bool ModelManager::loadVisionModel(const char* mmproj_path) {
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = true;  // Enable GPU by default
    mparams.print_timings = true;
    mparams.n_threads = 1;
    mparams.verbosity = GGML_LOG_LEVEL_INFO;
    
    ctx_vision = mtmd_init_from_file(mmproj_path, model, mparams);
    if (!ctx_vision) {
        LOGe("Failed to load vision model from %s", mmproj_path);
        return false;
    }
    return true;
}

bool ModelManager::initializeContext() {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;  // Adjust based on your needs
    ctx_params.n_batch = n_batch;
    
    lctx = llama_new_context_with_model(model, ctx_params);
    if (!lctx) {
        LOGe("Failed to create language context");
        return false;
    }
    return true;
}

bool ModelManager::initializeBatch() {
    // This is a struct, I have no idea how this could realistically fail
    batch = llama_batch_init(n_batch, 0, 1);
    return true;
}

bool ModelManager::initializeSampler() {
    common_params_sampling sampling_params;
    sampling_params.temp = 0.2f;  // Lower temperature for better quality
    
    sampler = common_sampler_init(model, sampling_params);
    if (!sampler) {
        LOGe("Failed to initialize sampler");
        return false;
    }
    return true;
}

bool ModelManager::processImage(const char* image_path) {
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(image_path));
    if (!bmp.ptr) {
        LOGe("Failed to load image from %s", image_path);
        return false;
    }
    
    bitmaps.entries.push_back(std::move(bmp));
    return true;
}

void ModelManager::addBitmap(mtmd::bitmap&& bmp) {
    bitmaps.entries.push_back(std::move(bmp));
}

std::string ModelManager::generateResponse(const char* prompt, int max_tokens) {
    if (!evalMessage(prompt, true)) {  // Add BOS token for first message
        return "Error: Failed to evaluate message";
    }

    std::string response;
    llama_tokens generated_tokens;
    int n_predict = max_tokens;

    for (int i = 0; i < n_predict; i++) {
        llama_token token_id = common_sampler_sample(sampler, lctx, -1);
        generated_tokens.push_back(token_id);
        common_sampler_accept(sampler, token_id, true);

        if (llama_vocab_is_eog(vocab, token_id) || checkAntiprompt(generated_tokens)) {
            break;
        }

        response += common_token_to_piece(lctx, token_id);

        // Evaluate the token
        common_batch_clear(batch);
        common_batch_add(batch, token_id, n_past++, {0}, true);
        if (llama_decode(lctx, batch)) {
            LOGe("failed to decode token");
            return "Error: Failed to decode token";
        }
    }

    return response;
}

bool ModelManager::evalMessage(const char* prompt, bool add_bos) {
    if (!tmpls) {
        LOGe("Chat templates not initialized");
        return false;
    }

    // Create chat message
    common_chat_msg msg;
    msg.role = "user";
    msg.content = prompt;

    // Format chat message using templates
    common_chat_templates_inputs tmpl_inputs;
    tmpl_inputs.messages = {msg};
    tmpl_inputs.add_generation_prompt = true;
    tmpl_inputs.use_jinja = false;  // jinja is buggy here
    auto formatted_chat = common_chat_templates_apply(tmpls.get(), tmpl_inputs);
    LOGi("formatted_chat.prompt: %s", formatted_chat.prompt.c_str());

    mtmd_input_text text;
    text.text = formatted_chat.prompt.c_str();
    text.add_special = add_bos;
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto& bitmaps = getBitmaps();  // Use non-const reference since c_ptr() isn't const
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(ctx_vision,
                               chunks.ptr.get(),
                               &text,
                               bitmaps_c_ptr.data(),
                               bitmaps_c_ptr.size());
    if (res != 0) {
        LOGe("Unable to tokenize prompt, res = %d", res);
        return false;
    }

    // Clear bitmaps after tokenization
    bitmaps.entries.clear();

    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(ctx_vision,
                               lctx,
                               chunks.ptr.get(),
                               n_past,
                               0,  // seq_id
                               n_batch,
                               true,  // logits_last
                               &new_n_past)) {
        LOGe("Unable to eval prompt");
        return false;
    }

    n_past = new_n_past;
    return true;
}

bool ModelManager::initializeChatTemplate(const char* template_name) {
    if (!model) {
        LOGe("Model not loaded");
        return false;
    }

    // Check if model has built-in chat template
    if (!llama_model_chat_template(model, nullptr) && !template_name) {
        LOGe("Model does not have chat template and no template name provided");
        return false;
    }

    // Initialize chat templates
    tmpls = common_chat_templates_init(model, template_name);
    if (!tmpls) {
        LOGe("Failed to initialize chat templates");
        return false;
    }

    // Load antiprompt tokens for legacy templates
    if (template_name) {
        if (strcmp(template_name, "vicuna") == 0) {
            antiprompt_tokens = common_tokenize(lctx, "ASSISTANT:", false, true);
        } else if (strcmp(template_name, "deepseek") == 0) {
            antiprompt_tokens = common_tokenize(lctx, "###", false, true);
        }
    }

    return true;
}

bool ModelManager::checkAntiprompt(const llama_tokens& generated_tokens) const {
    if (antiprompt_tokens.empty() || generated_tokens.size() < antiprompt_tokens.size()) {
        return false;
    }
    return std::equal(
        generated_tokens.end() - antiprompt_tokens.size(),
        generated_tokens.end(),
        antiprompt_tokens.begin()
    );
}
