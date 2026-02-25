#include "common.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "clip.h"
#include "model_manager.h"
#include <android/log.h>
#include <jni.h>
#include <chrono>

// Global flag to control generation
std::atomic<bool> g_should_stop{false};

// Initialize static member
JavaVM* ModelManager::javaVM = nullptr;

#undef TAG
#define TAG "model_manager.cpp"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static jmethodID method_onTextGenerated = nullptr;
static jmethodID method_onGenerationComplete = nullptr;
static jmethodID method_onGenerationError = nullptr;

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
        llama_model_free(model);
        model = nullptr;
    }
    vocab = nullptr;
    n_past = 0;
    bitmaps.entries.clear();
}

void ModelManager::onTextGenerated(const std::string& text, JNIEnv* env, jobject callback) {
    if (!method_onTextGenerated) {
        jclass callbackClass = env->GetObjectClass(callback);
        method_onTextGenerated = env->GetMethodID(callbackClass, "onTextGenerated", "(Ljava/lang/String;)V");
        env->DeleteLocalRef(callbackClass);
    }
    
    jstring jtext = env->NewStringUTF(text.c_str());
    env->CallVoidMethod(callback, method_onTextGenerated, jtext);
    env->DeleteLocalRef(jtext);
}

void ModelManager::onGenerationComplete(JNIEnv* env, jobject callback) {
    if (!method_onGenerationComplete) {
        jclass callbackClass = env->GetObjectClass(callback);
        method_onGenerationComplete = env->GetMethodID(callbackClass, "onGenerationComplete", "()V");
        env->DeleteLocalRef(callbackClass);
    }
    env->CallVoidMethod(callback, method_onGenerationComplete);
}

void ModelManager::onGenerationError(const std::string& error, JNIEnv* env, jobject callback) {
    if (!method_onGenerationError) {
        jclass callbackClass = env->GetObjectClass(callback);
        method_onGenerationError = env->GetMethodID(callbackClass, "onGenerationError", "(Ljava/lang/String;)V");
        env->DeleteLocalRef(callbackClass);
    }
    
    jstring jerror = env->NewStringUTF(error.c_str());
    env->CallVoidMethod(callback, method_onGenerationError, jerror);
    env->DeleteLocalRef(jerror);
}

bool ModelManager::loadLanguageModel(const char* model_path) {
    cleanup();  // Clean up any existing models first
    
    llama_model_params model_params = llama_model_default_params();
    // Let's try something here
    model_params.n_gpu_layers = gpu_layers;
    model = llama_model_load_from_file(model_path, model_params);
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

    ctx_vision.reset(mtmd_init_from_file(mmproj_path, model, mparams));
    if (!ctx_vision.get()) {
        LOGe("Failed to load vision model from %s", mmproj_path);
        return false;
    }
    return true;
}

bool ModelManager::initializeContext() {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;  // Adjust based on your needs
    ctx_params.n_batch = n_batch;
    ctx_params.swa_full = false;  // Match CLI behavior

    lctx = llama_init_from_model(model, ctx_params);
    if (!lctx) {
        LOGe("Failed to create language context");
        return false;
    }

    // Warmup: let backends compile and validate compute graphs before real data.
    // Without this, the Hexagon backend can crash on the first real decode.
    llama_set_warmup(lctx, true);
    {
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);
        std::vector<llama_token> tmp;
        if (bos != LLAMA_TOKEN_NULL) tmp.push_back(bos);
        if (eos != LLAMA_TOKEN_NULL) tmp.push_back(eos);
        if (tmp.empty()) tmp.push_back(0);
        if (llama_decode(lctx, llama_batch_get_one(tmp.data(), tmp.size()))) {
            LOGe("Warmup decode failed");
        }
    }
    llama_set_warmup(lctx, false);
    llama_memory_clear(llama_get_memory(lctx), true);

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
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), image_path));
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

void ModelManager::setCurrentCallback(JNIEnv* env, jobject callback) {
    // Store JavaVM pointer if not already stored
    if (!javaVM) {
        env->GetJavaVM(&javaVM);
    }

    if (currentCallback) {
        env->DeleteGlobalRef(currentCallback);
    }
    currentCallback = env->NewGlobalRef(callback);
}

void ModelManager::clearCurrentCallback(JNIEnv* env) {
    if (currentCallback) {
        env->DeleteGlobalRef(currentCallback);
        currentCallback = nullptr;
    }
}

JNIEnv* ModelManager::getJNIEnv() {
    if (!javaVM) {
        return nullptr;
    }

    JNIEnv* env = nullptr;
    jint result = javaVM->GetEnv((void**)&env, JNI_VERSION_1_6);
    if (result == JNI_EDETACHED) {
        // Thread is not attached to JVM, attach it
        result = javaVM->AttachCurrentThread(&env, nullptr);
        if (result != JNI_OK) {
            LOGe("Failed to attach thread to JVM");
            return nullptr;
        }
    } else if (result != JNI_OK) {
        LOGe("Failed to get JNIEnv");
        return nullptr;
    }
    return env;
}

void ModelManager::generateResponseAsync(const char* prompt, int max_tokens, JNIEnv* env, jobject callback) {
    // Store the callback
    setCurrentCallback(env, callback);

    // This ate up literal days of my life
    std::string str_prompt(prompt);
    if(str_prompt.find("<__image__>") == std::string::npos) {
        str_prompt = " <__image__> " + str_prompt;
    }

    // Create chat message
    common_chat_msg msg;
    msg.role = "user";
    msg.content = str_prompt;

    if (!evalMessage(msg, true)) {  // Add BOS token for first message
        onGenerationError("Failed to evaluate message", env, callback);
        clearCurrentCallback(env);
        return;
    }

    llama_tokens generated_tokens;
    int n_predict = max_tokens;

    for (int i = 0; i < n_predict; i++) {
        // Check if we should stop
        if (g_should_stop) {
            onGenerationComplete(env, callback);
            break;
        }

        llama_token token_id = common_sampler_sample(sampler, lctx, -1);
        generated_tokens.push_back(token_id);
        common_sampler_accept(sampler, token_id, true);

        if (llama_vocab_is_eog(vocab, token_id) || checkAntiprompt(generated_tokens)) {
            onGenerationComplete(env, callback);
            break;
        }

        // Convert token to text
        std::string token_text = common_token_to_piece(lctx, token_id);
        if (!token_text.empty()) {
            onTextGenerated(token_text, env, callback);
        }

        // Check if we've generated enough tokens
        if (i >= n_predict - 1) {
            onGenerationComplete(env, callback);
            break;
        }

        // Check again before decoding
        if (g_should_stop) {
            onGenerationComplete(env, callback);
            break;
        }

        // Evaluate the token
        common_batch_clear(batch);
        common_batch_add(batch, token_id, n_past++, {0}, true);
        if (llama_decode(lctx, batch)) {
            LOGe("failed to decode token");
            onGenerationError("Failed to decode token", env, callback);
            break;
        }
    }

    // Clean up the callback at the end
    clearCurrentCallback(env);
}

bool ModelManager::evalMessage(common_chat_msg& msg, bool add_bos) {
    if (!tmpls) {
        LOGe("Chat templates not initialized");
        return false;
    }

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
    auto& bitmaps = getBitmaps();
    auto bitmaps_c_ptr = bitmaps.c_ptr();

    // Get JNIEnv for the current thread
    JNIEnv* env = getJNIEnv();
    
    // Send progress update for tokenization
    if (env && currentCallback) {
        onTextGenerated("PROGRESS:Tokenizing input...:10", env, currentCallback);
    }
    
    int32_t res = mtmd_tokenize(ctx_vision.get(),
                               chunks.ptr.get(),
                               &text,
                               bitmaps_c_ptr.data(),
                               bitmaps_c_ptr.size());

    if (res != 0) {
        LOGe("Unable to tokenize prompt, res = %d", res);
        return false;
    }

    // Send progress update for evaluation
    if (env && currentCallback) {
        onTextGenerated("PROGRESS:Evaluating chunks...:30", env, currentCallback);
    }

    llama_pos new_n_past;
    // This is our method, it sends progress updates, which is Android-specific
    if (evalChunksWithProgress(ctx_vision.get(),
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

    // Send progress update for completion
    if (env && currentCallback) {
        onTextGenerated("PROGRESS:Processing complete:100", env, currentCallback);
    }

    n_past = new_n_past;

    // Live dangerously
    bitmaps.entries.clear();
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

/*
 * Our platform-specific replacement for mtmd_helper_eval_chunks.
 
 * This method is our custom eval chunks method, it sends progress updates, which is Android-specific, 
 * and uses OpenMP for parallel processing, which the MTMD and llama.cpp libraries do not support on Android
 * due to the lack of support for OpenMP in early Android NDK versions.
 * 
 * However, I don't have that problem, so I can use OpenMP to parallelize the evaluation of the chunks.
 */

int32_t ModelManager::evalChunksWithProgress(mtmd_context * ctx,
                                struct llama_context * lctx,
                                const mtmd_input_chunks * chunks,
                                llama_pos n_past,
                                llama_seq_id seq_id,
                                int32_t n_batch,
                                bool logits_last,
                                llama_pos * new_n_past) {
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    if (n_chunks == 0) {
        LOGe("no chunks to eval\n");
        return 0;
    }

    // Get JNIEnv for the current thread
    JNIEnv* env = getJNIEnv();
    if (env && currentCallback) {
        onTextGenerated("PROGRESS:Analyzing image content...:35", env, currentCallback);
    }

    // Process chunks sequentially
    for (size_t i = 0; i < n_chunks; i++) {
        LOGi("Processing chunk %zu/%zu", i+1, n_chunks);
        bool chunk_logits_last = (i == n_chunks - 1) && logits_last;
        auto chunk = mtmd_input_chunks_get(chunks, i);

        // Log chunk type for debugging
        auto chunk_type = mtmd_input_chunk_get_type(chunk);
        const char* type_name = (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) ? "TEXT" :
                               (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) ? "IMAGE" :
                               (chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) ? "AUDIO" : "UNKNOWN";
        LOGi("Chunk %zu type: %s", i+1, type_name);

        int32_t res = mtmd_helper_eval_chunk_single(ctx, lctx, chunk, n_past, seq_id,
                                                     n_batch, chunk_logits_last, &n_past);
        if (res != 0) {
            LOGe("failed to eval chunk %zu\n", i);
            return res;
        }
        *new_n_past = n_past;
        LOGi("Completed chunk %zu/%zu", i+1, n_chunks);

    }

    if (env && currentCallback) {
        onTextGenerated("PROGRESS:Generating description...:70", env, currentCallback);
    }

    return 0;
}