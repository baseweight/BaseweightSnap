/**
 * @file mtmd-android.cpp
 * @brief Main file for the MTMD Android application
 * 
 * This file contains the JNI methods for loading and processing models,
 * as well as the initialization and cleanup of the MTMD context.
 */

#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <math.h>
#include <string>
#include <unistd.h>
#include "llama.h"
#include "common.h"
#include "mtmd.h"
#include "clip.h"
#include "model_manager.h"

#define TAG "mtmd-android.cpp"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

jclass la_int_var;
jmethodID la_int_var_value;
jmethodID la_int_var_inc;

std::string cached_token_chars;

static std::atomic<bool> g_should_stop{false};

bool is_valid_utf8(const char * string) {
    if (!string) {
        return true;
    }

    const unsigned char * bytes = (const unsigned char *)string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }

    return true;
}

static void log_callback(ggml_log_level level, const char * fmt, void * data) {
    if (level == GGML_LOG_LEVEL_ERROR)     __android_log_print(ANDROID_LOG_ERROR, TAG, fmt, data);
    else if (level == GGML_LOG_LEVEL_INFO) __android_log_print(ANDROID_LOG_INFO, TAG, fmt, data);
    else if (level == GGML_LOG_LEVEL_WARN) __android_log_print(ANDROID_LOG_WARN, TAG, fmt, data);
    else __android_log_print(ANDROID_LOG_DEFAULT, TAG, fmt, data);
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_ai_baseweight_baseweightsnap_MTMD_1Android_load_1models(
        JNIEnv *env,
        jobject,
        jstring language_model_path,
        jstring mmproj_path) {
    
    auto& manager = ModelManager::getInstance();
    
    const char *lang_model_path = env->GetStringUTFChars(language_model_path, 0);
    const char *mmproj_model_path = env->GetStringUTFChars(mmproj_path, 0);
    
    bool success = manager.loadLanguageModel(lang_model_path) &&
                  manager.loadVisionModel(mmproj_model_path) &&
                  manager.initializeContext() &&
                  manager.initializeBatch() &&
                  manager.initializeSampler() &&
                  manager.initializeChatTemplate("vicuna");  // Use vicuna template by default

    env->ReleaseStringUTFChars(language_model_path, lang_model_path);
    env->ReleaseStringUTFChars(mmproj_path, mmproj_model_path);

    if (!success) {
        LOGe("Failed to initialize models. Language model: %s, Vision model: %s", lang_model_path, mmproj_model_path);
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Failed to initialize models");
        return JNI_FALSE;
    }

    LOGi("Successfully initialized models");
    return JNI_TRUE;
}

extern "C"
JNIEXPORT void JNICALL
Java_ai_baseweight_baseweightsnap_MTMD_1Android_free_1models(
        JNIEnv *,
        jobject) {
    ModelManager::getInstance().cleanup();
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_ai_baseweight_baseweightsnap_MTMD_1Android_process_1image(
        JNIEnv *env,
        jobject,
        jstring image_path) {

    auto& manager = ModelManager::getInstance();
    
    // Check if models are loaded
    if (!manager.areModelsLoaded()) {
        LOGe("process_image(): models not loaded");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Models not loaded");
        return JNI_FALSE;
    }

    auto path_to_image = env->GetStringUTFChars(image_path, 0);
    LOGi("Processing image from %s", path_to_image);

    // Clear any existing bitmaps
    manager.clearBitmaps();

    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(path_to_image));
    env->ReleaseStringUTFChars(image_path, path_to_image);

    if (!bmp.ptr) {
        LOGe("Failed to load image from %s", path_to_image);
        return JNI_FALSE;
    }

    // Store the bitmap in the manager
    manager.addBitmap(std::move(bmp));
    LOGi("Successfully processed image");

    return JNI_TRUE;
}

extern "C"
JNIEXPORT void JNICALL
Java_ai_baseweight_baseweightsnap_MTMD_1Android_backend_1init(JNIEnv *, jobject) {
    llama_backend_init();
}

extern "C"
JNIEXPORT void JNICALL
Java_ai_baseweight_baseweightsnap_MTMD_1Android_backend_1free(JNIEnv *, jobject) {
    llama_backend_free();
}

extern "C"
JNIEXPORT void JNICALL
Java_ai_baseweight_baseweightsnap_MTMD_1Android_log_1to_1android(JNIEnv *, jobject) {
    llama_log_set(log_callback, NULL);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_ai_baseweight_baseweightsnap_MTMD_1Android_system_1info(JNIEnv *env, jobject) {
    return env->NewStringUTF(llama_print_system_info());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_ai_baseweight_baseweightsnap_MTMD_1Android_generate_1response(
        JNIEnv *env,
        jobject,
        jstring prompt,
        jint max_tokens) {

    auto& manager = ModelManager::getInstance();
    if (!manager.areModelsLoaded()) {
        LOGe("generate_response(): models not loaded");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Models not loaded");
        return nullptr;
    }

    if (manager.getBitmaps().entries.empty()) {
        LOGe("generate_response(): no image processed");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "No image processed");
        return nullptr;
    }

    const char* prompt_str = env->GetStringUTFChars(prompt, 0);
    
    // Create input text
    mtmd_input_text text;
    text.text = prompt_str;
    text.add_special = true;
    text.parse_special = true;

    // Tokenize the input
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto& bitmaps = manager.getBitmaps();  // Use non-const reference since c_ptr() isn't const
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(manager.getVisionContext(),
                               chunks.ptr.get(),
                               &text,
                               bitmaps_c_ptr.data(),
                               bitmaps_c_ptr.size());

    env->ReleaseStringUTFChars(prompt, prompt_str);

    if (res != 0) {
        LOGe("Unable to tokenize prompt, res = %d", res);
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Failed to tokenize prompt");
        return nullptr;
    }

    // Evaluate the tokens
    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(manager.getVisionContext(),
                               manager.getLanguageContext(),
                               chunks.ptr.get(),
                               manager.getNPast(),
                               0,
                               manager.getNBatch(),
                               true,
                               &new_n_past)) {
        LOGe("Unable to eval prompt");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Failed to evaluate prompt");
        return nullptr;
    }

    manager.setNPast(new_n_past);

    // Generate response
    std::string response;
    for (int i = 0; i < max_tokens; i++) {
        if (g_should_stop) {
            g_should_stop = false;  // Reset for next time
            break;
        }
        
        llama_token token_id = common_sampler_sample(manager.getSampler(), manager.getLanguageContext(), -1);
        common_sampler_accept(manager.getSampler(), token_id, true);

        if (llama_vocab_is_eog(manager.getVocab(), token_id)) {
            break;
        }

        response += common_token_to_piece(manager.getLanguageContext(), token_id);

        // Evaluate the token
        llama_batch& batch = manager.getBatch();
        common_batch_clear(batch);
        llama_pos n_past = manager.getNPast();
        common_batch_add(batch, token_id, n_past, {0}, true);
        manager.setNPast(n_past + 1);
        
        if (llama_decode(manager.getLanguageContext(), batch)) {
            LOGe("failed to decode token");
            env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Failed to decode token");
            return nullptr;
        }
    }

    return env->NewStringUTF(response.c_str());
}

extern "C"
JNIEXPORT jint JNICALL
Java_ai_baseweight_baseweightsnap_MTMD_1Android_get_1token_1count(
        JNIEnv *env,
        jobject,
        jstring text) {

    auto& manager = ModelManager::getInstance();
    if (!manager.areModelsLoaded()) {
        LOGe("get_token_count(): models not loaded");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Models not loaded");
        return -1;
    }

    const char* text_str = env->GetStringUTFChars(text, 0);
    
    // Create input text
    mtmd_input_text input_text;
    input_text.text = text_str;
    input_text.add_special = false;
    input_text.parse_special = true;

    // Tokenize the input
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto& bitmaps = manager.getBitmaps();  // Use non-const reference since c_ptr() isn't const
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(manager.getVisionContext(),
                               chunks.ptr.get(),
                               &input_text,
                               bitmaps_c_ptr.data(),
                               bitmaps_c_ptr.size());

    env->ReleaseStringUTFChars(text, text_str);

    if (res != 0) {
        LOGe("Unable to tokenize text, res = %d", res);
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Failed to tokenize text");
        return -1;
    }

    // For now, just return 0 since token counting isn't part of mtmd-cli.cpp
    return 0;
}

extern "C"
JNIEXPORT void JNICALL
Java_ai_baseweight_baseweightsnap_MTMD_1Android_stop_1generation(
        JNIEnv *,
        jobject) {
    g_should_stop = true;
}

extern "C"
JNIEXPORT void JNICALL
Java_ai_baseweight_baseweightsnap_MTMD_1Android_reset_1stop_1flag(
        JNIEnv *,
        jobject) {
    g_should_stop = false;
}

