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
#include "mtmd-helper.h"
#include "clip.h"
#include "model_manager.h"

#undef TAG
#define TAG "mtmd-android.cpp"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

jclass la_int_var;
jmethodID la_int_var_value;
jmethodID la_int_var_inc;

std::string cached_token_chars;

JavaVM* g_jvm;
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {

    // Initialize GGML and common
    ggml_time_init();
    common_init();
    
    // Initialize your ModelManager here
    ModelManager::getInstance();  // This ensures the singleton is created
    
    // You can also initialize any JNI-related things here
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }
    
    // Store the JavaVM pointer for later use if needed
    g_jvm = vm;
    
    return JNI_VERSION_1_6;
}



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

    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(manager.getVisionContext(), path_to_image));
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
JNIEXPORT void JNICALL
Java_ai_baseweight_baseweightsnap_MTMD_1Android_generate_1response(
        JNIEnv *env,
        jobject,
        jstring prompt,
        jint max_tokens,
        jobject callback) {

    auto& manager = ModelManager::getInstance();
    if (!manager.areModelsLoaded()) {
        LOGe("generate_response(): models not loaded");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Models not loaded");
        return;
    }

    if (manager.getBitmaps().entries.empty()) {
        LOGe("generate_response(): no image processed");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "No image processed");
        return;
    }

    const char* c_prompt = env->GetStringUTFChars(prompt, nullptr);
    ModelManager::getInstance().generateResponseAsync(c_prompt, max_tokens, env, callback);
    env->ReleaseStringUTFChars(prompt, c_prompt);
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


extern "C"
JNIEXPORT jboolean JNICALL
Java_ai_baseweight_baseweightsnap_MTMD_1Android_process_1image_1from_1byteBuff(JNIEnv *env,
                                                                               jobject thiz,
                                                                               jobject arr,
                                                                               jint width,
                                                                               jint height) {
    // TODO: implement process_image_from_byteBuff()
    jbyte* buff = (jbyte*)env->GetDirectBufferAddress(arr);
    jlong buff_len = env->GetDirectBufferCapacity(arr);
    if (buff_len != (jlong)(width * height * 4)) {
        LOGe("Buffer size mismatch: expected %d, got %ld", width * height * 4, (long)buff_len);
        return JNI_FALSE;
    }

    size_t len = width * height * 3;
    // Allocate memory for RGB buffer (raw pointer since mtmd takes ownership)
    uint8_t* rgb_buffer = new uint8_t[len];
    
    // Convert BGRA to RGB
    for (size_t i = 0; i < width * height; ++i) {
        size_t src_idx = i * 4;
        size_t dst_idx = i * 3;
        
        // Skip alpha channel (BGR -> RGB)
        rgb_buffer[dst_idx + 0] = buff[src_idx + 0]; // R
        rgb_buffer[dst_idx + 1] = buff[src_idx + 1]; // G
        rgb_buffer[dst_idx + 2] = buff[src_idx + 2]; // B;
    }

    // We load directly from the buffer since mtmd takes ownership
    // this is better than copying to file or messing around with PNG decoding
    mtmd::bitmap bmp(width, height, rgb_buffer);

    // Store the bitmap in the manager
    ModelManager::getInstance().addBitmap(std::move(bmp));
    LOGi("Successfully processed image");

    return JNI_TRUE;
}
