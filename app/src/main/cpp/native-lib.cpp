#include <jni.h>
#include <string>
#include <android/log.h>
#include "SmolVLM.h"

// Logging aliases
#define LOG_TAG "SmolVLM"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_baseweight_baseweightsnap_MainActivity_initializeSmolVLM(
        JNIEnv* env,
        jobject /* this */,
        jstring vision_model_path,
        jstring embed_model_path,
        jstring decoder_model_path,
        jstring vocab_path) {
    
    try {
        // Convert Java strings to C++ strings
        const char* vision_path = env->GetStringUTFChars(vision_model_path, nullptr);
        const char* embed_path = env->GetStringUTFChars(embed_model_path, nullptr);
        const char* decoder_path = env->GetStringUTFChars(decoder_model_path, nullptr);
        const char* vocab_path_str = env->GetStringUTFChars(vocab_path, nullptr);

        LOGI("Initializing SmolVLM with models:");
        LOGI("Vision model: %s", vision_path);
        LOGI("Embed model: %s", embed_path);
        LOGI("Decoder model: %s", decoder_path);
        LOGI("Vocab file: %s", vocab_path_str);

        // Initialize the SmolVLM singleton
        SmolVLM::initialize(
            std::string(vision_path),
            std::string(embed_path),
            std::string(decoder_path),
            std::string(vocab_path_str)
        );

        // Release the strings
        env->ReleaseStringUTFChars(vision_model_path, vision_path);
        env->ReleaseStringUTFChars(embed_model_path, embed_path);
        env->ReleaseStringUTFChars(decoder_model_path, decoder_path);
        env->ReleaseStringUTFChars(vocab_path, vocab_path_str);

        LOGI("SmolVLM initialized successfully");
        return JNI_TRUE;
    } catch (const std::exception& e) {
        LOGE("Initialization failed: %s", e.what());
        return JNI_FALSE;
    }
}