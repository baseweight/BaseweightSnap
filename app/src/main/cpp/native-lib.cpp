#include <jni.h>
#include <string>
#include <android/log.h>
#include "SmolVLM.h"
#include <opencv2/opencv.hpp>

// Logging aliases
#define LOG_TAG "SmolVLM"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_baseweight_baseweightsnap_ModelDownloader_initializeSmolVLM(
        JNIEnv* env,
        jobject /* this */,
        jstring vision_model_path,
        jstring embed_model_path,
        jstring decoder_model_path,
        jstring vocab_path,
        jstring tokenizer_path) {
    
    try {
        // Convert Java strings to C++ strings
        const char* vision_path = env->GetStringUTFChars(vision_model_path, nullptr);
        const char* embed_path = env->GetStringUTFChars(embed_model_path, nullptr);
        const char* decoder_path = env->GetStringUTFChars(decoder_model_path, nullptr);
        const char* vocab_path_str = env->GetStringUTFChars(vocab_path, nullptr);
        const char* tokenizer_path_str = env->GetStringUTFChars(tokenizer_path, nullptr);

        LOGI("Initializing SmolVLM with models:");
        LOGI("Vision model: %s", vision_path);
        LOGI("Embed model: %s", embed_path);
        LOGI("Decoder model: %s", decoder_path);
        LOGI("Vocab file: %s", vocab_path_str);
        LOGI("Tokenizer config: %s", tokenizer_path_str);

        // Initialize the SmolVLM singleton
        SmolVLM::initialize(
            std::string(vision_path),
            std::string(embed_path),
            std::string(decoder_path),
            std::string(vocab_path_str),
            std::string(tokenizer_path_str)
        );

        // Release the strings
        env->ReleaseStringUTFChars(vision_model_path, vision_path);
        env->ReleaseStringUTFChars(embed_model_path, embed_path);
        env->ReleaseStringUTFChars(decoder_model_path, decoder_path);
        env->ReleaseStringUTFChars(vocab_path, vocab_path_str);
        env->ReleaseStringUTFChars(tokenizer_path, tokenizer_path_str);

        LOGI("SmolVLM initialized successfully");
        return JNI_TRUE;
    } catch (const std::exception& e) {
        LOGE("Initialization failed: %s", e.what());
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_baseweight_baseweightsnap_MainActivity_describeImage(
        JNIEnv* env,
        jobject /* this */,
        jobject image_buffer,
        jint width,
        jint height,
        jstring prompt) {
    try {
        // Check if SmolVLM is initialized
        if (!SmolVLM::isInitialized()) {
            LOGE("SmolVLM not initialized. Call initialize() first.");
            return env->NewStringUTF("Error: Models not initialized. Please try again.");
        }

        // Convert Java strings to C++ strings
        const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);
        LOGI("Describing image with prompt: %s", prompt_str);

        // Get direct buffer address and capacity
        void* buffer_ptr = env->GetDirectBufferAddress(image_buffer);
        jlong buffer_capacity = env->GetDirectBufferCapacity(image_buffer);

        if (!buffer_ptr || buffer_capacity <= 0) {
            LOGE("Invalid buffer: ptr=%p, capacity=%lld", buffer_ptr, buffer_capacity);
            env->ReleaseStringUTFChars(prompt, prompt_str);
            return env->NewStringUTF("Error: Invalid image buffer");
        }

        LOGI("Buffer info: ptr=%p, capacity=%lld, width=%d, height=%d", 
             buffer_ptr, buffer_capacity, width, height);

        // Create OpenCV Mat from the buffer
        // Note: Assuming RGBA format from Android Bitmap
        cv::Mat image(height, width, CV_8UC4, buffer_ptr);
        if (image.empty()) {
            LOGE("Failed to create OpenCV Mat from buffer");
            env->ReleaseStringUTFChars(prompt, prompt_str);
            return env->NewStringUTF("Error: Failed to process image");
        }
        
        // Convert from RGBA to BGR (OpenCV's preferred format)
        cv::Mat bgr_image;
        cv::cvtColor(image, bgr_image, cv::COLOR_RGBA2BGR);

        // Get the SmolVLM instance
        SmolVLM& smolvlm = SmolVLM::getInstance();

        // Generate the description
        std::string description = smolvlm.generateText(prompt_str, bgr_image, 100);

        // Release the strings
        env->ReleaseStringUTFChars(prompt, prompt_str);

        // Return the description
        return env->NewStringUTF(description.c_str());
    } catch (const std::exception& e) {
        LOGE("Error describing image: %s", e.what());
        return env->NewStringUTF("Error: Failed to describe image");
    }
}