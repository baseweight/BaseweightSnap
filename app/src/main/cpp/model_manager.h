#pragma once

#include <memory>
#include <vector>
#include <string>
#include "llama.h"
#include "mtmd.h"
#include "common.h"
#include "sampling.h"
#include "chat.h"

class ModelManager {
public:
    // Delete copy constructor and assignment operator
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

    // Get singleton instance
    static ModelManager& getInstance() {
        static ModelManager instance;
        return instance;
    }

    // Cleanup existing models
    void cleanup();

    // Model loading
    bool loadLanguageModel(const char* model_path);
    bool loadVisionModel(const char* mmproj_path);
    bool initializeContext();
    bool initializeBatch();
    bool initializeSampler();
    bool initializeChatTemplate(const char* template_name = nullptr);

    // Image processing
    bool processImage(const char* image_path);
    void addBitmap(mtmd::bitmap&& bmp);
    void clearBitmaps() { bitmaps.entries.clear(); }
    bool areModelsLoaded() const { return model != nullptr && ctx_vision != nullptr && lctx != nullptr; }

    // Text generation
    std::string generateResponse(const char* prompt, int max_tokens);
    bool evalMessage(const char* prompt, bool add_bos = false);

    // Getters
    mtmd_context* getVisionContext() const { return ctx_vision; }
    llama_context* getLanguageContext() const { return lctx; }
    llama_model* getModel() const { return model; }
    const llama_vocab* getVocab() const { return vocab; }
    llama_batch& getBatch() { return batch; }
    int getNBatch() const { return n_batch; }
    void setNBatch(int batch_size) { n_batch = batch_size; }
    llama_pos getNPast() const { return n_past; }
    void setNPast(llama_pos past) { n_past = past; }
    common_sampler* getSampler() const { return sampler; }
    mtmd::bitmaps& getBitmaps() { return bitmaps; }

private:
    // Private constructor for singleton
    ModelManager() = default;
    ~ModelManager();

    // Vision context
    mtmd_context* ctx_vision = nullptr;
    
    // Language model
    llama_model* model = nullptr;
    llama_context* lctx = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_batch batch;
    int n_batch = 512;  // Default to a larger batch size for better performance
    llama_pos n_past = 0;
    
    // Sampler
    common_sampler* sampler = nullptr;
    
    // Image processing
    mtmd::bitmaps bitmaps;

    // Chat template handling
    common_chat_templates_ptr tmpls;
    llama_tokens antiprompt_tokens;
    bool checkAntiprompt(const llama_tokens& generated_tokens) const;
};
