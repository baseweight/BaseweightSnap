#pragma once

#include "llama.h"
#include "mtmd.h"
#include <memory>

class ModelManager {
private:
    std::unique_ptr<llama_model> model;
    std::unique_ptr<llama_context> context;
    std::unique_ptr<mtmd_context> vision_context;
    std::unique_ptr<llama_batch> batch;
    std::unique_ptr<llama_sampler> sampler;
    mtmd::bitmaps bitmaps;  // Store processed images here

public:
    ModelManager() = default;
    ~ModelManager() = default;

    bool loadLanguageModel(const char* path);
    bool loadVisionModel(const char* path);
    bool initializeContext();
    bool initializeBatch();
    bool initializeSampler();

    // Getters for JNI
    llama_model* getModel() const { return model.get(); }
    llama_context* getContext() const { return context.get(); }
    mtmd_context* getVisionContext() const { return vision_context.get(); }
    llama_batch* getBatch() const { return batch.get(); }
    llama_sampler* getSampler() const { return sampler.get(); }
    const mtmd::bitmaps& getBitmaps() const { return bitmaps; }
    
    // Add bitmap to the collection
    void addBitmap(mtmd::bitmap&& bmp) {
        bitmaps.entries.push_back(std::move(bmp));
    }
    
    // Clear all bitmaps
    void clearBitmaps() {
        bitmaps.entries.clear();
    }
};
