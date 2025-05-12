#include "model_manager.h"
#include <unistd.h>

bool ModelManager::loadLanguageModel(const char* path) {
    llama_model_params model_params = llama_model_default_params();
    model.reset(llama_model_load_from_file(path, model_params));
    if (!model) {
        return false;
    }
    return true;
}

bool ModelManager::loadVisionModel(const char* path) {
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = true;
    mparams.n_threads = std::max(1, std::min(8, (int) sysconf(_SC_NPROCESSORS_ONLN) - 2));
    
    vision_context.reset(mtmd_init_from_file(path, model.get(), mparams));
    return vision_context != nullptr;
}

bool ModelManager::initializeContext() {
    int n_threads = std::max(1, std::min(8, (int) sysconf(_SC_NPROCESSORS_ONLN) - 2));
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    context.reset(llama_new_context_with_model(model.get(), ctx_params));
    return context != nullptr;
}

bool ModelManager::initializeBatch() {
    batch.reset(new llama_batch());
    // Initialize batch parameters...
    return batch != nullptr;
}

bool ModelManager::initializeSampler() {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    sampler.reset(llama_sampler_chain_init(sparams));
    if (sampler) {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    }
    return sampler != nullptr;
}