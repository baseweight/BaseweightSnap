#ifndef NANOVLM_PREPROCESSOR_H
#define NANOVLM_PREPROCESSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to tokenizer
typedef struct TokenizerHandle TokenizerHandle;

// Tokenization result structure
typedef struct {
    int64_t* token_ids;
    size_t num_tokens;
    size_t* image_token_positions;
    size_t num_image_tokens;
} TokenizationResult;

// Image data structure
typedef struct {
    float* data;
    size_t width;
    size_t height;
    size_t channels;
} ImageData;

// Multiple images with grid info
typedef struct {
    ImageData* images;
    size_t num_images;
    size_t grid_h;
    size_t grid_w;
} MultiImageData;

// Load a tokenizer from JSON file
// Returns NULL on failure
TokenizerHandle* nanovlm_load_tokenizer(
    const char* tokenizer_path,
    const char* image_token
);

// Free tokenizer handle
void nanovlm_free_tokenizer(TokenizerHandle* handle);

// Tokenize text with image token placeholders
// image_token_length: how many times to repeat the image token
// Caller must free result with nanovlm_free_tokenization_result
TokenizationResult nanovlm_tokenize(
    TokenizerHandle* handle,
    const char* text,
    size_t image_token_length
);

// Free tokenization result
void nanovlm_free_tokenization_result(TokenizationResult result);

// Preprocess image to CHW format normalized to [0, 1]
// Returns image data in CHW layout (channels, height, width)
// Caller must free result with nanovlm_free_image_data
ImageData nanovlm_preprocess_image(
    const char* image_path,
    size_t target_size
);

// Preprocess image with splitting (global + patches)
// max_side_len: max dimension (e.g., 2048)
// patch_size: size of each patch (e.g., 512)
// resize_to_max: if true, resize to exactly max_side_len; if false, don't upscale
// Returns multiple images: [global_view, patch_0_0, patch_0_1, ...]
// Caller must free result with nanovlm_free_multi_image_data
MultiImageData nanovlm_preprocess_image_with_splitting(
    const char* image_path,
    size_t max_side_len,
    size_t patch_size,
    int resize_to_max
);

// Free image data
void nanovlm_free_image_data(ImageData image_data);

// Free multiple image data
void nanovlm_free_multi_image_data(MultiImageData multi_image_data);

// Decode token IDs back to text
// Returns newly allocated C string that must be freed with nanovlm_free_string
char* nanovlm_decode(
    TokenizerHandle* handle,
    const int64_t* token_ids,
    size_t num_tokens
);

// Free string returned by nanovlm_decode
void nanovlm_free_string(char* str);

#ifdef __cplusplus
}
#endif

#endif // NANOVLM_PREPROCESSOR_H
