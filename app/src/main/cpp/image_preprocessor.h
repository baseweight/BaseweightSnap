#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>

/**
 * Standalone image preprocessing for nanoVLM
 * Based on llama.cpp's bicubic resize and Python's preprocessing logic
 * No GGML dependencies
 */

namespace nanovlm {

// Image structure (RGB uint8)
struct Image {
    int width;
    int height;
    std::vector<uint8_t> data; // RGB format: RGBRGBRGB...

    Image() : width(0), height(0) {}
    Image(int w, int h) : width(w), height(h), data(3 * w * h) {}

    size_t size() const { return data.size(); }
};

// Preprocessed image data (CHW float32 format, normalized to [0, 1])
struct PreprocessedImage {
    std::vector<float> data; // CHW format
    int width;
    int height;
    int channels;

    PreprocessedImage() : width(0), height(0), channels(0) {}
    PreprocessedImage(int c, int h, int w)
        : data(c * h * w), width(w), height(h), channels(c) {}
};

// Multiple images with grid information
struct MultiImageResult {
    std::vector<PreprocessedImage> images; // [global_view, patch1, patch2, ...]
    size_t grid_h; // Number of patches in height
    size_t grid_w; // Number of patches in width

    MultiImageResult() : grid_h(0), grid_w(0) {}
};

/**
 * Load image from file using stb_image
 * Returns Image in RGB format
 */
Image load_image(const std::string& image_path);

/**
 * Bicubic resize (matches Python's torchvision BICUBIC)
 * Based on llama.cpp implementation
 */
void bicubic_resize(const Image& src, Image& dst, int target_width, int target_height);

/**
 * Convert RGB uint8 image to CHW float32 format, normalized to [0, 1]
 */
PreprocessedImage rgb_to_chw_normalized(const Image& img);

/**
 * Compute dynamic resize dimensions (matches Python's DynamicResize)
 *
 * @param orig_h Original height
 * @param orig_w Original width
 * @param max_side_len Maximum side length
 * @param patch_size Patch size for alignment
 * @param resize_to_max If true, always resize long side to max_side_len
 * @return Pair of (new_height, new_width)
 */
std::pair<int, int> compute_dynamic_resize(
    int orig_h, int orig_w,
    int max_side_len, int patch_size,
    bool resize_to_max
);

/**
 * Crop image to specified region
 */
Image crop_image(const Image& src, int x, int y, int width, int height);

/**
 * Preprocess image with dynamic resizing and splitting
 * Matches Python's preprocessing pipeline:
 * 1. Dynamic resize preserving aspect ratio
 * 2. Split into patches
 * 3. Create global view (bicubic downsampled) + patches
 *
 * @param image_path Path to image file
 * @param max_side_len Maximum side length (e.g., 2048)
 * @param patch_size Size of each patch (e.g., 512)
 * @param resize_to_max If true, always resize to max_side_len
 * @return MultiImageResult with global view + patches
 */
MultiImageResult preprocess_image_with_splitting(
    const std::string& image_path,
    int max_side_len,
    int patch_size,
    bool resize_to_max
);

/**
 * Preprocess image from ARGB buffer (for Android)
 * Converts ARGB8888 to RGB, then applies same preprocessing as above
 *
 * @param argb_buffer ARGB8888 buffer from Android
 * @param width Image width
 * @param height Image height
 * @param max_side_len Maximum side length (e.g., 2048)
 * @param patch_size Size of each patch (e.g., 512)
 * @param resize_to_max If true, always resize to max_side_len
 * @return MultiImageResult with global view + patches
 */
MultiImageResult preprocess_image_from_argb_buffer(
    const uint8_t* argb_buffer,
    int width,
    int height,
    int max_side_len,
    int patch_size,
    bool resize_to_max
);

} // namespace nanovlm
