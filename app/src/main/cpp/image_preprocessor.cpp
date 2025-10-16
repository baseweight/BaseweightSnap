#include "image_preprocessor.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

// Define STB_IMAGE_IMPLEMENTATION before including stb_image.h
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace nanovlm {

// Helper: clip value to range [lower, upper]
static inline int clip(int x, int lower, int upper) {
    return std::max(lower, std::min(x, upper));
}

// Helper: align to nearest multiple (rounds up)
static inline int align_up(int x, int n) {
    return ((x + n - 1) / n) * n;
}

Image load_image(const std::string& image_path) {
    int width, height, channels;
    unsigned char* data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);

    if (!data) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    Image img(width, height);

    // Copy RGB data
    size_t data_size = 3 * width * height;
    std::copy(data, data + data_size, img.data.begin());

    stbi_image_free(data);

    return img;
}

void bicubic_resize(const Image& src, Image& dst, int target_width, int target_height) {
    const int nx = src.width;
    const int ny = src.height;

    dst.width = target_width;
    dst.height = target_height;
    dst.data.resize(3 * target_width * target_height);

    float Cc;
    float C[5] = {};
    float d0, d2, d3, a0, a1, a2, a3;
    int i, j, k, jj;
    int x, y;
    float dx, dy;
    float tx, ty;

    tx = (float)nx / (float)target_width;
    ty = (float)ny / (float)target_height;

    // Bicubic interpolation; adapted from ViT.cpp, inspired from:
    //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
    //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

    for (i = 0; i < target_height; i++) {
        for (j = 0; j < target_width; j++) {
            x = (int)(tx * j);
            y = (int)(ty * i);

            dx = tx * j - x;
            dy = ty * i - y;

            for (k = 0; k < 3; k++) {
                for (jj = 0; jj <= 3; jj++) {
                    d0 = src.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k]
                       - src.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d2 = src.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k]
                       - src.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d3 = src.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k]
                       - src.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    a0 = src.data[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;

                    C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                    d0 = C[0] - C[1];
                    d2 = C[2] - C[1];
                    d3 = C[3] - C[1];
                    a0 = C[1];
                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;
                    Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                    const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                    dst.data[(i * target_width + j) * 3 + k] = Cc2;
                }
            }
        }
    }
}

PreprocessedImage rgb_to_chw_normalized(const Image& img) {
    PreprocessedImage result(3, img.height, img.width);

    // Convert from HWC (RGB interleaved) to CHW format and normalize to [0, 1]
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < img.height; y++) {
            for (int x = 0; x < img.width; x++) {
                int src_idx = (y * img.width + x) * 3 + c;
                int dst_idx = c * img.height * img.width + y * img.width + x;
                result.data[dst_idx] = img.data[src_idx] / 255.0f;
            }
        }
    }

    return result;
}

std::pair<int, int> compute_dynamic_resize(
    int orig_h, int orig_w,
    int max_side_len, int patch_size,
    bool resize_to_max
) {
    int long_side = std::max(orig_w, orig_h);
    int short_side = std::min(orig_w, orig_h);

    // Compute target long side
    int target_long;
    if (resize_to_max) {
        target_long = max_side_len;
    } else {
        target_long = std::min(max_side_len, align_up(long_side, patch_size));
    }

    // Scale factor
    double scale = static_cast<double>(target_long) / static_cast<double>(long_side);

    // Compute short side with ceiling to never undershoot
    int target_short = std::max(
        patch_size,
        static_cast<int>(std::ceil(short_side * scale / patch_size) * patch_size)
    );

    // Return (height, width)
    if (orig_w >= orig_h) {
        return {target_short, target_long};
    } else {
        return {target_long, target_short};
    }
}

Image crop_image(const Image& src, int x, int y, int width, int height) {
    if (x < 0 || y < 0 || x + width > src.width || y + height > src.height) {
        throw std::runtime_error("Crop region out of bounds");
    }

    Image dst(width, height);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int src_idx = ((y + i) * src.width + (x + j)) * 3;
            int dst_idx = (i * width + j) * 3;
            dst.data[dst_idx]     = src.data[src_idx];
            dst.data[dst_idx + 1] = src.data[src_idx + 1];
            dst.data[dst_idx + 2] = src.data[src_idx + 2];
        }
    }

    return dst;
}

MultiImageResult preprocess_image_with_splitting(
    const std::string& image_path,
    int max_side_len,
    int patch_size,
    bool resize_to_max
) {
    MultiImageResult result;

    // 1. Load image
    Image img = load_image(image_path);
    std::cout << "Loaded image: " << img.width << "x" << img.height << std::endl;

    // 2. Compute dynamic resize dimensions
    auto [new_h, new_w] = compute_dynamic_resize(
        img.height, img.width,
        max_side_len, patch_size,
        resize_to_max
    );
    std::cout << "Dynamic resize to: " << new_w << "x" << new_h << std::endl;

    // 3. Resize image
    Image resized;
    bicubic_resize(img, resized, new_w, new_h);
    std::cout << "Resized image: " << resized.width << "x" << resized.height << std::endl;

    // 4. Calculate grid dimensions
    int grid_h = new_h / patch_size;
    int grid_w = new_w / patch_size;
    result.grid_h = grid_h;
    result.grid_w = grid_w;

    std::cout << "Grid: " << grid_h << "x" << grid_w << " (" << (grid_h * grid_w) << " patches)" << std::endl;

    // 5. Create images
    if (grid_h == 1 && grid_w == 1) {
        // Only one patch - don't add global view
        PreprocessedImage patch = rgb_to_chw_normalized(resized);
        result.images.push_back(std::move(patch));
        std::cout << "Single patch mode (no global view)" << std::endl;
    } else {
        // Multiple patches - add global view first (bicubic downsampled)
        Image global_img;
        bicubic_resize(resized, global_img, patch_size, patch_size);
        PreprocessedImage global_preprocessed = rgb_to_chw_normalized(global_img);
        result.images.push_back(std::move(global_preprocessed));
        std::cout << "Added global view: " << patch_size << "x" << patch_size << std::endl;

        // Add all patches
        for (int row = 0; row < grid_h; row++) {
            for (int col = 0; col < grid_w; col++) {
                int x = col * patch_size;
                int y = row * patch_size;

                Image patch = crop_image(resized, x, y, patch_size, patch_size);
                PreprocessedImage patch_preprocessed = rgb_to_chw_normalized(patch);
                result.images.push_back(std::move(patch_preprocessed));
            }
        }
        std::cout << "Added " << (grid_h * grid_w) << " patches" << std::endl;
    }

    std::cout << "Total images: " << result.images.size() << std::endl;

    return result;
}

MultiImageResult preprocess_image_from_argb_buffer(
    const uint8_t* argb_buffer,
    int width,
    int height,
    int max_side_len,
    int patch_size,
    bool resize_to_max
) {
    // Convert ARGB to RGB
    Image img(width, height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int src_idx = (y * width + x) * 4;  // ARGB: 4 bytes per pixel
            int dst_idx = (y * width + x) * 3;  // RGB: 3 bytes per pixel

            // ARGB -> RGB (skip alpha channel)
            img.data[dst_idx + 0] = argb_buffer[src_idx + 1];  // R
            img.data[dst_idx + 1] = argb_buffer[src_idx + 2];  // G
            img.data[dst_idx + 2] = argb_buffer[src_idx + 3];  // B
        }
    }

    // Now apply same preprocessing as file-based version
    MultiImageResult result;

    // 1. Load completed (already have RGB data)
    std::cout << "Loaded image from buffer: " << width << "x" << height << std::endl;

    // 2. Compute dynamic resize dimensions
    auto [new_h, new_w] = compute_dynamic_resize(
        img.height, img.width,
        max_side_len, patch_size,
        resize_to_max
    );
    std::cout << "Dynamic resize to: " << new_w << "x" << new_h << std::endl;

    // 3. Resize image
    Image resized;
    bicubic_resize(img, resized, new_w, new_h);
    std::cout << "Resized image: " << resized.width << "x" << resized.height << std::endl;

    // 4. Calculate grid dimensions
    int grid_h = new_h / patch_size;
    int grid_w = new_w / patch_size;
    result.grid_h = grid_h;
    result.grid_w = grid_w;

    std::cout << "Grid: " << grid_h << "x" << grid_w << " (" << (grid_h * grid_w) << " patches)" << std::endl;

    // 5. Create images
    if (grid_h == 1 && grid_w == 1) {
        // Only one patch - don't add global view
        PreprocessedImage patch = rgb_to_chw_normalized(resized);
        result.images.push_back(std::move(patch));
        std::cout << "Single patch mode (no global view)" << std::endl;
    } else {
        // Multiple patches - add global view first (bicubic downsampled)
        Image global_img;
        bicubic_resize(resized, global_img, patch_size, patch_size);
        PreprocessedImage global_preprocessed = rgb_to_chw_normalized(global_img);
        result.images.push_back(std::move(global_preprocessed));
        std::cout << "Added global view: " << patch_size << "x" << patch_size << std::endl;

        // Add all patches
        for (int row = 0; row < grid_h; row++) {
            for (int col = 0; col < grid_w; col++) {
                int x = col * patch_size;
                int y = row * patch_size;

                Image patch = crop_image(resized, x, y, patch_size, patch_size);
                PreprocessedImage patch_preprocessed = rgb_to_chw_normalized(patch);
                result.images.push_back(std::move(patch_preprocessed));
            }
        }
        std::cout << "Added " << (grid_h * grid_w) << " patches" << std::endl;
    }

    std::cout << "Total images: " << result.images.size() << std::endl;

    return result;
}

} // namespace nanovlm
