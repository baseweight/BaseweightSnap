use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::Array5;
use std::collections::HashMap;

const MAX_IMAGE_SIZE: u32 = 4096; // 4k resolution as absolute maximum
const SMOLVLM_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const SMOLVLM_STD: [f32; 3] = [0.5, 0.5, 0.5];

#[derive(Debug, Clone)]
pub struct SmolVLMImageProcessor {
    do_convert_rgb: bool,
    do_resize: bool,
    size: HashMap<String, u32>,
    do_image_splitting: bool,
    max_image_size: HashMap<String, u32>,
    do_rescale: bool,
    rescale_factor: f32,
    do_normalize: bool,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    do_pad: bool,
}

impl Default for SmolVLMImageProcessor {
    fn default() -> Self {
        Self {
            do_convert_rgb: true,
            do_resize: true,
            size: HashMap::from([("longest_edge".to_string(), 2048)]),
            do_image_splitting: true,
            max_image_size: HashMap::from([("longest_edge".to_string(), 512)]),
            do_rescale: true,
            rescale_factor: 1.0 / 255.0,
            do_normalize: true,
            image_mean: SMOLVLM_MEAN,
            image_std: SMOLVLM_STD,
            do_pad: true,
        }
    }
}

impl SmolVLMImageProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    fn resize_output_size_rescale_to_max_len(
        height: u32,
        width: u32,
        max_len: u32,
    ) -> (u32, u32) {
        let max_original_size = height.max(width);
        let ratio = max_len as f32 / max_original_size as f32;
        let width = (width as f32 * ratio).round() as u32;
        let height = (height as f32 * ratio).round() as u32;
        (height, width)
    }

    fn rescale_size(size: (u32, u32), max_size: u32) -> (u32, u32) {
        let (height, width) = size;
        if height.max(width) <= max_size {
            return size;
        }
        Self::resize_output_size_rescale_to_max_len(height, width, max_size)
    }

    fn split_image(&self, image: &DynamicImage) -> Result<Vec<DynamicImage>> {
        let (width, height) = image.dimensions();
        let max_edge = self.max_image_size.get("longest_edge").unwrap_or(&512);

        if width <= *max_edge && height <= *max_edge {
            return Ok(vec![image.clone()]);
        }

        let mut patches = Vec::new();
        let patch_size = *max_edge;

        for y in (0..height).step_by(patch_size as usize) {
            for x in (0..width).step_by(patch_size as usize) {
                let crop_width = (patch_size).min(width - x);
                let crop_height = (patch_size).min(height - y);

                if crop_width > 0 && crop_height > 0 {
                    let patch = image.crop_imm(x, y, crop_width, crop_height);
                    patches.push(patch);
                }
            }
        }

        Ok(patches)
    }

    pub fn preprocess(&self, image: &DynamicImage) -> Result<Array5<f32>> {
        // Convert to RGB if needed
        let image = if self.do_convert_rgb {
            image.to_rgb8()
        } else {
            image.to_rgb8()
        };
        let image = DynamicImage::ImageRgb8(image);

        // Resize
        let image = if self.do_resize {
            let (width, height) = image.dimensions();
            let target_size = self.size.get("longest_edge").unwrap_or(&2048);
            let (new_height, new_width) = Self::rescale_size((height, width), *target_size);
            image.resize_exact(new_width, new_height, image::imageops::FilterType::CatmullRom)
        } else {
            image
        };

        // Split image if needed
        let patches = if self.do_image_splitting {
            self.split_image(&image)?
        } else {
            vec![image]
        };

        let num_patches = patches.len();
        let patch_size = 384; // Fixed patch size for SmolVLM

        let mut result = Array5::<f32>::zeros((1, num_patches, 3, patch_size, patch_size));

        for (patch_idx, patch) in patches.iter().enumerate() {
            // Resize patch to fixed size
            let resized_patch = patch.resize_exact(
                patch_size as u32,
                patch_size as u32,
                image::imageops::FilterType::CatmullRom,
            );
            let rgb_patch = resized_patch.to_rgb8();

            // Convert to array with CHW format
            for c in 0..3 {
                for y in 0..patch_size {
                    for x in 0..patch_size {
                        let pixel = rgb_patch.get_pixel(x as u32, y as u32);
                        let mut value = pixel[c] as f32;

                        // Rescale
                        if self.do_rescale {
                            value *= self.rescale_factor;
                        }

                        // Normalize
                        if self.do_normalize {
                            value = (value - self.image_mean[c]) / self.image_std[c];
                        }

                        result[[0, patch_idx, c, y, x]] = value;
                    }
                }
            }
        }

        Ok(result)
    }
}