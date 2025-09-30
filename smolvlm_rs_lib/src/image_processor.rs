use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array4, Array5};
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
        min_len: Option<u32>,
        max_len: Option<u32>,
    ) -> (u32, u32) {
        let max_len = max_len.unwrap_or_else(|| height.max(width));
        let aspect_ratio = width as f32 / height as f32;

        let (mut width, mut height) = if width >= height {
            let width = max_len;
            let height = (width as f32 / aspect_ratio).round() as u32;
            (width, height)
        } else {
            let height = max_len;
            let width = (height as f32 * aspect_ratio).round() as u32;
            (width, height)
        };

        // Avoid resizing to a size smaller than min_len
        let min_len = min_len.unwrap_or(1);
        height = height.max(min_len);
        width = width.max(min_len);

        (height, width)
    }

    fn resize_output_size_scale_below_upper_bound(
        height: u32,
        width: u32,
        max_len: Option<u32>,
    ) -> (u32, u32) {
        let max_len = max_len.unwrap_or_else(|| height.max(width));
        let aspect_ratio = width as f32 / height as f32;

        let (mut width, mut height) = if width >= height && width > max_len {
            let width = max_len;
            let height = (width as f32 / aspect_ratio).round() as u32;
            (width, height)
        } else if height > width && height > max_len {
            let height = max_len;
            let width = (height as f32 * aspect_ratio).round() as u32;
            (width, height)
        } else {
            (width, height)
        };

        // Avoid resizing to a size smaller than 1
        height = height.max(1);
        width = width.max(1);

        (height, width)
    }

    fn get_resize_output_image_size(
        image: &DynamicImage,
        resolution_max_side: u32,
    ) -> (u32, u32) {
        let (height, width) = (image.height(), image.width());
        
        // Only resize if the image is larger than resolution_max_side
        if height <= resolution_max_side && width <= resolution_max_side {
            return (height, width);
        }

        // Find the output size, when rescaling the longest edge to max_len and preserving the aspect ratio
        let (height, width) = Self::resize_output_size_rescale_to_max_len(height, width, None, Some(resolution_max_side));
        
        // Find the output size when scaling the image to be below the MAX_IMAGE_SIZE
        let (height, width) = Self::resize_output_size_scale_below_upper_bound(height, width, Some(MAX_IMAGE_SIZE));
        
        (height, width)
    }

    fn convert_to_rgb(&self, image: DynamicImage) -> DynamicImage {
        image.to_rgb8().into()
    }

    fn resize(&self, image: DynamicImage, size: HashMap<String, u32>) -> Result<DynamicImage> {
        let (height, width) = if let Some(longest_edge) = size.get("longest_edge") {
            Self::get_resize_output_image_size(&image, *longest_edge)
        } else if let (Some(height), Some(width)) = (size.get("height"), size.get("width")) {
            (*height, *width)
        } else {
            return Err(anyhow::anyhow!("size must be a dictionary with key 'longest_edge' or 'height' and 'width'"));
        };

        Ok(image.resize_exact(width, height, image::imageops::FilterType::Lanczos3))
    }

    fn split_image(
        &self,
        image: DynamicImage,
        max_image_size: &HashMap<String, u32>,
    ) -> Result<(Vec<DynamicImage>, u32, u32)> {
        let (height, width) = (image.height(), image.width());
        let max_size = max_image_size.get("longest_edge").unwrap_or(&512);

        let mut frames = Vec::new();
        
        // Always do a 4x4 split
        let num_splits_h = 4;
        let num_splits_w = 4;
        
        // Calculate optimal split sizes
        let optimal_height = (height as f32 / num_splits_h as f32).ceil() as u32;
        let optimal_width = (width as f32 / num_splits_w as f32).ceil() as u32;

        for r in 0..num_splits_h {
            for c in 0..num_splits_w {
                let start_x = c * optimal_width;
                let start_y = r * optimal_height;
                let end_x = (start_x + optimal_width).min(width);
                let end_y = (start_y + optimal_height).min(height);

                let cropped = image.crop_imm(start_x, start_y, end_x - start_x, end_y - start_y);
                // Resize each cropped frame to 512x512 (hardcoded for vision encoder requirement)
                let resized = self.resize(
                    cropped,
                    HashMap::from([
                        ("height".to_string(), 512u32),
                        ("width".to_string(), 512u32),
                    ]),
                )?;
                frames.push(resized);
            }
        }

        // Add the original image, resized to 512x512 (hardcoded for vision encoder requirement)
        let resized = self.resize(
            image,
            HashMap::from([
                ("height".to_string(), 512u32),
                ("width".to_string(), 512u32),
            ]),
        )?;
        frames.push(resized);

        Ok((frames, num_splits_h, num_splits_w))
    }

    pub fn preprocess(&self, image: DynamicImage) -> Result<(Array5<f32>, Array4<i64>)> {
        let mut image = image;

        // Convert to RGB if needed
        if self.do_convert_rgb {
            image = self.convert_to_rgb(image);
        }

        // Resize if needed - first resize to size["longest_edge"] while preserving aspect ratio
        if self.do_resize {
            image = self.resize(image, self.size.clone())?;
        }

        // Split image if needed
        let (frames, num_splits_h, num_splits_w) = if self.do_image_splitting {
            // First resize to be multiples of max_image_size while preserving aspect ratio
            let max_size = self.max_image_size.get("longest_edge").unwrap_or(&512);
            let (height, width) = (image.height(), image.width());
            let aspect_ratio = width as f32 / height as f32;
            
            let (new_width, new_height) = if width >= height {
                let new_width = ((width as f32 / *max_size as f32).ceil() * *max_size as f32) as u32;
                let new_height = (new_width as f32 / aspect_ratio).round() as u32;
                let new_height = ((new_height as f32 / *max_size as f32).ceil() * *max_size as f32) as u32;
                (new_width, new_height)
            } else {
                let new_height = ((height as f32 / *max_size as f32).ceil() * *max_size as f32) as u32;
                let new_width = (new_height as f32 * aspect_ratio).round() as u32;
                let new_width = ((new_width as f32 / *max_size as f32).ceil() * *max_size as f32) as u32;
                (new_width, new_height)
            };

            let resized = self.resize(
                image,
                HashMap::from([
                    ("height".to_string(), new_height),
                    ("width".to_string(), new_width),
                ]),
            )?;
            
            self.split_image(resized, &self.max_image_size)?
        } else {
            // If not splitting, just resize to 512x512 (hardcoded for vision encoder requirement)
            let resized = self.resize(
                image,
                HashMap::from([
                    ("height".to_string(), 512u32),
                    ("width".to_string(), 512u32),
                ]),
            )?;
            (vec![resized], 0, 0)
        };

        // Convert frames to arrays and normalize
        let mut processed_frames = Vec::new();
        for frame in frames {
            let mut array = Array5::<f32>::zeros((1, 1, 3, frame.height() as usize, frame.width() as usize));

            for (y, x, pixel) in frame.pixels() {
                for c in 0..3 {
                    let val = pixel[c] as f32;
                    let val = if self.do_rescale {
                        val * self.rescale_factor
                    } else {
                        val
                    };
                    let val = if self.do_normalize {
                        (val - self.image_mean[c]) / self.image_std[c]
                    } else {
                        val
                    };
                    array[[0, 0, c, y as usize, x as usize]] = val;
                }
            }
            processed_frames.push(array);
        }

        // Stack frames into a batch with shape (batch, num_frames, channels, height, width)
        let batch = Array5::from_shape_fn(
            (1, processed_frames.len(), 3, processed_frames[0].shape()[3], processed_frames[0].shape()[4]),
            |(_, i, c, y, x)| {
                let frame = &processed_frames[i];
                frame[[0, 0, c, y, x]]
            },
        );

        // Create attention mask with shape (batch, num_frames, height, width)
        let height = processed_frames[0].shape()[3];
        let width = processed_frames[0].shape()[4];
        let attention_mask = Array4::ones((1, processed_frames.len(), height, width));

        Ok((batch, attention_mask))
    }
} 