use image::ImageReader;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float};
use std::ptr;
use tokenizers::Tokenizer;
use fast_image_resize::images::Image;

/// Opaque handle to a tokenizer
pub struct TokenizerHandle {
    tokenizer: Tokenizer,
    image_token_id: u32,
    image_token: String,
}

/// C-compatible struct for returning tokenization results
#[repr(C)]
pub struct TokenizationResult {
    pub token_ids: *mut i64,
    pub num_tokens: usize,
    pub image_token_positions: *mut usize,
    pub num_image_tokens: usize,
}

/// C-compatible struct for returning image data
#[repr(C)]
pub struct ImageData {
    pub data: *mut c_float,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

/// C-compatible struct for returning multiple images with grid info
#[repr(C)]
pub struct MultiImageData {
    pub images: *mut ImageData,
    pub num_images: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

/// Load a tokenizer from a JSON file and return an opaque handle
///
/// # Safety
/// - `tokenizer_path` must be a valid null-terminated C string
/// - `image_token` must be a valid null-terminated C string
/// - Returns null pointer on failure
#[no_mangle]
pub unsafe extern "C" fn nanovlm_load_tokenizer(
    tokenizer_path: *const c_char,
    image_token: *const c_char,
) -> *mut TokenizerHandle {
    if tokenizer_path.is_null() || image_token.is_null() {
        return ptr::null_mut();
    }

    let path_cstr = unsafe { CStr::from_ptr(tokenizer_path) };
    let path = match path_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let image_token_cstr = unsafe { CStr::from_ptr(image_token) };
    let img_token = match image_token_cstr.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return ptr::null_mut(),
    };

    let mut tokenizer = match Tokenizer::from_file(path) {
        Ok(t) => t,
        Err(_) => return ptr::null_mut(),
    };

    // Add the image token as a special token (like Python's extra_special_tokens)
    use tokenizers::AddedToken;

    // Add tokens in same order as Python to get consistent token IDs
    // Python's vlm_extra_tokens dict has keys: image_token, global_image_token, r1c1, r1c2, ...
    let mut special_tokens = vec![
        // First: main image token
        AddedToken::from(img_token.clone(), true),
        // Second: global image token
        AddedToken::from("<|global_image|>".to_string(), true),
    ];

    // Third: all row/col tokens (8x8 grid max)
    for row in 1..=8 {
        for col in 1..=8 {
            let token = format!("<row_{}_col_{}>", row, col);
            special_tokens.push(AddedToken::from(token, true));
        }
    }

    tokenizer.add_special_tokens(&special_tokens);

    // Now get the ID of the image token
    let image_token_id = match tokenizer.token_to_id(&img_token) {
        Some(id) => id,
        None => return ptr::null_mut(),
    };

    let handle = Box::new(TokenizerHandle {
        tokenizer,
        image_token_id,
        image_token: img_token,
    });

    Box::into_raw(handle)
}

/// Free a tokenizer handle
///
/// # Safety
/// - `handle` must be a valid pointer returned by `nanovlm_load_tokenizer`
/// - After calling this, the handle must not be used again
#[no_mangle]
pub unsafe extern "C" fn nanovlm_free_tokenizer(handle: *mut TokenizerHandle) {
    if !handle.is_null() {
        let _ = unsafe { Box::from_raw(handle) };
    }
}

/// Tokenize text with image token placeholders
///
/// # Safety
/// - `handle` must be a valid tokenizer handle
/// - `text` must be a valid null-terminated C string
/// - `image_token_length` specifies how many times to repeat the image token
/// - Caller must free the result using `nanovlm_free_tokenization_result`
#[no_mangle]
pub unsafe extern "C" fn nanovlm_tokenize(
    handle: *mut TokenizerHandle,
    text: *const c_char,
    image_token_length: usize,
) -> TokenizationResult {
    if handle.is_null() || text.is_null() {
        return TokenizationResult {
            token_ids: ptr::null_mut(),
            num_tokens: 0,
            image_token_positions: ptr::null_mut(),
            num_image_tokens: 0,
        };
    }

    let tokenizer_handle = unsafe { &*handle };
    let text_cstr = unsafe { CStr::from_ptr(text) };
    let prompt = match text_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return TokenizationResult {
            token_ids: ptr::null_mut(),
            num_tokens: 0,
            image_token_positions: ptr::null_mut(),
            num_image_tokens: 0,
        },
    };

    // Create full prompt with image tokens
    let image_tokens_repeated = tokenizer_handle.image_token.repeat(image_token_length);
    let full_prompt = format!("{}{}", image_tokens_repeated, prompt);

    // Encode
    let encoding = match tokenizer_handle.tokenizer.encode(full_prompt, false) {
        Ok(enc) => enc,
        Err(_) => return TokenizationResult {
            token_ids: ptr::null_mut(),
            num_tokens: 0,
            image_token_positions: ptr::null_mut(),
            num_image_tokens: 0,
        },
    };

    let token_ids = encoding.get_ids();

    // Find positions for ONLY <|image|> tokens
    // Special tokens like <|global_image|> and <row_X_col_Y> are context tokens
    // that should NOT be replaced with embeddings - they tell the model which patch to expect
    let mut image_positions = Vec::new();
    for (idx, &token_id) in token_ids.iter().enumerate() {
        if token_id == tokenizer_handle.image_token_id {
            image_positions.push(idx);
        }
    }

    // Convert to i64 for compatibility with ExecuTorch
    let token_ids_i64: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();
    let num_tokens = token_ids_i64.len();
    let num_image_tokens = image_positions.len();

    // Allocate and copy token IDs
    let mut token_ids_boxed = token_ids_i64.into_boxed_slice();
    let token_ids_ptr = token_ids_boxed.as_mut_ptr();
    std::mem::forget(token_ids_boxed);

    // Allocate and copy image positions
    let mut image_positions_boxed = image_positions.into_boxed_slice();
    let image_positions_ptr = image_positions_boxed.as_mut_ptr();
    std::mem::forget(image_positions_boxed);

    TokenizationResult {
        token_ids: token_ids_ptr,
        num_tokens,
        image_token_positions: image_positions_ptr,
        num_image_tokens,
    }
}

/// Free tokenization result
///
/// # Safety
/// - `result` must be a valid TokenizationResult returned by `nanovlm_tokenize`
#[no_mangle]
pub unsafe extern "C" fn nanovlm_free_tokenization_result(result: TokenizationResult) {
    if !result.token_ids.is_null() {
        let _ = unsafe { Box::from_raw(std::slice::from_raw_parts_mut(result.token_ids, result.num_tokens)) };
    }
    if !result.image_token_positions.is_null() {
        let _ = unsafe { Box::from_raw(std::slice::from_raw_parts_mut(result.image_token_positions, result.num_image_tokens)) };
    }
}

/// Preprocess an image to CHW format (channels, height, width) normalized to [0, 1]
///
/// # Safety
/// - `image_path` must be a valid null-terminated C string pointing to an image file
/// - `target_size` is the size to resize the image to (square)
/// - Caller must free the result using `nanovlm_free_image_data`
#[no_mangle]
pub unsafe extern "C" fn nanovlm_preprocess_image(
    image_path: *const c_char,
    target_size: usize,
) -> ImageData {
    if image_path.is_null() {
        return ImageData {
            data: ptr::null_mut(),
            width: 0,
            height: 0,
            channels: 0,
        };
    }

    let path_cstr = unsafe { CStr::from_ptr(image_path) };
    let path = match path_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return ImageData {
            data: ptr::null_mut(),
            width: 0,
            height: 0,
            channels: 0,
        },
    };

    // Load and decode image
    let img = match ImageReader::open(path) {
        Ok(reader) => match reader.decode() {
            Ok(img) => img,
            Err(_) => return ImageData {
                data: ptr::null_mut(),
                width: 0,
                height: 0,
                channels: 0,
            },
        },
        Err(_) => return ImageData {
            data: ptr::null_mut(),
            width: 0,
            height: 0,
            channels: 0,
        },
    };

    // Resize to target size using CatmullRom (closest to PIL's BICUBIC)
    let resized = img.resize_exact(
        target_size as u32,
        target_size as u32,
        image::imageops::FilterType::CatmullRom,
    );

    // Convert to RGB
    let rgb = resized.to_rgb8();

    // Convert to CHW format and normalize to [0, 1]
    let channels = 3;
    let total_size = channels * target_size * target_size;
    let mut data = vec![0.0f32; total_size];

    for c in 0..channels {
        for y in 0..target_size {
            for x in 0..target_size {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let idx = c * target_size * target_size + y * target_size + x;
                data[idx] = pixel[c] as f32 / 255.0;
            }
        }
    }

    let mut data_boxed = data.into_boxed_slice();
    let data_ptr = data_boxed.as_mut_ptr();
    std::mem::forget(data_boxed);

    ImageData {
        data: data_ptr,
        width: target_size,
        height: target_size,
        channels,
    }
}

/// Free image data
///
/// # Safety
/// - `image_data` must be a valid ImageData returned by `nanovlm_preprocess_image`
#[no_mangle]
pub unsafe extern "C" fn nanovlm_free_image_data(image_data: ImageData) {
    if !image_data.data.is_null() {
        let total_size = image_data.channels * image_data.width * image_data.height;
        let _ = unsafe { Box::from_raw(std::slice::from_raw_parts_mut(image_data.data, total_size)) };
    }
}

/// Preprocess image with dynamic resizing and splitting
///
/// # Safety
/// - `image_path` must be a valid null-terminated C string
/// - Returns MultiImageData with global view + patches
#[no_mangle]
pub unsafe extern "C" fn nanovlm_preprocess_image_with_splitting(
    image_path: *const c_char,
    max_side_len: usize,
    patch_size: usize,
    resize_to_max: i32,
) -> MultiImageData {
    if image_path.is_null() {
        return MultiImageData {
            images: ptr::null_mut(),
            num_images: 0,
            grid_h: 0,
            grid_w: 0,
        };
    }

    let path_cstr = unsafe { CStr::from_ptr(image_path) };
    let path = match path_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return MultiImageData {
            images: ptr::null_mut(),
            num_images: 0,
            grid_h: 0,
            grid_w: 0,
        },
    };

    // Load image
    let img = match ImageReader::open(path) {
        Ok(reader) => match reader.decode() {
            Ok(img) => img,
            Err(_) => return MultiImageData {
                images: ptr::null_mut(),
                num_images: 0,
                grid_h: 0,
                grid_w: 0,
            },
        },
        Err(_) => return MultiImageData {
            images: ptr::null_mut(),
            num_images: 0,
            grid_h: 0,
            grid_w: 0,
        },
    };

    let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);

    // 1. Dynamic resize
    let (new_h, new_w) = compute_dynamic_resize(orig_h, orig_w, max_side_len, patch_size, resize_to_max != 0);

    let resized = img.resize_exact(
        new_w as u32,
        new_h as u32,
        image::imageops::FilterType::CatmullRom,
    );

    // 2. Split into patches
    let grid_h = new_h / patch_size;
    let grid_w = new_w / patch_size;
    let _num_patches = grid_h * grid_w;

    // 3. Create global view + patches (unless only 1 patch)
    let mut image_list = Vec::new();

    if grid_h == 1 && grid_w == 1 {
        // Only one patch - don't add global view
        let patch = process_image_to_chw(&resized, patch_size, patch_size);
        image_list.push(patch);
    } else {
        // Multiple patches - add global view first using BICUBIC interpolation
        // This matches Python's torchvision.transforms.functional.resize with BICUBIC
        let global_data = resize_bicubic(&resized, patch_size, patch_size);
        image_list.push(global_data);

        // Add all patches
        for row in 0..grid_h {
            for col in 0..grid_w {
                let x = col * patch_size;
                let y = row * patch_size;

                let patch = resized.crop_imm(x as u32, y as u32, patch_size as u32, patch_size as u32);
                let patch_data = process_image_to_chw(&patch, patch_size, patch_size);
                image_list.push(patch_data);
            }
        }
    }

    // Convert to C-compatible format
    let num_images = image_list.len();
    let images_vec: Vec<ImageData> = image_list.into_iter().map(|(data, w, h, c)| {
        ImageData {
            data,
            width: w,
            height: h,
            channels: c,
        }
    }).collect();

    let images_boxed = images_vec.into_boxed_slice();
    let images_ptr = Box::into_raw(images_boxed) as *mut ImageData;

    MultiImageData {
        images: images_ptr,
        num_images,
        grid_h,
        grid_w,
    }
}

/// Free multiple image data
///
/// # Safety
/// - `multi_image_data` must be a valid MultiImageData
#[no_mangle]
pub unsafe extern "C" fn nanovlm_free_multi_image_data(multi_image_data: MultiImageData) {
    if !multi_image_data.images.is_null() {
        let images_slice = unsafe {
            std::slice::from_raw_parts_mut(multi_image_data.images, multi_image_data.num_images)
        };

        // Free each image's data
        for img in images_slice.iter() {
            if !img.data.is_null() {
                let total_size = img.channels * img.width * img.height;
                let _ = unsafe { Box::from_raw(std::slice::from_raw_parts_mut(img.data, total_size)) };
            }
        }

        // Free the images array itself
        let _ = unsafe { Box::from_raw(images_slice) };
    }
}

// Helper function: compute dynamic resize dimensions
fn compute_dynamic_resize(h: usize, w: usize, max_side_len: usize, patch_size: usize, resize_to_max: bool) -> (usize, usize) {
    let (long, short) = if w >= h { (w, h) } else { (h, w) };

    // Target long side
    let target_long = if resize_to_max {
        max_side_len
    } else {
        max_side_len.min((long + patch_size - 1) / patch_size * patch_size)
    };

    // Scale factor
    let scale = target_long as f64 / long as f64;

    // Compute short side with ceiling to never undershoot
    let target_short = ((short as f64 * scale / patch_size as f64).ceil() as usize * patch_size).max(patch_size);

    if w >= h {
        (target_short, target_long)
    } else {
        (target_long, target_short)
    }
}

// Helper function: resize image using bicubic interpolation (matches Python's BICUBIC)
fn resize_bicubic(img: &image::DynamicImage, target_w: usize, target_h: usize) -> (*mut c_float, usize, usize, usize) {
    use fast_image_resize as fr;

    let rgb = img.to_rgb8();
    let (src_w, src_h) = rgb.dimensions();

    // Create source image for fast_image_resize
    let src_image = Image::from_vec_u8(
        src_w,
        src_h,
        rgb.into_raw(),
        fr::PixelType::U8x3,
    ).unwrap();

    // Create destination image
    let mut dst_image = Image::new(
        target_w as u32,
        target_h as u32,
        fr::PixelType::U8x3,
    );

    // Resize with bicubic (CatmullRom) - v5.1.4 API
    let mut resizer = fr::Resizer::new();
    resizer.resize(
        &src_image,
        &mut dst_image,
        &fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::CatmullRom))
    ).unwrap();

    // Convert to CHW format and normalize to [0, 1]
    let channels = 3;
    let total_size = channels * target_h * target_w;
    let mut chw_data = vec![0.0f32; total_size];
    let dst_buffer = dst_image.buffer();

    for c in 0..channels {
        for y in 0..target_h {
            for x in 0..target_w {
                let src_idx = (y * target_w + x) * 3 + c;
                let dst_idx = c * target_h * target_w + y * target_w + x;
                chw_data[dst_idx] = dst_buffer[src_idx] as f32 / 255.0;
            }
        }
    }

    let mut data_boxed = chw_data.into_boxed_slice();
    let data_ptr = data_boxed.as_mut_ptr();
    std::mem::forget(data_boxed);

    (data_ptr, target_w, target_h, channels)
}

// Helper function: process image to CHW format
fn process_image_to_chw(img: &image::DynamicImage, target_w: usize, target_h: usize) -> (*mut c_float, usize, usize, usize) {
    let rgb = img.to_rgb8();

    let channels = 3;
    let total_size = channels * target_h * target_w;
    let mut data = vec![0.0f32; total_size];

    for c in 0..channels {
        for y in 0..target_h {
            for x in 0..target_w {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let idx = c * target_h * target_w + y * target_w + x;
                data[idx] = pixel[c] as f32 / 255.0;
            }
        }
    }

    let mut data_boxed = data.into_boxed_slice();
    let data_ptr = data_boxed.as_mut_ptr();
    std::mem::forget(data_boxed);

    (data_ptr, target_w, target_h, channels)
}

/// Decode token IDs back to text
///
/// # Safety
/// - `handle` must be a valid tokenizer handle
/// - `token_ids` must be a valid array of `num_tokens` elements
/// - Returns a newly allocated C string that must be freed with `nanovlm_free_string`
#[no_mangle]
pub unsafe extern "C" fn nanovlm_decode(
    handle: *mut TokenizerHandle,
    token_ids: *const i64,
    num_tokens: usize,
) -> *mut c_char {
    if handle.is_null() || token_ids.is_null() {
        return ptr::null_mut();
    }

    let tokenizer_handle = unsafe { &*handle };
    let ids_slice = unsafe { std::slice::from_raw_parts(token_ids, num_tokens) };
    let ids_u32: Vec<u32> = ids_slice.iter().map(|&id| id as u32).collect();

    let text = match tokenizer_handle.tokenizer.decode(&ids_u32, true) {
        Ok(t) => t,
        Err(_) => return ptr::null_mut(),
    };

    match CString::new(text) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a string returned by nanovlm_decode
///
/// # Safety
/// - `str` must be a valid pointer returned by `nanovlm_decode`
#[no_mangle]
pub unsafe extern "C" fn nanovlm_free_string(str: *mut c_char) {
    if !str.is_null() {
        let _ = unsafe { CString::from_raw(str) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;

    #[test]
    fn test_image_preprocessing_matches_python() {
        // Load Python reference data
        let python_tensor_path = "../test_image_tensor.npy";
        let python_tensor_file = File::open(python_tensor_path)
            .expect("Failed to open test_image_tensor.npy - run dump_preprocessing_reference.py first");
        let python_tensor: Array3<f32> = Array3::read_npy(python_tensor_file)
            .expect("Failed to read numpy array");

        println!("Python tensor shape: {:?}", python_tensor.shape());

        // Run Rust preprocessing
        let image_path = "../assets/image.png";
        let target_size = 512;

        let image_path_cstr = CString::new(image_path).unwrap();
        let image_data = unsafe {
            nanovlm_preprocess_image(image_path_cstr.as_ptr(), target_size)
        };

        assert!(!image_data.data.is_null(), "Image preprocessing failed");
        assert_eq!(image_data.width, target_size);
        assert_eq!(image_data.height, target_size);
        assert_eq!(image_data.channels, 3);

        // Convert to slice for comparison
        let total_size = image_data.channels * image_data.width * image_data.height;
        let rust_data = unsafe {
            std::slice::from_raw_parts(image_data.data, total_size)
        };

        // Compare against Python output
        let python_flat = python_tensor.as_slice().unwrap();

        let mut max_diff = 0.0f32;
        let mut num_mismatches = 0;
        let tolerance = 1e-6;

        for (i, (&rust_val, &python_val)) in rust_data.iter().zip(python_flat.iter()).enumerate() {
            let diff = (rust_val - python_val).abs();
            if diff > tolerance {
                num_mismatches += 1;
                if diff > max_diff {
                    max_diff = diff;
                    if num_mismatches <= 5 {
                        println!("Mismatch at index {}: rust={:.6}, python={:.6}, diff={:.6}",
                                 i, rust_val, python_val, diff);
                    }
                }
            }
        }

        println!("Max difference: {:.6}", max_diff);
        println!("Number of mismatches (tolerance={:.6}): {}/{}", tolerance, num_mismatches, total_size);

        // Cleanup
        unsafe { nanovlm_free_image_data(image_data); }

        // CatmullRom is close to BICUBIC but not identical - allow up to 2% difference
        // This is acceptable for inference (4/255 quantization difference)
        assert!(max_diff < 0.02, "Image preprocessing differs from Python by {:.6}", max_diff);
    }

    #[test]
    fn test_tokenization_matches_python() {
        // Load Python reference data
        let token_ids_path = "../test_token_ids.npy";
        let token_ids_file = File::open(token_ids_path)
            .expect("Failed to open test_token_ids.npy - run dump_preprocessing_reference.py first");
        let python_token_ids: ndarray::Array1<i64> = ndarray::Array1::read_npy(token_ids_file)
            .expect("Failed to read token IDs");

        let positions_path = "../test_image_token_positions.npy";
        let positions_file = File::open(positions_path)
            .expect("Failed to open test_image_token_positions.npy");
        let python_positions: ndarray::Array1<i64> = ndarray::Array1::read_npy(positions_file)
            .expect("Failed to read positions");

        println!("Python tokens: {}", python_token_ids.len());
        println!("Python image token positions: {}", python_positions.len());

        // Run Rust tokenization
        let tokenizer_path = "/tmp/tokenizer/tokenizer.json";
        let image_token = "<|image|>";
        let prompt = "What is in this image?";
        let image_token_length = 256;

        let tokenizer_path_cstr = CString::new(tokenizer_path).unwrap();
        let image_token_cstr = CString::new(image_token).unwrap();
        let prompt_cstr = CString::new(prompt).unwrap();

        let tokenizer_handle = unsafe {
            nanovlm_load_tokenizer(tokenizer_path_cstr.as_ptr(), image_token_cstr.as_ptr())
        };
        assert!(!tokenizer_handle.is_null(), "Failed to load tokenizer");

        let tok_result = unsafe {
            nanovlm_tokenize(tokenizer_handle, prompt_cstr.as_ptr(), image_token_length)
        };

        assert!(!tok_result.token_ids.is_null(), "Tokenization failed");

        // Compare token IDs
        let rust_tokens = unsafe {
            std::slice::from_raw_parts(tok_result.token_ids, tok_result.num_tokens)
        };

        println!("Rust tokens: {}", rust_tokens.len());

        assert_eq!(rust_tokens.len(), python_token_ids.len(),
                   "Token count mismatch: Rust={}, Python={}",
                   rust_tokens.len(), python_token_ids.len());

        let mut num_mismatches = 0;
        for (i, (&rust_id, &python_id)) in rust_tokens.iter().zip(python_token_ids.iter()).enumerate() {
            if rust_id != python_id {
                num_mismatches += 1;
                if num_mismatches <= 5 {
                    println!("Token mismatch at index {}: rust={}, python={}", i, rust_id, python_id);
                }
            }
        }

        assert_eq!(num_mismatches, 0, "Found {} token mismatches", num_mismatches);

        // Compare image token positions
        let rust_positions = unsafe {
            std::slice::from_raw_parts(tok_result.image_token_positions, tok_result.num_image_tokens)
        };

        println!("Rust image token positions: {}", rust_positions.len());

        assert_eq!(rust_positions.len(), python_positions.len() as usize,
                   "Image token position count mismatch");

        for (i, (&rust_pos, &python_pos)) in rust_positions.iter().zip(python_positions.iter()).enumerate() {
            assert_eq!(rust_pos, python_pos as usize,
                       "Position mismatch at index {}: rust={}, python={}",
                       i, rust_pos, python_pos);
        }

        println!("✅ Tokenization matches Python exactly!");

        // Cleanup
        unsafe {
            nanovlm_free_tokenization_result(tok_result);
            nanovlm_free_tokenizer(tokenizer_handle);
        }
    }
}
