mod image_processor;

#[cfg(target_os = "android")]
#[allow(non_snake_case)]
pub mod smolvlm_snap {
    extern crate jni;

    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::value::Value;
    use ort::execution_providers::XNNPACKExecutionProvider;
    use image::{RgbImage, DynamicImage};
    use anyhow::{Result as AnyhowResult, Context};
    use ndarray::{Array2, s};
    use jni::JNIEnv;
    use jni::objects::{JClass, JObject, JString};
    use jni::sys::{jstring, jint, jboolean};
    use std::sync::Mutex;
    use std::collections::HashMap;
    use tokenizers::Tokenizer;
    use crate::image_processor::SmolVLMImageProcessor;

    use std::sync::OnceLock;
    static SMOLVLM_INSTANCE: OnceLock<Mutex<SmolVLMProcessor>> = OnceLock::new();

    pub struct SmolVLMProcessor {
        vision_session: Option<Session>,
        embed_session: Option<Session>,
        decoder_session: Option<Session>,
        tokenizer: Option<Tokenizer>,
        image_processor: SmolVLMImageProcessor,
        initialized: bool,
    }

    impl SmolVLMProcessor {
        fn new() -> AnyhowResult<Self> {
            Ok(SmolVLMProcessor {
                vision_session: None,
                embed_session: None,
                decoder_session: None,
                tokenizer: None,
                image_processor: SmolVLMImageProcessor::new(),
                initialized: false,
            })
        }

        fn load_models(&mut self, model_dir: &str, tokenizer_path: &str) -> AnyhowResult<bool> {
            // Load tokenizer
            let tokenizer = Tokenizer::from_file(tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

            // Configure session builder with XNNPack
            let session_builder = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([XNNPACKExecutionProvider::default().build()])?;

            // Load vision encoder
            let vision_path = format!("{}/vision_encoder_q4.onnx", model_dir);
            let vision_session = session_builder
                .clone()
                .commit_from_file(&vision_path)
                .context("Failed to load vision encoder")?;

            // Load embed tokens model
            let embed_path = format!("{}/embed_tokens_q4.onnx", model_dir);
            let embed_session = session_builder
                .clone()
                .commit_from_file(&embed_path)
                .context("Failed to load embed tokens model")?;

            // Load decoder model
            let decoder_path = format!("{}/decoder_model_merged_q4.onnx", model_dir);
            let decoder_session = session_builder
                .commit_from_file(&decoder_path)
                .context("Failed to load decoder model")?;

            self.vision_session = Some(vision_session);
            self.embed_session = Some(embed_session);
            self.decoder_session = Some(decoder_session);
            self.tokenizer = Some(tokenizer);
            self.initialized = true;
            Ok(true)
        }

        fn process_image_from_buffer(&mut self, buffer: &[u8], width: i32, height: i32) -> AnyhowResult<bool> {
            if !self.initialized {
                return Ok(false);
            }

            // Convert ARGB8888 buffer to RGB image
            let mut rgb_data = Vec::with_capacity((height * width * 3) as usize);
            for i in 0..(height * width) as usize {
                let idx = i * 4;
                // ARGB8888 format: [B, G, R, A] in memory
                rgb_data.push(buffer[idx + 2]); // R
                rgb_data.push(buffer[idx + 1]); // G
                rgb_data.push(buffer[idx]);     // B
            }

            let img = RgbImage::from_raw(
                width as u32,
                height as u32,
                rgb_data
            ).ok_or_else(|| anyhow::anyhow!("Failed to create image from buffer"))?;

            let dynamic_img = DynamicImage::ImageRgb8(img);

            // Use the SmolVLM image processor for proper preprocessing
            let processed_images = self.image_processor.preprocess(&dynamic_img)?;

            // Run vision encoder
            if let Some(ref mut session) = self.vision_session {
                let mut inputs = HashMap::new();

                // SmolVLM expects input shape [batch, num_patches, channels, height, width]
                // We need to process each patch separately for the vision encoder
                let (batch_size, num_patches, channels, height, width) = processed_images.dim();

                for patch_idx in 0..num_patches {
                    let patch = processed_images.slice(s![.., patch_idx..patch_idx+1, .., .., ..]);
                    let patch_owned = patch.to_owned();
                    let patch_4d = patch_owned.to_shape((batch_size, channels, height, width))?;

                    inputs.insert("pixel_values", Value::from_array(patch_4d.to_owned())?);
                    let _outputs = session.run(inputs.clone())?;
                }

                return Ok(true);
            }

            Ok(false)
        }

        fn generate_response(&self, prompt: &str, max_tokens: i32) -> AnyhowResult<String> {
            if !self.initialized {
                return Ok("Error: Models not loaded".to_string());
            }

            let tokenizer = self.tokenizer.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;

            // Encode the prompt
            let encoding = tokenizer.encode(prompt, false)
                .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;
            let input_ids = encoding.get_ids();

            // Convert to ndarray
            let input_tensor = Array2::from_shape_vec(
                (1, input_ids.len()),
                input_ids.iter().map(|&id| id as i64).collect(),
            )?;

            // For now, implement a simple generation loop
            let _current_ids = input_tensor.clone();
            let _generated_text = String::new();

            for _ in 0..max_tokens.min(50) { // Limit to prevent infinite loops
                // For now, just break after a few iterations to avoid infinite loops
                // In a full implementation, we'd run the embed and decoder models properly
                break;
            }

            // For now, return a simplified response
            Ok(format!("SmolVLM response to: {}", prompt.chars().take(50).collect::<String>()))
        }
    }

    fn get_or_create_instance() -> &'static Mutex<SmolVLMProcessor> {
        SMOLVLM_INSTANCE.get_or_init(|| {
            Mutex::new(SmolVLMProcessor::new().expect("Failed to create SmolVLM processor"))
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn Java_ai_baseweight_baseweightsnap_SmolVLMAndroid_nativeLoadModels(
        env: JNIEnv,
        _class: JClass,
        model_dir: JString,
        tokenizer_path: JString,
    ) -> jboolean {
        let model_dir_str = match env.get_string(model_dir) {
            Ok(s) => s.to_string_lossy().to_string(),
            Err(_) => return false as jboolean,
        };

        let tokenizer_path_str = match env.get_string(tokenizer_path) {
            Ok(s) => s.to_string_lossy().to_string(),
            Err(_) => return false as jboolean,
        };

        let instance = get_or_create_instance();
        match instance.lock() {
            Ok(mut processor) => {
                match processor.load_models(&model_dir_str, &tokenizer_path_str) {
                    Ok(success) => success as jboolean,
                    Err(_) => false as jboolean,
                }
            }
            Err(_) => false as jboolean,
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn Java_ai_baseweight_baseweightsnap_SmolVLMAndroid_processImageFromBuffer(
        env: JNIEnv,
        _class: JClass,
        buffer: JObject,
        width: jint,
        height: jint,
    ) -> jboolean {
        let buffer_ptr = match env.get_direct_buffer_address(buffer.into()) {
            Ok(ptr) => ptr,
            Err(_) => return false as jboolean,
        };

        let buffer_slice = unsafe {
            std::slice::from_raw_parts(
                buffer_ptr as *const u8,
                (height * width * 4) as usize
            )
        };

        let instance = get_or_create_instance();
        match instance.lock() {
            Ok(mut processor) => {
                match processor.process_image_from_buffer(buffer_slice, width, height) {
                    Ok(success) => success as jboolean,
                    Err(_) => false as jboolean,
                }
            }
            Err(_) => false as jboolean,
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn Java_ai_baseweight_baseweightsnap_SmolVLMAndroid_nativeGenerateResponse(
        env: JNIEnv,
        _class: JClass,
        prompt: JString,
        max_tokens: jint,
    ) -> jstring {
        let prompt_str = match env.get_string(prompt) {
            Ok(s) => s.to_string_lossy().to_string(),
            Err(_) => return env.new_string("Error: Failed to get prompt").unwrap().into_raw(),
        };

        let instance = get_or_create_instance();
        match instance.lock() {
            Ok(processor) => {
                match processor.generate_response(&prompt_str, max_tokens) {
                    Ok(result) => env.new_string(&result).unwrap().into_raw(),
                    Err(e) => env.new_string(&format!("Error: {}", e)).unwrap().into_raw(),
                }
            }
            Err(_) => env.new_string("Error: Failed to lock processor").unwrap().into_raw(),
        }
    }
}