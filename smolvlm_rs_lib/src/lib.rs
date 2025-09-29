#[cfg(target_os = "android")]
#[allow(non_snake_case)]
pub mod smolvlm_snap {
    extern crate jni;

    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::value::Tensor;
    use image::{ImageBuffer, Rgb, RgbImage, DynamicImage, GenericImageView};
    use anyhow::{Result as AnyhowResult, Context};
    use ndarray::{Array, Array4};
    use jni::JNIEnv;
    use jni::objects::{JClass, JObject, JString};
    use jni::sys::{jstring, jint, jboolean};
    use tokenizers::Tokenizer;
    use std::sync::Mutex;
    use std::collections::HashMap;

    static mut SMOLVLM_INSTANCE: Option<Mutex<SmolVLMProcessor>> = None;

    pub struct SmolVLMProcessor {
        vision_session: Option<Session>,
        embed_session: Option<Session>,
        decoder_session: Option<Session>,
        tokenizer: Option<Tokenizer>,
        model_paths: HashMap<String, String>,
    }

    impl SmolVLMProcessor {
        fn new() -> AnyhowResult<Self> {
            Ok(SmolVLMProcessor {
                vision_session: None,
                embed_session: None,
                decoder_session: None,
                tokenizer: None,
                model_paths: HashMap::new(),
            })
        }

        fn load_models(&mut self, model_dir: &str, tokenizer_path: &str) -> AnyhowResult<bool> {
            // Load tokenizer
            let tokenizer = Tokenizer::from_file(tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
            self.tokenizer = Some(tokenizer);

            // Load vision encoder
            let vision_path = format!("{}/vision_encoder_q4.onnx", model_dir);
            let vision_session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(&vision_path)
                .context("Failed to load vision encoder")?;
            self.vision_session = Some(vision_session);

            // Load embed tokens
            let embed_path = format!("{}/embed_tokens_q4.onnx", model_dir);
            let embed_session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(&embed_path)
                .context("Failed to load embed tokens")?;
            self.embed_session = Some(embed_session);

            // Load decoder
            let decoder_path = format!("{}/decoder_model_merged_q4.onnx", model_dir);
            let decoder_session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(&decoder_path)
                .context("Failed to load decoder")?;
            self.decoder_session = Some(decoder_session);

            Ok(true)
        }

        fn process_image_from_buffer(&self, buffer: &[u8], width: i32, height: i32) -> AnyhowResult<Array4<f32>> {
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

            self.preprocess_image(DynamicImage::ImageRgb8(img))
        }

        fn preprocess_image(&self, image: DynamicImage) -> AnyhowResult<Array4<f32>> {
            // Resize to model input size (typically 384x384 for SmolVLM)
            let resized = image.resize_exact(384, 384, image::imageops::FilterType::CatmullRom);
            let rgb = resized.to_rgb8();

            // Convert to CHW format and normalize
            let mut data = Array4::<f32>::zeros((1, 3, 384, 384));

            for c in 0..3 {
                for y in 0..384 {
                    for x in 0..384 {
                        let pixel = rgb.get_pixel(x, y);
                        // Normalize to [0, 1] and then to model expected range
                        let normalized = (pixel[c] as f32 / 255.0 - 0.5) / 0.5;
                        data[[0, c, y as usize, x as usize]] = normalized;
                    }
                }
            }

            Ok(data)
        }

        fn generate_response(&self, prompt: &str, max_tokens: i32, callback: Box<dyn Fn(&str) + Send>) -> AnyhowResult<String> {
            let tokenizer = self.tokenizer.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;

            // Tokenize prompt
            let encoding = tokenizer.encode(prompt, false)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
            let input_ids = encoding.get_ids();

            // Convert to tensor format
            let input_tensor = Array::from_shape_vec(
                (1, input_ids.len()),
                input_ids.iter().map(|&x| x as i64).collect()
            )?;

            // Get embeddings
            let embed_session = self.embed_session.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Embed session not loaded"))?;

            let embed_input = Tensor::from_array(input_tensor)?;
            let embed_outputs = embed_session.run(ort::inputs!["input_ids" => embed_input])?;
            let embeddings = embed_outputs[0].try_extract_tensor::<f32>()?;

            // Generation loop
            let mut generated_tokens = Vec::new();
            let embeddings_array = Array::from_shape_vec(embeddings.0.dims()?, embeddings.1.to_vec())?;
            let mut current_embeddings = embeddings_array;

            for _ in 0..max_tokens {
                // Run decoder
                let decoder_session = self.decoder_session.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Decoder session not loaded"))?;

                let decoder_input = Tensor::from_array(current_embeddings.view())?;
                let decoder_outputs = decoder_session.run(ort::inputs!["inputs_embeds" => decoder_input])?;
                let logits = decoder_outputs[0].try_extract_tensor::<f32>()?;

                // Get next token (argmax for simplicity)
                let logits_array = Array::from_shape_vec(logits.0.dims()?, logits.1.to_vec())?;
                let last_logits = logits_array.slice(ndarray::s![0, -1, ..]);
                let next_token = last_logits.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap();

                generated_tokens.push(next_token);

                // Check for EOS token (assuming 2 is EOS)
                if next_token == 2 {
                    break;
                }

                // Decode partial result and call callback
                if let Ok(decoded) = tokenizer.decode(&generated_tokens, false) {
                    callback(&decoded);
                }

                // Update embeddings for next iteration
                let next_token_array = Array::from_shape_vec((1, 1), vec![next_token as i64])?;
                let next_embed_input = Tensor::from_array(next_token_array)?;
                let next_embed_outputs = embed_session.run(ort::inputs!["input_ids" => next_embed_input])?;
                let next_embeddings = next_embed_outputs[0].try_extract_tensor::<f32>()?;

                // Concatenate embeddings
                current_embeddings = ndarray::concatenate![
                    ndarray::Axis(1),
                    current_embeddings.view(),
                    next_embeddings.view()
                ];
            }

            // Final decode
            tokenizer.decode(&generated_tokens, false)
                .map_err(|e| anyhow::anyhow!("Final decode failed: {}", e))
        }
    }

    fn get_or_create_instance() -> &'static Mutex<SmolVLMProcessor> {
        unsafe {
            if SMOLVLM_INSTANCE.is_none() {
                SMOLVLM_INSTANCE = Some(Mutex::new(
                    SmolVLMProcessor::new().expect("Failed to create SmolVLM processor")
                ));
            }
            SMOLVLM_INSTANCE.as_ref().unwrap()
        }
    }

    #[no_mangle]
    pub extern "C" fn Java_ai_baseweight_baseweightsnap_SmolVLMAndroid_loadModels(
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

    #[no_mangle]
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
            Ok(processor) => {
                match processor.process_image_from_buffer(buffer_slice, width, height) {
                    Ok(_) => true as jboolean,
                    Err(_) => false as jboolean,
                }
            }
            Err(_) => false as jboolean,
        }
    }

    #[no_mangle]
    pub extern "C" fn Java_ai_baseweight_baseweightsnap_SmolVLMAndroid_generateResponse(
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
                let callback = Box::new(|_text: &str| {
                    // For now, we'll just accumulate and return the final result
                    // In a more sophisticated implementation, we'd use a callback mechanism
                });

                match processor.generate_response(&prompt_str, max_tokens, callback) {
                    Ok(result) => env.new_string(&result).unwrap().into_raw(),
                    Err(e) => env.new_string(&format!("Error: {}", e)).unwrap().into_raw(),
                }
            }
            Err(_) => env.new_string("Error: Failed to lock processor").unwrap().into_raw(),
        }
    }
}