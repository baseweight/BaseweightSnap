mod image_processor;

#[cfg(target_os = "android")]
#[allow(non_snake_case)]
pub mod smolvlm_snap {
    extern crate jni;

    use log::{info, error};
    use android_logger::Config;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn init_logger() {
        INIT.call_once(|| {
            android_logger::init_once(
                Config::default()
                    .with_max_level(log::LevelFilter::Info)
                    .with_tag("SmolVLM")
            );
        });
    }

    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::value::Value;
    use ort::execution_providers::{XNNPACKExecutionProvider, ExecutionProvider};
    use image::{RgbImage, DynamicImage};
    use anyhow::{Result as AnyhowResult, Context};
    use ndarray::{Array2, Array3};
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
        image_embeddings: Option<ndarray::Array2<f32>>,
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
                image_embeddings: None,
                initialized: false,
            })
        }

        fn load_models(&mut self, model_dir: &str, tokenizer_path: &str) -> AnyhowResult<bool> {
            init_logger();
            info!("SmolVLM: Starting model loading from {} with tokenizer {}", model_dir, tokenizer_path);

            // Load tokenizer
            info!("SmolVLM: Loading tokenizer from {}", tokenizer_path);
            let tokenizer = Tokenizer::from_file(tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
            info!("SmolVLM: Tokenizer loaded successfully");

            // Configure session builder with XNNPack
            info!("SmolVLM: Configuring session builder with XNNPack");

            // Check if XNNPack is available
            match XNNPACKExecutionProvider::default().is_available() {
                Ok(true) => info!("SmolVLM: XNNPack execution provider is AVAILABLE"),
                Ok(false) => error!("SmolVLM: XNNPack execution provider is NOT AVAILABLE - will fall back to CPU!"),
                Err(e) => error!("SmolVLM: Error checking XNNPack availability: {}", e),
            }

            // Try to enable XNNPack with explicit configuration
            let xnnpack = XNNPACKExecutionProvider::default()
                .with_intra_op_num_threads(std::num::NonZeroUsize::new(4).unwrap())
                .build();

            let session_builder = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .with_execution_providers([xnnpack])?;
            info!("SmolVLM: Session builder configured with XNNPack");

            // Load vision encoder - use uint8 for better XNNPack support
            let vision_path = format!("{}/vision_encoder_uint8.onnx", model_dir);
            info!("SmolVLM: Loading vision encoder (uint8) from {}", vision_path);
            let vision_session = session_builder
                .clone()
                .commit_from_file(&vision_path)
                .context("Failed to load vision encoder")?;

            info!("SmolVLM: Vision encoder loaded successfully");

            // Load embed tokens model - use uint8 for better XNNPack support
            let embed_path = format!("{}/embed_tokens_uint8.onnx", model_dir);
            info!("SmolVLM: Loading embed tokens (uint8) from {}", embed_path);
            let embed_session = session_builder
                .clone()
                .commit_from_file(&embed_path)
                .context("Failed to load embed tokens model")?;
            info!("SmolVLM: Embed tokens loaded successfully");

            // Load decoder model - use uint8 for better XNNPack support
            let decoder_path = format!("{}/decoder_model_merged_uint8.onnx", model_dir);
            info!("SmolVLM: Loading decoder (uint8) from {}", decoder_path);
            let decoder_session = session_builder
                .commit_from_file(&decoder_path)
                .context("Failed to load decoder model")?;
            info!("SmolVLM: Decoder loaded successfully");

            self.vision_session = Some(vision_session);
            self.embed_session = Some(embed_session);
            self.decoder_session = Some(decoder_session);
            self.tokenizer = Some(tokenizer);
            self.initialized = true;
            info!("SmolVLM: All models loaded successfully, processor initialized");
            Ok(true)
        }

        fn process_image_from_buffer(&mut self, buffer: &[u8], width: i32, height: i32) -> AnyhowResult<bool> {
            if !self.initialized {
                info!("SmolVLM: Cannot process image - models not initialized");
                return Ok(false);
            }

            info!("SmolVLM: Processing image {}x{}", width, height);

            // Convert ARGB8888 buffer to RGB image
            // Android's ARGB_8888 with copyPixelsToBuffer gives us RGBA in memory
            let mut rgb_data = Vec::with_capacity((height * width * 3) as usize);
            for i in 0..(height * width) as usize {
                let idx = i * 4;
                // RGBA format: [R, G, B, A] in memory
                rgb_data.push(buffer[idx]);     // R
                rgb_data.push(buffer[idx + 1]); // G
                rgb_data.push(buffer[idx + 2]); // B
                // Skip alpha channel (idx + 3)
            }

            let img = RgbImage::from_raw(
                width as u32,
                height as u32,
                rgb_data
            ).ok_or_else(|| anyhow::anyhow!("Failed to create image from buffer"))?;

            let dynamic_img = DynamicImage::ImageRgb8(img);

            // Use the SmolVLM image processor for proper preprocessing
            info!("SmolVLM: Preprocessing image with SmolVLM processor");
            let (processed_images, attention_mask) = self.image_processor.preprocess(dynamic_img)?;

            // Run vision encoder
            if let Some(ref mut session) = self.vision_session {
                info!("SmolVLM: Running vision encoder");
                let vision_start = std::time::Instant::now();
                let mut inputs: HashMap<&str, Value> = HashMap::new();

                // For SmolVLM2, we process the full image tensor at once
                let shape = processed_images.shape();
                info!("SmolVLM: Image tensor shape: {:?}", shape);

                // Keep the 5D shape (batch_size, num_patches, channels, height, width) as expected by the vision encoder
                inputs.insert("pixel_values", Value::from_array(processed_images.clone())?.into());

                // Convert attention mask to boolean while maintaining shape (batch, num_frames, height, width)
                let pixel_attention_mask_bool = attention_mask.map(|&x| x != 0);
                inputs.insert("pixel_attention_mask", Value::from_array(pixel_attention_mask_bool)?.into());

                info!("SmolVLM: About to run vision encoder with pixel_values and pixel_attention_mask inputs");
                let outputs = match session.run(inputs) {
                    Ok(outputs) => {
                        let vision_elapsed = vision_start.elapsed();
                        info!("SmolVLM: Vision encoder ran successfully in {:.2}ms, got {} outputs", vision_elapsed.as_millis(), outputs.len());
                        // Log all available output keys
                        let output_keys: Vec<String> = outputs.keys().map(|k| k.to_string()).collect();
                        info!("SmolVLM: Vision encoder output keys: {:?}", output_keys);
                        outputs
                    }
                    Err(e) => {
                        error!("SmolVLM: Vision encoder failed: {}", e);
                        return Ok(false);
                    }
                };

                // Extract image embeddings from vision encoder output (take first output)
                if let Some((_, embeddings_value)) = outputs.into_iter().next() {
                    let embeddings_array = embeddings_value.try_extract_tensor::<f32>()?;
                    let (shape, data) = embeddings_array;
                    info!("SmolVLM: Vision encoder output shape: {:?}", shape);

                    // Convert to ndarray for easier manipulation
                    let total_len = data.len();
                    let last_dim = shape[shape.len() - 1] as usize;
                    let seq_len = total_len / last_dim;

                    let embeddings_2d = ndarray::Array2::from_shape_vec((seq_len, last_dim), data.to_vec())?;
                    self.image_embeddings = Some(embeddings_2d);
                    info!("SmolVLM: Image embeddings extracted, shape: {:?}", (seq_len, last_dim));
                    return Ok(true);
                } else {
                    error!("SmolVLM: Vision encoder returned no outputs");
                    return Ok(false);
                }
            } else {
                error!("SmolVLM: Vision session not available");
                return Ok(false);
            }
        }

        fn generate_response(&mut self, prompt: &str, max_tokens: i32) -> AnyhowResult<String> {
            if !self.initialized {
                info!("SmolVLM: Cannot generate response - models not initialized");
                return Ok("Error: Models not loaded".to_string());
            }

            let image_embeddings = match &self.image_embeddings {
                Some(embeddings) => embeddings,
                None => {
                    info!("SmolVLM: No image processed yet");
                    return Ok("Please capture and process an image first.".to_string());
                }
            };

            info!("SmolVLM: Starting text generation for prompt: {}", prompt);

            let tokenizer = self.tokenizer.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;

            // Get image token ID
            let image_token = "<image>";
            let image_token_id = tokenizer.token_to_id(image_token)
                .ok_or_else(|| anyhow::anyhow!("Failed to get image token ID"))?;

            // SmolVLM2 uses 4x4 grid with 64 tokens per patch
            let image_rows = 4;
            let image_cols = 4;
            let image_seq_len = 64;

            // Generate the full image prompt string with proper structure
            let mut image_prompt_str = String::new();
            for n_h in 0..image_rows {
                for n_w in 0..image_cols {
                    image_prompt_str.push_str(&format!(
                        "<fake_token_around_image><row_{}_col_{}>{}",
                        n_h + 1,
                        n_w + 1,
                        image_token.repeat(image_seq_len)
                    ));
                }
                image_prompt_str.push('\n');
            }
            image_prompt_str.push_str(&format!(
                "\n<fake_token_around_image><global-img>{}",
                image_token.repeat(image_seq_len)
            ));
            image_prompt_str.push_str("<fake_token_around_image>");

            // Create the full prompt with expanded image tokens
            let full_prompt = format!("<|im_start|>User:{}\n{}<end_of_utterance>\nAssistant:",
                                     image_prompt_str, prompt);

            info!("SmolVLM: Full prompt length: {}", full_prompt.len());

            // Encode the expanded prompt
            let encoding = tokenizer.encode(full_prompt.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;
            let input_ids = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();

            info!("SmolVLM: Encoded prompt to {} tokens", input_ids.len());

            // Convert input IDs to ndarray
            let input_ids_array = ndarray::Array::from_vec(input_ids.iter().map(|&x| x as i64).collect())
                .into_shape_with_order((1, input_ids.len()))?;
            let attention_mask_array = ndarray::Array::from_vec(attention_mask.iter().map(|&x| x as i64).collect())
                .into_shape_with_order((1, attention_mask.len()))?;

            // Get embeddings for all tokens
            let mut input_embeds = {
                let mut embed_inputs: HashMap<&str, Value> = HashMap::new();
                embed_inputs.insert("input_ids", Value::from_array(input_ids_array.clone())?.into());

                let embed_outputs = if let Some(ref mut embed_session) = self.embed_session {
                    embed_session.run(embed_inputs)?
                } else {
                    return Ok("Error: Embedding model not loaded".to_string());
                };

                let embeddings_value = embed_outputs.values().next()
                    .ok_or_else(|| anyhow::anyhow!("No embedding output"))?;
                let tensor_data = embeddings_value.try_extract_tensor::<f32>()?;
                let (shape, data) = tensor_data;
                ndarray::Array3::from_shape_vec(
                    (shape[0] as usize, shape[1] as usize, shape[2] as usize),
                    data.to_vec()
                )?
            };

            info!("SmolVLM: Input embeddings shape: {:?}", input_embeds.shape());

            // Replace image token embeddings with actual image features
            let mut feature_idx = 0;
            for i in 0..input_ids_array.shape()[1] {
                if input_ids_array[[0, i]] == image_token_id as i64 {
                    if feature_idx < image_embeddings.shape()[0] {
                        for j in 0..image_embeddings.shape()[1] {
                            input_embeds[[0, i, j]] = image_embeddings[[feature_idx, j]];
                        }
                        feature_idx += 1;
                    }
                }
            }

            info!("SmolVLM: Replaced {} image tokens with vision features", feature_idx);

            let combined_embeddings = input_embeds;

            // Generate tokens using decoder model with proper past key values
            if let Some(ref mut decoder_session) = self.decoder_session {
                info!("SmolVLM: Starting generation with proper past key values");

                // Auto-detect model configuration from decoder inputs
                let decoder_inputs = &decoder_session.inputs;
                let mut max_layer = 0;
                let detected_kv_heads = 5; // Based on error messages - updated from 3 to 5

                for input in decoder_inputs {
                    if input.name.starts_with("past_key_values.") {
                        if let Some(layer_str) = input.name.split('.').nth(1) {
                            if let Ok(layer_num) = layer_str.parse::<usize>() {
                                max_layer = max_layer.max(layer_num);
                            }
                        }
                    }
                }

                let num_hidden_layers = max_layer + 1;
                let head_dim = 64; // Standard head dimension for SmolVLM
                info!("SmolVLM: Auto-detected {} layers, {} kv heads", num_hidden_layers, detected_kv_heads);

                // Initialize past key values
                let batch_size = 1;
                let mut past_key_values: HashMap<String, ndarray::ArrayD<f32>> = HashMap::new();
                for layer in 0..num_hidden_layers {
                    let key_array = ndarray::Array::zeros((batch_size, detected_kv_heads, 0, head_dim)).into_dyn();
                    let value_array = ndarray::Array::zeros((batch_size, detected_kv_heads, 0, head_dim)).into_dyn();
                    past_key_values.insert(format!("past_key_values.{}.key", layer), key_array);
                    past_key_values.insert(format!("past_key_values.{}.value", layer), value_array);
                }

                // Use the properly tokenized input_ids for the initial prompt
                let mut input_ids = input_ids_array.clone();

                // Use the attention mask from tokenization (already matches the sequence)
                let mut attention_mask = attention_mask_array.clone();

                // Calculate cumulative position_ids
                let mut position_ids_vec = Vec::new();
                let mut cumsum = 0i64;
                for &mask_val in attention_mask.iter() {
                    cumsum += mask_val;
                    position_ids_vec.push(cumsum);
                }
                let mut position_ids = ndarray::Array2::from_shape_vec((1, attention_mask.len()), position_ids_vec)?;

                let mut generated_tokens = Vec::new();
                let max_new_tokens = max_tokens.min(50);

                for step in 0..max_new_tokens {
                    // Get input embeddings
                    let input_embeds = if step == 0 {
                        // Use the pre-computed combined embeddings for first step
                        combined_embeddings.clone()
                    } else {
                        // Get embeddings for the new token
                        if let Some(ref mut embed_session) = self.embed_session {
                            let mut embed_inputs: HashMap<&str, Value> = HashMap::new();
                            embed_inputs.insert("input_ids", Value::from_array(input_ids.clone())?.into());
                            let embed_outputs = embed_session.run(embed_inputs)?;

                            if let Some(embeddings_value) = embed_outputs.values().next() {
                                let tensor_data = embeddings_value.try_extract_tensor::<f32>()?;
                                let (shape, data) = tensor_data;
                                // Convert to ndarray
                                Array3::from_shape_vec(
                                    (shape[0] as usize, shape[1] as usize, shape[2] as usize),
                                    data.to_vec(),
                                )?
                            } else {
                                return Err(anyhow::anyhow!("No embeddings output"));
                            }
                        } else {
                            return Err(anyhow::anyhow!("Embed session not available"));
                        }
                    };

                    // Prepare decoder inputs
                    let mut decoder_inputs: HashMap<&str, Value> = HashMap::new();
                    decoder_inputs.insert("inputs_embeds", Value::from_array(input_embeds.clone())?.into());
                    decoder_inputs.insert("attention_mask", Value::from_array(attention_mask.clone())?.into());
                    decoder_inputs.insert("position_ids", Value::from_array(position_ids.clone())?.into());

                    // Add past key values
                    for (key, value) in &past_key_values {
                        decoder_inputs.insert(key, Value::from_array(value.clone())?.into());
                    }

                    info!("SmolVLM: Running decoder step {}", step);
                    let decoder_outputs = decoder_session.run(decoder_inputs)?;

                    // Get logits (first output)
                    let logits_tensor = decoder_outputs[0].try_extract_tensor::<f32>()?;
                    let (logits_shape, logits_data) = logits_tensor;
                    let logits = Array3::from_shape_vec(
                        (logits_shape[0] as usize, logits_shape[1] as usize, logits_shape[2] as usize),
                        logits_data.to_vec(),
                    )?;
                    info!("SmolVLM: Decoder logits shape: {:?}", logits.shape());

                    // Get next token from last position
                    let last_idx = logits.shape()[1] - 1;
                    let logits_slice = logits.slice(ndarray::s![0, last_idx, ..]);

                    // Use argmax to get the most likely token
                    let next_token = logits_slice
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i as i64)
                        .unwrap();

                    generated_tokens.push(next_token as u32);

                    // Check for end-of-sequence token
                    if next_token == 2 {
                        info!("SmolVLM: End-of-sequence token reached at step {}", step);
                        break;
                    }

                    info!("SmolVLM: Generated token {} at step {}", next_token, step);

                    // Update inputs for next iteration (following working reference implementation)
                    input_ids = ndarray::Array::from_vec(vec![next_token]).into_shape_with_order((1, 1))?;

                    // Concatenate new attention mask (following working reference implementation)
                    let new_attention = ndarray::Array::<i64, _>::ones((1, 1));
                    let total_length = attention_mask.len() + new_attention.len();
                    let mut attention_vec = attention_mask.iter().copied().collect::<Vec<_>>();
                    attention_vec.extend(new_attention.iter().copied());
                    attention_mask = ndarray::Array::from_vec(attention_vec).into_shape_with_order((1, total_length))?;

                    // Update position_ids (following working reference implementation)
                    let current_pos = position_ids[[0, position_ids.shape()[1] - 1]] + 1;
                    position_ids = ndarray::Array::from_vec(vec![current_pos]).into_shape_with_order((1, 1))?;

                    // Update past key values from decoder outputs
                    for i in 0..num_hidden_layers {
                        let key = format!("past_key_values.{}.key", i);
                        let value = format!("past_key_values.{}.value", i);

                        if let Some(past_key) = past_key_values.get_mut(&key) {
                            if i * 2 + 1 < decoder_outputs.len() {
                                let present_key_tensor = decoder_outputs[i * 2 + 1].try_extract_tensor::<f32>()?;
                                let (shape, data) = present_key_tensor;
                                let present_key = ndarray::ArrayD::from_shape_vec(
                                    shape.iter().map(|&s| s as usize).collect::<Vec<_>>(),
                                    data.to_vec(),
                                )?;
                                *past_key = present_key;
                            }
                        }

                        if let Some(past_value) = past_key_values.get_mut(&value) {
                            if i * 2 + 2 < decoder_outputs.len() {
                                let present_value_tensor = decoder_outputs[i * 2 + 2].try_extract_tensor::<f32>()?;
                                let (shape, data) = present_value_tensor;
                                let present_value = ndarray::ArrayD::from_shape_vec(
                                    shape.iter().map(|&s| s as usize).collect::<Vec<_>>(),
                                    data.to_vec(),
                                )?;
                                *past_value = present_value;
                            }
                        }
                    }

                    if step >= 10 { // Generate at least 10 tokens for a meaningful response
                        break;
                    }
                }

                // Decode generated tokens back to text
                if !generated_tokens.is_empty() {
                    match tokenizer.decode(&generated_tokens, true) {
                        Ok(generated_text) => {
                            info!("SmolVLM: Generated text: {}", generated_text);
                            Ok(generated_text)
                        }
                        Err(e) => {
                            error!("SmolVLM: Failed to decode tokens: {}", e);
                            Ok(format!("Generated {} tokens but failed to decode", generated_tokens.len()))
                        }
                    }
                } else {
                    Ok("No tokens generated.".to_string())
                }
            } else {
                Ok("Error: Decoder model not loaded".to_string())
            }
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
        init_logger();
        info!("SmolVLM: nativeLoadModels called from JNI");

        let model_dir_str = match env.get_string(model_dir) {
            Ok(s) => s.to_string_lossy().to_string(),
            Err(e) => {
                error!("SmolVLM: Failed to get model_dir string: {:?}", e);
                return false as jboolean;
            }
        };

        let tokenizer_path_str = match env.get_string(tokenizer_path) {
            Ok(s) => s.to_string_lossy().to_string(),
            Err(e) => {
                error!("SmolVLM: Failed to get tokenizer_path string: {:?}", e);
                return false as jboolean;
            }
        };

        info!("SmolVLM: JNI parameters - model_dir: {}, tokenizer_path: {}", model_dir_str, tokenizer_path_str);

        let instance = get_or_create_instance();
        match instance.lock() {
            Ok(mut processor) => {
                match processor.load_models(&model_dir_str, &tokenizer_path_str) {
                    Ok(success) => {
                        info!("SmolVLM: load_models returned: {}", success);
                        success as jboolean
                    }
                    Err(e) => {
                        error!("SmolVLM: load_models failed with error: {}", e);
                        false as jboolean
                    }
                }
            }
            Err(e) => {
                error!("SmolVLM: Failed to lock processor: {:?}", e);
                false as jboolean
            }
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
            Ok(mut processor) => {
                match processor.generate_response(&prompt_str, max_tokens) {
                    Ok(result) => env.new_string(&result).unwrap().into_raw(),
                    Err(e) => env.new_string(&format!("Error: {}", e)).unwrap().into_raw(),
                }
            }
            Err(_) => env.new_string("Error: Failed to lock processor").unwrap().into_raw(),
        }
    }
}