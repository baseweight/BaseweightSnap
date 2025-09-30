# BaseweightSnap ONNX Migration Status

## Project Overview
Migration of BaseweightSnap Android app from llama.cpp to ONNX Runtime for running SmolVLM2-500M-Video-Instruct multimodal vision-language model inference.

## Current Status: Vision Encoder Working, Decoder Generation Loop Needs Past Key Values

The vision encoder is now working correctly, but the decoder is failing because it expects `past_key_values.0.key` inputs that we're not providing in the generation loop.

### Latest Error
```
Error: Non-zero status code returned while running Shape node. Name:'/model/attn_mask_reformat/past_key_subgraph/Shape' Status Message: Missing Input: past_key_values.0.key
```

## Working Components

### ✅ ONNX Runtime Infrastructure
- **Status**: WORKING
- ONNX Runtime 2.0.0-rc.10 with XNNPack execution provider
- Android NDK cross-compilation for arm64-v8a architecture
- JNI integration between Rust and Kotlin
- Model loading (3 second load time, all models load successfully)

### ✅ Vision Encoder
- **Status**: WORKING
- Successfully processes images and outputs vision features
- Uses working image processor from `/home/bowserj/rust_onnx_explore/smolvlm/src/image_processor.rs`
- Provides both `pixel_values` and `pixel_attention_mask` inputs
- Handles 5D tensor shapes `[1, 17, 3, 512, 512]` (17 patches = 4x4 grid + original)
- Converts attention mask to boolean: `pixel_attention_mask_bool = attention_mask.map(|&x| x != 0)`

### ✅ Text Embedder
- **Status**: WORKING
- Successfully converts text tokens to embeddings
- Properly handles multimodal embedding fusion (vision + text)

### ❌ Decoder Generation Loop
- **Status**: BROKEN - Missing past_key_values
- The decoder expects `past_key_values.{layer}.key` and `past_key_values.{layer}.value` inputs
- We need to implement the full generation loop from the working reference implementation

## Architecture

### Models
- `vision_encoder_q4.onnx` - Processes images into vision features
- `embed_tokens_q4.onnx` - Converts text tokens to embeddings
- `decoder_model_merged_q4.onnx` - Generates text tokens autoregressively

### File Structure
```
/home/bowserj/baseweight_download/BaseweightSnap/
├── smolvlm_rs_lib/                    # Rust library
│   ├── src/
│   │   ├── lib_simple.rs              # Main SmolVLM implementation (NEEDS FIXING)
│   │   └── image_processor.rs         # Working image processor (copied from reference)
│   └── Cargo.toml
├── app/src/main/
│   ├── java/ai/baseweight/baseweightsnap/
│   │   ├── MainActivity.kt            # Android UI
│   │   ├── SmolVLMAndroid.kt         # JNI wrapper
│   │   └── ModelManager.kt           # Model management
│   └── jniLibs/arm64-v8a/
│       └── libsmolvlm_snap.so        # Rust library
└── app/build.gradle.kts              # Android build config
```

### Reference Implementation
Working desktop implementation at `/home/bowserj/rust_onnx_explore/smolvlm/src/main.rs` that we should copy from.

## Key Technical Details

### Vision Encoder Inputs
```rust
inputs.insert("pixel_values", Value::from_array(processed_images.clone())?.into());
let pixel_attention_mask_bool = attention_mask.map(|&x| x != 0);
inputs.insert("pixel_attention_mask", Value::from_array(pixel_attention_mask_bool)?.into());
```

### Image Processing
- Uses 512x512 patch size (hardcoded throughout image processor)
- Creates 4x4 grid split + original image = 17 total patches
- Returns `(Array5<f32>, Array4<i64>)` for pixel_values and attention_mask

### Threading
- `processImage()` and `generateResponse()` both run on the same `runLoop` thread
- Both are suspend functions in Kotlin

## Next Steps to Fix

### 1. Fix Decoder Generation Loop
The working reference implementation (`/home/bowserj/rust_onnx_explore/smolvlm/src/main.rs`) shows how to:

1. **Auto-detect model configuration** from decoder inputs:
```rust
let decoder_inputs = &decoder_session.inputs;
let mut max_layer = 0;
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
```

2. **Initialize past_key_values** before generation loop:
```rust
let mut past_key_values: HashMap<String, Array<f32, _>> = HashMap::new();
for layer in 0..num_hidden_layers {
    let key_array = Array::zeros((batch_size, num_key_value_heads, 0, head_dim)).into_dyn();
    let value_array = Array::zeros((batch_size, num_key_value_heads, 0, head_dim)).into_dyn();
    past_key_values.insert(format!("past_key_values.{}.key", layer), key_array);
    past_key_values.insert(format!("past_key_values.{}.value", layer), value_array);
}
```

3. **Add past_key_values to each decoder call**:
```rust
for (key, value) in &past_key_values {
    decoder_inputs.insert(key, Value::from_array(value.clone())?.into());
}
```

4. **Update past_key_values after each decoder call**:
```rust
for i in 0..num_hidden_layers {
    let key = format!("past_key_values.{}.key", i);
    let value = format!("past_key_values.{}.value", i);

    if let Some(past_key) = past_key_values.get_mut(&key) {
        if i * 2 + 1 < decoder_outputs.len() {
            let present_key = decoder_outputs[i * 2 + 1].try_extract_tensor::<f32>()?.to_owned();
            *past_key = present_key.into_dyn();
        }
    }
    // Similar for values...
}
```

### 2. Build and Deploy Process
```bash
# 1. Build Rust library
cd /home/bowserj/baseweight_download/BaseweightSnap/smolvlm_rs_lib
ORT_LIB_LOCATION=/home/bowserj/ort/onnxruntime/build/Android/Debug cargo ndk -t arm64-v8a build --release

# 2. Copy to Android
cp target/aarch64-linux-android/release/libsmolvlm_snap.so ../app/src/main/jniLibs/arm64-v8a/

# 3. Build APK
cd /home/bowserj/baseweight_download/BaseweightSnap
./gradlew clean assembleDebug

# 4. Install
adb install app/build/outputs/apk/debug/app-debug.apk
```

### 3. Testing
```bash
# Launch app
adb shell monkey -p ai.baseweight.baseweightsnap -c android.intent.category.LAUNCHER 1

# Monitor logs
adb logcat | grep "SmolVLM"
```

## Model Configuration (Detected)
- SmolVLM2-500M-Video-Instruct
- Approximately 18+ layers (based on past_key_values inputs)
- 3 key-value heads
- 64 head dimension
- Vocabulary size: ~32000+ tokens

## Known Working Values
- Batch size: 1
- Image tensor shape: `[1, 17, 3, 512, 512]`
- Attention mask shape: `[1, 17, 512, 512]`
- Vision encoder outputs: image_features tensor
- EOS token: 2

## Dependencies
- ONNX Runtime 2.0.0-rc.10
- Android NDK
- Rust with `cargo-ndk`
- Android SDK API 28+

## Important: Copy Working Implementation
Don't reinvent the wheel - copy the working generation loop from `/home/bowserj/rust_onnx_explore/smolvlm/src/main.rs` lines 285-410 which handles past_key_values correctly.

## Debugging History

### Problems Solved
1. **Threading Issue**: Image processing happened on main thread while text generation happened on `runLoop` thread, causing image embeddings to be unavailable. Fixed by making `processImage()` a suspend function.

2. **Vision Encoder Input Rank Mismatch**: "Invalid rank for input: pixel_values Got: 4 Expected: 5". Fixed by copying working image processor with proper 5D tensor shapes.

3. **Vision Encoder Dimension Mismatch**: "Got: 384 Expected: 512" for dimensions 3 and 4. Fixed by hardcoding 512x512 patch sizes throughout the image processor.

4. **Missing Attention Mask Input**: "Missing Input: pixel_attention_mask". Fixed by using working implementation with boolean attention masks.

5. **Wrong Output Key Name**: Fixed by taking first output instead of looking for specific key name.

6. **Decoder Input Name Error**: "Invalid input name: hidden_states". Fixed by using correct input names: "inputs_embeds", "attention_mask", and "position_ids".

### Current Problem
7. **Missing Past Key Values**: "Missing Input: past_key_values.0.key" - The decoder generation loop needs to be completely rewritten to match the working reference implementation.