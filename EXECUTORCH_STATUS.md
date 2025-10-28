# ExecuTorch nanoVLM Implementation Status

## ✅ Completed

### 1. Build System
- **ExecuTorch v1.0.0** built successfully for Android arm64-v8a
  - Location: `/Users/bowserj/executorch_repos/executorch/cmake-out-android-arm64-v8a`
  - XNNPACK backend enabled with KleidiAI optimizations
  - 18 static libraries linked (~130MB total)
  - CMake integration complete

### 2. Rust Tokenizer
- **Cross-compilation** for Android working
  - Built for `aarch64-linux-android` target
  - NDK toolchain configured in `.cargo/config.toml`
  - FFI bindings tested and working
  - Library size: 13MB
  - Location: `rust-tokenizer/target/aarch64-linux-android/release/libnanovlm_preprocessor.a`

### 3. Native Library Integration
- **CMakeLists.txt** properly configured
  - All ExecuTorch libraries linked with `--whole-archive` for operator registration
  - Rust tokenizer linked
  - Image preprocessor integrated
  - Build produces 9.9MB `libbaseweightsnap.so`

### 4. Threading Architecture
- **Single-threaded execution** implemented
  - Dedicated "NanoVLM-RunLoop" thread created on startup
  - All native operations (load models, process image, generate) run on same thread
  - Kotlin coroutines with `withContext(runLoop)` for thread safety
  - Fixed crash from cross-thread ExecuTorch access

### 5. Module Loading
- **ExecuTorch v1.0.0 API** correctly implemented
  - Explicit `Module::load()` calls after construction
  - Using `Module::LoadMode::Mmap` (not `MmapUseMlock` to avoid OOM)
  - Error handling with `Error::Ok` checks
  - All 6 modules load successfully:
    - ✅ Vision encoder (vision_encoder.pte)
    - ✅ Modality projector (modality_projector.pte)
    - ✅ Prefill decoder (language_decoder_prefill.pte)
    - ✅ Decode decoder (language_decoder_decode.pte)
    - ✅ Token embedding (token_embedding.pte)
    - ✅ LM head (lm_head.pte)
  - ✅ Tokenizer (Rust FFI)

### 6. Application Structure
- **APK builds successfully**
  - Debug APK: 25MB
  - Native library: 9.9MB (includes ExecuTorch + XNNPACK + Rust)
  - Installs and launches without crashes
  - Models load in ~150ms

---

## 🚨 Current Blocker: Vision Encoder Freeze

### Problem
The vision encoder **hangs indefinitely** on the first `forward()` call during image processing:

```
10-20 19:58:39.109 I nanovlm-android: Running vision encoder on image 0: shape [1, 3, 512, 512], data range [0.000, 1.000]
[HANGS HERE - no further output]
```

**Code location**: `nanovlm_android.cpp:162`
```cpp
auto vision_result = vision_encoder_->forward(vision_inputs);  // <- FREEZES
```

### What We Know
1. ✅ Models load successfully (no errors)
2. ✅ Image preprocessing completes (17 tiles, 4x4 grid, 512x512)
3. ✅ Input tensor created correctly (shape [1, 3, 512, 512], data range [0.0, 1.0])
4. ❌ **Freeze happens during ExecuTorch forward pass**
5. ❌ No crash, no error - just infinite hang
6. ❌ Thread doesn't return (verified via logcat)

### Root Cause Analysis

The freeze is **NOT** due to:
- ❌ Threading issues (all on single dedicated thread)
- ❌ Memory issues (Mmap mode works, loads succeed)
- ❌ Module loading (all modules load without errors)
- ❌ Input data issues (tensor shape/range correct)

The freeze is **LIKELY** due to one of:

#### **Option 1: XNNPACK Delegate Not Active** (Most Likely)
The vision encoder model may not have XNNPACK delegate properly embedded:
- Models might be using **portable operators** instead of XNNPACK
- XNNPACK kernels may be deadlocking on first execution
- Delegate initialization may be incomplete

**Evidence**:
- CMakeLists.txt links XNNPACK libraries
- No XNNPACK initialization code in C++
- No verification that models actually use XNNPACK

**Fix Required**:
```bash
# Re-export vision encoder with XNNPACK delegate
python export_vision_encoder.py --use-xnnpack --num-threads=4
```

#### **Option 2: XNNPACK Threading Issue**
XNNPACK pthreadpool may not be initialized properly for Android:
- Default pthreadpool configuration might be incompatible
- Thread count might be misconfigured
- Threadpool might not be created

**Evidence**:
- ExecuTorch built with `extension_threadpool` support
- No explicit pthreadpool initialization in code
- Hang suggests waiting on thread synchronization

**Fix Required**:
```cpp
// In loadModels(), after loading vision encoder:
#include <executorch/extension/threadpool/threadpool.h>

// Initialize XNNPACK threadpool
auto threadpool = executorch::extension::threadpool::get_threadpool();
if (threadpool) {
    LOGi("XNNPACK threadpool initialized with %d threads", threadpool->num_threads());
}
```

#### **Option 3: Missing Operator Registration**
Some operators in the vision encoder might not be registered:
- XNNPACK ops might be missing from the build
- Custom ops might be needed but not linked
- Operator dispatch might be failing silently

**Evidence**:
- Using `--whole-archive` for operator libs
- All portable/custom ops linked
- But might be missing XNNPACK-specific ops

**Fix Required**:
```bash
# Check what ops the model uses
python -m executorch.exir.print_program vision_encoder.pte --ops-only
```

---

## 📋 TODO: Fix Vision Encoder Freeze

### Immediate Actions (Priority Order)

1. **Verify Model XNNPACK Export** ⭐ HIGHEST PRIORITY
   ```bash
   # Check model metadata
   python -c "
   import torch
   from executorch.exir import ExecutorchProgram
   prog = ExecutorchProgram.load('vision_encoder.pte')
   print('Delegates:', prog.program.backend_delegate)
   "

   # List operators
   python -m executorch.exir.print_program vision_encoder.pte --ops-only

   # If no XNNPACK delegate, re-export:
   python export_models.py \
       --model vision_encoder \
       --quantize \
       --use-xnnpack \
       --xnnpack-num-threads 4
   ```

2. **Add XNNPACK Initialization Code**
   ```cpp
   // In nanovlm_android.cpp, add to loadModels():

   #include <xnnpack.h>

   // Initialize XNNPACK (call once at startup)
   xnn_status status = xnn_initialize(nullptr);
   if (status != xnn_status_success) {
       LOGe("Failed to initialize XNNPACK: %d", status);
       return false;
   }
   LOGi("XNNPACK initialized successfully");
   ```

3. **Enable ExecuTorch Debug Logging**
   ```cpp
   // In CMakeLists.txt, change:
   add_definitions(-DET_LOG_ENABLED=1)  // Enable logging
   add_definitions(-DET_MIN_LOG_LEVEL=Debug)  // Verbose output

   // This will show what's happening during forward()
   ```

4. **Test with Portable Ops First** (Fallback)
   ```cpp
   // If XNNPACK is the issue, verify basic execution works:
   // Re-export model WITHOUT XNNPACK delegate
   // Should use portable CPU ops (slower but should work)
   ```

5. **Add Timeout & Logging**
   ```cpp
   // In processImageFromBuffer(), add detailed logging:
   LOGi("About to call vision encoder forward...");
   auto start = std::chrono::steady_clock::now();

   auto vision_result = vision_encoder_->forward(vision_inputs);

   auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
       std::chrono::steady_clock::now() - start).count();
   LOGi("Vision encoder forward completed in %ld ms", elapsed);
   ```

### Investigation Steps

1. **Check Model Files**
   - Verify `.pte` files are actually using XNNPACK delegate
   - Check file sizes (XNNPACK models should be larger due to delegate code)
   - Validate quantization format (XNNPACK requires specific quant format)

2. **Test Minimal Forward Pass**
   ```cpp
   // Create simple test in loadModels():
   std::vector<float> test_data(1 * 3 * 512 * 512, 0.5f);
   std::vector<int32_t> test_shape = {1, 3, 512, 512};
   auto test_tensor = from_blob(test_data.data(), test_shape, ScalarType::Float);

   LOGi("Testing vision encoder with dummy input...");
   auto test_result = vision_encoder_->forward({test_tensor});
   if (test_result.ok()) {
       LOGi("Vision encoder test forward succeeded!");
   } else {
       LOGe("Vision encoder test forward failed: %d", (int)test_result.error());
   }
   ```

3. **Verify XNNPACK Libraries**
   ```bash
   # Check if XNNPACK libs are actually in APK:
   unzip -l app/build/outputs/apk/debug/app-debug.apk | grep libbaseweightsnap

   # Check symbols in native library:
   aarch64-linux-android-nm libbaseweightsnap.so | grep xnn
   ```

---

## 🔧 Alternative Approaches (If Above Fails)

### Plan B: Use Portable Ops Only
If XNNPACK continues to cause issues:
1. Export all models **without** XNNPACK delegate
2. Use portable CPU operators
3. Accept slower performance (~5-10x slower)
4. Still functional, just not optimized

### Plan C: Split Vision Encoder
If vision encoder is too large:
1. Split into smaller chunks (patch-based processing)
2. Process image tiles one at a time
3. Aggregate results
4. Reduces memory pressure

### Plan D: Use PyTorch Mobile
If ExecuTorch proves too unstable:
1. Switch to PyTorch Mobile (Lite Interpreter)
2. Use TorchScript models instead of `.pte`
3. More mature, better documented
4. Larger binary size but more stable

---

## 📊 Current Build Configuration

### CMake Flags
```cmake
-DC10_USING_CUSTOM_GENERATED_MACROS
-DET_ENABLE_PROGRAM_VERIFICATION=0
-DET_LOG_ENABLED=0  # ⚠️ SHOULD ENABLE FOR DEBUGGING
-DET_MIN_LOG_LEVEL=Info  # ⚠️ SHOULD SET TO Debug
-march=armv8.4a+dotprod
```

### ExecuTorch Build
```bash
# Location: /Users/bowserj/executorch_repos/executorch
# Branch: v1.0.0
# Build dir: cmake-out-android-arm64-v8a
# Options:
EXECUTORCH_BUILD_XNNPACK=ON
EXECUTORCH_XNNPACK_ENABLE_KLEIDI=ON
EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE=ON
EXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON
```

### Linked Libraries (18 total)
```
libexecutorch.a (632K)
libexecutorch_core.a (2.7M)
libextension_module_static.a (1.5M)
libextension_tensor.a (1.1M)
libportable_ops_lib.a (1.7M)
libportable_kernels.a (89M)
libcustom_ops.a (2.8M)
liboptimized_native_cpu_ops_lib.a
liboptimized_kernels.a
libcpublas.a
libeigen_blas.a
libxnnpack_backend.a (3.7M)
libXNNPACK.a (13M)
libxnnpack-microkernels-prod.a (21M)
libcpuinfo.a
libpthreadpool.a
libkleidiai.a
libextension_threadpool.a
```

---

## 🎯 Next Steps (Recommended Order)

1. **Enable debug logging** in CMakeLists.txt → Rebuild → See what happens during forward()
2. **Verify XNNPACK delegate** in vision_encoder.pte
3. **Add XNNPACK initialization** code if delegate is present
4. **Re-export model** with XNNPACK if delegate is missing
5. **Test portable ops** as fallback if XNNPACK doesn't work
6. **Consider PyTorch Mobile** if ExecuTorch remains unstable

---

## 📝 Files Modified

### C++/Native
- `app/src/main/cpp/nanovlm_android.cpp` - Main inference engine
- `app/src/main/cpp/CMakeLists.txt` - Build configuration
- `app/src/main/cpp/image_preprocessor.cpp` - Image preprocessing
- `app/src/main/cpp/config_loader.h` - Config loading

### Kotlin/Android
- `app/src/main/java/ai/baseweight/baseweightsnap/NanoVLM_Android.kt` - JNI wrapper

### Rust
- `rust-tokenizer/src/lib.rs` - Tokenizer FFI
- `rust-tokenizer/.cargo/config.toml` - Android cross-compilation
- `rust-tokenizer/Cargo.toml` - Dependencies

---

## 🐛 Known Issues

1. **Vision encoder freezes on forward()** - BLOCKER
2. Models may not have XNNPACK delegate properly embedded
3. No error messages during freeze (silent hang)
4. No timeout mechanism implemented
5. Debug logging disabled (hard to diagnose)

---

## ✨ Success Criteria

- [ ] Vision encoder completes forward pass without freezing
- [ ] Image processing completes in <2 seconds per image
- [ ] Full inference pipeline works end-to-end
- [ ] Generates coherent text output
- [ ] Stable across multiple runs
- [ ] No memory leaks
- [ ] No crashes

---

## 📚 Resources

- ExecuTorch v1.0.0 docs: https://pytorch.org/executorch/stable/
- XNNPACK delegate: https://pytorch.org/executorch/stable/build-run-xnnpack.html
- Android integration: https://pytorch.org/executorch/stable/demo-apps-android.html
- Debugging guide: https://pytorch.org/executorch/stable/debugging.html

---

**Last Updated**: 2025-10-20
**Status**: 🔴 BLOCKED on vision encoder freeze
**Next Action**: Enable debug logging and verify XNNPACK delegate
