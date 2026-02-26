# Building BaseweightSnap with Hexagon NPU Support

## Overview

This guide explains how to build BaseweightSnap with Qualcomm Hexagon NPU acceleration using the Snapdragon toolchain Docker image.

**Why Hexagon?**
- Dedicated NPU for AI inference on Snapdragon devices
- Significant performance boost over CPU-only inference
- Power efficiency for mobile/edge deployments

---

## Prerequisites

1. **Docker** installed on your build machine
2. **BaseweightSnap repo** cloned with submodules:
   ```bash
   git clone --recursive https://github.com/yourorg/BaseweightSnap.git
   # or if already cloned:
   git submodule update --init --recursive
   ```
3. **Android NDK** (r28b or later) - included in Docker image

---

## Architecture

```
BaseweightSnap/
├── app/src/main/cpp/
│   ├── llama.cpp/          ← Git submodule
│   ├── CMakeLists.txt      ← Modified for Hexagon
│   └── model_manager.cpp   ← JNI interface
└── ...
```

The Snapdragon Docker image provides:
- Android NDK r28b
- Hexagon SDK 6.4.0.2
- OpenCL SDK
- CMake
- Cross-compilation toolchain for ARM64 + Hexagon

---

## Build Process

### Step 1: Prepare the Build Environment

```bash
cd ~/BaseweightSnap

# Ensure llama.cpp submodule is up to date
cd app/src/main/cpp/llama.cpp
git pull origin master  # or specific tag
cd ../../../../../
```

### Step 2: Run the Snapdragon Toolchain Container

```bash
# Run the toolchain container with your source mounted
docker run -it \
  -u $(id -u):$(id -g) \
  --volume $(pwd):/workspace \
  --platform linux/amd64 \
  ghcr.io/snapdragon-toolchain/arm64-android:v0.3

# Inside the container:
[d]> cd /workspace/app/src/main/cpp/llama.cpp
```

### Step 3: Copy Snapdragon CMake Preset

```bash
# Inside the container:
[d]/workspace/app/src/main/cpp/llama.cpp> \
  cp docs/backend/snapdragon/CMakeUserPresets.json .
```

### Step 4: Build llama.cpp with Hexagon Support

```bash
# Configure with Snapdragon preset (includes Hexagon + OpenCL)
[d]/workspace/app/src/main/cpp/llama.cpp> \
  cmake --preset arm64-android-snapdragon-release \
  -B build-hexagon

# You should see:
# -- Including OpenCL backend
# -- Including Hexagon backend
# -- Hexagon SDK: /opt/hexagon/6.4.0.2

# Build
[d]/workspace/app/src/main/cpp/llama.cpp> \
  cmake --build build-hexagon -j$(nproc)
```

**Expected outputs:**
```
build-hexagon/ggml/src/ggml-hexagon/
├── libggml-hexagon.so      ← Hexagon backend
├── libggml-htp-v73.so      ← HTP (Hexagon Tensor Processor) v73
├── libggml-htp-v75.so      ← HTP v75
├── libggml-htp-v79.so      ← HTP v79
└── libggml-htp-v81.so      ← HTP v81
```

### Step 5: Package for Android

```bash
# Install to package directory
[d]/workspace/app/src/main/cpp/llama.cpp> \
  cmake --install build-hexagon --prefix pkg-hexagon/llama.cpp

# Verify the package
[d]/workspace/app/src/main/cpp/llama.cpp> \
  ls -la pkg-hexagon/llama.cpp/lib/
```

**Expected libraries:**
```
libggml.so              ← Core GGML
libggml-cpu.so          ← CPU backend
libggml-opencl.so       ← Adreno GPU backend
libggml-hexagon.so      ← Hexagon NPU backend
libggml-htp-v73.so      ← HTP implementations
libggml-htp-v75.so
libggml-htp-v79.so
libggml-htp-v81.so
libllama.so             ← Main llama library
```

### Step 6: Copy Libraries to BaseweightSnap

```bash
# Exit the container first (or open new terminal)
exit

# Copy Hexagon libraries to BaseweightSnap jniLibs
cd ~/BaseweightSnap

mkdir -p app/src/main/jniLibs/arm64-v8a

cp app/src/main/cpp/llama.cpp/pkg-hexagon/llama.cpp/lib/*.so \
   app/src/main/jniLibs/arm64-v8a/

# Verify
cd app/src/main/jniLibs/arm64-v8a
ls -la *.so
```

### Step 7: Update CMakeLists.txt for Hexagon

Modify `app/src/main/cpp/CMakeLists.txt` to link Hexagon libraries:

```cmake
cmake_minimum_required(VERSION 3.22.1)
project("baseweightsnap")

# Find all libraries
set(JNI_LIBS_DIR ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})

# Import llama.cpp libraries as imported targets
add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/libggml.so)

add_library(ggml-cpu SHARED IMPORTED)
set_target_properties(ggml-cpu PROPERTIES
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/libggml-cpu.so)

add_library(ggml-opencl SHARED IMPORTED)
set_target_properties(ggml-opencl PROPERTIES
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/libggml-opencl.so)

# Hexagon backend
add_library(ggml-hexagon SHARED IMPORTED)
set_target_properties(ggml-hexagon PROPERTIES
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/libggml-hexagon.so)

# HTP variants
add_library(ggml-htp-v73 SHARED IMPORTED)
set_target_properties(ggml-htp-v73 PROPERTIES
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/libggml-htp-v73.so)

add_library(ggml-htp-v75 SHARED IMPORTED)
set_target_properties(ggml-htp-v75 PROPERTIES
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/libggml-htp-v75.so)

add_library(ggml-htp-v79 SHARED IMPORTED)
set_target_properties(ggml-htp-v79 PROPERTIES
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/libggml-htp-v79.so)

add_library(ggml-htp-v81 SHARED IMPORTED)
set_target_properties(ggml-htp-v81 PROPERTIES
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/libggml-htp-v81.so)

add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/libllama.so)

# Main native library
add_library(model_manager SHARED model_manager.cpp)

# Link everything
target_link_libraries(model_manager
    llama
    ggml
    ggml-cpu
    ggml-opencl
    ggml-hexagon
    ggml-htp-v73
    ggml-htp-v75
    ggml-htp-v79
    ggml-htp-v81
    android
    log
    EGL
    GLESv3
    OpenCL  # For Adreno GPU fallback
)
```

### Step 8: Update ModelManager.kt for Hexagon

Modify `ModelManager.kt` to select Hexagon backend when available:

```kotlin
class ModelManager(private val context: Context) {
    
    // Initialize with Hexagon support
    init {
        System.loadLibrary("ggml")
        System.loadLibrary("ggml-cpu")
        System.loadLibrary("ggml-opencl")
        System.loadLibrary("ggml-hexagon")
        System.loadLibrary("ggml-htp-v73")
        System.loadLibrary("ggml-htp-v75")
        System.loadLibrary("ggml-htp-v79")
        System.loadLibrary("ggml-htp-v81")
        System.loadLibrary("llama")
        System.loadLibrary("model_manager")
    }
    
    fun loadModel(modelPath: String, useHexagon: Boolean = true): Long {
        val params = if (useHexagon && isHexagonAvailable()) {
            // Use Hexagon NPU with GPU fallback
            "--device HTP0 --n-gpu-layers 99"
        } else {
            // CPU/GPU only
            "--n-gpu-layers 50"
        }
        return nativeLoadModel(modelPath, params)
    }
    
    private fun isHexagonAvailable(): Boolean {
        // Check if running on Snapdragon with Hexagon
        return Build.HARDWARE.contains("qcom") || 
               Build.BOARD.lowercase().contains("snapdragon")
    }
    
    private external fun nativeLoadModel(path: String, params: String): Long
}
```

### Step 9: Build the Android APK

```bash
cd ~/BaseweightSnap

# Build release APK with Hexagon support
./gradlew assembleRelease

# Or debug build for testing
./gradlew assembleDebug
```

---

## Deployment to Device

### Test on Snapdragon Device

```bash
# Install the APK
adb install -r app/build/outputs/apk/release/app-release.apk

# Push a test model
adb push /path/to/model-Q4_0.gguf /sdcard/Android/data/ai.baseweight.baseweightsnap/files/models/

# Run and check logs for Hexagon initialization
adb logcat -s "BaseweightSnap" "LLAMA" "GGML" | grep -i hexagon
```

**Expected log output:**
```
GGML: Hexagon backend (experimental) : allocating new registry : ndev 1
GGML: Hexagon Arch version v79
GGML: allocating new session: HTP0
GGML: new session: HTP0 : session-id 0 domain-id 3
```

---

## Performance Tuning

### Environment Variables (Runtime)

Set these in your app before loading the model:

```kotlin
// Use multiple Hexagon sessions for large models (8B+)
ProcessBuilder()
    .environment()["GGML_HEXAGON_NDEV"] = "2"  // Use HTP0 and HTP1

// Enable verbose logging for debugging
ProcessBuilder()
    .environment()["GGML_HEXAGON_VERBOSE"] = "1"

// Control HVX threads (usually leave at default)
ProcessBuilder()
    .environment()["GGML_HEXAGON_NHVX"] = "0"  // 0 = use all
```

### Model Selection

| Model Size | Sessions | Device |
|------------|----------|--------|
| 1-3B | 1 | HTP0 |
| 4-7B | 2 | HTP0, HTP1 |
| 8-13B | 3-4 | HTP0-3 |
| 20B+ | 4 | HTP0-3 |

---

## Troubleshooting

### Issue: "Hexagon backend not found"
**Solution:** Ensure all `libggml-htp-*.so` files are in `jniLibs/arm64-v8a/`

### Issue: "Failed to load model on Hexagon"
**Solution:** Check model quantization — Hexagon works best with Q4_0, Q4_K_M, Q5_0

### Issue: "HTP session allocation failed"
**Solution:** Reduce model size or increase `GGML_HEXAGON_NDEV`

### Issue: Build fails in Docker
**Solution:** Ensure you're using `--platform linux/amd64` flag

---

## Automation Script

Create `scripts/build-hexagon.sh`:

```bash
#!/bin/bash
set -e

echo "Building BaseweightSnap with Hexagon support..."

# Run Docker build
docker run -it --rm \
  -u $(id -u):$(id -g) \
  -v $(pwd):/workspace \
  --platform linux/amd64 \
  ghcr.io/snapdragon-toolchain/arm64-android:v0.3 \
  bash -c "
    cd /workspace/app/src/main/cpp/llama.cpp
    cp docs/backend/snapdragon/CMakeUserPresets.json .
    cmake --preset arm64-android-snapdragon-release -B build-hexagon
    cmake --build build-hexagon -j\$(nproc)
    cmake --install build-hexagon --prefix pkg-hexagon/llama.cpp
  "

# Copy libraries
mkdir -p app/src/main/jniLibs/arm64-v8a
cp app/src/main/cpp/llama.cpp/pkg-hexagon/llama.cpp/lib/*.so \
   app/src/main/jniLibs/arm64-v8a/

echo "Hexagon libraries ready!"
echo "Now run: ./gradlew assembleRelease"
```

Make executable:
```bash
chmod +x scripts/build-hexagon.sh
./scripts/build-hexagon.sh
```

---

## References

- [Snapdragon Toolchain](https://github.com/snapdragon-toolchain)
- [llama.cpp Hexagon Backend](https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/snapdragon/README.md)
- [Hexagon SDK Documentation](https://developer.qualcomm.com/software/hexagon-dsp-sdk)
- [BaseweightSnap Repo](https://github.com/yourorg/BaseweightSnap)

---

*Last updated: 2026-02-23*
