# Native Library Variants

This project builds **two separate native library variants** to optimize performance based on device capabilities:

## Variants

### 1. CPU Variant (`libbaseweightsnap-cpu.so`)
- **Optimizations**: NEON, KleidiAI, ARM dotprod instructions
- **Use case**: Devices without Vulkan 1.2 support
- **Target**: ARMv8.4-a+ with dotprod

### 2. Vulkan Variant (`libbaseweightsnap-vulkan.so`)
- **Optimizations**: Vulkan GPU acceleration
- **Use case**: Devices with Vulkan 1.2+ support
- **Requirements**: Vulkan 1.2 hardware support

## Runtime Library Selection

The app automatically detects Vulkan support at runtime and loads the appropriate library:

1. **Detection** happens in `VulkanDetector.kt`:
   - Checks for `FEATURE_VULKAN_HARDWARE_LEVEL`
   - Verifies Vulkan 1.2+ version support

2. **Loading** happens in `MTMD_Android.kt`:
   - Calls `VulkanDetector.getLibraryName(context)`
   - Loads either `baseweightsnap-vulkan` or `baseweightsnap-cpu`

## Building

### Option 1: Build Script (Recommended)
```bash
./build-native-variants.sh
```

This script:
1. Builds the CPU variant with `BUILD_VARIANT=cpu`
2. Builds the Vulkan variant with `BUILD_VARIANT=vulkan`
3. Copies both `.so` files to `app/src/main/jniLibs/`

### Option 2: Manual Build

**CPU Variant:**
```bash
./gradlew clean
./gradlew :app:externalNativeBuildDebug -DBUILD_VARIANT=cpu
```

**Vulkan Variant:**
```bash
./gradlew clean
./gradlew :app:externalNativeBuildDebug -DBUILD_VARIANT=vulkan
```

Then manually copy the libraries from:
```
app/build/intermediates/cmake/debug/obj/arm64-v8a/
```
to:
```
app/src/main/jniLibs/arm64-v8a/
```

## CMake Configuration

The `CMakeLists.txt` uses a `BUILD_VARIANT` option to control which library is built:

- `BUILD_VARIANT=cpu`: Enables `GGML_CPU_KLEIDIAI`, disables `GGML_VULKAN`
- `BUILD_VARIANT=vulkan`: Enables `GGML_VULKAN`, disables `GGML_CPU_KLEIDIAI`

## Files Modified

- **app/src/main/cpp/CMakeLists.txt** - Conditional build logic
- **app/src/main/java/.../VulkanDetector.kt** - Runtime Vulkan detection
- **app/src/main/java/.../MTMD_Android.kt** - Dynamic library loading
- **app/build.gradle.kts** - Native build configuration
- **build-native-variants.sh** - Automated build script

## Testing

To test which variant is loaded on a device, check logcat:

```bash
adb logcat | grep VulkanDetector
```

You should see:
```
I/VulkanDetector: Vulkan 1.2+ support: true/false
I/VulkanDetector: Loading Vulkan-enabled library
# or
I/VulkanDetector: Loading CPU-optimized library (NEON/KleidiAI)
```

## Troubleshooting

**Both libraries must be present in the APK:**
- Ensure both `.so` files are in `app/src/main/jniLibs/arm64-v8a/`
- Check the APK contents: `unzip -l app-debug.apk | grep libbaseweightsnap`

**UnsatisfiedLinkError:**
- Verify library names match exactly
- Check that the correct ABI folder is used (arm64-v8a, x86_64, etc.)
- Ensure all dependencies (like `libc++_shared.so`) are available
