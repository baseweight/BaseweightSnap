# Native Build: Unified Single APK

The app ships **one APK** with all GPU/NPU backends and selects between them
at runtime (Settings → GPU Backend, or auto-detection).

## Backends

| Backend | How it's built | Target hardware |
|---------|----------------|-----------------|
| CPU | Statically, from the llama.cpp submodule | Fallback for all devices |
| Vulkan | Statically, from the llama.cpp submodule | Non-Qualcomm GPUs (e.g. Pixel/Mali) |
| OpenCL | Statically, from the llama.cpp submodule | Qualcomm Adreno GPUs |
| Hexagon | Prebuilt dynamic backend (`scripts/build-hexagon.sh`) | Snapdragon NPU |

## Runtime selection

1. `BackendDetector.kt` picks a backend: the user's saved preference from
   Settings, or auto-detection (`opencl` on Qualcomm SoCs, `vulkan` elsewhere).
2. `MTMD_Android.kt` passes the hint to `backend_init()` in
   `app/src/main/cpp/mtmd-android.cpp`, which:
   - sets `ADSP_LIBRARY_PATH` to the native lib dir (for Hexagon DSP skels);
   - disables unwanted statically linked backends via env vars read by the
     ggml backend registry (`GGML_DISABLE_VULKAN` upstream,
     `GGML_DISABLE_OPENCL` from `patches/llama.cpp/`);
   - for `hexagon`, loads `libggml-hexagon.so` with `ggml_backend_load()`.

Changing the backend in Settings takes effect on the next app start (the
backend registry is initialized once per process).

## llama.cpp patches

`patches/llama.cpp/*.patch` are applied to the submodule automatically at
CMake configure time (idempotent; see `app/src/main/cpp/CMakeLists.txt`).
Current patches:

- `0001-ggml-add-GGML_DISABLE_OPENCL-env-gate.patch` — runtime opt-out for
  the OpenCL backend, mirroring upstream's `GGML_DISABLE_VULKAN` gate.
  Candidate for upstreaming to ggml-org/llama.cpp.

## Hexagon prebuilt libraries

`app/hexagon-only-libs/` is **gitignored** but packaged into the APK via
`jniLibs.srcDirs`. It must contain, for `arm64-v8a`:

- `libggml-hexagon.so` — the dynamic ggml backend (its `libggml-base.so`
  dependency resolves against the copy the app build already packages)
- `libggml-htp-v{68,69,73,75,79,81}.so` — Hexagon DSP skels

Regenerate them after cloning or after bumping the llama.cpp submodule:

```bash
./scripts/build-hexagon.sh   # needs Docker; ~10-15 min
```

The script builds inside Qualcomm's Snapdragon toolchain image at the exact
submodule commit recorded by the superproject (dynamic backends are only
ABI-compatible with the ggml they were built from) and copies the libraries
into place. The CMake preset it uses is tracked at
`scripts/snapdragon/CMakeUserPresets.json`.

## Building the APK

```bash
./gradlew assembleDebug     # or assembleRelease
```

No flavor/variant juggling — the single `externalNativeBuild` compiles
llama.cpp with Vulkan + OpenCL, and the Hexagon libs ride along as prebuilts.
